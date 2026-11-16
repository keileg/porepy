"""Configuration and recording interface for solver telemetry.

Telemetry is purely additive and off by default: a
:class:`NonlinearIterationRecorder` receives only values its callers have already
computed for their own purposes (residuals, convergence status, linear-solver status),
so recording never assembles the system. Beyond those values it evaluates only cheap
diagnostic operators: contact states, fracture properties and the state of the wells.

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast, runtime_checkable

import numpy as np

import porepy as pp
from porepy.numerics.solvers.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceStatusCollection,
)
from porepy.numerics.solvers.cycle_detection import ActiveSetHistory
from porepy.numerics.solvers.equation_variable_tags import EquationTag, VariableTag
from porepy.numerics.solvers.linear_solvers.linear_solver import (
    LinearSolverStatus,
    LinearSolverStatusSuccess,
)
from porepy.numerics.solvers.well_telemetry import (
    WellTopology,
    find_well_topology,
    well_state_fields,
)

# Reuses the numpy-aware JSON encoder SolverStatistics already relies on, rather than
# reimplementing it. This module is imported last by porepy.numerics.solvers.__init__
# (see the comment there) specifically so this import is safe: solver_statistics.py
# pulls in porepy.time_stepper.time_step_status, whose dataclass field annotations are
# evaluated eagerly against names this package only exposes after its own `import *`
# lines have run.
from porepy.viz.solver_statistics import _NumpyJSONEncoder

if TYPE_CHECKING:
    from porepy.time_stepper.time_step_status import TimeStepperStatus

__all__ = [
    "TelemetryConfig",
    "NonlinearIterationRecorder",
    "SCHEMA_VERSION",
    "JSONLWriter",
    "read_jsonl_records",
    "validate_record",
    "NonlinearTelemetryRecorder",
    "ContactState",
    "CONTACT_STATE_OPEN",
    "CONTACT_STATE_STICK",
    "CONTACT_STATE_SLIP",
    "evaluate_contact_state",
    "contact_state_counts",
    "contact_state_counts_by_subdomain",
    "FractureFlowProperties",
    "evaluate_fracture_flow_properties",
    "property_change_fields",
]


@dataclass
class TelemetryConfig:
    """Configuration for solver telemetry recording.

    Parameters:
        output_path: Where the telemetry record (JSONL and any sidecar files) is
            written.
        verbosity: 0 disables recording; 1 records scalar per-iteration fields
            (residual norms, convergence status, contact-state counts, ...); 2
            additionally writes full per-cell spatial fields to a sidecar file per
            iteration.
        equation_tags: Which equation blocks to record. An empty list (default) means
            every equation the model exposes -- the same "empty means all" convention
            already used by :class:`~porepy.numerics.solvers.newton_solver.
            NewtonSolver`'s own `equation_tags` parameter.
        variable_tags: As `equation_tags`, for variable blocks.
        cycle_window: Number of trailing iteration hashes kept for cycle/chattering
            detection. A cycle longer than this is not detected.
        flip_threshold: How many contact-state changes a fracture cell may accumulate
            within one nonlinear loop before being reported as chattering. A heuristic
            starting point rather than a tuned value.
        admissibility_bounds: Per-variable `(lower, upper)` bounds used for read-only
            admissibility counting. Keyed by variable name; a variable absent from
            this mapping is not checked.
        record_wells: Whether iteration records carry the state of every well, its
            junctions and their interfaces (see
            :mod:`~porepy.numerics.solvers.well_telemetry`). On by default: wells are a
            frequent source of nonlinear difficulty, and their state is invisible in
            the global fields. A model without wells records `null`.

    Raises:
        ValueError: If `verbosity` is not 0, 1, or 2; if `cycle_window` is not
            positive; if `flip_threshold` is negative; or if any
            `admissibility_bounds` entry has a lower bound exceeding its upper bound.

    """

    output_path: Path
    verbosity: int = 1
    equation_tags: list[EquationTag] = field(default_factory=list)
    variable_tags: list[VariableTag] = field(default_factory=list)
    cycle_window: int = 8
    flip_threshold: int = 2
    admissibility_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    record_wells: bool = True

    def __post_init__(self) -> None:
        if self.verbosity not in (0, 1, 2):
            raise ValueError(f"verbosity must be 0, 1, or 2, got {self.verbosity}.")
        if self.cycle_window <= 0:
            raise ValueError(f"cycle_window must be positive, got {self.cycle_window}.")
        if self.flip_threshold < 0:
            raise ValueError(
                f"flip_threshold must not be negative, got {self.flip_threshold}."
            )
        for name, (lower, upper) in self.admissibility_bounds.items():
            if lower > upper:
                raise ValueError(
                    f"admissibility_bounds[{name!r}] has lower bound {lower} greater "
                    f"than upper bound {upper}."
                )


@runtime_checkable
class NonlinearIterationRecorder(Protocol):
    """Interface for recording solver telemetry.

    Implementations must not assemble the system or solve anything: everything the
    solver already computed is passed in, and recomputing it would both cost a full
    assembly per iteration and risk recording a different state than the solver saw.
    Evaluating a cheap diagnostic operator is allowed -- the contact-state fields do
    exactly that, at two AD evaluations per iteration -- but that cost is charged to
    every iteration of every recorded run, so it stays the exception.

    `NewtonSolver`, `SequentialNonlinearSolver` and `TimeStepper` each hold an
    `Optional[NonlinearIterationRecorder]`, defaulting to `None`, in which case no
    method here is ever called.

    """

    def before_nonlinear_loop(self, model: pp.PorePyModel) -> None:
        """Reset any state scoped to a single nonlinear loop.

        Called once per `NewtonSolver.solve`/`SequentialNonlinearSolver.solve`
        invocation, before the first iteration -- for example to clear the previous
        loop's per-block residuals (needed for the reduction-factor field) or
        per-cell flip counters (needed for cycle detection).

        Parameters:
            model: The model instance the loop is being run on.

        """
        ...

    def record_iteration(
        self,
        model: pp.PorePyModel,
        iteration_index: int,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        equation_indexer: pp.ad.EquationIndexer,
        variable_indexer: pp.ad.VariableIndexer,
        linear_solver_status: LinearSolverStatus,
        convergence_status: ConvergenceStatusCollection,
        divergence_status: ConvergenceStatusCollection,
        convergence_info: ConvergenceInfoCollection,
    ) -> None:
        """Record telemetry for a single Newton iteration.

        Parameters:
            model: The model instance being solved.
            iteration_index: The Newton iteration counter (`NewtonSolver.
                iteration_index`).
            nonlinear_increment: The Newton step just taken.
            residual: Residual vector for this solver's active equations (`NewtonSolver.
                get_active_equations`), evaluated at the state after this iteration's
                increment. Not necessarily the full model's residual: a `NewtonSolver`
                used as a subsolver of a `SequentialNonlinearSolver` is scoped to a
                subset of the model's equations.
            equation_indexer: Indexer whose block offsets match `residual`'s layout --
                i.e. already restricted to this solver's active equations, in the same
                order `residual` was assembled in. Slicing per-block values must go
                through this indexer, not `model.equation_system.equation_indexer`,
                whenever this solver is scoped to fewer than all equations.
            variable_indexer: As `equation_indexer`, for the variables: its block
                offsets match `nonlinear_increment`'s layout.
            linear_solver_status: Status returned by the linear solve for this
                iteration.
            convergence_status: Per-criterion convergence status for this iteration.
            divergence_status: Per-criterion divergence status for this iteration.
            convergence_info: Norms and other convergence-criteria details for this
                iteration (e.g. `inc_abs`, `res_abs`).

        """
        ...

    def record_outer_iteration(
        self,
        model: pp.PorePyModel,
        iteration_index: int,
        nonlinear_increment: np.ndarray,
        full_residual: np.ndarray,
        equation_indexer: pp.ad.EquationIndexer,
        convergence_status: ConvergenceStatusCollection,
        divergence_status: ConvergenceStatusCollection,
        convergence_info: ConvergenceInfoCollection,
    ) -> None:
        """Record telemetry for a single outer (sequential-solver) iteration.

        Called from `SequentialNonlinearSolver.solve`'s own convergence check on the
        full system, structurally separate from any subsolver's `record_iteration`
        calls for its own inner iterations.

        Parameters:
            model: The model instance being solved.
            iteration_index: The outer iteration counter.
            nonlinear_increment: Change in the full solution vector over this outer
                iteration.
            full_residual: Residual of the full system (the union of all subsolvers'
                active equations, `SequentialNonlinearSolver.get_active_equations`) at
                this outer iteration.
            equation_indexer: Indexer whose block offsets match `full_residual`'s
                layout, analogous to `record_iteration`'s `equation_indexer`.
            convergence_status: Per-criterion convergence status for this outer
                iteration.
            divergence_status: Per-criterion divergence status for this outer
                iteration.
            convergence_info: Norms and other convergence-criteria details for this
                outer iteration.

        """
        ...

    def record_time_step(
        self, model: pp.PorePyModel, time_step_status: TimeStepperStatus
    ) -> None:
        """Record telemetry for a single time-step attempt.

        Called once per attempt, converged or not, from `TimeStepper.
        perform_time_step`.

        Parameters:
            model: The model instance being solved.
            time_step_status: The outcome of this time-step attempt.

        """
        ...


SCHEMA_VERSION = 2
"""Version of the telemetry JSONL record format produced by this module.

Bump on any field addition, removal, or type change to a record this module produces,
and record the change in `.claude/solver-telemetry/schema.md`. A reader can use a
record's own `schema_version` field to tell which shape to expect; `validate_record`
rejects any record whose `schema_version` is newer than this module supports.

"""

_RUN_RECORD_TYPE = "run"
"""`record_type` value for the once-per-run record (git commits, configuration hash).

See `validate_record`/`.claude/solver-telemetry/schema.md`. Other record types (per-
iteration, per-outer-iteration, per-time-step) are added here, and validated by
`validate_record`, as the commits producing them land.

"""

_DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE = 0.3
"""Fallback when a model doesn't set `params["constraint_violation_tolerance"]`.

Matches `ConstraintLineSearch.constraint_weights`'s own default for the same parameter:
`NonlinearTelemetryRecorder` reuses this same model parameter for its contact-state
near-transition counts, so the two agree on what counts as "close to a sign change" for
the same two indicator functions.

"""

CONTACT_STATE_OPEN = 0
"""Contact-state label for an open fracture cell."""
CONTACT_STATE_STICK = 1
"""Contact-state label for a closed, sticking fracture cell."""
CONTACT_STATE_SLIP = 2
"""Contact-state label for a closed, slipping fracture cell."""


@dataclass(frozen=True)
class ContactState:
    """The contact state of every fracture cell at one iterate.

    Parameters:
        labels: Per-cell state, one of `CONTACT_STATE_OPEN`/`_STICK`/`_SLIP`. Held as
            `int8` so its byte representation is fixed-width and platform-independent,
            which matters because it is hashed to identify the active set.
        opening: Raw `opening_indicator` values, kept for near-transition counting.
        sliding: Raw `sliding_indicator` values, kept for the same reason.

    """

    labels: np.ndarray
    opening: np.ndarray
    sliding: np.ndarray


class JSONLWriter:
    """Append-only writer for a telemetry record stream, one JSON object per line.

    Each write is flushed and `fsync`'d immediately, so a killed process leaves every
    already-written record durable on disk (at worst, the file's last line is a
    partially-written record; see `read_jsonl_records`).

    Parameters:
        path: File to append records to. Parent directory must already exist.

    """

    def __init__(self, path: Path) -> None:
        self.path = path
        """File this writer appends to."""
        self._file = open(path, "a", encoding="utf-8")

    def write_record(self, record: dict[str, Any]) -> None:
        """Append one record to the file.

        Parameters:
            record: A JSON-serializable mapping. Numpy scalars and arrays are
                supported (encoded the same way as `SolverStatistics`'s own output).

        """
        json.dump(record, self._file, cls=_NumpyJSONEncoder)
        self._file.write("\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    def __enter__(self) -> JSONLWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read a telemetry JSONL file, tolerating a truncated final line.

    A telemetry run may be killed mid-write (crash, out-of-memory, `SIGKILL`); the
    file's last line can then be a partial JSON object. Every complete line is parsed;
    a malformed final line is dropped rather than raising. A malformed line anywhere
    else in the file is not tolerated, since that indicates real corruption rather than
    an interrupted write.

    Parameters:
        path: JSONL file to read.

    Returns:
        One dict per complete line, in file order.

    Raises:
        json.JSONDecodeError: If a malformed line is found before the last one.

    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    records: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            if line_index == len(lines) - 1:
                continue
            raise
    return records


def validate_record(record: dict[str, Any]) -> None:
    """Validate a telemetry record's envelope, and its fields for known record types.

    Every record type must carry `schema_version` and `record_type`. Beyond that,
    only the "run" record type (`.claude/solver-telemetry/schema.md`) is validated
    field-by-field so far; other record types are extended here as the commits
    producing them land.

    Parameters:
        record: A single decoded JSONL line.

    Raises:
        ValueError: If a required envelope field is missing, a known field has the
            wrong type, or `schema_version` is newer than this module supports.

    """
    for key in ("schema_version", "record_type"):
        if key not in record:
            raise ValueError(f"Telemetry record is missing required field {key!r}.")
    if not isinstance(record["schema_version"], int):
        raise ValueError("schema_version must be an int.")
    if record["schema_version"] > SCHEMA_VERSION:
        raise ValueError(
            f"Record schema_version {record['schema_version']} is newer than this "
            f"module supports ({SCHEMA_VERSION})."
        )
    if record["record_type"] == _RUN_RECORD_TYPE:
        _validate_run_record(record)


def _validate_run_record(record: dict[str, Any]) -> None:
    """Validate the fields specific to a "run" record.

    See `.claude/solver-telemetry/schema.md` for the field-by-field description.

    """
    for key in ("porepy_commit", "pp_solvers_commit", "config_hash"):
        if key not in record:
            raise ValueError(f"Run record is missing required field {key!r}.")
    if not isinstance(record["porepy_commit"], str):
        raise ValueError("porepy_commit must be a str.")
    if record["pp_solvers_commit"] is not None and not isinstance(
        record["pp_solvers_commit"], str
    ):
        raise ValueError("pp_solvers_commit must be a str or None.")
    if not isinstance(record["config_hash"], str):
        raise ValueError("config_hash must be a str.")


class NonlinearTelemetryRecorder:
    """Concrete `NonlinearIterationRecorder`, writing JSONL to `config.output_path`.

    Covers the raw per-block residual fields (L-inf norm, reduction factor, argmax cell
    location), per-variable-block increment and iterate norms (both rely only on the
    equation/variable block structure every PorePy model exposes through
    `EquationOnDomain`/`Variable`/`EquationIndexer`), linear-solver status,
    admissibility counts (config-declared bounds, pure counting), the range and largest
    per-iteration change of fracture aperture and permeability for a fractured model,
    the state of every well, its junctions and interfaces for a model with wells,
    and three duck-typed fields specific to contact-mechanics models: contact-state
    counts, active-set cycle/chattering detection, and line-search telemetry. A model
    that doesn't expose the corresponding hook simply gets `null` for that field,
    rather than an error. Level-2 spatial fields (`.npz` sidecar) are added by a later
    commit.

    Important:
        The contact-state and cycle fields need `opening_indicator`/
        `sliding_indicator`, which come from
        :class:`~porepy.models.solution_strategy.ContactIndicators`. That is an opt-in
        mixin: no standard model (`MomentumBalance`, `Poromechanics`,
        `Thermoporomechanics`) includes it, so on a stock contact model both fields
        are `null` until it is mixed in, as the shipped examples that use the
        constraint line search do. Likewise, the line-search field only appears when
        the solve actually runs `ConstraintLineSearch`, which writes the
        `constraint_weights` this reads.

    Parameters:
        config: Telemetry configuration.

    """

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        """Telemetry configuration."""
        self._writer = JSONLWriter(config.output_path)
        # Inner and outer state is kept separate, never shared. One recorder instance
        # is meant to serve both a SequentialNonlinearSolver and its subsolvers, and
        # before_nonlinear_loop fires on *every* NewtonSolver.solve() call, including
        # each subsolver's -- i.e. several times per outer iteration, not once per
        # outer loop. Resetting shared state there would wipe the outer loop's own
        # history out from under it every time any subsolver starts a fresh inner
        # loop. So: inner state is reset by before_nonlinear_loop, as its name
        # promises; outer state is reset by record_outer_iteration itself, on that
        # call's own first iteration.
        self._previous_inner_block_residual_norms: dict[str, float] = {}
        self._previous_outer_block_residual_norms: dict[str, float] = {}
        self._inner_active_sets = ActiveSetHistory(
            window=config.cycle_window, flip_threshold=config.flip_threshold
        )
        # Fracture aperture and permeability at the previous inner iteration, or at the
        # start of the loop before its first iteration.
        self._previous_fracture_flow: Optional[FractureFlowProperties] = None
        # The wells of the md-grid they were found in. Looked up again when the model
        # gets a new md-grid, as it does each time it is prepared anew.
        self._well_topology: tuple[
            Optional[pp.MixedDimensionalGrid], Optional[WellTopology]
        ] = (None, None)
        self._outer_active_sets = ActiveSetHistory(
            window=config.cycle_window, flip_threshold=config.flip_threshold
        )

    def before_nonlinear_loop(self, model: pp.PorePyModel) -> None:
        """Reset the state used for inner reduction factors, cycle detection and
        fracture property changes (`record_iteration`). Called once per
        `NewtonSolver.solve()`, which for a subsolver of `SequentialNonlinearSolver` is
        once per outer iteration, not once per outer loop -- see `__init__`.

        """
        self._previous_inner_block_residual_norms = {}
        self._inner_active_sets.reset()
        self._previous_fracture_flow = evaluate_fracture_flow_properties(model)

    def record_iteration(
        self,
        model: pp.PorePyModel,
        iteration_index: int,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        equation_indexer: pp.ad.EquationIndexer,
        variable_indexer: pp.ad.VariableIndexer,
        linear_solver_status: LinearSolverStatus,
        convergence_status: ConvergenceStatusCollection,
        divergence_status: ConvergenceStatusCollection,
        convergence_info: ConvergenceInfoCollection,
    ) -> None:
        """Write one "iteration" record. See `NonlinearIterationRecorder`."""
        if self.config.verbosity == 0:
            return
        contact_state = evaluate_contact_state(model)
        self._writer.write_record(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "iteration",
                "iteration_index": iteration_index,
                "convergence_status": _stringify_status(convergence_status),
                "divergence_status": _stringify_status(divergence_status),
                "convergence_info": dict(convergence_info),
                "linear_solver": _linear_solver_fields(linear_solver_status),
                "blocks": self._block_residual_fields(
                    model,
                    residual,
                    equation_indexer,
                    self._previous_inner_block_residual_norms,
                ),
                "increments": self._increment_fields(
                    model, nonlinear_increment, variable_indexer
                ),
                "admissibility": self._admissibility_counts(model),
                "contact_state": self._contact_state_fields(model, contact_state),
                "cycle": _cycle_fields(self._inner_active_sets, contact_state),
                "fracture_flow": self._fracture_flow_fields(model),
                "line_search": self._line_search_fields(model),
                "wells": self._well_fields(
                    model,
                    nonlinear_increment,
                    residual,
                    equation_indexer,
                    variable_indexer,
                ),
            }
        )

    def _well_fields(
        self,
        model: pp.PorePyModel,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        equation_indexer: pp.ad.EquationIndexer,
        variable_indexer: pp.ad.VariableIndexer,
    ) -> Optional[dict[str, list[dict[str, Any]]]]:
        """State of the wells, their junctions and interfaces; `None` if recording
        them is switched off or the model has no wells.

        Costs a few operator evaluations on the well grids per iteration.

        """
        if not self.config.record_wells:
            return None
        mdg, topology = self._well_topology
        if mdg is not model.mdg:
            topology = find_well_topology(model)
            self._well_topology = (model.mdg, topology)
        if topology is None:
            return None
        return well_state_fields(
            model,
            topology,
            nonlinear_increment,
            residual,
            equation_indexer,
            variable_indexer,
        )

    def _fracture_flow_fields(
        self, model: pp.PorePyModel
    ) -> Optional[dict[str, Optional[dict[str, Any]]]]:
        """Fracture aperture and permeability, and their largest change since the
        previous iteration; `None` for a model without fractures.

        These trace the shear dilation -> aperture -> permeability coupling, which can
        amplify a Newton step or, when the flux discretisation lags it, make Newton
        converge linearly. Costs two operator evaluations per iteration.

        """
        current = evaluate_fracture_flow_properties(model)
        if current is None:
            return None
        previous = self._previous_fracture_flow
        self._previous_fracture_flow = current
        fractures = model.mdg.subdomains(dim=model.nd - 1)
        return {
            "aperture": property_change_fields(
                current.aperture,
                None if previous is None else previous.aperture,
                fractures,
            ),
            "permeability": (
                None
                if current.permeability is None
                else property_change_fields(
                    current.permeability,
                    None if previous is None else previous.permeability,
                    fractures,
                )
            ),
        }

    def _contact_state_fields(
        self, model: pp.PorePyModel, state: Optional[ContactState]
    ) -> Optional[dict[str, Any]]:
        """Contact-state counts, overall and per fracture, or `None` for a model
        without fracture contact mechanics.

        """
        if state is None:
            return None
        tolerance = model.params.get(
            "constraint_violation_tolerance", _DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE
        )
        fields: dict[str, Any] = dict(contact_state_counts(state, tolerance))
        fields["by_subdomain"] = contact_state_counts_by_subdomain(
            state, model.mdg.subdomains(dim=model.nd - 1)
        )
        return fields

    def record_outer_iteration(
        self,
        model: pp.PorePyModel,
        iteration_index: int,
        nonlinear_increment: np.ndarray,
        full_residual: np.ndarray,
        equation_indexer: pp.ad.EquationIndexer,
        convergence_status: ConvergenceStatusCollection,
        divergence_status: ConvergenceStatusCollection,
        convergence_info: ConvergenceInfoCollection,
    ) -> None:
        """Write one "outer_iteration" record. See `NonlinearIterationRecorder`.

        Resets the outer reduction-factor and cycle state on `iteration_index == 1`
        (the first outer iteration of a fresh `SequentialNonlinearSolver.solve()` call,
        matching the off-by-one `iteration_index + 1` convention already used there)
        instead of relying on `before_nonlinear_loop`, since that hook also fires once
        per subsolver's own inner loop -- see `__init__`.

        """
        if self.config.verbosity == 0:
            return
        if iteration_index == 1:
            self._previous_outer_block_residual_norms = {}
            self._outer_active_sets.reset()
        contact_state = evaluate_contact_state(model)
        self._writer.write_record(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "outer_iteration",
                "iteration_index": iteration_index,
                "convergence_status": _stringify_status(convergence_status),
                "divergence_status": _stringify_status(divergence_status),
                "convergence_info": dict(convergence_info),
                "blocks": self._block_residual_fields(
                    model,
                    full_residual,
                    equation_indexer,
                    self._previous_outer_block_residual_norms,
                ),
                "admissibility": self._admissibility_counts(model),
                "contact_state": self._contact_state_fields(model, contact_state),
                "cycle": _cycle_fields(self._outer_active_sets, contact_state),
            }
        )

    def record_time_step(
        self, model: pp.PorePyModel, time_step_status: TimeStepperStatus
    ) -> None:
        """Write one "time_step" record. See `NonlinearIterationRecorder`."""
        if self.config.verbosity == 0:
            return
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "time_step",
            "status": time_step_status.serialize(),
        }
        # Duck-typed: TimeStepperStatus's three subclasses each carry a different
        # subset of these fields, none declared on the (never-subclassed-elsewhere)
        # abstract base.
        for optional_field in ("dt", "time", "reason"):
            value = getattr(time_step_status, optional_field, None)
            if value is not None:
                record[optional_field] = value
        attempts = getattr(time_step_status, "attempts", None)
        if attempts:
            record["num_attempts"] = len(attempts)
            record["nonlinear_solver_status"] = attempts[
                -1
            ].nonlinear_solve_status.serialize()
        self._writer.write_record(record)

    def _block_residual_fields(
        self,
        model: pp.PorePyModel,
        residual: np.ndarray,
        equation_indexer: pp.ad.EquationIndexer,
        previous_norms: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        """Per-equation-block residual telemetry: L-inf norm, reduction factor, and
        the cell location of the largest-magnitude entry.

        Parameters:
            model: The model being solved.
            residual: Residual vector matching `equation_indexer`'s layout.
            equation_indexer: Indexer scoped to `residual` (see `record_iteration`).
            previous_norms: The caller's own previous-iteration L-inf norms, by block
                name, updated in place. Callers with different iteration cadences
                (inner vs. outer) must pass different dicts -- see `__init__`.

        """
        if self.config.equation_tags:
            selected, _ = equation_indexer.filter_by_tags(
                tags=self.config.equation_tags, model=model
            )
            selected_equations = set(selected)
        else:
            selected_equations = set(equation_indexer.indices)

        domains_by_name: dict[str, dict[pp.GridLike, np.ndarray]] = {}
        for equation, indices in equation_indexer.indices.items():
            if equation not in selected_equations or indices.size == 0:
                continue
            domains_by_name.setdefault(equation.name, {})[equation.domain] = indices

        fields: dict[str, dict[str, Any]] = {}
        for name, domains_to_indices in domains_by_name.items():
            block_indices = np.concatenate(list(domains_to_indices.values()))
            block_residual = residual[block_indices]
            norm = float(np.max(np.abs(block_residual)))

            previous_norm = previous_norms.get(name)
            reduction_factor = (
                norm / previous_norm
                if previous_norm is not None and previous_norm != 0.0
                else None
            )
            previous_norms[name] = norm

            fields[name] = {
                "residual_inf_norm": norm,
                "reduction_factor": reduction_factor,
                "argmax_location": _argmax_location(
                    domains_to_indices,
                    block_residual,
                    model.equation_system.equations[name].target.dof_info.cells,
                ),
            }
        return fields

    def _increment_fields(
        self,
        model: pp.PorePyModel,
        nonlinear_increment: np.ndarray,
        variable_indexer: pp.ad.VariableIndexer,
    ) -> dict[str, dict[str, Any]]:
        """Per-variable-block L-inf norms of the increment and of the updated iterate,
        and the cell of the largest increment.

        Successive increment norms of a block give its observed convergence rate, and
        unlike residual norms they are not held up by a round-off floor until the
        increment itself reaches round-off relative to the iterate, which the second
        norm makes visible. The location tells which part of the model the Newton step
        moves most, which the residual locations do not: a cell can be balanced while
        its value swings.

        Parameters:
            model: The model being solved, holding the iterate after the increment.
            nonlinear_increment: The increment, laid out as `variable_indexer`.
            variable_indexer: Indexer whose block offsets match `nonlinear_increment`.

        """
        if self.config.variable_tags:
            selected, _ = variable_indexer.filter_by_tags(
                tags=self.config.variable_tags, model=model
            )
            selected_variables = set(selected)
        else:
            selected_variables = set(variable_indexer.indices)

        iterate = model.equation_system.get_variable_values(iterate_index=0)
        full_indices = model.equation_system.variable_indexer.indices
        increment_indices: dict[str, dict[pp.GridLike, np.ndarray]] = {}
        iterate_parts: dict[str, list[np.ndarray]] = {}
        num_components: dict[str, int] = {}
        for variable, indices in variable_indexer.indices.items():
            if variable not in selected_variables or indices.size == 0:
                continue
            increment_indices.setdefault(variable.name, {})[variable.domain] = indices
            iterate_parts.setdefault(variable.name, []).append(
                iterate[full_indices[variable]]
            )
            num_components[variable.name] = variable.target.dof_info.cells

        fields: dict[str, dict[str, Any]] = {}
        for name, domains_to_indices in increment_indices.items():
            block_increment = nonlinear_increment[
                np.concatenate(list(domains_to_indices.values()))
            ]
            fields[name] = {
                "increment_inf_norm": float(np.max(np.abs(block_increment))),
                "value_inf_norm": float(
                    np.max(np.abs(np.concatenate(iterate_parts[name])))
                ),
                "argmax_location": _argmax_location(
                    domains_to_indices, block_increment, num_components[name]
                ),
            }
        return fields

    def _admissibility_counts(self, model: pp.PorePyModel) -> dict[str, dict[str, int]]:
        """Count, per configured variable, how many entries fall outside its bounds.

        Read-only counting: `TelemetryConfig.admissibility_bounds` declares the bounds,
        this only reports how many degrees of freedom violate them. Acting on a
        violation (clamping, relaxing) is D1 territory, not this recorder's job.

        """
        if not self.config.admissibility_bounds:
            return {}

        all_values = model.equation_system.get_variable_values(iterate_index=0)
        variable_groups = model.equation_system.variable_indexer.group_by_name()

        counts: dict[str, dict[str, int]] = {}
        for name, (lower, upper) in self.config.admissibility_bounds.items():
            domains_to_indices = variable_groups.get(name)
            if not domains_to_indices:
                raise ValueError(
                    f"admissibility_bounds declares a bound for variable {name!r}, "
                    "but the model has no such variable."
                )
            indices = np.concatenate(list(domains_to_indices.values()))
            values = all_values[indices]
            counts[name] = {
                "num_below": int(np.sum(values < lower)),
                "num_above": int(np.sum(values > upper)),
                "num_total": int(values.size),
            }
        return counts

    def _line_search_fields(self, model: pp.PorePyModel) -> dict[str, float] | None:
        """Duck-typed line-search telemetry: the accepted step's relaxation factor.

        `None` if the model has no `constraint_weights` in its subdomain data, i.e.
        this solve isn't using `ConstraintLineSearch`. Per the discovery report, "alpha"
        is the accepted per-DOF relaxation vector reduced to a scalar (its min) --
        not read off the solver as a single existing float, since none exists.

        """
        weight_arrays = [
            model.mdg.subdomain_data(sd)["constraint_weights"]
            for sd in model.mdg.subdomains()
            if "constraint_weights" in model.mdg.subdomain_data(sd)
        ]
        if not weight_arrays:
            return None
        return {"alpha": float(min(np.min(weights) for weights in weight_arrays))}


def evaluate_contact_state(model: pp.PorePyModel) -> Optional[ContactState]:
    """Evaluate the contact state of every fracture cell, once.

    Duck-typed: `None` if the model doesn't expose `opening_indicator`/
    `sliding_indicator` (`ContactIndicators`, or an equivalent) or has no fracture
    subdomains. Costs exactly two AD evaluations, so callers that need both
    state counts and the labels must share one result rather than calling twice.

    Parameters:
        model: The model whose current iterate is to be classified.

    Returns:
        The evaluated state, or `None` for a model without fracture contact
        mechanics.

    """
    if not (
        hasattr(model, "opening_indicator") and hasattr(model, "sliding_indicator")
    ):
        return None
    fracture_subdomains = model.mdg.subdomains(dim=model.nd - 1)
    if not fracture_subdomains:
        return None

    opening = model.equation_system.evaluate(
        model.opening_indicator(fracture_subdomains)
    )
    sliding = model.equation_system.evaluate(
        model.sliding_indicator(fracture_subdomains)
    )

    is_open = opening < 0
    is_slip = ~is_open & (sliding >= 0)
    labels = np.full(opening.size, CONTACT_STATE_STICK, dtype=np.int8)
    labels[is_open] = CONTACT_STATE_OPEN
    labels[is_slip] = CONTACT_STATE_SLIP

    return ContactState(labels=labels, opening=opening, sliding=sliding)


def contact_state_counts(state: ContactState, tolerance: float) -> dict[str, int]:
    """Count fracture cells per contact state, and cells close to a transition.

    Parameters:
        state: Evaluated contact state, from :func:`evaluate_contact_state`.
        tolerance: How close to zero an indicator must be for its cell to count as
            near a transition. Does not affect the state classification itself.

    Returns:
        Counts per state, the total, and the two near-transition counts.

    """
    return {
        "num_open": int(np.sum(state.labels == CONTACT_STATE_OPEN)),
        "num_stick": int(np.sum(state.labels == CONTACT_STATE_STICK)),
        "num_slip": int(np.sum(state.labels == CONTACT_STATE_SLIP)),
        "num_total": int(state.labels.size),
        "num_near_opening_transition": int(np.sum(np.abs(state.opening) < tolerance)),
        "num_near_sliding_transition": int(np.sum(np.abs(state.sliding) < tolerance)),
    }


@dataclass
class FractureFlowProperties:
    """Cell-wise fracture properties coupling deformation to flow."""

    aperture: np.ndarray
    """Aperture of every fracture cell."""
    permeability: Optional[np.ndarray]
    """Largest component of every fracture cell's permeability tensor, or `None` for a
    model without fluid flow."""


def evaluate_fracture_flow_properties(
    model: pp.PorePyModel,
) -> Optional[FractureFlowProperties]:
    """Evaluate aperture and permeability on all fracture subdomains.

    Parameters:
        model: The model whose current iterate is evaluated.

    Returns:
        The properties, or `None` for a model without fractures or without an
        aperture.

    """
    fractures = model.mdg.subdomains(dim=model.nd - 1)
    if not fractures or not hasattr(model, "aperture"):
        return None
    aperture = cast(
        np.ndarray, model.equation_system.evaluate(model.aperture(fractures))
    )
    permeability = None
    if hasattr(model, "permeability"):
        # Permeability evaluates to a full tensor per cell, nine entries each.
        tensor = cast(
            np.ndarray, model.equation_system.evaluate(model.permeability(fractures))
        )
        permeability = tensor.reshape(9, -1, order="F").max(axis=0)
    return FractureFlowProperties(aperture=aperture, permeability=permeability)


def property_change_fields(
    current: np.ndarray, previous: Optional[np.ndarray], subdomains: list[pp.Grid]
) -> dict[str, Any]:
    """Range of a cell-wise fracture property, and the largest factor by which any
    cell's value changed since `previous`.

    The change is measured as a factor, `max(new / old, old / new)`, so that a doubling
    and a halving count alike and the measure is independent of units. Cells where
    either value is not positive have no such factor and are left out.

    Parameters:
        current: The property on every cell of `subdomains`, in order.
        previous: The same at the previous iteration, or `None` if there is none.
        subdomains: The fracture subdomains the values are laid out on.

    Returns:
        `min` and `max` of `current`; `max_change_factor` and `max_change_location`
        (dim, domain id, cell), both `None` when no factor can be formed.

    """
    fields: dict[str, Any] = {
        "min": float(np.min(current)),
        "max": float(np.max(current)),
        "max_change_factor": None,
        "max_change_location": None,
    }
    if previous is None:
        return fields
    valid = (current > 0) & (previous > 0)
    if not np.any(valid):
        return fields
    ratio = np.ones_like(current)
    ratio[valid] = current[valid] / previous[valid]
    factor = np.maximum(ratio, 1.0 / ratio)
    cell = int(np.argmax(np.where(valid, factor, 0.0)))
    fields["max_change_factor"] = float(factor[cell])
    fields["max_change_location"] = _cell_location(cell, subdomains)
    return fields


def _cell_location(cell: int, subdomains: list[pp.Grid]) -> dict[str, int]:
    """Locate a cell given by its index into values laid out over `subdomains`."""
    offset = 0
    for subdomain in subdomains:
        if cell < offset + subdomain.num_cells:
            return {
                "dim": subdomain.dim,
                "domain_id": subdomain.id,
                "cell": cell - offset,
            }
        offset += subdomain.num_cells
    raise ValueError(f"Cell {cell} lies beyond the given subdomains.")


def contact_state_counts_by_subdomain(
    state: ContactState, subdomains: list[pp.Grid]
) -> dict[str, dict[str, int]]:
    """Count fracture cells per contact state, separately for each fracture.

    Overall counts cannot say which fracture is slipping, and on a model with several
    fractures the difficult region is usually one of them.

    Parameters:
        state: Evaluated contact state, from :func:`evaluate_contact_state`.
        subdomains: The fracture subdomains `state` was evaluated on, in the same
            order.

    Returns:
        Open, stick and slip counts, keyed by the subdomain id as a string (the form a
        JSON object key takes anyway).

    """
    counts: dict[str, dict[str, int]] = {}
    offset = 0
    for subdomain in subdomains:
        labels = state.labels[offset : offset + subdomain.num_cells]
        offset += subdomain.num_cells
        counts[str(subdomain.id)] = {
            "num_open": int(np.sum(labels == CONTACT_STATE_OPEN)),
            "num_stick": int(np.sum(labels == CONTACT_STATE_STICK)),
            "num_slip": int(np.sum(labels == CONTACT_STATE_SLIP)),
        }
    return counts


def _cycle_fields(
    history: ActiveSetHistory, state: Optional[ContactState]
) -> Optional[dict[str, Any]]:
    """Advance the active-set history by one iteration and report what it found.

    `None` for a model without fracture contact mechanics, where there is no active
    set to track.

    """
    if state is None:
        return None
    detection = history.record(state.labels)
    return {
        "active_set_hash": detection.active_set_hash,
        "repeat_length": detection.repeat_length,
        "cycling_cells": detection.cycling_cells,
        "num_flipped_cells": detection.num_flipped_cells,
        "chattering_cells": detection.chattering_cells,
    }


def _argmax_location(
    domains_to_indices: dict[pp.GridLike, np.ndarray],
    block_values: np.ndarray,
    num_components: int,
) -> dict[str, Any] | None:
    """Locate the largest-magnitude entry of one block's values, as a cell.

    Parameters:
        domains_to_indices: The block's index arrays per domain, in the order
            `block_values` concatenates them.
        block_values: The block's values, a residual or an increment.
        num_components: Degrees of freedom per cell of the block. Zero or less if the
            block is not cell-centered (e.g. defined on faces/nodes), which
            `pp.ad.dof_to_cell` does not apply to.

    Returns:
        The domain's dimension and id, and the cell; `None` for a block that is not
        cell-centered.

    """
    if num_components <= 0:
        return None

    argmax_local = int(np.argmax(np.abs(block_values)))
    offset = 0
    for domain, indices in domains_to_indices.items():
        if argmax_local < offset + indices.size:
            local_in_domain = argmax_local - offset
            return {
                "dim": domain.dim,
                "domain_id": domain.id,
                "cell": pp.ad.dof_to_cell(local_in_domain, domain, num_components),
            }
        offset += indices.size
    raise AssertionError(
        "argmax index not found in any domain; block_indices/domains_to_indices "
        "are inconsistent."
    )


def _stringify_status(status: ConvergenceStatusCollection) -> dict[str, str]:
    return {key: str(value) for key, value in status.items()}


def _linear_solver_fields(status: LinearSolverStatus) -> dict[str, Any]:
    """Success flag, and the solve time or why the solve fell short.

    Any status other than success is recorded as unsuccessful, with its class name as
    ``outcome``, rather than checked against a fixed set of classes: a status class the
    recorder does not know must not end the run it observes.

    """
    if isinstance(status, LinearSolverStatusSuccess):
        return {"success": True, "solve_time": status.solve_time}
    return {
        "success": False,
        "outcome": type(status).__name__,
        "reason": getattr(status, "reason", None),
    }
