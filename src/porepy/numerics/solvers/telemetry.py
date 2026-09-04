"""Configuration and recording interface for solver telemetry.

Telemetry is purely additive and off by default: a
:class:`NonlinearIterationRecorder` receives only values its callers have already
computed for their own purposes (residuals, convergence status, linear-solver status),
so recording must never trigger additional AD evaluation or system assembly.

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

import porepy as pp
from porepy.numerics.solvers.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceStatusCollection,
)
from porepy.numerics.solvers.equation_variable_tags import EquationTag, VariableTag
from porepy.numerics.solvers.linear_solvers.linear_solver import (
    LinearSolverStatus,
    LinearSolverStatusFailure,
    LinearSolverStatusSuccess,
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
            detection.
        admissibility_bounds: Per-variable `(lower, upper)` bounds used for read-only
            admissibility counting. Keyed by variable name; a variable absent from
            this mapping is not checked.

    Raises:
        ValueError: If `verbosity` is not 0, 1, or 2; if `cycle_window` is not
            positive; or if any `admissibility_bounds` entry has a lower bound
            exceeding its upper bound.

    """

    output_path: Path
    verbosity: int = 1
    equation_tags: list[EquationTag] = field(default_factory=list)
    variable_tags: list[VariableTag] = field(default_factory=list)
    cycle_window: int = 8
    admissibility_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verbosity not in (0, 1, 2):
            raise ValueError(f"verbosity must be 0, 1, or 2, got {self.verbosity}.")
        if self.cycle_window <= 0:
            raise ValueError(f"cycle_window must be positive, got {self.cycle_window}.")
        for name, (lower, upper) in self.admissibility_bounds.items():
            if lower > upper:
                raise ValueError(
                    f"admissibility_bounds[{name!r}] has lower bound {lower} greater "
                    f"than upper bound {upper}."
                )


@runtime_checkable
class NonlinearIterationRecorder(Protocol):
    """Interface for recording solver telemetry.

    Implementations receive only values their callers have already computed and must
    not trigger additional AD evaluation or system assembly. `NewtonSolver`,
    `SequentialNonlinearSolver` and `TimeStepper` each hold an
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


SCHEMA_VERSION = 1
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

    Physics-agnostic: relies only on the equation block structure every PorePy model
    exposes through `EquationOnDomain`/`EquationIndexer`. This first implementation
    covers the raw per-block residual fields (L-inf norm, reduction factor, argmax
    cell location) plus linear-solver status; duck-typed fields (line-search
    telemetry, contact-state counts, admissibility counts) and level-2 spatial fields
    are added by later commits -- constructing this recorder for a model that doesn't
    expose those hooks simply leaves the corresponding record fields absent for now.

    Parameters:
        config: Telemetry configuration.

    """

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        """Telemetry configuration."""
        self._writer = JSONLWriter(config.output_path)
        self._previous_block_residual_norms: dict[str, float] = {}

    def before_nonlinear_loop(self, model: pp.PorePyModel) -> None:
        """Reset the previous-iteration residual norms used for reduction factors."""
        self._previous_block_residual_norms = {}

    def record_iteration(
        self,
        model: pp.PorePyModel,
        iteration_index: int,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        equation_indexer: pp.ad.EquationIndexer,
        linear_solver_status: LinearSolverStatus,
        convergence_status: ConvergenceStatusCollection,
        divergence_status: ConvergenceStatusCollection,
        convergence_info: ConvergenceInfoCollection,
    ) -> None:
        """Write one "iteration" record. See `NonlinearIterationRecorder`."""
        if self.config.verbosity == 0:
            return
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
                    model, residual, equation_indexer
                ),
            }
        )

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
        """Write one "outer_iteration" record. See `NonlinearIterationRecorder`."""
        if self.config.verbosity == 0:
            return
        self._writer.write_record(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "outer_iteration",
                "iteration_index": iteration_index,
                "convergence_status": _stringify_status(convergence_status),
                "divergence_status": _stringify_status(divergence_status),
                "convergence_info": dict(convergence_info),
                "blocks": self._block_residual_fields(
                    model, full_residual, equation_indexer
                ),
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
    ) -> dict[str, dict[str, Any]]:
        """Per-equation-block residual telemetry: L-inf norm, reduction factor, and
        the cell location of the largest-magnitude entry.

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

            previous_norm = self._previous_block_residual_norms.get(name)
            reduction_factor = (
                norm / previous_norm
                if previous_norm is not None and previous_norm != 0.0
                else None
            )
            self._previous_block_residual_norms[name] = norm

            fields[name] = {
                "residual_inf_norm": norm,
                "reduction_factor": reduction_factor,
                "argmax_location": self._argmax_location(
                    model, name, domains_to_indices, block_residual
                ),
            }
        return fields

    def _argmax_location(
        self,
        model: pp.PorePyModel,
        name: str,
        domains_to_indices: dict[pp.GridLike, np.ndarray],
        block_residual: np.ndarray,
    ) -> dict[str, Any] | None:
        """Locate the largest-magnitude entry of one block's residual, as a cell.

        `None` if this equation isn't cell-centered (e.g. defined on faces/nodes),
        since `pp.ad.dof_to_cell` doesn't apply then.

        """
        num_components = model.equation_system.equations[name].target.dof_info.cells
        if num_components <= 0:
            return None

        argmax_local = int(np.argmax(np.abs(block_residual)))
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
    if isinstance(status, LinearSolverStatusSuccess):
        return {"success": True, "solve_time": status.solve_time}
    assert isinstance(status, LinearSolverStatusFailure), (
        "LinearSolverStatus is documented as a closed enum of two cases."
    )
    return {"success": False, "reason": status.reason}
