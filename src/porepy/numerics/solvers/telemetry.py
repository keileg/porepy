"""Configuration and recording interface for solver telemetry.

Telemetry is purely additive and off by default: a
:class:`NonlinearIterationRecorder` receives only values its callers have already
computed for their own purposes (residuals, convergence status, linear-solver status),
so recording must never trigger additional AD evaluation or system assembly.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

import porepy as pp
from porepy.numerics.solvers.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceStatusCollection,
)
from porepy.numerics.solvers.equation_variable_tags import EquationTag, VariableTag
from porepy.numerics.solvers.linear_solvers.linear_solver import LinearSolverStatus

if TYPE_CHECKING:
    # Imported only for type checking. porepy.time_stepper.time_step_status imports
    # porepy.numerics.solvers at module level, so importing it here unconditionally
    # would be a runtime circular import: this module is itself part of
    # porepy.numerics.solvers.
    from porepy.time_stepper.time_step_status import TimeStepperStatus

__all__ = [
    "TelemetryConfig",
    "NonlinearIterationRecorder",
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
            full_residual: Residual of the full system (all subsolvers' equations
                together) at this outer iteration.
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
