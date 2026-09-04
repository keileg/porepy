"""Unit tests for `TelemetryConfig` and the `NonlinearIterationRecorder` interface.

No live model is used: `TelemetryConfig` is exercised directly, and
`NonlinearIterationRecorder` conformance is checked against hand-built fake recorders
and hand-built fake convergence/linear-solver/time-step objects.

"""

from pathlib import Path

import numpy as np
import pytest

import porepy as pp
from porepy.numerics.solvers.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceStatus,
    ConvergenceStatusCollection,
)
from porepy.numerics.solvers.linear_solvers.linear_solver import (
    LinearSolverStatusSuccess,
)
from porepy.numerics.solvers.newton_solver import NewtonSolverConverged
from porepy.numerics.solvers.telemetry import (
    NonlinearIterationRecorder,
    TelemetryConfig,
)
from porepy.time_stepper.time_step_status import (
    TimeStepperAttemptData,
    TimeStepperStatusSuccess,
)


class TestTelemetryConfig:
    """Unit tests for `TelemetryConfig` defaults and validation."""

    def test_defaults(self, tmp_path: Path) -> None:
        config = TelemetryConfig(output_path=tmp_path)
        assert config.verbosity == 1
        assert config.equation_tags == []
        assert config.variable_tags == []
        assert config.cycle_window == 8
        assert config.admissibility_bounds == {}

    @pytest.mark.parametrize("verbosity", [-1, 3])
    def test_invalid_verbosity_raises(self, tmp_path: Path, verbosity: int) -> None:
        with pytest.raises(ValueError, match="verbosity"):
            TelemetryConfig(output_path=tmp_path, verbosity=verbosity)

    @pytest.mark.parametrize("cycle_window", [0, -1])
    def test_non_positive_cycle_window_raises(
        self, tmp_path: Path, cycle_window: int
    ) -> None:
        with pytest.raises(ValueError, match="cycle_window"):
            TelemetryConfig(output_path=tmp_path, cycle_window=cycle_window)

    def test_admissibility_bounds_lower_above_upper_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="pressure"):
            TelemetryConfig(
                output_path=tmp_path, admissibility_bounds={"pressure": (1.0, 0.0)}
            )

    def test_admissibility_bounds_equal_bounds_is_valid(self, tmp_path: Path) -> None:
        # Lower == upper is a degenerate but not invalid interval.
        config = TelemetryConfig(
            output_path=tmp_path, admissibility_bounds={"pressure": (1.0, 1.0)}
        )
        assert config.admissibility_bounds == {"pressure": (1.0, 1.0)}


class _FakeRecorder:
    """A minimal, hand-built stand-in for `NonlinearIterationRecorder`.

    Records every call it receives instead of doing anything with the data, so tests
    can assert on what a real caller would hand a recorder.

    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def before_nonlinear_loop(self, model: pp.PorePyModel) -> None:
        self.calls.append(("before_nonlinear_loop", (model,)))

    def record_iteration(
        self,
        model,
        iteration_index,
        nonlinear_increment,
        linear_solver_status,
        convergence_status,
        divergence_status,
        convergence_info,
    ) -> None:
        self.calls.append(
            (
                "record_iteration",
                (
                    model,
                    iteration_index,
                    nonlinear_increment,
                    linear_solver_status,
                    convergence_status,
                    divergence_status,
                    convergence_info,
                ),
            )
        )

    def record_outer_iteration(
        self,
        model,
        iteration_index,
        nonlinear_increment,
        full_residual,
        convergence_status,
        divergence_status,
        convergence_info,
    ) -> None:
        self.calls.append(
            (
                "record_outer_iteration",
                (
                    model,
                    iteration_index,
                    nonlinear_increment,
                    full_residual,
                    convergence_status,
                    divergence_status,
                    convergence_info,
                ),
            )
        )

    def record_time_step(self, model, time_step_status) -> None:
        self.calls.append(("record_time_step", (model, time_step_status)))


class TestNonlinearIterationRecorderConformance:
    """Structural (runtime_checkable) conformance tests for the protocol."""

    def test_fake_recorder_satisfies_protocol(self) -> None:
        assert isinstance(_FakeRecorder(), NonlinearIterationRecorder)

    @pytest.mark.parametrize(
        "missing_method",
        [
            "before_nonlinear_loop",
            "record_iteration",
            "record_outer_iteration",
            "record_time_step",
        ],
    )
    def test_recorder_missing_a_method_does_not_satisfy_protocol(
        self, missing_method: str
    ) -> None:
        incomplete_recorder_type = type(
            "IncompleteRecorder",
            (_FakeRecorder,),
            {missing_method: None},
        )
        assert not isinstance(incomplete_recorder_type(), NonlinearIterationRecorder)


class TestNonlinearIterationRecorderInvocation:
    """Smoke tests that a recorder can be invoked with realistic, hand-built inputs.

    A failure here most likely means the protocol's method signatures drifted from
    what the values available at the eventual call sites (`NewtonSolver.
    after_nonlinear_iteration`/`nonlinear_loop`, `SequentialNonlinearSolver.solve`,
    `TimeStepper._update_model_after_trial`) actually look like.

    """

    def test_record_iteration(self) -> None:
        recorder = _FakeRecorder()
        model = object()
        convergence_status = ConvergenceStatusCollection(
            {"res_abs": ConvergenceStatus.CONTINUE_ITERATING}
        )
        divergence_status = ConvergenceStatusCollection(
            {"max_iter": ConvergenceStatus.CONTINUE_ITERATING}
        )
        convergence_info: ConvergenceInfoCollection = {"res_abs": {"norm": 1.0e-3}}
        linear_solver_status = LinearSolverStatusSuccess(solve_time=0.01)
        nonlinear_increment = np.ones(3)

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=nonlinear_increment,
            linear_solver_status=linear_solver_status,
            convergence_status=convergence_status,
            divergence_status=divergence_status,
            convergence_info=convergence_info,
        )

        [(name, args)] = recorder.calls
        assert name == "record_iteration"
        (
            recorded_model,
            recorded_iteration_index,
            recorded_increment,
            recorded_linear_status,
            recorded_convergence_status,
            recorded_divergence_status,
            recorded_convergence_info,
        ) = args
        assert recorded_model is model
        assert recorded_iteration_index == 1
        np.testing.assert_array_equal(recorded_increment, nonlinear_increment)
        assert recorded_linear_status is linear_solver_status
        assert recorded_convergence_status is convergence_status
        assert recorded_divergence_status is divergence_status
        assert recorded_convergence_info is convergence_info

    def test_record_outer_iteration(self) -> None:
        recorder = _FakeRecorder()
        model = object()
        convergence_status = ConvergenceStatusCollection(
            {"res_abs": ConvergenceStatus.CONVERGED}
        )
        divergence_status = ConvergenceStatusCollection()
        convergence_info: ConvergenceInfoCollection = {"res_abs": {"norm": 1.0e-11}}

        recorder.record_outer_iteration(
            model=model,
            iteration_index=2,
            nonlinear_increment=np.zeros(2),
            full_residual=np.ones(2) * 1e-11,
            convergence_status=convergence_status,
            divergence_status=divergence_status,
            convergence_info=convergence_info,
        )

        [(name, _)] = recorder.calls
        assert name == "record_outer_iteration"

    def test_record_time_step(self) -> None:
        recorder = _FakeRecorder()
        model = object()
        nonlinear_solver_status = NewtonSolverConverged(
            linear_solver_statuses=[LinearSolverStatusSuccess(solve_time=0.01)],
            convergence_statuses=ConvergenceStatusCollection(
                {"res_abs": ConvergenceStatus.CONVERGED}
            ),
            divergence_statuses=ConvergenceStatusCollection(),
        )
        time_step_status = TimeStepperStatusSuccess(
            time=1.0,
            attempts=[
                TimeStepperAttemptData(
                    dt=1.0, nonlinear_solve_status=nonlinear_solver_status
                )
            ],
        )

        recorder.record_time_step(model=model, time_step_status=time_step_status)

        [(name, args)] = recorder.calls
        assert name == "record_time_step"
        assert args == (model, time_step_status)

    def test_before_nonlinear_loop(self) -> None:
        recorder = _FakeRecorder()
        model = object()
        recorder.before_nonlinear_loop(model)
        assert recorder.calls == [("before_nonlinear_loop", (model,))]
