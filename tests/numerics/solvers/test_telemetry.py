"""Unit tests for the telemetry configuration, recording interface, and JSONL I/O.

No live model is used: `TelemetryConfig` is exercised directly,
`NonlinearIterationRecorder` conformance is checked against hand-built fake recorders
and hand-built fake convergence/linear-solver/time-step objects, and `JSONLWriter`/
`read_jsonl_records`/`validate_record` are exercised against hand-built records.

"""

import json
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
    SCHEMA_VERSION,
    JSONLWriter,
    NonlinearIterationRecorder,
    TelemetryConfig,
    read_jsonl_records,
    validate_record,
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


def _run_record(**overrides) -> dict:
    """A minimal, valid "run" record, for tests to override fields on."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run",
        "porepy_commit": "deadbeef",
        "pp_solvers_commit": "cafef00d",
        "config_hash": "abc123",
    }
    record.update(overrides)
    return record


class TestJSONLWriter:
    """Unit tests for `JSONLWriter`, isolated from any real telemetry recorder."""

    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.jsonl"
        records = [_run_record(config_hash="a"), _run_record(config_hash="b")]

        with JSONLWriter(path) as writer:
            for record in records:
                writer.write_record(record)

        assert read_jsonl_records(path) == records

    def test_write_record_supports_numpy_types(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.jsonl"
        record = _run_record(
            config_hash="a",
            extra={"residual": np.array([1.0, 2.0]), "count": np.int64(3)},
        )

        with JSONLWriter(path) as writer:
            writer.write_record(record)

        [read_back] = read_jsonl_records(path)
        assert read_back["extra"]["residual"] == [1.0, 2.0]
        assert read_back["extra"]["count"] == 3

    def test_each_write_is_flushed_before_the_next(self, tmp_path: Path) -> None:
        # A crash between two writes must not lose the first: read back what is on
        # disk after only one write, using a second, independent file handle.
        path = tmp_path / "telemetry.jsonl"
        writer = JSONLWriter(path)
        writer.write_record(_run_record())
        assert len(read_jsonl_records(path)) == 1
        writer.close()


class TestReadJsonlRecords:
    """Unit tests for `read_jsonl_records`, in particular truncated-file handling.

    A failure here most likely means a telemetry run killed mid-write (crash,
    SIGKILL, out-of-memory) would either lose complete records that were already on
    disk, or make the reader raise instead of tolerating the partial last line.

    """

    def test_truncated_final_line_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.jsonl"
        complete = [_run_record(config_hash="a"), _run_record(config_hash="b")]
        with open(path, "w", encoding="utf-8") as f:
            for record in complete:
                f.write(json.dumps(record) + "\n")
            # Simulate being killed mid-write of a third record.
            f.write('{"schema_version": 1, "record_type": "run", "porepy_com')

        assert read_jsonl_records(path) == complete

    def test_malformed_line_before_the_end_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"schema_version": 1, "record_type": "run", "porepy_com\n')
            f.write(json.dumps(_run_record()) + "\n")

        with pytest.raises(json.JSONDecodeError):
            read_jsonl_records(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.jsonl"
        record = _run_record()
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n\n")

        assert read_jsonl_records(path) == [record]

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.jsonl"
        path.touch()
        assert read_jsonl_records(path) == []


class TestValidateRecord:
    """Unit tests for `validate_record`."""

    def test_valid_run_record_passes(self) -> None:
        validate_record(_run_record())

    def test_valid_run_record_with_none_pp_solvers_commit_passes(self) -> None:
        validate_record(_run_record(pp_solvers_commit=None))

    @pytest.mark.parametrize("missing_key", ["schema_version", "record_type"])
    def test_missing_envelope_field_raises(self, missing_key: str) -> None:
        record = _run_record()
        del record[missing_key]
        with pytest.raises(ValueError, match=missing_key):
            validate_record(record)

    def test_non_int_schema_version_raises(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            validate_record(_run_record(schema_version="1"))

    def test_newer_schema_version_raises(self) -> None:
        with pytest.raises(ValueError, match="newer"):
            validate_record(_run_record(schema_version=SCHEMA_VERSION + 1))

    @pytest.mark.parametrize(
        "missing_key", ["porepy_commit", "pp_solvers_commit", "config_hash"]
    )
    def test_missing_run_field_raises(self, missing_key: str) -> None:
        record = _run_record()
        del record[missing_key]
        with pytest.raises(ValueError, match=missing_key):
            validate_record(record)

    def test_wrong_type_porepy_commit_raises(self) -> None:
        with pytest.raises(ValueError, match="porepy_commit"):
            validate_record(_run_record(porepy_commit=123))

    def test_wrong_type_config_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="config_hash"):
            validate_record(_run_record(config_hash=123))

    def test_wrong_type_pp_solvers_commit_raises(self) -> None:
        with pytest.raises(ValueError, match="pp_solvers_commit"):
            validate_record(_run_record(pp_solvers_commit=123))

    def test_unknown_record_type_only_checks_envelope(self) -> None:
        # A record type not yet known to this module (added by a later commit) is
        # not rejected outright -- only its envelope is checked.
        validate_record(
            {"schema_version": SCHEMA_VERSION, "record_type": "not_yet_implemented"}
        )
