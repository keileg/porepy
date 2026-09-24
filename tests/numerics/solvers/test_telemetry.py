"""Unit tests for the telemetry configuration, recording interface, and JSONL I/O.

`TelemetryConfig`, the `NonlinearIterationRecorder` interface, and `JSONLWriter`/
`read_jsonl_records`/`validate_record` are exercised against hand-built fakes, no live
model needed. `NonlinearTelemetryRecorder`, the concrete recorder, needs a real
`EquationSystem` to slice residuals against, so its tests use a small fixture model
(`SinglePhaseFlow`) instead; its two contact-mechanics duck-typed fields need a model
with a fracture (`SquareDomainOrthogonalFractures` + `pp.MomentumBalance`).

"""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.numerics.solvers.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceStatus,
    ConvergenceStatusCollection,
)
from porepy.numerics.solvers.linear_solvers.linear_solver import (
    LinearSolverStatusFailure,
    LinearSolverStatusSuccess,
)
from porepy.numerics.solvers.newton_solver import NewtonSolverConverged
from porepy.numerics.solvers.telemetry import (
    CONTACT_STATE_OPEN,
    CONTACT_STATE_SLIP,
    CONTACT_STATE_STICK,
    SCHEMA_VERSION,
    JSONLWriter,
    NonlinearIterationRecorder,
    NonlinearTelemetryRecorder,
    TelemetryConfig,
    contact_state_counts,
    evaluate_contact_state,
    property_change_fields,
    read_jsonl_records,
    validate_record,
)
from porepy.time_stepper.time_step_status import (
    TimeStepperAttemptData,
    TimeStepperStatusContinueIterating,
    TimeStepperStatusFailure,
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
        assert config.flip_threshold == 2
        assert config.admissibility_bounds == {}

    def test_negative_flip_threshold_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="flip_threshold"):
            TelemetryConfig(output_path=tmp_path, flip_threshold=-1)

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
        residual,
        equation_indexer,
        variable_indexer,
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
                    residual,
                    equation_indexer,
                    variable_indexer,
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
        equation_indexer,
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
                    equation_indexer,
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
        domain = pp.CartGrid([3])
        equation = pp.ad.EquationOnDomain("mass_balance_equation", domain)
        residual = np.array([1.0, 2.0, 3.0])
        equation_indexer = pp.ad.EquationIndexer(indices={equation: np.arange(3)})
        variable = pp.ad.Variable("pressure", {pp.ad.GridEntity.cells: 1}, domain)
        variable_indexer = pp.ad.VariableIndexer(indices={variable: np.arange(3)})

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=nonlinear_increment,
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=variable_indexer,
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
            recorded_residual,
            recorded_equation_indexer,
            recorded_variable_indexer,
            recorded_linear_status,
            recorded_convergence_status,
            recorded_divergence_status,
            recorded_convergence_info,
        ) = args
        assert recorded_model is model
        assert recorded_iteration_index == 1
        np.testing.assert_array_equal(recorded_increment, nonlinear_increment)
        np.testing.assert_array_equal(recorded_residual, residual)
        assert recorded_equation_indexer is equation_indexer
        assert recorded_variable_indexer is variable_indexer
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
        domain = pp.CartGrid([2])
        equation = pp.ad.EquationOnDomain("mass_balance_equation", domain)
        equation_indexer = pp.ad.EquationIndexer(indices={equation: np.arange(2)})

        recorder.record_outer_iteration(
            model=model,
            iteration_index=2,
            nonlinear_increment=np.zeros(2),
            full_residual=np.ones(2) * 1e-11,
            equation_indexer=equation_indexer,
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


def _converged_attempt(dt: float = 1.0) -> TimeStepperAttemptData:
    """One successful time-step attempt, as the TimeStepper records it."""
    return TimeStepperAttemptData(
        dt=dt,
        nonlinear_solve_status=NewtonSolverConverged(
            linear_solver_statuses=[],
            convergence_statuses=ConvergenceStatusCollection(),
            divergence_statuses=ConvergenceStatusCollection(),
        ),
    )


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


def _prepared_model() -> pp.PorePyModel:
    """A small, prepared model with exactly one equation block: `mass_balance_equation`
    on a single 2x2 Cartesian subdomain (4 cells), so per-block telemetry has exactly
    one block to reason about.

    """
    model = SinglePhaseFlow({"times_to_export": []})
    model.prepare_simulation()
    return model


def _perturb_pressure(model: pp.PorePyModel) -> None:
    """Give the model a non-uniform state, so the residual isn't trivially zero/tied."""
    pressure = model.pressure(model.mdg.subdomains())
    model.equation_system.set_variable_values(
        np.array([1.0, 2.0, 5.0, 0.5]),
        variables=[pressure],
        iterate_index=0,
        additive=False,
    )


def _full_residual_and_indexer(
    model: pp.PorePyModel,
) -> tuple[np.ndarray, pp.ad.EquationIndexer]:
    """The full-system residual and its (already-matching) equation indexer."""
    residual = model.equation_system.assemble(evaluate_jacobian=False)
    return residual, model.equation_system.equation_indexer


class TestNonlinearTelemetryRecorder:
    """Tests for the concrete recorder, against a small fixture model.

    A failure here most likely means either the per-block residual math (L-inf norm,
    reduction factor, argmax cell location) is wrong, or a record no longer matches
    what `.claude/solver-telemetry/schema.md` documents for the "iteration"/
    "outer_iteration"/"time_step" record types.

    """

    @staticmethod
    def _recorder(tmp_path: Path, **config_overrides) -> NonlinearTelemetryRecorder:
        config = TelemetryConfig(
            output_path=tmp_path / "telemetry.jsonl", **config_overrides
        )
        return NonlinearTelemetryRecorder(config)

    def test_record_iteration_writes_expected_block_fields(
        self, tmp_path: Path
    ) -> None:
        model = _prepared_model()
        _perturb_pressure(model)
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(tmp_path)

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.02),
            convergence_status=ConvergenceStatusCollection(
                {"res_abs": ConvergenceStatus.CONTINUE_ITERATING}
            ),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={"res_abs": {"norm": 1.0e-3}},
        )

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        validate_record(record)
        assert record["record_type"] == "iteration"
        assert record["iteration_index"] == 1
        assert record["linear_solver"] == {"success": True, "solve_time": 0.02}
        assert record["convergence_status"] == {"res_abs": "continue_iterating"}
        assert record["convergence_info"] == {"res_abs": {"norm": 1.0e-3}}

        assert list(record["blocks"]) == ["mass_balance_equation"]
        block = record["blocks"]["mass_balance_equation"]
        expected_cell = int(np.argmax(np.abs(residual)))
        assert block["residual_inf_norm"] == pytest.approx(
            float(np.max(np.abs(residual)))
        )
        assert block["reduction_factor"] is None  # Nothing to divide by yet.
        subdomain = model.mdg.subdomains()[0]
        assert block["argmax_location"] == {
            "dim": subdomain.dim,
            "domain_id": subdomain.id,
            "cell": expected_cell,
        }

    def test_reduction_factor_uses_previous_iteration(self, tmp_path: Path) -> None:
        model = _prepared_model()
        _perturb_pressure(model)
        recorder = self._recorder(tmp_path)

        def record(residual: np.ndarray, iteration_index: int) -> None:
            recorder.record_iteration(
                model=model,
                iteration_index=iteration_index,
                nonlinear_increment=np.zeros_like(residual),
                residual=residual,
                equation_indexer=model.equation_system.equation_indexer,
                variable_indexer=model.equation_system.variable_indexer,
                linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
                convergence_status=ConvergenceStatusCollection(),
                divergence_status=ConvergenceStatusCollection(),
                convergence_info={},
            )

        record(np.array([1.0, -2.0, 4.0, 0.0]), iteration_index=1)  # inf-norm 4.0
        record(np.array([1.0, -1.0, 2.0, 0.0]), iteration_index=2)  # inf-norm 2.0

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert records[0]["blocks"]["mass_balance_equation"]["reduction_factor"] is None
        assert records[1]["blocks"]["mass_balance_equation"][
            "reduction_factor"
        ] == pytest.approx(0.5)

    def test_before_nonlinear_loop_resets_reduction_factor_state(
        self, tmp_path: Path
    ) -> None:
        model = _prepared_model()
        _perturb_pressure(model)
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(tmp_path)

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )
        recorder.before_nonlinear_loop(model)
        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        for record in records:
            assert record["blocks"]["mass_balance_equation"]["reduction_factor"] is None

    def test_inner_before_nonlinear_loop_does_not_disturb_outer_reduction_factor(
        self, tmp_path: Path
    ) -> None:
        """Regression test: `before_nonlinear_loop` used to reset a single shared
        reduction-factor dict, so a subsolver starting a fresh inner Newton loop (which
        calls `before_nonlinear_loop` on the same shared recorder, per
        SequentialNonlinearSolver's design) would silently wipe out the outer loop's
        own "previous iteration" state. A failure here means that regressed.

        """
        model = _prepared_model()
        recorder = self._recorder(tmp_path)

        def record_outer(residual: np.ndarray, iteration_index: int) -> None:
            recorder.record_outer_iteration(
                model=model,
                iteration_index=iteration_index,
                nonlinear_increment=np.zeros_like(residual),
                full_residual=residual,
                equation_indexer=model.equation_system.equation_indexer,
                convergence_status=ConvergenceStatusCollection(),
                divergence_status=ConvergenceStatusCollection(),
                convergence_info={},
            )

        # Outer iteration 1, then a subsolver's inner Newton loop starts (as
        # SequentialNonlinearSolver's subsolvers do on every outer iteration),
        # calling before_nonlinear_loop on this same, shared recorder.
        record_outer(np.array([1.0, -2.0, 4.0, 0.0]), iteration_index=1)  # inf-norm 4.0
        recorder.before_nonlinear_loop(model)
        record_outer(np.array([1.0, -1.0, 2.0, 0.0]), iteration_index=2)  # inf-norm 2.0

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert records[0]["blocks"]["mass_balance_equation"]["reduction_factor"] is None
        assert records[1]["blocks"]["mass_balance_equation"][
            "reduction_factor"
        ] == pytest.approx(0.5)

    def test_record_outer_iteration_resets_on_first_iteration_of_a_new_solve(
        self, tmp_path: Path
    ) -> None:
        """A second `SequentialNonlinearSolver.solve()` call starts its own
        `iteration_index` back at 1; the outer reduction factor must not carry over
        from the previous solve's last iteration.

        """
        model = _prepared_model()
        recorder = self._recorder(tmp_path)

        def record_outer(residual: np.ndarray, iteration_index: int) -> None:
            recorder.record_outer_iteration(
                model=model,
                iteration_index=iteration_index,
                nonlinear_increment=np.zeros_like(residual),
                full_residual=residual,
                equation_indexer=model.equation_system.equation_indexer,
                convergence_status=ConvergenceStatusCollection(),
                divergence_status=ConvergenceStatusCollection(),
                convergence_info={},
            )

        # First "solve()": two outer iterations, ending with inf-norm 2.0.
        record_outer(np.array([1.0, -2.0, 4.0, 0.0]), iteration_index=1)
        record_outer(np.array([1.0, -1.0, 2.0, 0.0]), iteration_index=2)
        # Second "solve()": iteration_index restarts at 1.
        record_outer(np.array([1.0, -1.0, 3.0, 0.0]), iteration_index=1)

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert records[2]["blocks"]["mass_balance_equation"]["reduction_factor"] is None

    def test_verbosity_zero_disables_recording(self, tmp_path: Path) -> None:
        model = _prepared_model()
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(tmp_path, verbosity=0)

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )
        recorder.record_outer_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            full_residual=residual,
            equation_indexer=equation_indexer,
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )
        recorder.record_time_step(
            model=model,
            time_step_status=TimeStepperStatusSuccess(
                time=1.0, attempts=[_converged_attempt()]
            ),
        )

        assert (
            not (tmp_path / "telemetry.jsonl").exists()
            or read_jsonl_records(tmp_path / "telemetry.jsonl") == []
        )

    def test_equation_tags_restrict_recorded_blocks(self, tmp_path: Path) -> None:
        model = _prepared_model()
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(
            tmp_path,
            equation_tags=[pp.solvers.EquationTag(name="not_a_real_equation")],
        )

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert record["blocks"] == {}

    def test_record_outer_iteration_writes_block_fields(self, tmp_path: Path) -> None:
        model = _prepared_model()
        _perturb_pressure(model)
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(tmp_path)

        recorder.record_outer_iteration(
            model=model,
            iteration_index=3,
            nonlinear_increment=np.zeros_like(residual),
            full_residual=residual,
            equation_indexer=equation_indexer,
            convergence_status=ConvergenceStatusCollection(
                {"res_abs": ConvergenceStatus.CONVERGED}
            ),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={"res_abs": {"norm": 1.0e-11}},
        )

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        validate_record(record)
        assert record["record_type"] == "outer_iteration"
        assert record["iteration_index"] == 3
        assert list(record["blocks"]) == ["mass_balance_equation"]

    @pytest.mark.parametrize(
        "time_step_status,expected_fields",
        [
            (
                lambda: TimeStepperStatusSuccess(
                    time=1.5, attempts=[_converged_attempt(dt=0.5)]
                ),
                {
                    "status": "successful",
                    "dt": 0.5,
                    "time": 1.5,
                    "num_attempts": 1,
                    "nonlinear_solver_status": "successful",
                },
            ),
            (
                lambda: TimeStepperStatusFailure(
                    reason="dt below minimum",
                    time=1.5,
                    attempts=[_converged_attempt(dt=0.5)],
                ),
                {"status": "failed", "reason": "dt below minimum", "num_attempts": 1},
            ),
            (
                lambda: TimeStepperStatusContinueIterating(
                    attempts=[_converged_attempt(), _converged_attempt()]
                ),
                {"status": "in_progress", "num_attempts": 2},
            ),
        ],
        ids=["success", "failure", "continue_iterating"],
    )
    def test_record_time_step(
        self, tmp_path: Path, time_step_status, expected_fields: dict
    ) -> None:
        model = _prepared_model()
        recorder = self._recorder(tmp_path)

        recorder.record_time_step(model=model, time_step_status=time_step_status())

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        validate_record(record)
        assert record["record_type"] == "time_step"
        for key, value in expected_fields.items():
            assert record[key] == value

    def test_linear_solver_failure_is_recorded(self, tmp_path: Path) -> None:
        model = _prepared_model()
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(tmp_path)

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusFailure(reason="singular matrix"),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert record["linear_solver"] == {
            "success": False,
            "reason": "singular matrix",
        }

    def test_no_admissibility_bounds_configured_gives_empty_dict(
        self, tmp_path: Path
    ) -> None:
        model = _prepared_model()
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(tmp_path)

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert record["admissibility"] == {}

    def test_admissibility_counts_violations(self, tmp_path: Path) -> None:
        model = _prepared_model()
        _perturb_pressure(model)  # pressure = [1.0, 2.0, 5.0, 0.5]
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(
            tmp_path, admissibility_bounds={"pressure": (0.75, 3.0)}
        )

        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=model.equation_system.variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        validate_record(record)
        assert record["admissibility"] == {
            "pressure": {"num_below": 1, "num_above": 1, "num_total": 4}
        }

    def test_admissibility_counts_are_also_written_for_outer_iteration(
        self, tmp_path: Path
    ) -> None:
        model = _prepared_model()
        _perturb_pressure(model)
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(
            tmp_path, admissibility_bounds={"pressure": (0.75, 3.0)}
        )

        recorder.record_outer_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=np.zeros_like(residual),
            full_residual=residual,
            equation_indexer=equation_indexer,
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )

        [record] = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert record["admissibility"] == {
            "pressure": {"num_below": 1, "num_above": 1, "num_total": 4}
        }

    def test_admissibility_bounds_for_unknown_variable_raises(
        self, tmp_path: Path
    ) -> None:
        model = _prepared_model()
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder = self._recorder(
            tmp_path, admissibility_bounds={"temperature": (0.0, 1.0)}
        )

        with pytest.raises(ValueError, match="temperature"):
            recorder.record_iteration(
                model=model,
                iteration_index=1,
                nonlinear_increment=np.zeros_like(residual),
                residual=residual,
                equation_indexer=equation_indexer,
                variable_indexer=model.equation_system.variable_indexer,
                linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
                convergence_status=ConvergenceStatusCollection(),
                divergence_status=ConvergenceStatusCollection(),
                convergence_info={},
            )


def _cell_array(values: np.ndarray, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
    """Wrap one value per cell of the given subdomains as an Ad array."""
    space = pp.ad.OperatorSpace.from_domains(subdomains, {pp.ad.GridEntity.cells: 1})
    return pp.ad.DenseArray(values, source=space, target=space)


def _fixed_indicator_model(
    opening_values: np.ndarray, sliding_values: np.ndarray, **params
) -> pp.PorePyModel:
    """A prepared, fractured contact-mechanics model whose opening/sliding indicators
    are hard-coded to the given arrays, for controlled testing of contact-state
    counting -- following the same pattern `tests/numerics/solvers/test_line_search.py`
    uses to test `ConstraintLineSearch` without depending on `ContactIndicators`'s real
    (traction/gap-based) computation.

    """

    class FixedIndicators:
        def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            return _cell_array(opening_values, subdomains)

        def sliding_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            return _cell_array(sliding_values, subdomains)

    class Model(SquareDomainOrthogonalFractures, pp.MomentumBalance, FixedIndicators):
        """One vertical fracture -> 2 fracture cells by default."""

    model = Model({"times_to_export": [], **params})
    model.prepare_simulation()
    return model


def _mutable_indicator_model() -> tuple[pp.PorePyModel, Callable[..., None]]:
    """As `_fixed_indicator_model`, but the caller can change the indicators between
    recorded iterations -- needed to drive the solver through a repeating active set.

    Returns:
        The prepared model, and a callable setting the two indicator arrays.

    """
    values = {
        "opening": np.array([0.9, 0.9]),
        "sliding": np.array([-0.9, -0.9]),
    }

    class MutableIndicators:
        def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            return _cell_array(values["opening"], subdomains)

        def sliding_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            return _cell_array(values["sliding"], subdomains)

    class Model(SquareDomainOrthogonalFractures, pp.MomentumBalance, MutableIndicators):
        """One vertical fracture -> 2 fracture cells by default."""

    model = Model({"times_to_export": []})
    model.prepare_simulation()

    def set_indicators(opening: np.ndarray, sliding: np.ndarray) -> None:
        values["opening"] = opening
        values["sliding"] = sliding

    return model, set_indicators


# Two distinct active sets on the default two-cell fracture, as (opening, sliding)
# indicator pairs. They differ only in cell 0: open in A, slipping in B.
_ACTIVE_SET_A = (np.array([-0.9, 0.9]), np.array([0.0, -0.9]))
_ACTIVE_SET_B = (np.array([0.9, 0.9]), np.array([0.5, -0.9]))


def _write_bare_iteration(
    recorder: NonlinearTelemetryRecorder,
    model: pp.PorePyModel,
    iteration_index: int = 1,
) -> None:
    """Call `record_iteration` with everything except `model` at its simplest."""
    residual, equation_indexer = _full_residual_and_indexer(model)
    recorder.record_iteration(
        model=model,
        iteration_index=iteration_index,
        nonlinear_increment=np.zeros_like(residual),
        residual=residual,
        equation_indexer=equation_indexer,
        variable_indexer=model.equation_system.variable_indexer,
        linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
        convergence_status=ConvergenceStatusCollection(),
        divergence_status=ConvergenceStatusCollection(),
        convergence_info={},
    )


def _record_bare_iteration(
    recorder: NonlinearTelemetryRecorder, model: pp.PorePyModel
) -> dict:
    """As `_write_bare_iteration`, returning the record just written."""
    _write_bare_iteration(recorder, model)
    return read_jsonl_records(recorder.config.output_path)[-1]


class TestIncrementFields:
    """Per-variable-block increment and iterate norms in the "iteration" record.

    A failure most likely means the increment is sliced against the wrong layout, which
    for a solver restricted to some variables silently attributes one block's increment
    to another.

    """

    @staticmethod
    def _record(
        tmp_path: Path,
        model: pp.PorePyModel,
        increment: np.ndarray,
        variable_indexer: pp.ad.VariableIndexer,
    ) -> dict:
        recorder = NonlinearTelemetryRecorder(
            TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        )
        residual, equation_indexer = _full_residual_and_indexer(model)
        recorder.record_iteration(
            model=model,
            iteration_index=1,
            nonlinear_increment=increment,
            residual=residual,
            equation_indexer=equation_indexer,
            variable_indexer=variable_indexer,
            linear_solver_status=LinearSolverStatusSuccess(solve_time=0.0),
            convergence_status=ConvergenceStatusCollection(),
            divergence_status=ConvergenceStatusCollection(),
            convergence_info={},
        )
        return read_jsonl_records(tmp_path / "telemetry.jsonl")[-1]

    def test_norms_of_increment_and_iterate(self, tmp_path: Path) -> None:
        model = _prepared_model()
        model.equation_system.set_variable_values(
            np.array([1.0, -5.0, 2.0, 3.0]), variables=["pressure"], iterate_index=0
        )
        increment = np.array([0.1, 0.0, -0.4, 0.2])

        record = self._record(
            tmp_path, model, increment, model.equation_system.variable_indexer
        )

        assert record["increments"] == {
            "pressure": {"increment_inf_norm": 0.4, "value_inf_norm": 5.0}
        }

    def test_restricted_layout(self, tmp_path: Path) -> None:
        model = pp.MassAndEnergyBalance({"times_to_export": []})
        model.prepare_simulation()
        equation_system = model.equation_system
        temperatures = equation_system.get_variables(["temperature"])
        equation_system.set_variable_values(
            np.full(equation_system.dofs_of(temperatures).size, 300.0),
            variables=temperatures,
            iterate_index=0,
        )
        restricted = pp.ad.construct_matching_restricted_indexer(
            equation_system.variable_indexer, temperatures
        )
        increment = np.zeros(equation_system.dofs_of(temperatures).size)
        increment[-1] = -7.0

        record = self._record(tmp_path, model, increment, restricted)

        assert record["increments"] == {
            "temperature": {"increment_inf_norm": 7.0, "value_inf_norm": 300.0}
        }


class TestFractureFlowFields:
    """The aperture and permeability fields tracing the dilation -> permeability
    coupling.

    A failure in the change factor or its location most likely means cells are
    compared against the wrong previous values or located in the wrong fracture, which
    would point a diagnosis at the wrong part of the fracture network.

    """

    def test_ranges_without_previous_values(self) -> None:
        grid = pp.CartGrid([3])
        fields = property_change_fields(np.array([1.0, 3.0, 2.0]), None, [grid])
        assert fields == {
            "min": 1.0,
            "max": 3.0,
            "max_change_factor": None,
            "max_change_location": None,
        }

    def test_halving_and_doubling_count_alike_and_are_located(self) -> None:
        first, second = pp.CartGrid([2]), pp.CartGrid([2])
        previous = np.array([1.0, 1.0, 1.0, 1.0])
        # A halving (factor 2) in the first fracture, a tripling in the second.
        current = np.array([0.5, 1.0, 1.0, 3.0])

        fields = property_change_fields(current, previous, [first, second])

        assert fields["max_change_factor"] == pytest.approx(3.0)
        assert fields["max_change_location"] == {
            "dim": 1,
            "domain_id": second.id,
            "cell": 1,
        }

    def test_cells_without_a_positive_value_are_left_out(self) -> None:
        grid = pp.CartGrid([2])
        fields = property_change_fields(
            np.array([0.0, 2.0]), np.array([1.0, 0.0]), [grid]
        )
        assert fields["max_change_factor"] is None

    def test_none_for_a_model_without_fractures(self, tmp_path: Path) -> None:
        recorder = NonlinearTelemetryRecorder(
            TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        )
        record = _record_bare_iteration(recorder, _prepared_model())
        assert record["fracture_flow"] is None

    def test_no_change_since_the_start_of_the_loop(self, tmp_path: Path) -> None:
        class Model(SquareDomainOrthogonalFractures, pp.Poromechanics):
            """One fracture, with both aperture and permeability."""

        model = Model({"times_to_export": []})
        model.prepare_simulation()
        recorder = NonlinearTelemetryRecorder(
            TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        )

        recorder.before_nonlinear_loop(model)
        record = _record_bare_iteration(recorder, model)

        for name in ("aperture", "permeability"):
            assert record["fracture_flow"][name]["max_change_factor"] == 1.0

    def test_no_permeability_without_fluid_flow(self, tmp_path: Path) -> None:
        model = _fixed_indicator_model(
            opening_values=np.array([0.9, 0.9]), sliding_values=np.array([-0.9, -0.9])
        )
        recorder = NonlinearTelemetryRecorder(
            TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        )
        record = _record_bare_iteration(recorder, model)
        assert record["fracture_flow"]["aperture"] is not None
        assert record["fracture_flow"]["permeability"] is None


class TestEvaluateContactState:
    """Tests for the shared contact-state evaluation the counts and the cycle
    detection both build on.

    A failure here most likely means the label encoding drifted from the sign
    convention `ContactIndicators.opening_indicator`/`sliding_indicator` document
    (negative opening = open, negative sliding = sticking).

    """

    def test_none_for_a_model_without_contact_indicators(self) -> None:
        assert evaluate_contact_state(_prepared_model()) is None

    def test_labels_follow_the_indicator_sign_convention(self) -> None:
        # Cell 0: opening < 0 -> open. Cell 1: closed, sliding < 0 -> stick.
        model = _fixed_indicator_model(
            opening_values=np.array([-0.9, 0.9]),
            sliding_values=np.array([0.0, -0.9]),
        )
        state = evaluate_contact_state(model)
        assert state is not None
        np.testing.assert_array_equal(
            state.labels, [CONTACT_STATE_OPEN, CONTACT_STATE_STICK]
        )

    def test_slip_label(self) -> None:
        model = _fixed_indicator_model(
            opening_values=np.array([0.9, 0.9]),
            sliding_values=np.array([0.1, -0.2]),
        )
        state = evaluate_contact_state(model)
        assert state is not None
        np.testing.assert_array_equal(
            state.labels, [CONTACT_STATE_SLIP, CONTACT_STATE_STICK]
        )

    def test_labels_are_int8_so_their_bytes_are_stable_to_hash(self) -> None:
        # The cycle detection hashes these bytes; a platform-dependent int width
        # would make the same active set hash differently on different machines.
        model = _fixed_indicator_model(
            opening_values=np.array([-0.9, 0.9]),
            sliding_values=np.array([0.0, -0.9]),
        )
        state = evaluate_contact_state(model)
        assert state is not None
        assert state.labels.dtype == np.int8

    def test_counts_derive_from_labels(self) -> None:
        model = _fixed_indicator_model(
            opening_values=np.array([-0.9, 0.2]),
            sliding_values=np.array([0.0, -0.9]),
        )
        state = evaluate_contact_state(model)
        assert state is not None

        counts = contact_state_counts(state, tolerance=0.3)

        assert counts["num_open"] == 1
        assert counts["num_stick"] == 1
        assert counts["num_slip"] == 0
        assert counts["num_total"] == 2
        # Only the second cell's opening indicator (0.2) is within the tolerance.
        assert counts["num_near_opening_transition"] == 1


class TestContactStateCounts:
    """Tests for the duck-typed `contact_state` field.

    A failure here most likely means either the sign convention no longer matches
    `ContactIndicators.opening_indicator`/`sliding_indicator`'s own documented
    convention (negative opening = open, negative sliding = sticking), or the
    near-transition tolerance no longer agrees with what
    `ConstraintLineSearch.constraint_weights` treats as close to a sign change for the
    same two indicators.

    """

    def test_none_for_a_model_without_contact_indicators(self, tmp_path: Path) -> None:
        model = _prepared_model()  # SinglePhaseFlow: no fractures, no indicators.
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        assert record["contact_state"] is None

    def test_open_and_stick_counts(self, tmp_path: Path) -> None:
        # Cell 0: opening < 0 -> open. Cell 1: opening > 0, sliding < 0 -> stick.
        model = _fixed_indicator_model(
            opening_values=np.array([-0.9, 0.9]),
            sliding_values=np.array([0.0, -0.9]),
        )
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        assert record["contact_state"]["num_open"] == 1
        assert record["contact_state"]["num_stick"] == 1
        assert record["contact_state"]["num_slip"] == 0
        assert record["contact_state"]["num_total"] == 2

    def test_slip_count(self, tmp_path: Path) -> None:
        # Both cells closed (opening > 0); both sliding (sliding >= 0) -> slip.
        model = _fixed_indicator_model(
            opening_values=np.array([0.9, 0.9]),
            sliding_values=np.array([0.1, 0.2]),
        )
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        assert record["contact_state"]["num_open"] == 0
        assert record["contact_state"]["num_stick"] == 0
        assert record["contact_state"]["num_slip"] == 2

    def test_near_transition_counts_use_default_tolerance(self, tmp_path: Path) -> None:
        # 0.2 < default tolerance (0.3): near the opening transition, though still
        # unambiguously classified as open/stick by the hard-zero sign.
        model = _fixed_indicator_model(
            opening_values=np.array([-0.2, 0.9]),
            sliding_values=np.array([0.0, -0.9]),
        )
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        assert record["contact_state"]["num_open"] == 1  # Classification unaffected.
        assert record["contact_state"]["num_near_opening_transition"] == 1

    def test_near_transition_counts_use_configured_tolerance(
        self, tmp_path: Path
    ) -> None:
        model = _fixed_indicator_model(
            opening_values=np.array([-0.2, 0.9]),
            sliding_values=np.array([0.0, -0.9]),
            constraint_violation_tolerance=0.1,
        )
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        # 0.2 > the configured tolerance of 0.1: no longer counted as near-transition.
        assert record["contact_state"]["num_near_opening_transition"] == 0

    def test_counts_per_fracture(self, tmp_path: Path) -> None:
        """A failure most likely means the indicator array is split at the wrong
        offsets, attributing one fracture's cells to another."""
        # Two fractures of two cells each: the first open and sticking, the second
        # slipping in both cells.
        model = _fixed_indicator_model(
            opening_values=np.array([-0.9, 0.9, 0.9, 0.9]),
            sliding_values=np.array([0.0, -0.9, 0.1, 0.2]),
            fracture_indices=[0, 1],
        )
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        first, second = (str(sd.id) for sd in model.mdg.subdomains(dim=model.nd - 1))
        assert record["contact_state"]["by_subdomain"] == {
            first: {"num_open": 1, "num_stick": 1, "num_slip": 0},
            second: {"num_open": 0, "num_stick": 0, "num_slip": 2},
        }


class TestLineSearchFields:
    """Tests for the duck-typed `line_search` field."""

    def test_none_without_constraint_weights(self, tmp_path: Path) -> None:
        model = _prepared_model()
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        assert record["line_search"] is None

    def test_alpha_is_the_minimum_relaxation_weight(self, tmp_path: Path) -> None:
        model = _prepared_model()
        for sd in model.mdg.subdomains():
            model.mdg.subdomain_data(sd)["constraint_weights"] = np.array(
                [1.0, 0.4, 0.7, 1.0]
            )
        config = TelemetryConfig(output_path=tmp_path / "telemetry.jsonl")
        recorder = NonlinearTelemetryRecorder(config)

        record = _record_bare_iteration(recorder, model)

        assert record["line_search"] == {"alpha": pytest.approx(0.4)}


class TestCycleField:
    """Tests for the `cycle` field the recorder derives from the active-set history.

    The history itself is unit-tested in `test_cycle_detection.py`; these tests cover
    it being fed the right labels, at the right times, with inner and outer histories
    kept apart.

    """

    @staticmethod
    def _recorder(tmp_path: Path, **config_overrides) -> NonlinearTelemetryRecorder:
        config = TelemetryConfig(
            output_path=tmp_path / "telemetry.jsonl", **config_overrides
        )
        return NonlinearTelemetryRecorder(config)

    def test_none_for_a_model_without_contact_indicators(self, tmp_path: Path) -> None:
        record = _record_bare_iteration(self._recorder(tmp_path), _prepared_model())

        assert record["cycle"] is None

    def test_first_iteration_reports_a_hash_but_no_repeat(self, tmp_path: Path) -> None:
        model, set_indicators = _mutable_indicator_model()
        set_indicators(*_ACTIVE_SET_A)

        record = _record_bare_iteration(self._recorder(tmp_path), model)

        assert record["cycle"]["repeat_length"] is None
        assert record["cycle"]["num_flipped_cells"] == 0
        assert len(record["cycle"]["active_set_hash"]) == 16

    def test_repeating_active_set_is_detected(self, tmp_path: Path) -> None:
        model, set_indicators = _mutable_indicator_model()
        recorder = self._recorder(tmp_path)

        for indicators in [_ACTIVE_SET_A, _ACTIVE_SET_B, _ACTIVE_SET_A]:
            set_indicators(*indicators)
            _write_bare_iteration(recorder, model)

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert [r["cycle"]["repeat_length"] for r in records] == [None, None, 2]
        # Cell 0 is the one changing state inside the cycle; cell 1 never moves.
        assert records[2]["cycle"]["cycling_cells"] == [0]

    def test_chattering_cells_reported_once_over_the_threshold(
        self, tmp_path: Path
    ) -> None:
        model, set_indicators = _mutable_indicator_model()
        recorder = self._recorder(tmp_path, flip_threshold=2)

        # Cell 0 flips on every iteration: three flips over four iterations.
        for indicators in [_ACTIVE_SET_A, _ACTIVE_SET_B, _ACTIVE_SET_A, _ACTIVE_SET_B]:
            set_indicators(*indicators)
            _write_bare_iteration(recorder, model)

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert [r["cycle"]["chattering_cells"] for r in records] == [[], [], [], [0]]

    def test_before_nonlinear_loop_resets_the_inner_history(
        self, tmp_path: Path
    ) -> None:
        model, set_indicators = _mutable_indicator_model()
        recorder = self._recorder(tmp_path)
        set_indicators(*_ACTIVE_SET_A)
        _write_bare_iteration(recorder, model)

        recorder.before_nonlinear_loop(model)
        _write_bare_iteration(recorder, model)

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        # Same active set, but across a loop boundary: not a cycle in either loop.
        assert records[1]["cycle"]["repeat_length"] is None

    def test_inner_reset_does_not_disturb_the_outer_history(
        self, tmp_path: Path
    ) -> None:
        """Regression test, mirroring the reduction-factor one: a subsolver starting
        its own inner loop calls `before_nonlinear_loop` on the shared recorder, which
        must not clear the outer loop's active-set history.

        """
        model, set_indicators = _mutable_indicator_model()
        recorder = self._recorder(tmp_path)
        residual, equation_indexer = _full_residual_and_indexer(model)

        def record_outer(iteration_index: int) -> None:
            recorder.record_outer_iteration(
                model=model,
                iteration_index=iteration_index,
                nonlinear_increment=np.zeros_like(residual),
                full_residual=residual,
                equation_indexer=equation_indexer,
                convergence_status=ConvergenceStatusCollection(),
                divergence_status=ConvergenceStatusCollection(),
                convergence_info={},
            )

        set_indicators(*_ACTIVE_SET_A)
        record_outer(iteration_index=1)
        recorder.before_nonlinear_loop(model)  # A subsolver's inner loop starting.
        set_indicators(*_ACTIVE_SET_B)
        record_outer(iteration_index=2)
        recorder.before_nonlinear_loop(model)
        set_indicators(*_ACTIVE_SET_A)
        record_outer(iteration_index=3)

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert [r["cycle"]["repeat_length"] for r in records] == [None, None, 2]

    def test_outer_history_resets_on_a_new_solve(self, tmp_path: Path) -> None:
        model, set_indicators = _mutable_indicator_model()
        recorder = self._recorder(tmp_path)
        residual, equation_indexer = _full_residual_and_indexer(model)

        def record_outer(iteration_index: int) -> None:
            recorder.record_outer_iteration(
                model=model,
                iteration_index=iteration_index,
                nonlinear_increment=np.zeros_like(residual),
                full_residual=residual,
                equation_indexer=equation_indexer,
                convergence_status=ConvergenceStatusCollection(),
                divergence_status=ConvergenceStatusCollection(),
                convergence_info={},
            )

        set_indicators(*_ACTIVE_SET_A)
        record_outer(iteration_index=1)
        record_outer(iteration_index=2)  # Would be a repeat within this solve.
        record_outer(iteration_index=1)  # A fresh solve restarts the count.

        records = read_jsonl_records(tmp_path / "telemetry.jsonl")
        assert [r["cycle"]["repeat_length"] for r in records] == [None, 1, None]
