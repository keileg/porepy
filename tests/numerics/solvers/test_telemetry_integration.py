"""End-to-end tests of solver telemetry on a running model.

Where `test_telemetry.py` drives the recorder directly with hand-built arguments, this
module runs a real simulation through `ModelRunner` and inspects the file that comes
out. It covers what only a full run can show: that recording does not perturb the
solution, that every record a real run emits satisfies the schema, that recording never
assembles the system behind the solver's back, and that the file survives the process
being killed mid-solve.

The fixture model is deliberately the smallest one that produces every field: a
poromechanics model needs time steps (so `"time_step"` records exist) and Newton
iterations, a fracture gives it contact mechanics, and `ContactIndicators` is what makes
the contact-state and cycle fields non-null -- without that opt-in mixin they stay
`null` even on a fractured model.

"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional
from unittest import mock

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.models.solution_strategy import ContactIndicators
from porepy.numerics.solvers.newton_solver import NewtonSolver
from porepy.numerics.solvers.telemetry import (
    NonlinearIterationRecorder,
    NonlinearTelemetryRecorder,
    TelemetryConfig,
    read_jsonl_records,
    validate_record,
)
from porepy.time_stepper.scheduler import assemble_default_time_scheduler
from porepy.time_stepper.time_stepper import TimeStepper


class TelemetryFixtureModel(
    SquareDomainOrthogonalFractures, ContactIndicators, pp.Poromechanics
):
    """Poromechanics on a coarse square with one fracture, with contact indicators."""


def run_model(
    telemetry: Optional[NonlinearIterationRecorder], num_time_steps: int = 2
) -> pp.PorePyModel:
    """Run the fixture model to completion, with the given recorder or none.

    Parameters:
        telemetry: Recorder passed to both the nonlinear solver and the time stepper,
            or `None` to run with telemetry switched off entirely.
        num_time_steps: Number of unit time steps to run.

    Returns:
        The model, in its final state.

    """
    model = TelemetryFixtureModel(
        {
            "times_to_export": [],
            "fracture_indices": [0],
            "meshing_arguments": {"cell_size": 0.5},
            "time_manager": pp.TimeManager(
                schedule=[0, num_time_steps], dt_init=1.0, constant_dt=True
            ),
        }
    )
    pp.ModelRunner(
        model,
        time_stepper=TimeStepper(
            scheduler=assemble_default_time_scheduler(time_manager=model.time_manager),
            telemetry=telemetry,
        ),
        nonlinear_solver=NewtonSolver(telemetry=telemetry),
    ).run()
    return model


def recorder_writing_to(path: Path, verbosity: int = 1) -> NonlinearTelemetryRecorder:
    """A recorder writing to `path` at the given verbosity."""
    return NonlinearTelemetryRecorder(
        TelemetryConfig(output_path=path, verbosity=verbosity)
    )


def final_values(model: pp.PorePyModel) -> np.ndarray:
    """The model's converged solution vector."""
    return model.equation_system.get_variable_values(iterate_index=0)


@pytest.fixture(scope="module")
def solution_without_telemetry() -> np.ndarray:
    """The solution the fixture model reaches with telemetry switched off."""
    return final_values(run_model(None))


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Every record written by one full run of the fixture model at verbosity 1."""
    path = tmp_path_factory.mktemp("telemetry") / "telemetry.jsonl"
    run_model(recorder_writing_to(path))
    return read_jsonl_records(path)


class TestNonInterference:
    """Telemetry must be observation only.

    A failure here means recording changed the computed solution -- the worst possible
    failure mode for a diagnostic, since it would invalidate every run made with it
    switched on.

    """

    @pytest.mark.parametrize("verbosity", [0, 1])
    def test_solution_is_bitwise_identical(
        self,
        verbosity: int,
        solution_without_telemetry: np.ndarray,
        tmp_path: Path,
    ) -> None:
        # Exact equality, not allclose: the recorder only reads state, so anything
        # short of bit-for-bit agreement means it perturbed the solve.
        recorder = recorder_writing_to(tmp_path / "telemetry.jsonl", verbosity)

        assert np.array_equal(
            final_values(run_model(recorder)), solution_without_telemetry
        )

    def test_verbosity_zero_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.jsonl"

        run_model(recorder_writing_to(path, verbosity=0))

        assert read_jsonl_records(path) == []

    def test_recording_never_assembles_the_system(self, tmp_path: Path) -> None:
        # Assembling inside a recorder call would cost a full extra assembly per
        # iteration, and could record a state other than the one the solver saw.
        # Cheap AD evaluation of a diagnostic operator stays allowed, as the
        # contact-state fields need it.
        recorder = _GuardedRecorder(recorder_writing_to(tmp_path / "telemetry.jsonl"))
        assemble = pp.ad.EquationSystem.assemble

        def guarded_assemble(self, *args, **kwargs):
            assert not recorder.in_progress, "Telemetry recording assembled the system."
            return assemble(self, *args, **kwargs)

        with mock.patch.object(pp.ad.EquationSystem, "assemble", guarded_assemble):
            run_model(recorder)


class _GuardedRecorder:
    """Recorder wrapper that reports whether a recorder call is in progress.

    Lets a test tell "the solver assembled the system" from "the recorder did", which
    is otherwise invisible from inside `EquationSystem.assemble`.

    """

    def __init__(self, recorder: NonlinearIterationRecorder) -> None:
        self._recorder = recorder
        self.in_progress = False

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self._recorder, name)

        def guarded(*args: Any, **kwargs: Any) -> Any:
            self.in_progress = True
            try:
                return method(*args, **kwargs)
            finally:
                self.in_progress = False

        return guarded


class TestRecordedRun:
    """What a real run's records look like.

    A failure here means the records a live simulation produces no longer match what
    `validate_record` and the schema reference describe, even though the unit tests
    driving the recorder directly still pass.

    """

    def test_every_record_validates(self, recorded_run: list[dict[str, Any]]) -> None:
        for record in recorded_run:
            validate_record(record)

    def test_both_record_types_are_produced(
        self, recorded_run: list[dict[str, Any]]
    ) -> None:
        types = [record["record_type"] for record in recorded_run]

        assert types.count("time_step") == 2
        assert types.count("iteration") > 0

    def test_iteration_indices_restart_after_each_time_step(
        self, recorded_run: list[dict[str, Any]]
    ) -> None:
        # Records are written in solve order, so each time_step record closes off a
        # run of iterations that has to start at 1 and increase by one.
        indices_per_step: list[list[int]] = [[]]
        for record in recorded_run:
            if record["record_type"] == "time_step":
                indices_per_step.append([])
            else:
                indices_per_step[-1].append(record["iteration_index"])

        for indices in indices_per_step:
            assert indices == list(range(1, len(indices) + 1))

    def test_contact_fields_are_populated(
        self, recorded_run: list[dict[str, Any]]
    ) -> None:
        # The fixture model mixes in ContactIndicators; without it both fields would
        # be null on every record, and this test is what would notice.
        iterations = [
            record for record in recorded_run if record["record_type"] == "iteration"
        ]

        for record in iterations:
            assert record["contact_state"]["num_total"] == 2
            assert record["cycle"]["active_set_hash"]


# Run in a subprocess so it can be killed outright. It imports this module rather than
# redefining the model, so the crashed run is the same run the other tests make.
_CRASHING_RUN = """
import sys

sys.path.insert(0, {test_directory!r})
from test_telemetry_integration import recorder_writing_to, run_model

run_model(recorder_writing_to({output_path!r}), num_time_steps=200)
"""


class TestCrashRobustness:
    """Records already written must survive the process dying mid-solve.

    A failure here means a run that crashes -- which is exactly the run whose telemetry
    is most worth having -- leaves a file that cannot be read.

    """

    def test_records_survive_sigkill(self, tmp_path: Path) -> None:
        output_path = tmp_path / "telemetry.jsonl"
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _CRASHING_RUN.format(
                    test_directory=str(Path(__file__).parent),
                    output_path=str(output_path),
                ),
            ]
        )
        try:
            _wait_for_records(output_path, count=3, process=process)
            os.kill(process.pid, signal.SIGKILL)
        finally:
            process.wait(timeout=60)

        records = read_jsonl_records(output_path)

        assert len(records) >= 3
        for record in records:
            validate_record(record)


def _wait_for_records(
    path: Path, count: int, process: subprocess.Popen, timeout: float = 300.0
) -> None:
    """Block until `path` holds at least `count` newline-terminated records.

    Raises:
        AssertionError: If the process exits first, or the timeout expires.

    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().count("\n") >= count:
            return
        assert process.poll() is None, "The run finished before it could be killed."
        time.sleep(0.05)
    raise AssertionError(f"No {count} records within {timeout} s.")
