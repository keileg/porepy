"""Tests for the classification of time-step attempts from telemetry records.

The records are built by hand, holding only the fields the rules read, so that each rule
is tested on exactly the evidence it is meant to act on. A failure most likely means a
rule has drifted from its documented threshold, which would silently relabel every
classified run.

"""

from typing import Any, Optional

import pytest

from porepy.numerics.solvers.step_classification import (
    ClassificationThresholds,
    classify_run,
)

THRESHOLDS = ClassificationThresholds(expensive_iterations=3)


def _iteration(
    index: int,
    *,
    linear_solve_succeeded: bool = True,
    diverged: bool = False,
    residual: float = 1.0,
    relative_increment: Optional[float] = None,
    violations: int = 0,
    flips: int = 0,
    cycling: bool = False,
    aperture_change: Optional[float] = None,
) -> dict[str, Any]:
    """An iteration record with only the fields the classification reads."""
    return {
        "record_type": "iteration",
        "iteration_index": index,
        "linear_solver": {"success": linear_solve_succeeded},
        "divergence_status": {"res_bound": "failed" if diverged else "converged"},
        "blocks": {"energy_balance_equation": {"residual_inf_norm": residual}},
        "increments": (
            None
            if relative_increment is None
            else {
                "temperature": {
                    "increment_inf_norm": relative_increment * 300.0,
                    "value_inf_norm": 300.0,
                }
            }
        ),
        "admissibility": {"temperature": {"num_below": violations, "num_above": 0}},
        "contact_state": {"num_total": 20},
        "cycle": {"num_flipped_cells": flips, "cycling_cells": [3] if cycling else []},
        "fracture_flow": {
            "aperture": {
                "max_change_factor": aperture_change,
                "max_change_location": {"dim": 2, "domain_id": 1, "cell": 4},
            },
            "permeability": None,
        },
    }


def _step(successful: bool, dt: Optional[float] = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": "time_step",
        "status": "successful" if successful else "in_progress",
    }
    if dt is not None:
        record["dt"] = dt
    return record


def _classify_one(
    iterations: list[dict[str, Any]], successful: bool
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Cost and labels (code -> evidence) of a single attempt."""
    [attempt] = classify_run([*iterations, _step(successful)], THRESHOLDS).attempts
    return attempt.cost, {label.code: label.evidence for label in attempt.labels}


class TestCost:
    def test_categories(self) -> None:
        records = [
            _iteration(1),
            _step(successful=True),
            *(_iteration(i) for i in range(1, 5)),
            _step(successful=True),
            _iteration(1),
            _step(successful=False),
            _iteration(1),
        ]
        costs = [a.cost for a in classify_run(records, THRESHOLDS).attempts]
        assert costs == ["normal", "expensive", "failed", "unfinished"]

    def test_normal_attempts_get_no_labels(self) -> None:
        cost, labels = _classify_one(
            [_iteration(1, linear_solve_succeeded=False)], successful=True
        )
        assert cost == "normal"
        assert labels == {}


class TestArtefactLabels:
    def test_linear_solver(self) -> None:
        _, labels = _classify_one(
            [_iteration(1), _iteration(2, linear_solve_succeeded=False)],
            successful=False,
        )
        assert labels["L"] == {"failed_solves_at_iterations": [2]}

    def test_criterion_without_blow_up(self) -> None:
        _, labels = _classify_one(
            [_iteration(1, residual=1.0), _iteration(2, residual=5.0, diverged=True)],
            successful=False,
        )
        assert labels == {"K": {"divergence_tests": ["res_bound"]}}

    @pytest.mark.parametrize(
        "iterations",
        [
            # One iteration: growth cannot be judged.
            [_iteration(1, diverged=True)],
            # The residual grew beyond the bound: a blow-up, not the criterion's doing.
            [_iteration(1, residual=1.0), _iteration(2, residual=1e4, diverged=True)],
            # An excursion explains the divergence.
            [_iteration(1, violations=4), _iteration(2, diverged=True)],
        ],
    )
    def test_no_criterion_label_when_evidence_is_missing_or_explained(
        self, iterations: list[dict[str, Any]]
    ) -> None:
        _, labels = _classify_one(iterations, successful=False)
        assert "K" not in labels


class TestExcursion:
    def test_violation_by_a_newton_step(self) -> None:
        _, labels = _classify_one(
            [_iteration(1), _iteration(2, violations=7)], successful=False
        )
        assert labels["D"] == {"first_iteration": 2, "max_violating_dofs": 7}

    def test_violation_after_a_failed_solve_is_left_to_the_solve(self) -> None:
        _, labels = _classify_one(
            [_iteration(1, linear_solve_succeeded=False, violations=7)],
            successful=False,
        )
        assert "D" not in labels
        assert "L" in labels


class TestContactLabels:
    def test_cycling_is_active_set_combinatorics(self) -> None:
        _, labels = _classify_one(
            [_iteration(1), _iteration(2, flips=1, cycling=True)], successful=False
        )
        assert labels["A"] == {"cycling_at_iterations": [2]}

    def test_flips_without_progress(self) -> None:
        increments = [1e-2, 9e-3, 8e-3, 7e-3]
        _, labels = _classify_one(
            [
                _iteration(i + 1, flips=5, relative_increment=value)
                for i, value in enumerate(increments)
            ],
            successful=False,
        )
        assert labels["A"] == {"stagnant_flipping_iterations": 3}
        assert "T" not in labels

    def test_flips_with_progress_are_a_transition(self) -> None:
        increments = [1e-2, 1e-3, 1e-5, 1e-9]
        _, labels = _classify_one(
            [
                _iteration(i + 1, flips=5, relative_increment=value)
                for i, value in enumerate(increments)
            ],
            successful=True,
        )
        assert "A" not in labels
        assert labels["T"] == {
            "max_flipped_cells": 5,
            "fracture_cells": 20,
            "flipping_iterations": 4,
        }


class TestLaggedCoupling:
    @staticmethod
    def _labels(increments: list[float], flips: Optional[list[int]] = None) -> dict:
        flips = flips or [0] * len(increments)
        _, labels = _classify_one(
            [
                _iteration(i + 1, relative_increment=value, flips=flip)
                for i, (value, flip) in enumerate(zip(increments, flips))
            ],
            successful=True,
        )
        return labels

    def test_linear_convergence(self) -> None:
        evidence = self._labels([1e-3 * 0.2**k for k in range(6)])["C-lag"]
        assert evidence["block"] == "temperature"
        assert evidence["rate"] == pytest.approx(0.2)
        assert evidence["order"] == pytest.approx(1.0)

    def test_quadratic_convergence_is_not_labelled(self) -> None:
        assert "C-lag" not in self._labels([1e-1, 1e-2, 1e-4, 1e-8, 1e-12])

    def test_only_the_tail_after_the_last_flip_counts(self) -> None:
        # Linear throughout, but contact states change until the fourth iteration,
        # leaving a settled tail too short to judge.
        increments = [1e-3 * 0.2**k for k in range(6)]
        labels = self._labels(increments, flips=[3, 3, 3, 3, 0, 0])
        assert "C-lag" not in labels

    def test_round_off_is_left_out(self) -> None:
        # Two linear steps, then the block sits at round-off: too little to judge.
        assert "C-lag" not in self._labels([1e-3, 2e-4, 4e-5, 1e-16, 1e-16, 1e-16])


class TestAmplification:
    def test_blow_up_with_a_large_aperture_change(self) -> None:
        _, labels = _classify_one(
            [
                _iteration(1, residual=1.0),
                _iteration(2, residual=1e5, aperture_change=5.0, diverged=True),
            ],
            successful=False,
        )
        assert labels["C-amp"] == {
            "quantity": "aperture",
            "factor": 5.0,
            "iteration": 2,
            "location": {"dim": 2, "domain_id": 1, "cell": 4},
        }

    def test_no_blow_up_no_amplification(self) -> None:
        _, labels = _classify_one(
            [_iteration(1), _iteration(2, aperture_change=5.0, diverged=True)],
            successful=False,
        )
        assert "C-amp" not in labels


class TestStepSizeInsensitivity:
    @staticmethod
    def _run(steps: list[tuple[float, int]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for dt, num_iterations in steps:
            records += [_iteration(i + 1) for i in range(num_iterations)]
            records.append(_step(successful=True, dt=dt))
        return classify_run(records, THRESHOLDS).step_size_insensitive

    def test_controller_cuts_without_fewer_iterations(self) -> None:
        stretches = self._run([(16.0, 11), (8.0, 11), (4.0, 10), (2.0, 10), (1.0, 10)])
        assert stretches == [
            {
                "first_attempt": 0,
                "last_attempt": 4,
                "dt": [16.0, 1.0],
                "iterations": [11, 10],
            }
        ]

    def test_iterations_that_fall_with_the_step_are_not_flagged(self) -> None:
        assert self._run([(16.0, 11), (8.0, 8), (4.0, 6), (2.0, 5), (1.0, 4)]) == []

    def test_a_drop_imposed_by_the_schedule_ends_the_stretch(self) -> None:
        # 1000 -> 1 in one step is a schedule point, not a controller cut.
        assert self._run([(1000.0, 11), (1.0, 11), (0.5, 11)]) == []
