"""Classification of time-step attempts from solver telemetry.

Every attempt recorded in a telemetry log gets a cost category, and every costly one
(failed or expensive) gets zero or more mechanism labels, each carrying the numbers that
triggered it. Every rule is a threshold on recorded quantities, collected in
:class:`ClassificationThresholds`, so a classification is reproducible and can be
recalibrated without reading residual histories by hand.

Mechanism labels, checked for costly attempts only:

- ``L``: a linear solve failed.
- ``K``: a divergence test ended the attempt without any sign of a blow-up.
- ``D``: an iterate violated a declared variable bound.
- ``A``: contact states cycled, or kept flipping while the iteration made no progress.
- ``T``: many contact states changed, without the stagnation of ``A``.
- ``C-lag``: once the contact state had settled, most variable blocks converged only
  linearly, at a steady rate, the signature of a coupling the Jacobian lags.
- ``O``: once the contact state had settled, some variable block's increment
  alternated up and down by large factors instead of decreasing.
- ``C-amp``: the attempt blew up while fracture aperture or permeability changed by a
  large factor within one iteration.

The first two are artefacts of the linear solver and of the convergence criteria; they
are labels of their own because they are otherwise read as physics. A costly attempt
with no label is reported as such.

Across attempts, :func:`classify_run` also flags stretches where the step size was cut
by a large factor without the iteration count falling: a physical nonlinearity gets
easier as the step shrinks, while a lagged derivative or an over-strict convergence
criterion does not.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, cast

import numpy as np

__all__ = [
    "ClassificationThresholds",
    "Label",
    "AttemptClassification",
    "RunClassification",
    "classify_run",
]

FAILED = "failed"
EXPENSIVE = "expensive"
NORMAL = "normal"
UNFINISHED = "unfinished"


@dataclass(frozen=True)
class ClassificationThresholds:
    """The thresholds of every classification rule.

    Parameters:
        expensive_iterations: An accepted attempt with more iterations than this is
            expensive. Set it to the upper end of the time step controller's optimal
            iteration range, above which the controller shrinks the step; the default
            is that of :class:`~porepy.TimeManager`.
        growth_bound: Factor by which a block's residual must grow within an attempt,
            relative to its first iteration, to count as a blow-up (``K``, ``C-amp``).
        stagnation_ratio: Ratio of successive largest relative increments at or above
            which an iteration counts as making no progress (``A``).
        stagnation_iterations: Consecutive flipping iterations without progress needed
            for ``A``.
        transition_fraction: Fraction of the fracture cells that must change contact
            state within one iteration for ``T``.
        round_off: Relative increment below which a variable block is converged to
            round-off, and left out of the convergence-rate estimate.
        significant_increment: Relative increment a block must exceed at the start of
            the settled tail to be assessed at all; smaller blocks carry no information
            about the rate.
        linear_tail_points: Increment ratios needed on a block's settled tail before
            its convergence is assessed (``C-lag``).
        linear_fit: Smallest coefficient of determination of a straight-line fit to the
            logarithm of a block's relative increments for the decay to count as steady
            (``C-lag``). Oscillating or erratic sequences fall below it.
        linear_rate_range: Range of the fitted reduction per iteration within which a
            steady decay counts as linear convergence (``C-lag``). Below the range it
            gains two orders of magnitude per iteration, too fast to cost iterations
            and indistinguishable from Newton's own convergence ending at round-off;
            above it the block is barely converging.
        linear_order_limit: Largest median ratio of successive logarithmic decrements
            for the decay to count as linear rather than accelerating; Newton's
            quadratic convergence doubles the decrement every iteration (``C-lag``).
        linear_block_fraction: Fraction of the assessed blocks that must converge
            linearly for ``C-lag``. A single linear block is as likely to be noise.
        oscillation_factor: Factor by which a block's relative increment must jump, up
            or down, for a step to count towards an oscillation (``O``).
        oscillation_steps: Consecutive alternating jumps needed for ``O``.
        amplification_factor: Factor by which fracture aperture or permeability must
            change in some cell within one iteration for ``C-amp``.
        insensitivity_dt_factor: Factor by which the step size must shrink over a run
            of accepted attempts before its effect on iterations is judged.
        largest_controller_cut: Largest factor by which the controller shrinks the step
            from one accepted attempt to the next. A larger drop is taken to be imposed
            by the schedule (a step shortened to hit a schedule point), and ends the
            run.
        insensitivity_iteration_drop: Largest fractional drop in iterations over such a
            run for it to count as insensitive to the step size.

    """

    expensive_iterations: int = 7
    growth_bound: float = 1e3
    stagnation_ratio: float = 0.5
    stagnation_iterations: int = 3
    transition_fraction: float = 0.1
    round_off: float = 1e-14
    significant_increment: float = 1e-10
    linear_tail_points: int = 3
    linear_fit: float = 0.9
    linear_rate_range: tuple[float, float] = (1e-2, 0.5)
    linear_order_limit: float = 1.5
    linear_block_fraction: float = 0.5
    oscillation_factor: float = 5.0
    oscillation_steps: int = 3
    amplification_factor: float = 2.0
    insensitivity_dt_factor: float = 8.0
    largest_controller_cut: float = 10.0
    insensitivity_iteration_drop: float = 0.25


@dataclass
class Label:
    """A mechanism label and the recorded numbers that triggered it."""

    code: str
    """One of ``L``, ``K``, ``D``, ``A``, ``T``, ``C-lag``, ``O``, ``C-amp``."""
    evidence: dict[str, Any]
    """The quantities the rule tested, by name."""


@dataclass
class AttemptClassification:
    """Classification of one time-step attempt."""

    index: int
    """Position of the attempt in the log, counting from 0."""
    cost: str
    """``failed``, ``expensive``, ``normal``, or ``unfinished`` for iterations the log
    ends in without a time-step record."""
    num_iterations: int
    """Nonlinear iterations the attempt took."""
    dt: Optional[float]
    """Step size, if the log records it (it does for accepted attempts)."""
    time: Optional[float]
    """Time at the start of the step, if recorded."""
    labels: list[Label] = field(default_factory=list)
    """Mechanism labels, for failed and expensive attempts only."""


@dataclass
class RunClassification:
    """Classification of every attempt in a log, and the run-level signature."""

    attempts: list[AttemptClassification]
    """One entry per attempt, in log order."""
    step_size_insensitive: list[dict[str, Any]]
    """Stretches of accepted attempts over which the step size shrank by at least
    ``insensitivity_dt_factor`` while iterations fell by at most
    ``insensitivity_iteration_drop``, with the step sizes and iteration counts at their
    ends."""


def classify_run(
    records: list[dict[str, Any]],
    thresholds: ClassificationThresholds = ClassificationThresholds(),
) -> RunClassification:
    """Classify every time-step attempt in a telemetry log.

    Parameters:
        records: The log's records, as read by
            :func:`~porepy.numerics.solvers.telemetry.read_jsonl_records`.
        thresholds: The thresholds of the rules.

    Returns:
        The classification.

    """
    attempts = [
        _classify_attempt(index, step, iterations, thresholds)
        for index, (step, iterations) in enumerate(_attempts(records))
    ]
    return RunClassification(
        attempts=attempts,
        step_size_insensitive=_step_size_insensitive_stretches(attempts, thresholds),
    )


def _attempts(
    records: list[dict[str, Any]],
) -> Iterator[tuple[Optional[dict[str, Any]], list[dict[str, Any]]]]:
    """Group iteration records into attempts, each closed by its time-step record.

    Iterations after the last time-step record form a final attempt without one.

    """
    iterations: list[dict[str, Any]] = []
    for record in records:
        if record["record_type"] == "iteration":
            iterations.append(record)
        elif record["record_type"] == "time_step":
            yield record, iterations
            iterations = []
    if iterations:
        yield None, iterations


def _classify_attempt(
    index: int,
    step: Optional[dict[str, Any]],
    iterations: list[dict[str, Any]],
    thresholds: ClassificationThresholds,
) -> AttemptClassification:
    """Assign the cost category and, for a costly attempt, the mechanism labels."""
    cost = _cost(step, len(iterations), thresholds)
    classification = AttemptClassification(
        index=index,
        cost=cost,
        num_iterations=len(iterations),
        dt=None if step is None else step.get("dt"),
        time=None if step is None else step.get("time"),
    )
    # An attempt can fail before its first iteration is recorded, e.g. when the first
    # linear solve fails; it then carries no evidence for any mechanism.
    if cost in (FAILED, EXPENSIVE) and iterations:
        classification.labels = _mechanism_labels(
            iterations, failed=cost == FAILED, thresholds=thresholds
        )
    return classification


def _cost(
    step: Optional[dict[str, Any]],
    num_iterations: int,
    thresholds: ClassificationThresholds,
) -> str:
    if step is None:
        return UNFINISHED
    if step["status"] != "successful":
        return FAILED
    if num_iterations > thresholds.expensive_iterations:
        return EXPENSIVE
    return NORMAL


def _mechanism_labels(
    iterations: list[dict[str, Any]],
    failed: bool,
    thresholds: ClassificationThresholds,
) -> list[Label]:
    """Apply every rule; artefact rules first, so that they are read first."""
    blow_up = _blow_up(iterations, thresholds)
    linear_solver = _linear_solver_label(iterations)
    excursion = _excursion_label(iterations)
    candidates = [
        linear_solver,
        (
            _criterion_label(iterations, blow_up)
            if failed and linear_solver is None and excursion is None
            else None
        ),
        excursion,
    ]
    active_set = _active_set_label(iterations, thresholds)
    candidates += [
        active_set,
        None if active_set else _transition_label(iterations, thresholds),
        _lagged_coupling_label(iterations, thresholds),
        _oscillation_label(iterations, thresholds),
        _amplification_label(iterations, thresholds) if blow_up else None,
    ]
    return [label for label in candidates if label is not None]


def _blow_up(
    iterations: list[dict[str, Any]], thresholds: ClassificationThresholds
) -> Optional[dict[str, Any]]:
    """The largest growth of any block's residual over the attempt, if it exceeds the
    bound or some residual or increment is not finite; otherwise `None`."""
    first, last = iterations[0], iterations[-1]
    largest_growth, block = 0.0, None
    for name, fields in last["blocks"].items():
        start = first["blocks"].get(name, {}).get("residual_inf_norm")
        end = fields["residual_inf_norm"]
        if not np.isfinite(end):
            return {"block": name, "growth": float("inf")}
        if start and end / start > largest_growth:
            largest_growth, block = end / start, name
    for record in iterations:
        for name, norms in (record.get("increments") or {}).items():
            if not np.isfinite(norms["increment_inf_norm"]):
                return {"block": name, "growth": float("inf")}
    if largest_growth > thresholds.growth_bound:
        return {"block": block, "growth": largest_growth}
    return None


def _linear_solver_label(iterations: list[dict[str, Any]]) -> Optional[Label]:
    failures = [
        record["iteration_index"]
        for record in iterations
        if not record["linear_solver"]["success"]
    ]
    if not failures:
        return None
    return Label("L", {"failed_solves_at_iterations": failures})


def _criterion_label(
    iterations: list[dict[str, Any]], blow_up: Optional[dict[str, Any]]
) -> Optional[Label]:
    """A divergence test that fired although nothing blew up.

    Growth can only be judged over at least two iterations; an attempt ended after one
    gives no evidence either way.

    """
    fired = [
        name
        for name, status in iterations[-1]["divergence_status"].items()
        if status == "failed"
    ]
    if not fired or blow_up is not None or len(iterations) < 2:
        return None
    return Label("K", {"divergence_tests": fired})


def _excursion_label(iterations: list[dict[str, Any]]) -> Optional[Label]:
    """Bound violations by iterates of successful Newton steps.

    A violation at or after a failed linear solve is that failure's consequence, since
    its update is not a Newton step, and is left to ``L``.

    """
    violations = [
        (
            record["iteration_index"],
            sum(
                counts["num_below"] + counts["num_above"]
                for counts in (record.get("admissibility") or {}).values()
            ),
        )
        for record in _before_first_failed_solve(iterations)
    ]
    violating = [(index, count) for index, count in violations if count > 0]
    if not violating:
        return None
    return Label(
        "D",
        {
            "first_iteration": violating[0][0],
            "max_violating_dofs": max(count for _, count in violating),
        },
    )


def _before_first_failed_solve(
    iterations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for position, record in enumerate(iterations):
        if not record["linear_solver"]["success"]:
            return iterations[:position]
    return iterations


def _active_set_label(
    iterations: list[dict[str, Any]], thresholds: ClassificationThresholds
) -> Optional[Label]:
    """Cycling, or flips persisting over iterations that make no progress."""
    cycling = [
        record["iteration_index"]
        for record in iterations
        if (record.get("cycle") or {}).get("cycling_cells")
    ]
    if cycling:
        return Label("A", {"cycling_at_iterations": cycling})

    progress = _progress_ratios(iterations)
    longest, current = 0, 0
    for record, ratio in zip(iterations, progress):
        flipping = (record.get("cycle") or {}).get("num_flipped_cells", 0) > 0
        stagnating = ratio is not None and ratio >= thresholds.stagnation_ratio
        current = current + 1 if flipping and stagnating else 0
        longest = max(longest, current)
    if longest >= thresholds.stagnation_iterations:
        return Label("A", {"stagnant_flipping_iterations": longest})
    return None


def _progress_ratios(iterations: list[dict[str, Any]]) -> list[Optional[float]]:
    """Ratio of each iteration's largest relative increment to the previous one's;
    `None` where either is missing."""
    largest = [_largest_relative_increment(record) for record in iterations]
    ratios: list[Optional[float]] = [None]
    for previous, current in zip(largest[:-1], largest[1:]):
        ratios.append(current / previous if previous and current is not None else None)
    return ratios


def _largest_relative_increment(record: dict[str, Any]) -> Optional[float]:
    relative = [
        norms["increment_inf_norm"] / norms["value_inf_norm"]
        for norms in (record.get("increments") or {}).values()
        if norms["value_inf_norm"] > 0
    ]
    return max(relative) if relative else None


def _transition_label(
    iterations: list[dict[str, Any]], thresholds: ClassificationThresholds
) -> Optional[Label]:
    flips = [
        (record.get("cycle") or {}).get("num_flipped_cells", 0) for record in iterations
    ]
    num_cells = max(
        (
            (record.get("contact_state") or {}).get("num_total", 0)
            for record in iterations
        ),
        default=0,
    )
    if num_cells == 0 or max(flips) < max(
        1, thresholds.transition_fraction * num_cells
    ):
        return None
    return Label(
        "T",
        {
            "max_flipped_cells": max(flips),
            "fracture_cells": num_cells,
            "flipping_iterations": sum(1 for count in flips if count > 0),
        },
    )


def _lagged_coupling_label(
    iterations: list[dict[str, Any]], thresholds: ClassificationThresholds
) -> Optional[Label]:
    """Steady linear convergence of most variable blocks once contact has settled.

    The settled tail is the iterations after the last one in which a contact state
    changed; before that, slow convergence is the active set's doing. A lagged coupling
    slows the whole coupled system alike, so the rule asks for most blocks to decay
    linearly, not one.

    """
    sequences = _settled_relative_increments(iterations, thresholds)
    assessed = {
        name: _linear_decay(values, thresholds) for name, values in sequences.items()
    }
    linear = {name: rate for name, rate in assessed.items() if rate is not None}
    if len(assessed) < 2 or len(linear) < thresholds.linear_block_fraction * len(
        assessed
    ):
        return None
    return Label(
        "C-lag",
        {
            "median_rate": float(np.median(list(linear.values()))),
            "linear_blocks": sorted(linear),
            "assessed_blocks": len(assessed),
        },
    )


def _settled_relative_increments(
    iterations: list[dict[str, Any]], thresholds: ClassificationThresholds
) -> dict[str, list[float]]:
    """Each significant block's relative increments over the settled tail, above
    round-off, for blocks with enough of them to assess."""
    last_flip = -1
    for position, record in enumerate(iterations):
        if (record.get("cycle") or {}).get("num_flipped_cells", 0) > 0:
            last_flip = position
    tail = iterations[last_flip + 1 :]
    sequences: dict[str, list[float]] = {}
    for name in (tail[0].get("increments") or {}) if tail else {}:
        values = [
            norms["increment_inf_norm"] / norms["value_inf_norm"]
            for norms in ((record.get("increments") or {}).get(name) for record in tail)
            if norms is not None and norms["value_inf_norm"] > 0
        ]
        values = [value for value in values if value > thresholds.round_off]
        if (
            len(values) > thresholds.linear_tail_points
            and values[0] >= thresholds.significant_increment
        ):
            sequences[name] = values
    return sequences


def _linear_decay(
    values: list[float], thresholds: ClassificationThresholds
) -> Optional[float]:
    """The reduction per iteration of a steady, non-accelerating decay; `None` if the
    decay is erratic, accelerating, or outside the linear range."""
    logs = np.log(values)
    steps = np.arange(len(logs))
    slope, intercept = np.polyfit(steps, logs, 1)
    residual = logs - (slope * steps + intercept)
    spread = np.sum((logs - logs.mean()) ** 2)
    fit = 1.0 - np.sum(residual**2) / spread if spread > 0 else 0.0
    decrements = -np.diff(logs)
    accelerations = decrements[1:] / decrements[:-1]
    rate = float(np.exp(slope))
    low, high = thresholds.linear_rate_range
    if (
        fit >= thresholds.linear_fit
        and low <= rate <= high
        and np.all(decrements > 0)
        and np.median(accelerations) <= thresholds.linear_order_limit
    ):
        return rate
    return None


def _oscillation_label(
    iterations: list[dict[str, Any]], thresholds: ClassificationThresholds
) -> Optional[Label]:
    """Blocks whose settled relative increments alternate up and down by large
    factors, and the longest such alternation."""
    oscillating: dict[str, int] = {}
    for name, values in _settled_relative_increments(iterations, thresholds).items():
        longest = _longest_alternation(values, thresholds.oscillation_factor)
        if longest >= thresholds.oscillation_steps:
            oscillating[name] = longest
    if not oscillating:
        return None
    return Label("O", {"alternating_steps_by_block": oscillating})


def _longest_alternation(values: list[float], factor: float) -> int:
    """Longest run of consecutive jumps by at least `factor`, alternating in
    direction."""
    longest, current, previous_direction = 0, 0, 0
    for before, after in zip(values[:-1], values[1:]):
        ratio = after / before
        direction = 1 if ratio >= factor else -1 if ratio <= 1.0 / factor else 0
        if direction != 0 and direction != previous_direction:
            current = current + 1 if previous_direction != 0 else 1
        else:
            current = 1 if direction != 0 else 0
        previous_direction = direction
        longest = max(longest, current)
    return longest


def _amplification_label(
    iterations: list[dict[str, Any]], thresholds: ClassificationThresholds
) -> Optional[Label]:
    """The largest one-iteration change of fracture aperture or permeability, if it
    reaches the amplification factor. As for ``D``, changes at or after a failed linear
    solve are that failure's consequence and are left out."""
    largest: Optional[dict[str, Any]] = None
    for record in _before_first_failed_solve(iterations):
        for quantity, fields in (record.get("fracture_flow") or {}).items():
            factor = (fields or {}).get("max_change_factor")
            if factor is not None and (largest is None or factor > largest["factor"]):
                largest = {
                    "quantity": quantity,
                    "factor": factor,
                    "iteration": record["iteration_index"],
                    "location": fields["max_change_location"],
                }
    if largest is None or largest["factor"] < thresholds.amplification_factor:
        return None
    return Label("C-amp", largest)


def _step_size_insensitive_stretches(
    attempts: list[AttemptClassification], thresholds: ClassificationThresholds
) -> list[dict[str, Any]]:
    """Stretches that shrink the step by the dt factor while iterations fall by at most
    the allowed drop."""
    flagged = []
    for stretch in _shrinking_stretches(attempts, thresholds):
        (first, first_dt), (last, last_dt) = stretch[0], stretch[-1]
        drop = (first.num_iterations - last.num_iterations) / first.num_iterations
        if (
            first_dt / last_dt >= thresholds.insensitivity_dt_factor
            and drop <= thresholds.insensitivity_iteration_drop
        ):
            flagged.append(
                {
                    "first_attempt": first.index,
                    "last_attempt": last.index,
                    "dt": [first_dt, last_dt],
                    "iterations": [first.num_iterations, last.num_iterations],
                }
            )
    return flagged


def _shrinking_stretches(
    attempts: list[AttemptClassification], thresholds: ClassificationThresholds
) -> list[list[tuple[AttemptClassification, float]]]:
    """Maximal runs of at least two consecutive accepted attempts whose step size never
    grows and never drops faster than the controller cuts, each attempt paired with its
    step size. A failed attempt ends a run: the step it cuts is chopped, not chosen by
    the controller."""
    stretches = []
    current: list[tuple[AttemptClassification, float]] = []
    for attempt in attempts:
        accepted = attempt.cost in (NORMAL, EXPENSIVE) and attempt.dt is not None
        if (
            accepted
            and current
            and current[-1][1] / thresholds.largest_controller_cut
            <= cast(float, attempt.dt)
            <= current[-1][1]
        ):
            current.append((attempt, cast(float, attempt.dt)))
            continue
        if len(current) >= 2:
            stretches.append(current)
        current = [(attempt, cast(float, attempt.dt))] if accepted else []
    if len(current) >= 2:
        stretches.append(current)
    return stretches
