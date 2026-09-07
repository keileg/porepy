"""Detection of repeated active sets and per-cell chattering in a nonlinear loop.

A contact-mechanics nonlinear solve can fail to converge by cycling: the set of open,
sticking and slipping cells returns to a state it already visited, so the solver
retraces the same iterates indefinitely. It can also converge slowly because a few
individual cells flip state on nearly every iteration while the rest of the domain is
settled. This module detects and reports both; acting on either is a separate concern.

Deliberately independent of the rest of the telemetry machinery: it works on a plain
array of per-cell integer state labels, and knows nothing about models, records, or how
those labels were produced.

"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "ActiveSetHistory",
    "CycleDetection",
    "hash_active_set",
]


def hash_active_set(labels: np.ndarray) -> str:
    """Hash a per-cell state-label array into a short, stable identifier.

    Uses `hashlib.blake2b` rather than the builtin `hash()`, which is salted per
    process (`PYTHONHASHSEED`): a builtin hash would differ between runs of the same
    simulation, making recorded active-set identifiers useless for comparing runs or
    for detecting a cycle across a restart.

    Parameters:
        labels: Per-cell integer state labels. The array's byte representation is
            hashed directly, so its dtype must be fixed-width for the digest to be
            reproducible across machines.

    Returns:
        A 16-character hex digest.

    """
    return hashlib.blake2b(labels.tobytes(), digest_size=8).hexdigest()


@dataclass(frozen=True)
class CycleDetection:
    """What the recent active-set history says about the current iteration.

    Parameters:
        active_set_hash: Identifier of the current active set, from
            :func:`hash_active_set`.
        repeat_length: How many iterations back an identical active set was last seen,
            or `None` if it was not seen within the retained window. A value of `k`
            means the solver has returned to the state it held `k` iterations ago,
            i.e. it is cycling with period `k`.
        cycling_cells: Cells whose state differs from the current one at some point
            inside the detected cycle -- the cells actually driving it. Empty when
            `repeat_length` is `None`.
        num_flipped_cells: Cells whose state changed since the previous iteration.
        chattering_cells: Cells that have changed state more often than the configured
            threshold since the history was last reset.

    """

    active_set_hash: str
    repeat_length: Optional[int] = None
    cycling_cells: list[int] = field(default_factory=list)
    num_flipped_cells: int = 0
    chattering_cells: list[int] = field(default_factory=list)


class ActiveSetHistory:
    """Rolling history of recent active sets, detecting repeats and chattering.

    Parameters:
        window: How many trailing iterations to keep and search for a repeat of the
            current active set. A cycle longer than this is not detected.
        flip_threshold: How many state changes a cell may accumulate before it is
            reported as chattering.

    Raises:
        ValueError: If `window` is not positive or `flip_threshold` is negative.

    """

    def __init__(self, window: int, flip_threshold: int) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}.")
        if flip_threshold < 0:
            raise ValueError(
                f"flip_threshold must not be negative, got {flip_threshold}."
            )
        self.window = window
        """Number of trailing iterations searched for a repeat."""
        self.flip_threshold = flip_threshold
        """State changes a cell may accumulate before being reported as chattering."""
        self._recent: deque[tuple[str, np.ndarray]] = deque(maxlen=window)
        self._flip_counts: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Forget the history, at the start of a new nonlinear loop.

        Without this, a repeat would be reported across the boundary between two
        separate solves that happen to pass through the same active set, which says
        nothing about either one cycling.

        """
        self._recent.clear()
        self._flip_counts = None

    def record(self, labels: np.ndarray) -> CycleDetection:
        """Add one iteration's active set to the history and report on it.

        Parameters:
            labels: Per-cell integer state labels for the current iterate.

        Returns:
            What the history says about this iteration.

        """
        active_set_hash = hash_active_set(labels)
        repeat_length, cycling_cells = self._find_repeat(active_set_hash, labels)
        num_flipped_cells, chattering_cells = self._count_flips(labels)

        self._recent.append((active_set_hash, labels.copy()))

        return CycleDetection(
            active_set_hash=active_set_hash,
            repeat_length=repeat_length,
            cycling_cells=cycling_cells,
            num_flipped_cells=num_flipped_cells,
            chattering_cells=chattering_cells,
        )

    def _find_repeat(
        self, active_set_hash: str, labels: np.ndarray
    ) -> tuple[Optional[int], list[int]]:
        """Look for the current active set among the retained ones.

        Returns the distance back to the most recent identical set, and the cells that
        take a different state somewhere inside the resulting cycle. Those intermediate
        cells are the informative part: the repeated set itself is by definition equal
        to the current one, so diffing against it would report nothing.

        """
        for distance, (previous_hash, _) in enumerate(reversed(self._recent), start=1):
            if previous_hash != active_set_hash:
                continue
            # Everything after the repeated entry is traversed within the cycle.
            intermediate = list(self._recent)[len(self._recent) - distance + 1 :]
            differing = np.zeros(labels.size, dtype=bool)
            for _, intermediate_labels in intermediate:
                differing |= intermediate_labels != labels
            return distance, [int(cell) for cell in np.flatnonzero(differing)]
        return None, []

    def _count_flips(self, labels: np.ndarray) -> tuple[int, list[int]]:
        """Update the per-cell flip counts against the previous iteration."""
        if not self._recent:
            self._flip_counts = np.zeros(labels.size, dtype=int)
            return 0, []

        _, previous_labels = self._recent[-1]
        flipped = previous_labels != labels
        assert self._flip_counts is not None, "Set on the first recorded iteration."
        self._flip_counts += flipped

        chattering = np.flatnonzero(self._flip_counts > self.flip_threshold)
        return int(np.sum(flipped)), [int(cell) for cell in chattering]
