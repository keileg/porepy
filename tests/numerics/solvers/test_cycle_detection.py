"""Unit tests for active-set cycle and chattering detection.

Exercised on hand-built label arrays: this module deliberately knows nothing about
models, so its tests need none either.

"""

import numpy as np
import pytest

from porepy.numerics.solvers.cycle_detection import ActiveSetHistory, hash_active_set


def _labels(*values: int) -> np.ndarray:
    """A label array in the fixed-width dtype the hashing relies on."""
    return np.array(values, dtype=np.int8)


class TestHashActiveSet:
    """Tests for `hash_active_set`.

    A failure of the pinned-digest test means the encoding changed, which silently
    invalidates comparison of active-set identifiers against any previously recorded
    run.

    """

    def test_digest_is_pinned(self) -> None:
        # Pinned rather than merely self-consistent: a change of dtype, byte order or
        # digest size would otherwise pass unnoticed while breaking cross-run
        # comparison.
        assert hash_active_set(_labels(0, 1, 2)) == "e879c20168a0a436"

    def test_equal_label_arrays_hash_equal(self) -> None:
        assert hash_active_set(_labels(0, 1, 2)) == hash_active_set(_labels(0, 1, 2))

    def test_different_label_arrays_hash_differently(self) -> None:
        assert hash_active_set(_labels(0, 1, 2)) != hash_active_set(_labels(0, 1, 1))

    def test_hash_is_not_the_salted_builtin(self) -> None:
        # The builtin hash() is salted per process, so a digest that agreed with it
        # would not be reproducible across runs.
        labels = _labels(0, 1, 2)
        assert hash_active_set(labels) != str(hash(labels.tobytes()))


class TestActiveSetHistoryConstruction:
    """Validation of the history's own parameters."""

    @pytest.mark.parametrize("window", [0, -1])
    def test_non_positive_window_raises(self, window: int) -> None:
        with pytest.raises(ValueError, match="window"):
            ActiveSetHistory(window=window, flip_threshold=2)

    def test_negative_flip_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="flip_threshold"):
            ActiveSetHistory(window=4, flip_threshold=-1)


class TestCycleDetection:
    """Tests for detecting a repeated active set.

    A failure here most likely means a genuinely cycling solve would no longer be
    flagged, or a converging one would be flagged spuriously.

    """

    def test_no_repeat_on_distinct_states(self) -> None:
        history = ActiveSetHistory(window=4, flip_threshold=2)

        assert history.record(_labels(0, 0)).repeat_length is None
        assert history.record(_labels(0, 1)).repeat_length is None
        assert history.record(_labels(1, 1)).repeat_length is None

    def test_settled_active_set_repeats_with_length_one(self) -> None:
        # A converging solve stops changing state, so every iteration after the first
        # repeats the previous one. Length 1 is therefore the signature of a settled
        # active set, not of cycling -- only length >= 2 means the solver left a state
        # and came back. Anything reading these records has to make that distinction.
        history = ActiveSetHistory(window=4, flip_threshold=2)
        history.record(_labels(0, 1))

        detection = history.record(_labels(0, 1))

        assert detection.repeat_length == 1
        # Nothing sits between the repeated state and this one.
        assert detection.cycling_cells == []

    def test_period_two_cycle(self) -> None:
        history = ActiveSetHistory(window=4, flip_threshold=2)
        history.record(_labels(0, 1))  # A
        history.record(_labels(0, 2))  # B

        detection = history.record(_labels(0, 1))  # A again

        assert detection.repeat_length == 2
        # Cell 1 is the one taking a different state inside the cycle; cell 0 never
        # moves and must not be reported as driving it.
        assert detection.cycling_cells == [1]

    def test_repeat_older_than_the_window_is_not_reported(self) -> None:
        history = ActiveSetHistory(window=2, flip_threshold=5)
        history.record(_labels(0, 0))  # Falls out of the window below.
        history.record(_labels(0, 1))
        history.record(_labels(0, 2))

        assert history.record(_labels(0, 0)).repeat_length is None

    def test_reset_forgets_previous_states(self) -> None:
        history = ActiveSetHistory(window=4, flip_threshold=2)
        history.record(_labels(0, 1))

        history.reset()

        # Without the reset this would be a repeat of length 1; across a loop
        # boundary it says nothing about either loop cycling.
        assert history.record(_labels(0, 1)).repeat_length is None


class TestFlipCounting:
    """Tests for per-cell chattering counts."""

    def test_first_iteration_has_no_flips(self) -> None:
        history = ActiveSetHistory(window=4, flip_threshold=1)

        detection = history.record(_labels(0, 1, 2))

        assert detection.num_flipped_cells == 0
        assert detection.chattering_cells == []

    def test_flips_counted_against_the_previous_iteration(self) -> None:
        history = ActiveSetHistory(window=4, flip_threshold=5)
        history.record(_labels(0, 1, 2))

        detection = history.record(_labels(1, 1, 0))

        assert detection.num_flipped_cells == 2

    def test_cells_over_the_threshold_are_reported_as_chattering(self) -> None:
        history = ActiveSetHistory(window=8, flip_threshold=2)
        # Cell 0 flips on every iteration, cell 1 never does.
        for labels in [(0, 0), (1, 0), (0, 0), (1, 0)]:
            detection = history.record(_labels(*labels))

        # Three flips, one more than the threshold allows.
        assert detection.chattering_cells == [0]

    def test_threshold_is_exclusive(self) -> None:
        history = ActiveSetHistory(window=8, flip_threshold=2)
        for labels in [(0, 0), (1, 0), (0, 0)]:
            detection = history.record(_labels(*labels))

        # Exactly two flips: at the threshold, not over it.
        assert detection.chattering_cells == []

    def test_reset_clears_flip_counts(self) -> None:
        history = ActiveSetHistory(window=8, flip_threshold=1)
        for labels in [(0, 0), (1, 0), (0, 0)]:
            history.record(_labels(*labels))

        history.reset()
        history.record(_labels(0, 0))
        detection = history.record(_labels(1, 0))

        # One flip since the reset, which is not over the threshold.
        assert detection.chattering_cells == []

    def test_recorded_labels_are_not_aliased(self) -> None:
        # The caller may reuse its array; the history must not end up comparing an
        # array against a mutated version of itself.
        history = ActiveSetHistory(window=4, flip_threshold=5)
        labels = _labels(0, 0)
        history.record(labels)
        labels[0] = 1

        assert history.record(_labels(0, 0)).num_flipped_cells == 0
