import math

from chessblunder.labelling import win_prob, win_prob_drop


def test_equal_position_is_fifty_percent():
    assert math.isclose(win_prob(0), 0.5)


def test_win_prob_is_monotonic_and_bounded():
    assert 0 < win_prob(-1000) < win_prob(0) < win_prob(1000) < 1


def test_white_blunder_has_large_positive_drop():
    # White goes from +1.00 to -3.00 pawns
    assert win_prob_drop(100, -300, mover_is_white=True) > 0.3


def test_black_perspective_is_flipped():
    # Eval moving from -1.00 to +3.00 (White's POV) is a blunder by Black
    assert win_prob_drop(-100, 300, mover_is_white=False) > 0.3


def test_improving_position_is_negative_drop():
    assert win_prob_drop(0, 200, mover_is_white=True) < 0
