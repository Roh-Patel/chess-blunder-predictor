"""Convert engine evaluations into win probabilities and blunder labels."""
import math

MATE_CP = 10_000      # centipawn value used for forced mates
BLUNDER_DROP = 0.15   # win-probability drop of 15% is a blunder


def win_prob(cp: float) -> float:
    """Lichess's centipawn-to-win-probability curve; cp is from the player's point of view."""
    return 0.5 + 0.5 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def win_prob_drop(cp_before_white: float, cp_after_white: float, mover_is_white: bool) -> float:
    """How much win probability the mover lost with their move (negative = gained)."""
    sign = 1 if mover_is_white else -1
    return win_prob(sign * cp_before_white) - win_prob(sign * cp_after_white)
