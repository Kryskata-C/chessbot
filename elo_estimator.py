"""Map between average centipawn loss (ACPL) and an ELO estimate.

The same curve powers two things:

- **Opponent estimation** — watch the opponent's moves, track their ACPL,
  and read off an ELO so the bot knows who it is playing.
- **Bot self-estimation** — feed the bot's *own* chosen-move losses through
  the identical curve to get a live "realized ELO", which the move selector
  compares against the user's target to self-correct.

Calibration anchors (blitz-ish): ACPL 10 -> ~2500, 45 -> ~1500, 95 -> ~1000.
"""

from __future__ import annotations

import math

MIN_ELO = 300
MAX_ELO = 3000


def acpl_to_elo(acpl: float) -> int:
    """Map average centipawn loss to an ELO estimate."""
    if acpl <= 0:
        return MAX_ELO
    elo = 4034 - 667 * math.log(acpl)
    return int(max(MIN_ELO, min(MAX_ELO, elo)))


def elo_to_acpl(elo: float) -> float:
    """Inverse of acpl_to_elo: the ACPL a player of this ELO averages."""
    elo = max(MIN_ELO, min(MAX_ELO, elo))
    return math.exp((4034 - elo) / 667)


def decided_factor(eval_cp: int) -> float:
    """1.0 in a live position, falling toward 0.15 as |eval| grows past
    +-300 — how much a centipawn matters once the game is decided."""
    e = abs(eval_cp)
    if e <= 300:
        return 1.0
    return max(0.15, 1.0 - (e - 300) / 900.0)


def effective_loss(loss_cp: float, eval_before: int) -> float:
    """Centipawn loss scaled by how much it matters in this position.
    A 200cp slip at +1500 is a rounding error; at 0.00 it is a mistake."""
    return loss_cp * decided_factor(eval_before)


def position_weight(eval_before: int) -> float:
    """How much a move from this position should count toward a rating
    estimate: fully when the game is live, little once it is decided."""
    return max(0.25, decided_factor(eval_before))


def blend_opponent_elo(prior: int | None, observed: int | None,
                       moves: int) -> int | None:
    """Combine the rating printed next to the opponent's name (prior) with
    the rating implied by their moves so far (observed). The prior anchors
    the early game; the observation takes over gradually, half-way after
    16 of their moves. Either may be missing."""
    if prior is None:
        return observed
    if observed is None:
        return prior
    # A printed rating is rarely far below the truth, but a bot (or a
    # sandbagger) can play well under it: allow more room down than up.
    observed = max(prior - 600, min(prior + 300, observed))
    w = moves / (moves + 16.0)
    return int(round(prior + w * (observed - prior)))


class EloEstimator:
    """Tracks move quality and estimates an ELO rating via ACPL.

    Uses an exponential moving average of centipawn loss so recent play
    weighs more heavily than the opening.
    """

    MIN_MOVES = 4       # minimum moves before showing an estimate
    EMA_ALPHA = 0.10    # smoothing factor for exponential moving average
    MAX_CPL = 500       # clamp individual CPL values
    BOOK_MOVES = 4      # a player's first moves are memorised, not judged
    BOOK_WEIGHT = 0.3

    def __init__(self):
        self._ema_cpl: float = 0.0
        self._move_count: int = 0

    def record_move(self, cpl: float, eval_before: int | None = None) -> None:
        """Record centipawn loss for one move.

        `eval_before` (side-to-move POV, cp) is the position the move was
        played from. Moves in decided positions say little about strength
        (a blunder when already lost, or a lazy move when +9), so their
        loss is damped and their weight in the average reduced.
        """
        cpl = max(0.0, min(cpl, self.MAX_CPL))
        weight = 1.0
        if eval_before is not None:
            cpl = effective_loss(cpl, eval_before)
            weight = position_weight(eval_before)
        if self._move_count < self.BOOK_MOVES:
            weight *= self.BOOK_WEIGHT
        # Errors are strong evidence, good moves weak evidence: anyone
        # finds a recapture, only a weak player hangs a piece. So a
        # blundering opponent is recognised within a few moves.
        if cpl >= 150:
            weight *= 2.4
        elif cpl >= 60:
            weight *= 1.6
        # Warm-up: a plain running mean for the first few moves (so one
        # early move can't dominate), then settle into the EMA.
        alpha = max(self.EMA_ALPHA, 1.0 / (self._move_count + 1)) * weight
        self._ema_cpl = alpha * cpl + (1 - alpha) * self._ema_cpl
        self._move_count += 1

    def get_estimate(self) -> int | None:
        """Return estimated ELO or None if not enough data."""
        if self._move_count < self.MIN_MOVES:
            return None
        return acpl_to_elo(self._ema_cpl)

    def get_move_count(self) -> int:
        return self._move_count

    def get_acpl(self) -> float | None:
        """Return current average (EMA) centipawn loss, or None if no data."""
        if self._move_count == 0:
            return None
        return self._ema_cpl

    def reset(self) -> None:
        """Clear all state for a new game."""
        self._ema_cpl = 0.0
        self._move_count = 0
