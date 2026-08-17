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


class EloEstimator:
    """Tracks move quality and estimates an ELO rating via ACPL.

    Uses an exponential moving average of centipawn loss so recent play
    weighs more heavily than the opening.
    """

    MIN_MOVES = 3       # minimum moves before showing an estimate
    EMA_ALPHA = 0.15    # smoothing factor for exponential moving average
    MAX_CPL = 500       # clamp individual CPL values

    def __init__(self):
        self._ema_cpl: float = 0.0
        self._move_count: int = 0

    def record_move(self, cpl: float) -> None:
        """Record centipawn loss for one move."""
        cpl = max(0.0, min(cpl, self.MAX_CPL))
        # Warm-up: a plain running mean for the first few moves (so one
        # early move can't dominate), then settle into the EMA.
        alpha = max(self.EMA_ALPHA, 1.0 / (self._move_count + 1))
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
