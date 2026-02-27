"""Estimate opponent ELO from centipawn loss per move."""

from __future__ import annotations

import math


class EloEstimator:
    """Tracks opponent move quality and estimates ELO rating.

    Uses exponential moving average of centipawn loss (CPL) mapped
    to an ELO estimate via: ELO = 4034 - 667 * ln(ACPL).
    """

    MIN_MOVES = 3       # minimum moves before showing an estimate
    EMA_ALPHA = 0.15    # smoothing factor for exponential moving average
    MIN_ELO = 300
    MAX_ELO = 3000
    MAX_CPL = 500        # clamp individual CPL values

    def __init__(self):
        self._ema_cpl: float = 0.0
        self._move_count: int = 0

    def record_move(self, cpl: float) -> None:
        """Record centipawn loss for one opponent move."""
        cpl = max(0.0, min(cpl, self.MAX_CPL))
        if self._move_count == 0:
            self._ema_cpl = cpl
        else:
            self._ema_cpl = self.EMA_ALPHA * cpl + (1 - self.EMA_ALPHA) * self._ema_cpl
        self._move_count += 1

    def get_estimate(self) -> int | None:
        """Return estimated ELO or None if not enough data."""
        if self._move_count < self.MIN_MOVES:
            return None
        return self._acpl_to_elo(self._ema_cpl)

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

    def _acpl_to_elo(self, acpl: float) -> int:
        """Map average centipawn loss to an ELO estimate."""
        if acpl <= 0:
            return self.MAX_ELO
        elo = 4034 - 667 * math.log(acpl)
        return int(max(self.MIN_ELO, min(self.MAX_ELO, elo)))
