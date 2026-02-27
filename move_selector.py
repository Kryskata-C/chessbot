"""Human-like move selection with adaptive accuracy.

Instead of always playing the engine's #1 move, this module selects from
the top N engine candidates using a softmax probability distribution.
The "temperature" (randomness) adapts dynamically based on:

- Game phase (opening/middle/endgame)
- Position pressure (winning vs losing)
- Eval trend (is our position deteriorating?)
- Move criticality (is there only one good move?)
- Anti-engine smoothing (avoid suspiciously perfect streaks)

When losing, temperature drops sharply — near-engine play to fight back.
When comfortable, temperature rises — more natural, human-like variance.
"""

from __future__ import annotations

import math
import random

from engine import ChessEngine


class HumanMoveSelector:
    """Selects moves that mimic human play while remaining competitive."""

    # Game phase thresholds (piece count)
    OPENING_PIECES = 28
    EARLY_MID_PIECES = 22
    MIDDLE_PIECES = 16
    # Below MIDDLE_PIECES = endgame

    NUM_CANDIDATES = 5
    TREND_WINDOW = 6  # how many evals to look back for trend detection

    def __init__(self, engine: ChessEngine):
        self.engine = engine
        self._eval_history: list[int] = []
        self._move_number: int = 0
        self._consecutive_best: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_move(self, fen: str, piece_count: int) -> str | None:
        """Pick a human-like move for the position.

        Returns a UCI move string, or None if no legal moves.
        """
        top_moves = self.engine.get_top_moves(fen, self.NUM_CANDIDATES)
        if not top_moves:
            # Fallback to basic best-move
            return self.engine.get_best_move(fen)
        if len(top_moves) == 1:
            self._record(top_moves, top_moves[0]["move"])
            return top_moves[0]["move"]

        best_eval = top_moves[0]["eval"]
        self._eval_history.append(best_eval)

        phase = self._get_phase(piece_count)
        pressure = self._compute_pressure(best_eval)
        trend = self._compute_trend_urgency()
        criticality = self._compute_criticality(top_moves)

        temperature = self._compute_temperature(
            phase, pressure, trend, criticality
        )

        chosen = self._weighted_select(top_moves, temperature)
        self._record(top_moves, chosen)

        # Logging for debugging
        delta = best_eval - next(
            m["eval"] for m in top_moves if m["move"] == chosen
        )
        tag = "*" if chosen == top_moves[0]["move"] else " "
        print(
            f"  [{tag}] move={chosen}  loss={delta}cp  "
            f"temp={temperature:.0f}  phase={phase}  "
            f"pressure={pressure:.2f}  trend={trend:.2f}  "
            f"crit={criticality:.2f}"
        )

        return chosen

    def reset(self):
        """Clear state for a new game."""
        self._eval_history.clear()
        self._move_number = 0
        self._consecutive_best = 0

    # ------------------------------------------------------------------
    # Phase detection
    # ------------------------------------------------------------------

    def _get_phase(self, piece_count: int) -> str:
        if piece_count >= self.OPENING_PIECES:
            return "opening"
        if piece_count >= self.EARLY_MID_PIECES:
            return "early_middle"
        if piece_count >= self.MIDDLE_PIECES:
            return "middlegame"
        return "endgame"

    # ------------------------------------------------------------------
    # Pressure: how much trouble are we in? (0 = desperate, 1+ = fine)
    # ------------------------------------------------------------------

    def _compute_pressure(self, eval_cp: int) -> float:
        """Return a 0–1.1 pressure factor from the current eval."""
        if eval_cp <= -400:
            return 0.0
        if eval_cp <= -200:
            # Linear ramp from 0.0 at -400 to 0.25 at -200
            return 0.25 * (eval_cp + 400) / 200
        if eval_cp <= -100:
            return 0.25 + 0.25 * (eval_cp + 200) / 100
        if eval_cp <= 0:
            return 0.50 + 0.25 * (eval_cp + 100) / 100
        if eval_cp <= 150:
            return 0.75 + 0.25 * eval_cp / 150
        if eval_cp <= 400:
            return 1.0 + 0.1 * (eval_cp - 150) / 250
        return 1.1

    # ------------------------------------------------------------------
    # Trend detection: is our eval sliding downhill?
    # ------------------------------------------------------------------

    def _compute_trend_urgency(self) -> float:
        """0 = stable/improving, approaching 1 = eval collapsing."""
        if len(self._eval_history) < 3:
            return 0.0

        window = self._eval_history[-self.TREND_WINDOW:]
        n = len(window)
        x_mean = (n - 1) / 2
        y_mean = sum(window) / n

        numer = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(window))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom == 0:
            return 0.0

        slope = numer / denom  # cp per move (negative = getting worse)
        if slope >= 0:
            return 0.0
        return min(1.0, abs(slope) / 100)

    # ------------------------------------------------------------------
    # Criticality: is there only one reasonable move?
    # ------------------------------------------------------------------

    def _compute_criticality(self, top_moves: list[dict]) -> float:
        """0 = many good options, 1 = one clearly best move.

        When criticality is high, even a human would find the right move
        because the position "screams" for it.
        """
        if len(top_moves) < 2:
            return 1.0

        best = top_moves[0]["eval"]
        second = top_moves[1]["eval"]
        gap = best - second

        # Small gap (<30cp) = many reasonable choices
        # Big gap (200cp+) = must find the best move
        if gap <= 30:
            return 0.0
        return min(1.0, (gap - 30) / 170)

    # ------------------------------------------------------------------
    # Temperature: the heart of humanization
    # ------------------------------------------------------------------

    def _compute_temperature(
        self,
        phase: str,
        pressure: float,
        trend_urgency: float,
        criticality: float,
    ) -> float:
        """Higher temperature = more random (human-like).
        Lower temperature = more engine-like (fighting back).
        """
        # Base temperature by game phase
        base = {
            "opening": 45,        # humans know theory, moderate variance
            "early_middle": 70,   # complex, humans err most here
            "middlegame": 60,     # still significant variance
            "endgame": 35,        # humans get more precise
        }[phase]

        # Opening book effect: first ~6 moves humans play memorized lines
        if self._move_number < 6:
            book_factor = 0.55 + (self._move_number / 6) * 0.45
            base *= book_factor

        # Pressure scaling (0=desperate→very low, 1.1=dominating→normal+)
        pressure_factor = 0.08 + 0.92 * pressure

        # Trend urgency: eval dropping → tighten play to stop bleeding
        trend_factor = 1.0 - 0.55 * trend_urgency

        # Criticality: obvious best move → human finds it too
        crit_factor = 1.0 - 0.65 * criticality

        # Anti-engine: too many consecutive best moves looks suspicious
        if self._consecutive_best >= 5:
            anti_engine = 1.0 + 0.07 * (self._consecutive_best - 4)
            anti_engine = min(anti_engine, 1.35)
        else:
            anti_engine = 1.0

        temperature = base * pressure_factor * trend_factor * crit_factor * anti_engine

        # Floor so there's always *some* chance of deviation
        if pressure <= 0.15:
            temperature = max(3, temperature)   # desperate: near-engine
        else:
            temperature = max(8, temperature)   # normal: small floor

        return temperature

    # ------------------------------------------------------------------
    # Weighted random selection (softmax over eval deltas)
    # ------------------------------------------------------------------

    def _weighted_select(self, moves: list[dict], temperature: float) -> str:
        best_eval = moves[0]["eval"]

        weights = []
        for m in moves:
            delta = best_eval - m["eval"]
            exp = -delta / max(temperature, 1)
            exp = max(exp, -20)  # prevent underflow
            weights.append(math.exp(exp))

        total = sum(weights)
        if total == 0:
            return moves[0]["move"]

        return random.choices(
            [m["move"] for m in moves], weights=weights, k=1
        )[0]

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _record(self, top_moves: list[dict], chosen: str):
        if chosen == top_moves[0]["move"]:
            self._consecutive_best += 1
        else:
            self._consecutive_best = 0
        self._move_number += 1
