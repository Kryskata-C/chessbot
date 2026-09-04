"""Cross-game governor: keep the bot's profile human over a whole session.

Per-game randomness (edge, margin, lapses) makes single games vary the way
a person's do, but over ten games the averages must also look right — a
best-move rate that sits at 55% for a month is a tell no single game
shows. The governor watches the last few games' best-move rate and
contested loss and nudges two levers for the next game: a temperature
multiplier and a shift of the per-game edge over the opponent.

State lives in a small JSON file (the live app uses one in the home
directory; self-play runs use a throwaway per run).
"""

from __future__ import annotations

import json
import os
import time

WINDOW = 6
BEST_HIGH, BEST_LOW = 52.0, 28.0     # best-move % band a human sits in
# Contested ACPL only flags the extremes: self-play showed ~20cp is this
# architecture's floor, and forcing it lower with heat only adds blunders
# (it.10: temp x1.5 gave 4 draws and 6 suspicious sacs, ACPL unchanged).
CACPL_LOW, CACPL_HIGH = 12.0, 45.0
STEP_TEMP, STEP_EDGE = 0.05, 15
TEMP_RANGE = (0.8, 1.25)
EDGE_RANGE = (-60, 60)


class SessionGovernor:
    def __init__(self, path: str | None):
        self.path = path
        self.games: list[dict] = []
        self.temp_mult: float = 1.0
        self.edge_shift: int = 0
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    d = json.load(f)
                self.games = list(d.get("games", []))[-24:]
                self.temp_mult = float(d.get("temp_mult", 1.0))
                self.edge_shift = int(d.get("edge_shift", 0))
            except (OSError, ValueError):
                pass

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record_game(self, best_pct: float | None, c_acpl: float | None,
                    score: float | None) -> str:
        """Fold one finished game in and re-aim the levers. Returns a
        one-line description of what changed (for the log)."""
        if not self.enabled:
            return "governor off"
        self.games.append({"best_pct": best_pct, "c_acpl": c_acpl,
                           "score": score, "t": time.time()})
        self.games = self.games[-24:]
        recent = [g for g in self.games[-WINDOW:]
                  if g.get("best_pct") is not None and g.get("c_acpl") is not None]
        if len(recent) < 3:
            self._save()
            return f"governor: {len(recent)} game(s) seen, holding neutral"
        best = sum(g["best_pct"] for g in recent) / len(recent)
        cacpl = sum(g["c_acpl"] for g in recent) / len(recent)
        too_strong = best > BEST_HIGH or cacpl < CACPL_LOW
        too_weak = best < BEST_LOW or cacpl > CACPL_HIGH
        if too_strong and not too_weak:
            self.temp_mult += STEP_TEMP
            self.edge_shift -= STEP_EDGE
            verdict = "too clean -> loosening"
        elif too_weak and not too_strong:
            self.temp_mult -= STEP_TEMP
            self.edge_shift += STEP_EDGE
            verdict = "too loose -> tightening"
        else:
            self.temp_mult += 0.25 * (1.0 - self.temp_mult)
            self.edge_shift = int(round(self.edge_shift * 0.75))
            verdict = "in band -> relaxing toward neutral"
        self.temp_mult = max(TEMP_RANGE[0], min(TEMP_RANGE[1], self.temp_mult))
        self.edge_shift = max(EDGE_RANGE[0], min(EDGE_RANGE[1], self.edge_shift))
        self._save()
        return (f"governor: last {len(recent)} games best {best:.0f}% "
                f"contested {cacpl:.0f}cp -> {verdict} "
                f"(temp x{self.temp_mult:.2f}, edge {self.edge_shift:+d})")

    def _save(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "w") as f:
                json.dump({"games": self.games, "temp_mult": self.temp_mult,
                           "edge_shift": self.edge_shift}, f)
        except OSError:
            pass
