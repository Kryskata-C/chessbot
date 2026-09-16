"""Upload each finished live game to the account server so the website's
dashboard can show games played / won / drawn / lost and per-game details.

Only the signed-in user's own row is written (row-level security checks
auth.uid() = user_id). Uploads run on a background thread and never block
or break the game loop: a failure is logged and the game stays in
live_games/ on disk anyway.
"""

from __future__ import annotations

import sys
import threading
import time

from version import __version__ as APP_VERSION


class GameUploader:
    def __init__(self, accounts):
        self._accounts = accounts  # auth.Accounts (has .client and .profile)

    @property
    def enabled(self) -> bool:
        return self._accounts is not None and self._accounts.profile is not None

    def upload(self, summary: dict) -> None:
        """summary comes from recorder.GameRecorder.summary()."""
        if not self.enabled:
            return
        if not summary.get("plies"):
            return  # nothing happened; don't record an empty game
        row = self._row(summary)
        threading.Thread(target=self._send, args=(row,), daemon=True).start()

    def _row(self, s: dict) -> dict:
        started = s.get("started") or time.time()
        return {
            "user_id": self._accounts.profile.id,
            "played_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started)) + "Z",
            "duration_s": int(max(0, (s.get("ended") or time.time()) - started)),
            "color": s.get("color"),
            "result": s.get("result") or "*",
            "termination": s.get("termination"),
            "score": s.get("score"),
            "plies": s.get("plies"),
            "target_elo": s.get("target_elo"),
            "opponent_rating": s.get("opp_rating"),
            "opponent_estimate": s.get("opp_estimate"),
            "accuracy": _r(s.get("accuracy")),
            "acpl": _r(s.get("acpl")),
            "contested_acpl": _r(s.get("contested_acpl")),
            "realized_elo": s.get("realized_elo"),
            "resyncs": s.get("resyncs") or 0,
            "platform": "win" if sys.platform == "win32" else "mac" if sys.platform == "darwin" else sys.platform,
            "app_version": APP_VERSION,
            "pgn": s.get("pgn"),
        }

    def _send(self, row: dict) -> None:
        try:
            self._accounts.client.table("games").insert(row).execute()
            print(f"Game uploaded ({row['result']}, {row['plies']} plies)")
        except Exception as e:
            print(f"Game upload failed (kept locally): {str(e)[:160]}")


def _r(v, nd: int = 1):
    return None if v is None else round(float(v), nd)
