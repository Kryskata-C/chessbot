"""Stockfish engine integration."""

from __future__ import annotations

import shutil
from stockfish import Stockfish


def find_stockfish() -> str:
    """Find the Stockfish binary: the copy bundled with the app first, then
    whatever is on PATH or in the usual Homebrew locations."""
    import os
    from paths import resource_path
    candidates = [
        resource_path("stockfish"),
        resource_path("bin", "stockfish"),
        shutil.which("stockfish") or "",
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(
        "Stockfish not found. Install with: brew install stockfish"
    )


class ChessEngine:
    def __init__(self, depth: int = 18, threads: int = 2):
        self.depth = depth
        self._threads = threads
        self._path = find_stockfish()
        self.engine = Stockfish(
            path=self._path,
            depth=depth,
            parameters={"Threads": threads, "Hash": 128},
        )

    def _restart(self):
        """Restart Stockfish process after a crash."""
        try:
            self.engine = Stockfish(
                path=self._path,
                depth=self.depth,
                parameters={"Threads": self._threads, "Hash": 128},
            )
            print("Stockfish restarted successfully.")
        except Exception as e:
            print(f"Stockfish restart failed: {e}")

    def _safe_call(self, fn, fallback=None):
        """Call fn, restarting Stockfish once on failure."""
        try:
            return fn()
        except Exception as e:
            print(f"Engine error ({e}), restarting Stockfish...")
            self._restart()
            try:
                return fn()
            except Exception as e2:
                print(f"Engine error after restart: {e2}")
                return fallback

    @staticmethod
    def _fen_ok(fen: str) -> bool:
        """Cheap sanity check: both kings must be on the board.

        Stockfish hard-crashes on king-less positions, and the wrapper's
        own validation raises — either way the call chain dies, so reject
        these before touching the engine process.
        """
        placement = fen.split()[0]
        return "K" in placement and "k" in placement

    def get_best_move(self, fen: str, depth: int | None = None) -> str | None:
        """Get the best move for the given FEN position.

        Args:
            depth: Optional one-off search depth (e.g. shallow lookahead
                for reply prediction); the engine's default is restored after.
        """
        if not self._fen_ok(fen):
            print(f"Rejecting FEN without both kings: {fen}")
            return None
        def _call():
            if depth is not None:
                self.engine.set_depth(depth)
            try:
                self.engine.set_fen_position(fen)
                return self.engine.get_best_move()
            finally:
                if depth is not None:
                    self.engine.set_depth(self.depth)
        return self._safe_call(_call, fallback=None)

    def get_top_moves(self, fen: str, n: int = 5, depth: int | None = None) -> list[dict]:
        """Get the top N moves with evaluations for the given position.
        `depth` overrides the default for this call (thin endgames are
        cheap to search deeper, and depth 12 sees no conversion plan)."""
        if not self._fen_ok(fen):
            print(f"Rejecting FEN without both kings: {fen}")
            return []
        def _call():
            self.engine.set_fen_position(fen)
            if depth is not None and depth != self.depth:
                self.engine.set_depth(depth)
            try:
                raw = self.engine.get_top_moves(n)
            finally:
                if depth is not None and depth != self.depth:
                    self.engine.set_depth(self.depth)
            result = []
            for m in raw:
                if m.get("Mate") is not None:
                    # Keep mate distance in the score: mate-in-2 must beat
                    # mate-in-8, or a selector choosing among "equal" mates
                    # shuffles plans forever and repetition-draws won games.
                    n_mate = min(abs(m["Mate"]), 99)
                    eval_cp = (100000 - 100 * n_mate if m["Mate"] > 0
                               else -100000 + 100 * n_mate)
                else:
                    eval_cp = m.get("Centipawn", 0)
                result.append({"move": m["Move"], "eval": eval_cp})
            return result
        return self._safe_call(_call, fallback=[])

    def get_evaluation(self, fen: str, depth: int = 10) -> int:
        """Evaluate a position and return score in centipawns from side-to-move POV."""
        if not self._fen_ok(fen):
            print(f"Rejecting FEN without both kings: {fen}")
            return 0
        def _call():
            self.engine.set_fen_position(fen)
            result = self.engine.get_evaluation()
            if result["type"] == "cp":
                return result["value"]
            elif result["type"] == "mate":
                mate_moves = result["value"]
                if mate_moves > 0:
                    return 100000
                elif mate_moves < 0:
                    return -100000
                else:
                    return 0
            return 0
        return self._safe_call(_call, fallback=0)

    def parse_move(self, uci_move: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Convert UCI move to (from_row, from_col), (to_row, to_col).

        Coordinates are 0-indexed from top-left of board (rank 8 = row 0).
        """
        from_file = ord(uci_move[0]) - ord("a")  # col 0-7
        from_rank = int(uci_move[1])              # rank 1-8
        to_file = ord(uci_move[2]) - ord("a")
        to_rank = int(uci_move[3])

        # Convert rank to row (rank 8 = row 0, rank 1 = row 7)
        from_row = 8 - from_rank
        from_col = from_file
        to_row = 8 - to_rank
        to_col = to_file

        return (from_row, from_col), (to_row, to_col)

    def move_to_screen_coords(
        self, uci_move: str, board: dict, white_on_bottom: bool = True
    ) -> tuple[dict, dict]:
        """Convert UCI move to screen pixel coordinates.

        Returns:
            (from_rect, to_rect) where each rect is {x, y, w, h}.
        """
        (from_row, from_col), (to_row, to_col) = self.parse_move(uci_move)

        if not white_on_bottom:
            from_row = 7 - from_row
            from_col = 7 - from_col
            to_row = 7 - to_row
            to_col = 7 - to_col

        sq = board["square_size"]
        bx, by = board["x"], board["y"]

        from_rect = {
            "x": round(bx + from_col * sq),
            "y": round(by + from_row * sq),
            "w": round(sq),
            "h": round(sq),
        }
        to_rect = {
            "x": round(bx + to_col * sq),
            "y": round(by + to_row * sq),
            "w": round(sq),
            "h": round(sq),
        }
        return from_rect, to_rect
