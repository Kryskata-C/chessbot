"""Stockfish engine integration."""

from __future__ import annotations

import shutil
from stockfish import Stockfish


def find_stockfish() -> str:
    """Find the Stockfish binary path."""
    path = shutil.which("stockfish")
    if path:
        return path
    # Common brew install locations
    for p in ["/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"]:
        import os
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Stockfish not found. Install with: brew install stockfish"
    )


class ChessEngine:
    def __init__(self, depth: int = 18, threads: int = 2):
        self.depth = depth
        path = find_stockfish()
        self.engine = Stockfish(
            path=path,
            depth=depth,
            parameters={"Threads": threads, "Hash": 128},
        )

    def get_best_move(self, fen: str) -> str | None:
        """Get the best move for the given FEN position.

        Returns:
            UCI move string (e.g., 'e2e4') or None if no move.
        """
        try:
            self.engine.set_fen_position(fen)
            move = self.engine.get_best_move()
            return move
        except Exception as e:
            print(f"Engine error: {e}")
            return None

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
            "x": int(bx + from_col * sq),
            "y": int(by + from_row * sq),
            "w": int(sq),
            "h": int(sq),
        }
        to_rect = {
            "x": int(bx + to_col * sq),
            "y": int(by + to_row * sq),
            "w": int(sq),
            "h": int(sq),
        }
        return from_rect, to_rect
