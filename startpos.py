"""Template-free reading of a board that shows the starting position.

Piece templates are cut from the starting position, so two things must be
known before any template exists: whether the board IS in the starting
position, and which colour sits on top. Both come from raw pixels:
occupied squares have texture (a piece drawn over a flat square) while
empty ones are flat, and white pieces are brighter than black ones on
every chess.com piece set.
"""

from __future__ import annotations

import cv2
import numpy as np

START_FEN_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _square_crops(screenshot: np.ndarray, board: dict, inset: float = 0.2):
    """Grayscale inner crops of all 64 squares, row-major from the top."""
    sq = board["square_size"]
    img_h, img_w = screenshot.shape[:2]
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    out = []
    for row in range(8):
        for col in range(8):
            x0 = int(round(board["x"] + (col + inset) * sq))
            y0 = int(round(board["y"] + (row + inset) * sq))
            x1 = int(round(board["x"] + (col + 1 - inset) * sq))
            y1 = int(round(board["y"] + (row + 1 - inset) * sq))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img_w, x1), min(img_h, y1)
            out.append(gray[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None)
    return out


def looks_like_start_position(screenshot: np.ndarray, board: dict) -> bool:
    """True when exactly the 32 squares of ranks 1-2 and 7-8 carry texture."""
    crops = _square_crops(screenshot, board)
    if any(c is None for c in crops):
        return False
    texture = np.array([float(c.std()) for c in crops])
    order = np.argsort(-texture)
    busy = set(int(i) for i in order[:32])
    expected = {r * 8 + c for r in (0, 1, 6, 7) for c in range(8)}
    if busy != expected:
        return False
    # The two groups must be clearly separated, not just ordered.
    lo_busy = texture[sorted(expected)].min()
    hi_empty = texture[[i for i in range(64) if i not in expected]].max()
    return lo_busy > 2.0 * hi_empty + 2.0


def white_on_top(screenshot: np.ndarray, board: dict) -> bool:
    """Which colour's back rank is at the top. The central 50% of an
    occupied square is mostly piece body: near-white for White's pieces,
    near-black for Black's, whatever the board theme, so the median
    brightness of those crops separates the two sides."""
    crops = _square_crops(screenshot, board, inset=0.25)

    def body(idx: int) -> float:
        c = crops[idx]
        return float(np.median(c)) if c is not None and c.size else 128.0

    top = np.median([body(i) for i in range(0, 16)])
    bottom = np.median([body(i) for i in range(48, 64)])
    return top > bottom


def start_layout(white_top: bool) -> list[list[str | None]]:
    """Piece names per square for a starting-position board, from the
    top-left of the screen, for either orientation."""
    from calibrate import STARTING_POSITION
    if not white_top:
        return STARTING_POSITION
    # Board seen from Black's side: ranks and files both run the other way,
    # so White's back rank is the top row, reading R N B K Q B N R.
    return [row[::-1] for row in STARTING_POSITION[::-1]]
