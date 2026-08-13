"""Identify chess pieces on each square using template matching."""

from __future__ import annotations

import os
import cv2
import numpy as np
import chess

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
TEMPLATE_SIZE = 80
MATCH_THRESHOLD = 0.55
# Relaxed threshold used only when a king is missing from the board.
# Overlays like chess.com's red check-glow can drag the king's normal
# match score below MATCH_THRESHOLD, making the king "vanish".
KING_RECOVERY_THRESHOLD = 0.25

# Base piece names -> FEN symbols
PIECE_NAMES = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}


def _load_templates() -> list[tuple[str, str, np.ndarray]]:
    """Load all piece template images.

    Returns:
        List of (base_name, fen_symbol, image) tuples.
        Multiple entries per piece if light/dark variants exist.
    """
    templates = []
    if not os.path.isdir(TEMPLATE_DIR):
        return templates
    for fname in os.listdir(TEMPLATE_DIR):
        if not fname.endswith(".png"):
            continue
        stem = fname[:-4]  # remove .png
        # Match against known piece names (with optional _light/_dark suffix)
        base = stem.replace("_light", "").replace("_dark", "")
        if base not in PIECE_NAMES:
            continue
        path = os.path.join(TEMPLATE_DIR, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            templates.append((base, PIECE_NAMES[base], img))
    return templates


_templates: list[tuple[str, str, np.ndarray]] | None = None


def get_templates() -> list[tuple[str, str, np.ndarray]]:
    global _templates
    if _templates is None:
        _templates = _load_templates()
    return _templates


def reload_templates():
    """Force reload templates from disk."""
    global _templates
    _templates = None
    return get_templates()


def recognize_square(square_img: np.ndarray) -> str | None:
    """Identify the piece on a single square image.

    Returns:
        FEN piece character (e.g., 'K', 'p') or None for empty.
    """
    templates = get_templates()
    if not templates:
        return None

    square_resized = cv2.resize(square_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))

    best_score = -1
    best_fen = None

    for _base, fen_sym, tmpl in templates:
        result = cv2.matchTemplate(square_resized, tmpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_fen = fen_sym

    if best_score >= MATCH_THRESHOLD and best_fen is not None:
        return best_fen
    return None


def _square_image(screenshot: np.ndarray, board: dict, row: int, col: int) -> np.ndarray | None:
    """Crop one square from the screenshot, clamped to image bounds."""
    sq = board["square_size"]
    img_h, img_w = screenshot.shape[:2]
    x = round(board["x"] + col * sq)
    y = round(board["y"] + row * sq)
    w = round(sq)
    h = round(sq)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    square_img = screenshot[y : y + h, x : x + w]
    return square_img if square_img.size > 0 else None


def _recover_missing_king(
    screenshot: np.ndarray, board: dict,
    positions: list[list[str | None]], king_sym: str,
) -> bool:
    """Place a missing king on the best-matching empty-looking square.

    A king can never actually leave the board, so if recognition lost one
    (check-glow, hover effects), rematch king templates over the squares
    that read as empty using a relaxed threshold. Squares already holding
    a recognized piece are never overridden.
    """
    king_templates = [t for t in get_templates() if t[1] == king_sym]
    if not king_templates:
        return False

    best_score = KING_RECOVERY_THRESHOLD
    best_rc: tuple[int, int] | None = None
    for row in range(8):
        for col in range(8):
            if positions[row][col] is not None:
                continue
            square_img = _square_image(screenshot, board, row, col)
            if square_img is None:
                continue
            square_resized = cv2.resize(square_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            for _base, _fen, tmpl in king_templates:
                score = cv2.matchTemplate(
                    square_resized, tmpl, cv2.TM_CCOEFF_NORMED
                ).max()
                if score > best_score:
                    best_score = score
                    best_rc = (row, col)

    if best_rc is not None:
        positions[best_rc[0]][best_rc[1]] = king_sym
        print(f"Recovered missing {king_sym!r} at row={best_rc[0]} col={best_rc[1]} (score {best_score:.2f})")
        return True
    return False


def recognize_board(screenshot: np.ndarray, board: dict) -> list[list[str | None]]:
    """Recognize all pieces on the board.

    Args:
        screenshot: Full screen BGR image.
        board: Board detection result dict.

    Returns:
        8x8 list, rows top-to-bottom, cols left-to-right.
        Each cell is a FEN piece char or None.
    """
    positions = []
    for row in range(8):
        rank = []
        for col in range(8):
            square_img = _square_image(screenshot, board, row, col)
            if square_img is None:
                rank.append(None)
            else:
                rank.append(recognize_square(square_img))
        positions.append(rank)

    # Kings can never be captured — if one wasn't recognized, try to
    # recover it with a relaxed match before the scan is used.
    flat = [p for rank in positions for p in rank]
    for king_sym in ("K", "k"):
        if king_sym not in flat:
            _recover_missing_king(screenshot, board, positions, king_sym)

    return positions


def board_to_fen(positions: list[list[str | None]], white_on_bottom: bool = True) -> str:
    """Convert 8x8 position array to FEN string."""
    fen_rows = []
    for row in positions:
        fen_row = ""
        empty = 0
        for piece in row:
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


def detect_orientation(positions: list[list[str | None]]) -> bool:
    """Detect whether white is on the bottom of the screen."""
    white_bottom = 0
    white_top = 0
    for col in range(8):
        for row in [6, 7]:
            p = positions[row][col]
            if p and p.isupper():
                white_bottom += 1
        for row in [0, 1]:
            p = positions[row][col]
            if p and p.isupper():
                white_top += 1
    return white_bottom >= white_top


def positions_to_fen(positions: list[list[str | None]], turn: str = "w") -> str:
    """Convert recognized positions to a full FEN string."""
    white_bottom = detect_orientation(positions)
    piece_placement = board_to_fen(positions, white_bottom)

    if not white_bottom:
        rows = piece_placement.split("/")
        rows = [row[::-1] for row in reversed(rows)]
        piece_placement = "/".join(rows)

    return f"{piece_placement} {turn} KQkq - 0 1"
