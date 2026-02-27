"""Identify chess pieces on each square using template matching."""

from __future__ import annotations

import os
import cv2
import numpy as np
import chess

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
TEMPLATE_SIZE = 80
MATCH_THRESHOLD = 0.55

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


def recognize_board(screenshot: np.ndarray, board: dict) -> list[list[str | None]]:
    """Recognize all pieces on the board.

    Args:
        screenshot: Full screen BGR image.
        board: Board detection result dict.

    Returns:
        8x8 list, rows top-to-bottom, cols left-to-right.
        Each cell is a FEN piece char or None.
    """
    sq = board["square_size"]
    img_h, img_w = screenshot.shape[:2]
    # Inset edge squares by a few pixels to avoid board-border artifacts
    # that cause noisy recognition on ranks 1/8 and files a/h
    inset = max(1, round(sq * 0.04))
    positions = []
    for row in range(8):
        rank = []
        for col in range(8):
            x = round(board["x"] + col * sq)
            y = round(board["y"] + row * sq)
            w = round(sq)
            h = round(sq)
            # Apply inset for edge squares
            if col == 0:
                x += inset
                w -= inset
            if col == 7:
                w -= inset
            if row == 0:
                y += inset
                h -= inset
            if row == 7:
                h -= inset
            # Clamp to image bounds so edge squares don't go out of frame
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            square_img = screenshot[y : y + h, x : x + w]
            if square_img.size == 0:
                rank.append(None)
            else:
                rank.append(recognize_square(square_img))
        positions.append(rank)
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
