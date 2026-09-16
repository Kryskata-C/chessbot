"""Read the outcome off chess.com's game-over dialog.

The board alone only proves checkmate, stalemate and material draws.
Resignations, flags, agreed draws, repetitions and aborts are announced
by the dialog chess.com drops over the board ("You Won" / "You Lost" /
"Draw" with "by resignation", "on time", "by agreement", ...). That
dialog hides the centre of the board, which is the cheap cue used here
to decide when to spend an OCR call (the OS's own OCR via native.py,
tesseract as a fallback — the same machinery as opponent_rating.py).
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from board_detector import GREEN_LOWER, GREEN_UPPER, BEIGE_LOWER, BEIGE_UPPER
from opponent_rating import _ocr, ocr_available

# Board-colour share of the central 4x4 squares. Pieces cover a fair bit
# of a busy centre, but never nearly all of it; the dialog covers all of it.
_DIALOG_BOARD_FRACTION = 0.12

_WON = re.compile(r"\byou\s*won\b", re.I)
_LOST = re.compile(r"\byou\s*lost\b", re.I)
_WHITE_WON = re.compile(r"\bwhite\s*won\b", re.I)
_BLACK_WON = re.compile(r"\bblack\s*won\b", re.I)
_DRAW = re.compile(r"\bdraw\b", re.I)
_ABORTED = re.compile(r"\baborted\b", re.I)
_SCORE = re.compile(r"(1|0|½|1/2)\s*[-–—]\s*(1|0|½|1/2)")

# (termination, pattern) — first match wins, so the specific ones go first.
_REASONS = (
    ("insufficient", re.compile(r"insufficient", re.I)),
    ("checkmate", re.compile(r"checkmate", re.I)),
    ("resignation", re.compile(r"resign", re.I)),
    ("timeout", re.compile(r"on\s*time|time\s*out|timeout|ran\s*out\s*of\s*time", re.I)),
    ("agreement", re.compile(r"agreement", re.I)),
    ("stalemate", re.compile(r"stalemate", re.I)),
    ("repetition", re.compile(r"repetition", re.I)),
    ("fifty_moves", re.compile(r"50[\s-]*move|fifty[\s-]*move", re.I)),
    ("abandoned", re.compile(r"abandon", re.I)),
    ("aborted", re.compile(r"abort", re.I)),
)


def looks_like_game_over(screenshot: np.ndarray, board: dict) -> bool:
    """True when the middle of the board is not board-coloured any more,
    i.e. something opaque (the game-over dialog) sits on top of it."""
    sq = board["square_size"]
    x0 = int(board["x"] + 2 * sq)
    y0 = int(board["y"] + 2 * sq)
    x1 = int(board["x"] + 6 * sq)
    y1 = int(board["y"] + 6 * sq)
    h, w = screenshot.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    hsv = cv2.cvtColor(screenshot[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER),
                          cv2.inRange(hsv, BEIGE_LOWER, BEIGE_UPPER))
    return cv2.countNonZero(mask) / mask.size < _DIALOG_BOARD_FRACTION


def parse_result(text: str, player_color: str | None) -> tuple[str, str | None] | None:
    """(result, termination) from the dialog's text, or None when the text
    does not announce a game outcome. result is '1-0' | '0-1' |
    '1/2-1/2' | '*' (aborted); termination is one of the keys in _REASONS."""
    if not text:
        return None
    text = text.replace("\n", " ")
    termination = next((name for name, rx in _REASONS if rx.search(text)), None)

    if _WHITE_WON.search(text):
        return "1-0", termination
    if _BLACK_WON.search(text):
        return "0-1", termination
    if _WON.search(text) and player_color:
        return ("1-0" if player_color == "w" else "0-1"), termination
    if _LOST.search(text) and player_color:
        return ("0-1" if player_color == "w" else "1-0"), termination
    if _ABORTED.search(text):
        return "*", "aborted"

    m = _SCORE.search(text)
    if m:
        a, b = m.groups()
        if a in ("½", "1/2"):
            return "1/2-1/2", termination
        if a == "1" and b == "0":
            return "1-0", termination
        if a == "0" and b == "1":
            return "0-1", termination
    # "Draw" alone also shows up in draw-offer banners ("X offers a draw"),
    # so a draw needs the dialog's reason line too.
    if _DRAW.search(text) and termination not in (None, "checkmate", "resignation", "aborted"):
        return "1/2-1/2", termination
    return None


def read_game_result(screenshot: np.ndarray, board: dict,
                     player_color: str | None) -> tuple[str, str | None] | None:
    """OCR the board area and parse the game-over dialog; None when no
    outcome is announced (or OCR is unavailable)."""
    if not ocr_available():
        return None
    sq = board["square_size"]
    x0 = int(board["x"] - 0.5 * sq)
    y0 = int(board["y"] - 0.5 * sq)
    x1 = int(board["x"] + board["width"] + 0.5 * sq)
    y1 = int(board["y"] + board["width"] + 0.5 * sq)
    h, w = screenshot.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 50 or y1 - y0 < 50:
        return None
    gray = cv2.cvtColor(screenshot[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if gray.shape[1] < 900:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    global last_text
    try:
        text = _ocr(gray, 6)
    except Exception:
        return None
    last_text = text or ""
    return parse_result(text, player_color)


last_text = ""   # what the most recent read_game_result() OCR'd (for the log)
