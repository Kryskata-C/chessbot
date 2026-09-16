"""Clock-aware think times: how long a human would plausibly take on this
move given the time control and the time left.

The old model used fixed bases (3-11 s, up to 30 s) whatever the clock
said, which in a 3-minute game spends the whole clock in 25 moves. This
one budgets the remaining time over the moves still expected, then bends
each move's share by the position (obvious moves fast, real decisions
slow) and by the clock (time trouble makes everyone quick).

The player's clock is read off chess.com with OCR (`read_clock`) at every
turn; between reads the budget carries an estimate. The time control is
either chosen in the menu or inferred from the clock at move one.
"""

from __future__ import annotations

import math
import random
import re

import cv2
import numpy as np

from opponent_rating import _ocr, ocr_available

# key -> (base seconds, increment seconds). "auto" = read it from the clock.
TIME_CONTROLS: dict[str, tuple[int, int] | None] = {
    "auto": None,
    "1+0": (60, 0), "2+1": (120, 1),
    "3+0": (180, 0), "3+2": (180, 2), "5+0": (300, 0), "5+3": (300, 3),
    "10+0": (600, 0), "15+10": (900, 10), "30+0": (1800, 0),
}
# What a full clock at move one most likely means (increment is invisible
# on the clock; the plain version is assumed since pacing for it is safer).
_BASE_TO_KEY = {60: "1+0", 120: "2+1", 180: "3+0", 300: "5+0",
                600: "10+0", 900: "15+10", 1800: "30+0"}
FALLBACK = "3+0"    # no clock read, nothing chosen: pace like blitz

_CLOCK_RE = re.compile(r"(?<![\d.])(\d{1,3}):(\d{2})(?:[.,](\d))?(?![\d])|(?<![\d:.])(\d{1,2})[.,](\d)(?![\d])")
# Vertical strips (in squares) off the board edge where chess.com prints
# the player rows; the clock sits at the right end of that row.
_STRIPS = ((0.62, 0.30), (0.75, 0.28), (1.0, 0.10))


def parse_clock(text: str) -> float | None:
    """Seconds shown by a clock string like '3:00', '0:09.8' or '9.8'."""
    if not text:
        return None
    best = None
    for m in _CLOCK_RE.finditer(text):
        if m.group(1) is not None:
            s = int(m.group(1)) * 60 + int(m.group(2)) + (int(m.group(3)) / 10 if m.group(3) else 0)
        else:
            s = int(m.group(4)) + int(m.group(5)) / 10
        if s <= 3 * 3600 and (best is None or s > best):
            best = s
    return best


def read_clock(screenshot: np.ndarray, board: dict, side: str = "bottom") -> float | None:
    """Seconds left on the clock of the player at the bottom (us) or top."""
    if not ocr_available():
        return None
    sq = board["square_size"]
    h_img, w_img = screenshot.shape[:2]
    x0 = int(board["x"] + 0.55 * board["width"])
    x1 = min(w_img, int(board["x"] + board["width"] + 0.2 * sq))
    for far, near in _STRIPS:
        if side == "bottom":
            y0 = int(board["y"] + board["width"] + near * sq)
            y1 = int(board["y"] + board["width"] + far * sq)
        else:
            y0 = int(board["y"] - far * sq)
            y1 = int(board["y"] - near * sq)
        if y0 < 0 or y1 > h_img or y1 - y0 < 6 or x1 - x0 < 20:
            continue
        crop = cv2.cvtColor(screenshot[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        fx = max(2, int(math.ceil(160 / max(1, crop.shape[0]))))
        gray = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=int(np.median(gray)))
        try:
            secs = parse_clock(_ocr(gray, 7))
        except Exception:
            return None
        if secs is not None:
            return secs
    return None


class ThinkTimer:
    """Per-game pacing state. `suggest()` is called once per own move."""

    def __init__(self, control: str = "auto"):
        self.set_control(control)
        self.new_game()

    # ---- configuration -------------------------------------------------
    def set_control(self, key: str) -> None:
        self.control_key = key if key in TIME_CONTROLS else "auto"

    def new_game(self) -> None:
        self.base, self.inc = TIME_CONTROLS[self.control_key] or (None, None)
        self.inferred: str | None = None
        self.remaining: float | None = None   # our clock, seconds (estimate between reads)
        self.moves = 0                        # our moves so far
        self.spent = 0.0                      # seconds this model has asked for so far

    @property
    def label(self) -> str:
        if self.control_key != "auto":
            return self.control_key
        return self.inferred or f"{FALLBACK} (assumed)"

    # ---- clock input ---------------------------------------------------
    def observe_clock(self, seconds: float) -> str | None:
        """A fresh OCR read of our clock. Returns the inferred time-control
        key when this read settled it (auto mode, first move)."""
        if seconds is None or seconds < 0:
            return None
        self.remaining = float(seconds)
        if self.control_key == "auto" and self.inferred is None and self.moves <= 1:
            # A fresh clock reads the full base time (2:59 after a slow
            # first move still rounds to it).
            for base, key in _BASE_TO_KEY.items():
                if base * 0.9 <= seconds <= base * 1.02:
                    self.inferred = key
                    self.base, self.inc = TIME_CONTROLS[key]
                    return key
        return None

    # ---- the model -----------------------------------------------------
    def suggest(self, crit: float, loss: float, move_number: int,
                piece_count: int, under_pressure: bool, near_equal: bool) -> float:
        """Seconds to wait before playing. `crit` is the move selector's
        criticality (1 = one obvious move), `loss` the chosen move's
        centipawn loss versus the best, `move_number` our move count."""
        base, inc = self.base or 0, self.inc or 0
        if not base:
            base, inc = TIME_CONTROLS[FALLBACK]
        if self.remaining is None:
            remaining = max(0.0, base - self.spent + inc * self.moves)
        else:
            remaining = self.remaining

        # Even split of what's left over the moves still expected, minus a
        # reserve so a long game does not end on the flag.
        bullet = base <= 120
        moves_left = max(18.0, 50.0 - move_number) if bullet else max(14.0, 42.0 - move_number)
        reserve = min(0.12 * base, 15.0)
        share = max(0.0, remaining - reserve) / moves_left + 0.8 * inc

        # Position: what kind of move is this?
        if move_number < 8:
            mult = 0.35          # opening: familiar territory
        elif crit >= 0.6:
            mult = 0.3           # one obvious move (recapture, only move)
        elif loss >= 60:
            mult = 0.55          # slips happen when moving fast
        elif crit < 0.15:
            mult = 2.2           # several playable moves: a real decision
        else:
            mult = 1.0
        if under_pressure:
            mult *= 1.3          # worse positions get more thought
        if piece_count <= 10:
            mult *= 0.6          # simple endgames go quicker
        if near_equal:
            mult *= 1.2          # two near-equal options

        # Time trouble: everyone speeds up, long thinks disappear.
        low = remaining < max(20.0, 0.1 * base)
        if low:
            mult = min(mult, 1.1)
            share = remaining / max(10.0, moves_left * 0.6) + 0.8 * inc

        seconds = share * mult * math.exp(random.gauss(0.0, 0.25 if bullet else 0.35))
        floor = 0.4 if bullet else 0.6
        cap = min(max(share * 4.0, floor), 0.2 * remaining + inc, 60.0 if base >= 600 else 45.0)
        seconds = max(floor, min(cap, seconds))
        if remaining < 10:
            # Seconds left: premove territory, whatever the position.
            seconds = min(seconds, max(0.25, remaining / 15.0))

        self.moves = move_number + 1
        self.spent += seconds
        if self.remaining is not None:
            # Carry the estimate forward: the next OCR read replaces it.
            self.remaining = max(0.0, self.remaining - seconds + inc)
        return seconds
