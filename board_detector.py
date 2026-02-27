"""Detect chess.com board boundaries on screen."""

from __future__ import annotations

import cv2
import numpy as np


# chess.com color ranges in HSV
# Green squares: #769656 → HSV ~(75, 42%, 59%)
# Beige squares: #EEEED2 → HSV ~(60, 12%, 93%)

# We use generous ranges to account for display variations
GREEN_LOWER = np.array([30, 40, 80])
GREEN_UPPER = np.array([90, 255, 200])

BEIGE_LOWER = np.array([20, 10, 180])
BEIGE_UPPER = np.array([45, 80, 255])


def detect_board(screenshot: np.ndarray) -> dict | None:
    """Find the chess board in the screenshot.

    Returns:
        Dict with keys: x, y, width, height, square_size,
        or None if no board found.
    """
    hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    beige_mask = cv2.inRange(hsv, BEIGE_LOWER, BEIGE_UPPER)
    combined = cv2.bitwise_or(green_mask, beige_mask)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the largest roughly-square contour
    best = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 10000:  # too small to be a board
            continue
        aspect = w / h if h > 0 else 0
        if 0.8 < aspect < 1.2 and area > best_area:
            best = (x, y, w, h)
            best_area = area

    if best is None:
        return None

    x, y, w, h = best
    # Make it exactly square, centering the shorter dimension
    size = max(w, h)
    if w < size:
        x -= (size - w) // 2
    if h < size:
        y -= (size - h) // 2
    square_size = size / 8

    return {
        "x": x,
        "y": y,
        "width": size,
        "height": size,
        "square_size": square_size,
    }


def get_square_coords(board: dict, row: int, col: int) -> tuple[int, int, int, int]:
    """Get pixel coordinates for a specific board square.

    Args:
        board: Board detection result dict.
        row: 0-7 (0 = top of screen).
        col: 0-7 (0 = left of screen).

    Returns:
        (x, y, w, h) pixel coordinates.
    """
    sq = board["square_size"]
    x = round(board["x"] + col * sq)
    y = round(board["y"] + row * sq)
    return x, y, round(sq), round(sq)
