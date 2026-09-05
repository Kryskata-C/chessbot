"""Read the opponent's displayed rating ("Isabel (1600)") off the screen.

chess.com prints the opponent's name and rating on the line just above
the board. That number is a far better prior for how strong they are
than a dozen moves of centipawn loss, so it anchors the opponent
estimate before their play says anything. Uses Apple's Vision framework
(built into macOS, so nothing to install for the packaged app) and falls
back to the tesseract CLI; silently unavailable if neither works.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

_TESSERACT = shutil.which("tesseract")

try:  # macOS Vision framework via pyobjc
    import Vision as _Vision
    import Quartz as _Quartz
    from Foundation import NSData as _NSData
    _HAVE_VISION = True
except Exception:  # not macOS, or pyobjc missing
    _HAVE_VISION = False

# Vertical windows above the board (in squares) that frame the name line
# on chess.com's layout; a couple of alternatives cover layout jitter.
# (top, bottom, tesseract page-segmentation mode). The tight single-line
# windows hit the name line on chess.com's layout at 1x and 2x capture;
# the last, wide one reads the whole strip as a block as a fallback.
_WINDOWS = ((0.62, 0.30, 7), (0.60, 0.32, 7), (0.75, 0.28, 7), (0.80, 0.25, 7), (1.0, 0.10, 6))
_RATING_RE = re.compile(r"\((\d{3,4})\)")


def ocr_available() -> bool:
    return _HAVE_VISION or _TESSERACT is not None


def _ocr_vision(gray: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", gray)
    if not ok:
        return ""
    data = _NSData.dataWithBytes_length_(png.tobytes(), len(png))
    src = _Quartz.CGImageSourceCreateWithData(data, None)
    img = _Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return ""
    req = _Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(_Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    handler = _Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    handler.performRequests_error_([req], None)
    lines = []
    for r in req.results() or []:
        cands = r.topCandidates_(1)
        if cands:
            lines.append(str(cands[0].string()))
    return " ".join(lines)


def _ocr_tesseract(gray: np.ndarray, psm: int = 7) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, gray)
        r = subprocess.run(
            [_TESSERACT, path, "stdout", "--psm", str(psm)],
            capture_output=True, timeout=5,
        )
        return r.stdout.decode("utf-8", "replace")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _ocr(gray: np.ndarray, psm: int = 7) -> str:
    if _HAVE_VISION:
        text = _ocr_vision(gray)
        if text or _TESSERACT is None:
            return text
    if _TESSERACT is not None:
        return _ocr_tesseract(gray, psm)
    return ""


def read_opponent_rating(screenshot: np.ndarray, board: dict) -> int | None:
    """Rating in parentheses on the opponent line above the board, or None."""
    if not ocr_available():
        return None
    sq = board["square_size"]
    x0 = int(board["x"] + 0.08 * board["width"])   # skip the avatar
    x1 = int(board["x"] + 0.6 * board["width"])
    img_h, img_w = screenshot.shape[:2]
    for top, bot, psm in _WINDOWS:
        y0 = int(board["y"] - top * sq)
        y1 = int(board["y"] - bot * sq)
        if y0 < 0 or y1 <= y0 or x1 <= x0 or x1 > img_w:
            continue
        crop = cv2.cvtColor(screenshot[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        # Tesseract wants ~30px+ glyphs: scale the strip to ~160px tall
        # whatever the capture resolution (1x live grabs, 2x screenshots).
        fx = max(2, int(math.ceil(160 / max(1, crop.shape[0]))))
        gray = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bitwise_not(gray)  # white-on-dark -> dark-on-white
        gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,
                                  cv2.BORDER_CONSTANT, value=255)
        try:
            text = _ocr(gray, psm)
        except Exception:
            return None
        m = _RATING_RE.search(text)
        if m:
            rating = int(m.group(1))
            if 100 <= rating <= 3500:
                return rating
    return None
