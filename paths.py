"""Where the app keeps its files.

Running from a checkout, everything lives next to the sources as before.
Inside the packaged .app the bundle is read-only (and signed), so anything
the app writes -- piece templates, live game logs, the session governor
state, the opening repertoire seed, the log file -- goes to
~/Library/Application Support/Chess Vision/. Bundled read-only resources
(the Stockfish binary) are looked up through resource_path().
"""

from __future__ import annotations

import os
import sys

APP_NAME = "Chess Vision"

FROZEN = bool(getattr(sys, "frozen", False))
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts: str) -> str:
    """A file shipped with the app (read-only)."""
    base = getattr(sys, "_MEIPASS", _SRC_DIR)
    return os.path.join(base, *parts)


def data_dir() -> str:
    """Writable per-user directory; created on first use."""
    if FROZEN or os.environ.get("CHESS_VISION_DATA_DIR"):
        base = os.environ.get("CHESS_VISION_DATA_DIR") or os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", APP_NAME)
    else:
        base = _SRC_DIR
    os.makedirs(base, exist_ok=True)
    return base


def data_path(*parts: str) -> str:
    return os.path.join(data_dir(), *parts)


TEMPLATE_DIR = data_path("templates")
LIVE_GAMES_DIR = data_path("live_games")
SESSION_FILE = data_path("session.json") if FROZEN else os.path.join(
    os.path.expanduser("~"), ".chess_vision_session.json")
REPERTOIRE_FILE = data_path("repertoire") if FROZEN else os.path.join(
    os.path.expanduser("~"), ".chess_vision_repertoire")
LOG_FILE = data_path("chess-vision.log")
