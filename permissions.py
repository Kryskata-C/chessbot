"""Ask for the macOS permissions the app needs before anything else.

Screen Recording is the one that matters: without it every capture is a
black frame and the board is never found, which looks like the app is
broken. macOS only lists an app under Privacy & Security → Screen
Recording once it has asked, and a freshly granted permission applies
only after the process restarts — so the card explains, asks, opens the
right settings pane, and offers a relaunch.
"""

from __future__ import annotations

import os
import subprocess
import sys

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

from ui_theme import Card, section

_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"


def screen_recording_granted() -> bool:
    try:
        import Quartz
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return True  # can't tell (not macOS / no pyobjc): don't block


def request_screen_recording() -> bool:
    """Trigger the system prompt; True if already/now granted."""
    try:
        import Quartz
        return bool(Quartz.CGRequestScreenCaptureAccess())
    except Exception:
        return True


def relaunch() -> None:
    """Start a fresh copy of the app and quit this one (a new Screen
    Recording grant only takes effect in a new process)."""
    exe = sys.executable
    if getattr(sys, "frozen", False):
        bundle = os.path.abspath(os.path.join(os.path.dirname(exe), "..", ".."))
        if bundle.endswith(".app"):
            subprocess.Popen(["/bin/sh", "-c", f'sleep 0.7; open -n "{bundle}"'])
        else:
            subprocess.Popen(["/bin/sh", "-c", f'sleep 0.7; "{exe}"'])
    else:
        subprocess.Popen(["/bin/sh", "-c",
                          "sleep 0.7; " + " ".join(f'"{a}"' for a in [exe, *sys.argv])])
    QApplication.quit()


class PermissionWindow(Card):
    """Blocks until Screen Recording is granted; emits `granted` (dev mode)
    or relaunches the app (packaged) once it is."""

    granted = pyqtSignal()

    def __init__(self):
        super().__init__(width=400, on_close=QApplication.quit)
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 22); lay.setSpacing(10)
        lay.addWidget(self.header())
        lay.addWidget(section("Before your first game"))
        body = QLabel(
            "Chess Vision reads the chess.com board off your screen, so macOS "
            "has to allow it to record the screen.\n\n"
            "1. Click Open System Settings.\n"
            "2. Turn on Chess Vision under Screen Recording.\n"
            "3. Come back here — the app relaunches itself.")
        body.setWordWrap(True); lay.addWidget(body)
        self.state = QLabel("Waiting for permission…"); self.state.setObjectName("dim")
        self.state.setWordWrap(True); lay.addWidget(self.state)
        row = QHBoxLayout(); row.setSpacing(8)
        self.open_btn = QPushButton("Open System Settings")
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.clicked.connect(self.open_settings)
        self.relaunch_btn = QPushButton("Relaunch Chess Vision")
        self.relaunch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.relaunch_btn.clicked.connect(relaunch)
        self.relaunch_btn.hide()
        row.addWidget(self.open_btn); row.addWidget(self.relaunch_btn)
        lay.addLayout(row)
        self._timer = QTimer(self); self._timer.timeout.connect(self._poll)
        self._done = False

    def start(self) -> None:
        """Ask macOS (this registers the app in the Screen Recording list
        and shows the system prompt), then keep checking."""
        if request_screen_recording():
            self._on_granted(); return
        self.show()
        self._timer.start(1000)

    def open_settings(self) -> None:
        subprocess.Popen(["open", _SETTINGS_URL])

    def _poll(self) -> None:
        if screen_recording_granted():
            self._timer.stop()
            self._on_granted()

    def _on_granted(self) -> None:
        if self._done:
            return
        self._done = True
        if getattr(sys, "frozen", False) and self.isVisible():
            # Granted while running: capture stays black until a new
            # process starts. Say so and relaunch.
            self.state.setText("Granted. Relaunching so it takes effect…")
            self.state.setObjectName("ok"); self.state.style().polish(self.state)
            self.open_btn.hide(); self.relaunch_btn.show()
            QTimer.singleShot(1500, relaunch)
            return
        self.hide()
        self.granted.emit()
