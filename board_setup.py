"""Board-setup reminder, shown after sign-in and before the setup card.

The recogniser only knows chess.com's default look (green board, Neo
pieces): other themes leave it blind. So before the first game we show a
picture of the board it expects, with the two settings to check, and a
button to open the right page on chess.com. "Don't show this again" is
remembered per user."""

from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt, QRectF, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QFont, QDesktopServices, QPen
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
                             QCheckBox, QApplication)

from paths import data_path
from ui_theme import Card, section, PANEL2, LINE, MUTED, INK, ACC

SETTINGS_URL = "https://www.chess.com/settings/board"
PREFS_FILE = data_path("prefs.json")
# chess.com "Green" board + Neo piece colours
LIGHT, DARK = QColor("#ebecd0"), QColor("#779556")
WHITE_PIECE, BLACK_PIECE = QColor("#f8f8f8"), QColor("#2b2b2b")
BACK_RANK = "♜♞♝♛♚♝♞♜"


def _prefs() -> dict:
    try:
        with open(PREFS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_pref(key: str, value) -> None:
    d = _prefs(); d[key] = value
    try:
        os.makedirs(os.path.dirname(PREFS_FILE), exist_ok=True)
        with open(PREFS_FILE, "w") as f:
            json.dump(d, f, indent=2)
    except OSError:
        pass


def reminder_dismissed() -> bool:
    return bool(_prefs().get("board_reminder_dismissed"))


class BoardPreview(QWidget):
    """A small painted chess.com board in the expected theme, start position."""

    def __init__(self, square: int = 22):
        super().__init__()
        self.sq = square
        self.setFixedSize(square * 8, square * 8)

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        clip = QPainterPath(); clip.addRoundedRect(QRectF(self.rect()), 8, 8)
        p.setClipPath(clip)
        s = self.sq
        for r in range(8):
            for c in range(8):
                p.fillRect(c * s, r * s, s, s, LIGHT if (r + c) % 2 == 0 else DARK)
        font = QFont("Apple Symbols, Segoe UI Symbol, DejaVu Sans"); font.setPixelSize(int(s * 0.86))
        p.setFont(font)
        rows = [(0, BACK_RANK, BLACK_PIECE), (1, "♟" * 8, BLACK_PIECE),
                (6, "♟" * 8, WHITE_PIECE), (7, BACK_RANK, WHITE_PIECE)]
        for r, glyphs, colour in rows:
            for c, g in enumerate(glyphs):
                cell = QRectF(c * s, r * s, s, s)
                # soft outline so white pieces read on the light squares
                p.setPen(QPen(QColor(0, 0, 0, 90) if colour == WHITE_PIECE else QColor(255, 255, 255, 40), 1.4))
                p.drawText(cell.translated(0, 1), Qt.AlignmentFlag.AlignCenter, g)
                p.setPen(colour)
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter, g)
        p.setClipping(False)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(clip)


def _step(n: int, title: str, detail: str) -> QWidget:
    w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(12)
    num = QLabel(str(n)); num.setFixedSize(24, 24); num.setAlignment(Qt.AlignmentFlag.AlignCenter)
    num.setStyleSheet(f"background: {PANEL2}; border: 1px solid {LINE}; border-radius: 12px; "
                      f"color: {ACC}; font-family: SF Mono, Menlo, Monaco; font-size: 11px; font-weight: 600;")
    col = QVBoxLayout(); col.setSpacing(1); col.setContentsMargins(0, 0, 0, 0)
    t = QLabel(title); t.setStyleSheet("font-size: 13px; font-weight: 600;")
    d = QLabel(detail); d.setObjectName("dim"); d.setWordWrap(True)
    col.addWidget(t); col.addWidget(d)
    h.addWidget(num, alignment=Qt.AlignmentFlag.AlignTop); h.addLayout(col, 1)
    return w


class BoardSetupWindow(Card):
    """Emits `done` when the user continues (and remembers the opt-out)."""

    done = pyqtSignal()

    def __init__(self):
        super().__init__(width=460, on_close=QApplication.quit)
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 22); lay.setSpacing(10)
        lay.addWidget(self.header())
        lay.addWidget(section("Before you play"))

        title = QLabel("Give chess.com its classic look.")
        title.setStyleSheet("font-family: Georgia, Iowan Old Style, Times New Roman; font-size: 24px; letter-spacing: -0.3px;")
        title.setWordWrap(True); lay.addWidget(title)
        body = QLabel("Chess Vision reads the board straight off your screen, and it only "
                      "recognises chess.com's default theme. Any other board or piece "
                      "style leaves it blind.")
        body.setObjectName("dim"); body.setWordWrap(True); lay.addWidget(body)
        lay.addSpacing(6)

        row = QHBoxLayout(); row.setSpacing(20)
        row.addWidget(BoardPreview(22), alignment=Qt.AlignmentFlag.AlignTop)
        steps = QVBoxLayout(); steps.setSpacing(12)
        steps.addWidget(_step(1, "Open Board & Pieces", "Settings on chess.com, or the button below."))
        steps.addWidget(_step(2, "Board: Green", "The default. Any other colour hides the board from us."))
        steps.addWidget(_step(3, "Pieces: Neo", "The default set, so every piece is recognised."))
        steps.addStretch()
        row.addLayout(steps, 1)
        lay.addLayout(row); lay.addSpacing(10)

        btns = QHBoxLayout(); btns.setSpacing(8)
        open_btn = QPushButton("Open chess.com settings"); open_btn.setObjectName("ghost")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(SETTINGS_URL)))
        go = QPushButton("My board is set"); go.setCursor(Qt.CursorShape.PointingHandCursor)
        go.clicked.connect(self._continue)
        btns.addWidget(open_btn); btns.addWidget(go, 1)
        lay.addLayout(btns)

        self.skip = QCheckBox("Don't show this again")
        self.skip.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self.skip, alignment=Qt.AlignmentFlag.AlignCenter)

    def _continue(self):
        if self.skip.isChecked():
            _save_pref("board_reminder_dismissed", True)
        self.done.emit(); self.hide()
