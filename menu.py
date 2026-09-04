"""Setup card: colour, strength, what to draw on the board, start."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QSlider, QButtonGroup, QWidget, QApplication)

from ui_theme import Card, section

VISUALS = [
    ("arrow", "Move arrow"), ("ghost", "Ghost piece"), ("reply", "Their reply"),
    ("pv", "Line preview"), ("candidates", "Alternatives"), ("threats", "Threats"),
    ("trail", "Move trail"), ("evalbar", "Eval bar"), ("timing", "Think timer"),
]
DEFAULT_OFF = {"pv", "candidates"}


def strength_label(elo: int) -> str:
    if elo < 800: return "beginner"
    if elo < 1200: return "casual"
    if elo < 1600: return "club player"
    if elo < 2000: return "strong club"
    if elo < 2400: return "expert"
    return "master"


class MenuWindow(Card):
    started = pyqtSignal(str, int, dict)   # (color "w"/"b"/"auto", target_elo, visuals)
    sign_out = pyqtSignal()

    MIN_ELO, MAX_ELO, STEP = 400, 2800, 50

    def __init__(self):
        super().__init__(width=420, on_close=QApplication.quit)
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 22); lay.setSpacing(10)
        lay.addWidget(self.header())
        tag = QLabel("Set up the game, then start scanning."); tag.setObjectName("dim")
        lay.addWidget(tag); lay.addSpacing(8)

        lay.addWidget(section("Play as"))
        row = QHBoxLayout(); row.setSpacing(8)
        self._color_group = QButtonGroup(self); self._color_group.setExclusive(True)
        self._color_btns = {}
        for key, text in (("auto", "Auto-detect"), ("w", "White"), ("b", "Black")):
            b = QPushButton(text); b.setObjectName("tile"); b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self._color_group.addButton(b); self._color_btns[key] = b; row.addWidget(b)
        self._color_btns["auto"].setChecked(True)
        lay.addLayout(row); lay.addSpacing(10)

        lay.addWidget(section("Strength"))
        srow = QHBoxLayout()
        self.elo_label = QLabel("1400"); self.elo_label.setObjectName("big")
        self.level_label = QLabel(strength_label(1400)); self.level_label.setObjectName("dim")
        self.level_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        srow.addWidget(self.elo_label); srow.addStretch(); srow.addWidget(self.level_label)
        lay.addLayout(srow)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(self.MIN_ELO, self.MAX_ELO); self.slider.setValue(1400)
        self.slider.setSingleStep(self.STEP); self.slider.setPageStep(200)
        self.slider.valueChanged.connect(self._elo_changed)
        lay.addWidget(self.slider)
        hint = QLabel("Rating it imitates. It adapts to each opponent from here."); hint.setObjectName("dim")
        lay.addWidget(hint); lay.addSpacing(10)

        lay.addWidget(section("Show on the board"))
        grid = QGridLayout(); grid.setSpacing(6)
        self._vis = {}
        for i, (key, text) in enumerate(VISUALS):
            b = QPushButton(text); b.setObjectName("pill"); b.setCheckable(True)
            b.setChecked(key not in DEFAULT_OFF); b.setCursor(Qt.CursorShape.PointingHandCursor)
            self._vis[key] = b; grid.addWidget(b, i // 3, i % 3)
        lay.addLayout(grid); lay.addSpacing(14)

        self.start_btn = QPushButton("Start scanning"); self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._on_start)
        lay.addWidget(self.start_btn)

        foot = QHBoxLayout()
        self.account_label = QLabel(""); self.account_label.setObjectName("dim")
        out = QPushButton("Sign out"); out.setObjectName("link"); out.setCursor(Qt.CursorShape.PointingHandCursor)
        out.clicked.connect(self.sign_out.emit)
        foot.addWidget(self.account_label); foot.addStretch(); foot.addWidget(out)
        lay.addLayout(foot)

    def set_account(self, email: str, status: str = "") -> None:
        self.account_label.setText(f"{email}  ·  {status}" if status else email)

    def _elo_changed(self, v: int):
        snapped = round(v / self.STEP) * self.STEP
        if snapped != v:
            self.slider.setValue(snapped); return
        self.elo_label.setText(str(snapped)); self.level_label.setText(strength_label(snapped))

    def color(self) -> str:
        return next(k for k, b in self._color_btns.items() if b.isChecked())

    def _on_start(self):
        visuals = {k: b.isChecked() for k, b in self._vis.items()}
        self.started.emit(self.color(), self.slider.value(), visuals)
        self.hide()

