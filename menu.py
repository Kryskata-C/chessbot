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
TIME_CONTROLS = ["auto", "1+0", "2+1", "3+0", "3+2", "5+0", "5+3", "10+0", "15+10", "30+0"]


def strength_label(elo: int) -> str:
    if elo < 800: return "beginner"
    if elo < 1200: return "casual"
    if elo < 1600: return "club player"
    if elo < 2000: return "strong club"
    if elo < 2400: return "expert"
    return "master"


class MenuWindow(Card):
    started = pyqtSignal(str, int, dict, str)   # (color "w"/"b"/"auto", target_elo, visuals, time control)
    sign_out = pyqtSignal()

    MIN_ELO, MAX_ELO, STEP = 400, 2800, 50
    DEFAULT_NOTE = "Set up the game, then start scanning."

    def __init__(self):
        super().__init__(width=420, on_close=QApplication.quit)
        lay = QVBoxLayout(self); lay.setContentsMargins(28, 24, 28, 22); lay.setSpacing(10)
        lay.addWidget(self.header())
        self.tag = QLabel(self.DEFAULT_NOTE); self.tag.setObjectName("dim"); self.tag.setWordWrap(True)
        lay.addWidget(self.tag); lay.addSpacing(8)

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

        lay.addWidget(section("Time control"))
        tgrid = QGridLayout(); tgrid.setSpacing(6)
        self._tc_group = QButtonGroup(self); self._tc_group.setExclusive(True)
        self._tc_btns = {}
        for i, key in enumerate(TIME_CONTROLS):
            b = QPushButton("Read clock" if key == "auto" else key); b.setObjectName("pill"); b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self._tc_group.addButton(b); self._tc_btns[key] = b; tgrid.addWidget(b, i // 5, i % 5)
        self._tc_btns["auto"].setChecked(True)
        lay.addLayout(tgrid)
        thint = QLabel("Paces the think timer so the clock lasts. Read clock = infer it from your clock at move one."); thint.setObjectName("dim"); thint.setWordWrap(True)
        lay.addWidget(thint); lay.addSpacing(10)

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

        self.update_label = QLabel(""); self.update_label.setObjectName("ok")
        self.update_label.setOpenExternalLinks(True); self.update_label.setWordWrap(True)
        self.update_label.hide()
        lay.addWidget(self.update_label)

        foot = QHBoxLayout()
        self.account_label = QLabel(""); self.account_label.setObjectName("dim")
        out = QPushButton("Sign out"); out.setObjectName("link"); out.setCursor(Qt.CursorShape.PointingHandCursor)
        out.clicked.connect(self.sign_out.emit)
        foot.addWidget(self.account_label); foot.addStretch(); foot.addWidget(out)
        lay.addLayout(foot)

    def set_note(self, text: str | None = None) -> None:
        """Line under the wordmark: why we're back here (game over, stopped)."""
        self.tag.setText(text or self.DEFAULT_NOTE)

    def set_update(self, version: str, url: str) -> None:
        """A newer build is on the website: say so, with a download link."""
        self.update_label.setText(
            f'Version {version} is out — <a href="{url}" style="color: #b9f24a;">download it</a>')
        self.update_label.show()

    def set_account(self, email: str, status: str = "") -> None:
        self.account_label.setText(f"{email}  ·  {status}" if status else email)

    def _elo_changed(self, v: int):
        snapped = round(v / self.STEP) * self.STEP
        if snapped != v:
            self.slider.setValue(snapped); return
        self.elo_label.setText(str(snapped)); self.level_label.setText(strength_label(snapped))

    def color(self) -> str:
        return next(k for k, b in self._color_btns.items() if b.isChecked())

    def time_control(self) -> str:
        return next(k for k, b in self._tc_btns.items() if b.isChecked())

    def _on_start(self):
        visuals = {k: b.isChecked() for k, b in self._vis.items()}
        self.started.emit(self.color(), self.slider.value(), visuals, self.time_control())
        self.hide()

