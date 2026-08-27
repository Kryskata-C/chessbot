"""Animated startup menu for Chess Vision.

Developed & owned by Kryskata-C.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QPointF, QRect, QRectF, pyqtSignal, pyqtProperty,
    QPropertyAnimation, QEasingCurve,
)
from PyQt6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPainterPath, QPixmap,
)
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QApplication, QGraphicsDropShadowEffect,
)

ACCENT = QColor(77, 163, 255)
ACCENT_DEEP = QColor(48, 110, 220)
GOLD = QColor(232, 184, 75)
BG_TOP = QColor(17, 22, 32)
BG_BOTTOM = QColor(9, 12, 19)
TEXT_MAIN = QColor(226, 233, 244)
TEXT_DIM = QColor(123, 138, 165)

# (key, icon, label, tooltip, default)
FEATURES = [
    ("arrow", "➤", "Best-move arrow",
     "Animated glowing arrow for the suggested move", True),
    ("ghost", "♞", "Ghost piece slide",
     "A translucent piece glides along the suggested move", True),
    ("reply", "⚔", "Enemy reply arrow",
     "Predicted opponent response, drawn dashed in orange", True),
    ("pv", "☰", "Line preview",
     "Chains the next plies of the engine line after the reply", False),
    ("candidates", "≡", "Candidate moves",
     "Faint arrows for alternative strong moves", False),
    ("threats", "☠", "Threat radar",
     "Pulses a red glow on your pieces that are in danger", True),
    ("trail", "✦", "Enemy move trail",
     "Fading trail showing the opponent's last move", True),
    ("evalbar", "▮", "Live eval bar",
     "Animated evaluation bar pinned beside the board", True),
]


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


def _with_alpha(c: QColor, alpha: int) -> QColor:
    out = QColor(c)
    out.setAlpha(alpha)
    return out


class ToggleSwitch(QWidget):
    """iOS-style animated toggle switch."""

    toggled = pyqtSignal(bool)

    def __init__(self, checked: bool = True, parent=None):
        super().__init__(parent)
        self._checked = checked
        self._knob = 1.0 if checked else 0.0
        self.setFixedSize(44, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._anim = QPropertyAnimation(self, b"knobPos", self)
        self._anim.setDuration(170)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, on: bool):
        if on == self._checked:
            return
        self._checked = on
        self._anim.stop()
        self._anim.setStartValue(self._knob)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()
        self.toggled.emit(on)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def getKnobPos(self) -> float:
        return self._knob

    def setKnobPos(self, v: float):
        self._knob = v
        self.update()

    knobPos = pyqtProperty(float, fget=getKnobPos, fset=setKnobPos)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = self._knob

        track = _lerp_color(QColor(52, 60, 76), ACCENT, t)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 3, 44, 18), 9, 9)

        # Knob with a soft shadow
        x = 3 + t * 20
        p.setBrush(QColor(0, 0, 0, 70))
        p.drawEllipse(QRectF(x + 1, 4.5, 18, 18))
        p.setBrush(QColor(245, 248, 252))
        p.drawEllipse(QRectF(x, 3, 18, 18))
        p.end()


class ColorSelect(QWidget):
    """Segmented Auto/White/Black picker with a sliding indicator."""

    OPTIONS = [("auto", "✦  Auto"), ("w", "♔  White"), ("b", "♚  Black")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = "auto"
        self._sel = 0.0  # segment index the indicator sits on (0..2)
        self.setFixedHeight(46)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._anim = QPropertyAnimation(self, b"selPos", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def color(self) -> str:
        return self._color

    def getSelPos(self) -> float:
        return self._sel

    def setSelPos(self, v: float):
        self._sel = v
        self.update()

    selPos = pyqtProperty(float, fget=getSelPos, fset=setSelPos)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        n = len(self.OPTIONS)
        idx = min(n - 1, int(event.position().x() / (self.width() / n)))
        color = self.OPTIONS[idx][0]
        if color == self._color:
            return
        self._color = color
        self._anim.stop()
        self._anim.setStartValue(self._sel)
        self._anim.setEndValue(float(idx))
        self._anim.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self.OPTIONS)

        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        p.setBrush(QColor(22, 29, 43))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 12, 12)

        # Sliding indicator
        seg = (w - 8) / n
        ix = 4 + self._sel * seg
        grad = QLinearGradient(ix, 0, ix + seg, 0)
        grad.setColorAt(0, ACCENT_DEEP)
        grad.setColorAt(1, ACCENT)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(ix, 4, seg, h - 8), 9, 9)

        # Labels
        font = QFont("Helvetica Neue", 12, QFont.Weight.DemiBold)
        p.setFont(font)
        for i, (_, label) in enumerate(self.OPTIONS):
            on = max(0.0, 1.0 - abs(self._sel - i))
            p.setPen(_lerp_color(TEXT_DIM, QColor(255, 255, 255), on))
            p.drawText(QRectF(i * w / n, 0, w / n, h),
                       Qt.AlignmentFlag.AlignCenter, label)
        p.end()


def _elo_tier(elo: int) -> str:
    if elo < 800:
        return "Novice"
    if elo < 1200:
        return "Beginner"
    if elo < 1600:
        return "Intermediate"
    if elo < 2000:
        return "Advanced"
    if elo < 2400:
        return "Expert"
    return "Master"


class StrengthSelect(QWidget):
    """Draggable ELO slider that sets the strength the bot imitates."""

    MIN_ELO = 400
    MAX_ELO = 2800
    STEP = 25

    def __init__(self, default: int = 1400, parent=None):
        super().__init__(parent)
        self._elo = default
        self.setFixedHeight(58)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._val_font = QFont("Helvetica Neue", 15, QFont.Weight.Black)
        self._tier_font = QFont("Helvetica Neue", 10, QFont.Weight.DemiBold)

    def value(self) -> int:
        return self._elo

    def _x_to_elo(self, x: float) -> int:
        frac = (x - 7) / max(1, self.width() - 14)
        frac = max(0.0, min(1.0, frac))
        raw = self.MIN_ELO + frac * (self.MAX_ELO - self.MIN_ELO)
        return int(round(raw / self.STEP) * self.STEP)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_from_x(event.position().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._set_from_x(event.position().x())

    def _set_from_x(self, x: float):
        elo = self._x_to_elo(x)
        if elo != self._elo:
            self._elo = elo
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Value + tier readout
        p.setFont(self._val_font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(0, 0, w, 24), Qt.AlignmentFlag.AlignLeft |
                   Qt.AlignmentFlag.AlignVCenter, f"{self._elo}")
        p.setFont(self._tier_font)
        p.setPen(ACCENT)
        p.drawText(QRectF(0, 0, w, 24), Qt.AlignmentFlag.AlignRight |
                   Qt.AlignmentFlag.AlignVCenter, _elo_tier(self._elo).upper())

        # Track
        track_y = h - 14
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(22, 29, 43))
        p.drawRoundedRect(QRectF(7, track_y - 3, w - 14, 6), 3, 3)

        frac = (self._elo - self.MIN_ELO) / (self.MAX_ELO - self.MIN_ELO)
        hx = 7 + frac * (w - 14)
        grad = QLinearGradient(7, 0, hx, 0)
        grad.setColorAt(0, ACCENT_DEEP)
        grad.setColorAt(1, ACCENT)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(7, track_y - 3, max(6, hx - 7), 6), 3, 3)

        # Handle
        p.setBrush(QColor(255, 255, 255))
        p.setPen(QPen(ACCENT, 2))
        p.drawEllipse(QPointF(hx, track_y), 8, 8)
        p.end()


class FeatureRow(QWidget):
    """One toggleable visual effect: icon, name, animated switch."""

    def __init__(self, icon: str, label: str, tooltip: str, checked: bool,
                 parent=None):
        super().__init__(parent)
        self._hover = False
        self.setFixedHeight(34)
        self.setToolTip(tooltip)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 8, 0)
        layout.setSpacing(10)

        icon_label = QLabel(icon)
        icon_label.setFixedWidth(22)
        icon_label.setStyleSheet(
            f"color: rgb({ACCENT.red()},{ACCENT.green()},{ACCENT.blue()});"
            "font-size: 15px; background: transparent;"
        )
        name_label = QLabel(label)
        name_label.setStyleSheet(
            "color: #dfe6f2; font-size: 13px; background: transparent;"
        )
        self.switch = ToggleSwitch(checked)

        layout.addWidget(icon_label)
        layout.addWidget(name_label)
        layout.addStretch()
        layout.addWidget(self.switch)

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def paintEvent(self, event):
        if self._hover:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 12))
            p.drawRoundedRect(self.rect(), 8, 8)
            p.end()


class MenuWindow(QDialog):
    """Animated startup menu: color pick + visual effect toggles."""

    started = pyqtSignal(str, int, dict)  # (color "w"/"b", target_elo, visuals)

    HEADER_H = 148
    FOOTER_H = 84

    def __init__(self):
        super().__init__()
        self._drag_pos: Optional[QPoint] = None
        self._t0 = time.time()
        self._entrance_done = False

        self.setWindowTitle("Chess Vision")
        self.setFixedSize(400, 762)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Drifting background chess pieces, confined to the animated
        # header/footer bands so per-frame repaints stay small.
        def _particle(band):
            return {
                "ch": random.choice("♞♜♝♛♚♟"
                                    "♘♖♗♕♔♙"),
                "x": random.random(),
                "y": random.random(),
                "band": band,  # (y_top_frac, y_bottom_frac) of the window
                "size": random.uniform(13, 28),
                "speed": random.uniform(0.010, 0.032),
                "sway": random.uniform(6, 18),
                "phase": random.uniform(0, math.tau),
                "alpha": random.randint(14, 36),
            }
        self._pieces = (
            [_particle((0.0, 0.25)) for _ in range(9)]
            + [_particle((0.86, 1.0)) for _ in range(4)]
        )

        # Cached paint resources — the card background never changes, and
        # fonts are expensive to rebuild 30x a second.
        self._bg_cache: Optional[QPixmap] = None
        self._title_font = QFont("Helvetica Neue", 23, QFont.Weight.Black)
        self._title_font.setLetterSpacing(
            QFont.SpacingType.AbsoluteSpacing, 5)
        self._sub_font = QFont("Helvetica Neue", 10)
        self._sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        self._tiny_font = QFont("Helvetica Neue", 8)
        self._tiny_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        self._brand_font = QFont("Helvetica Neue", 14, QFont.Weight.Black)
        self._brand_font.setLetterSpacing(
            QFont.SpacingType.AbsoluteSpacing, 3)
        self._copy_font = QFont("Helvetica Neue", 8)
        self._knight_font = QFont("Arial", 40)
        self._particle_fonts: dict[int, QFont] = {}

        # --- interactive content ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, self.HEADER_H, 26, self.FOOTER_H)
        layout.setSpacing(0)

        self.color_select = ColorSelect()
        layout.addWidget(self.color_select)
        layout.addSpacing(16)

        strength_label = QLabel("BOT STRENGTH")
        strength_label.setStyleSheet(
            "color: #7b8aa5; font-size: 10px; font-weight: 700;"
            "letter-spacing: 2px; background: transparent;"
        )
        layout.addWidget(strength_label)
        layout.addSpacing(2)
        self.strength_select = StrengthSelect()
        layout.addWidget(self.strength_select)
        layout.addSpacing(12)

        section = QLabel("VISUAL EFFECTS")
        section.setStyleSheet(
            "color: #7b8aa5; font-size: 10px; font-weight: 700;"
            "letter-spacing: 2px; background: transparent;"
        )
        layout.addWidget(section)
        layout.addSpacing(8)

        self._rows: dict[str, FeatureRow] = {}
        for key, icon, label, tooltip, default in FEATURES:
            row = FeatureRow(icon, label, tooltip, default)
            self._rows[key] = row
            layout.addWidget(row)
            layout.addSpacing(4)

        layout.addSpacing(10)
        self.start_btn = QPushButton("▶  START SCANNING")
        self.start_btn.setFixedHeight(46)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #306edc, stop:1 #4da3ff);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 800;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3f7de8, stop:1 #63b1ff);
            }
            QPushButton:pressed {
                background: #2a5fc4;
            }
        """)
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)

        # Soft static glow on the start button. (Animating the blur radius
        # re-rasterizes a Gaussian blur every frame — it alone cost ~50% of
        # a CPU core, so the glow stays static.)
        glow = QGraphicsDropShadowEffect(self.start_btn)
        glow.setOffset(0, 0)
        glow.setColor(_with_alpha(ACCENT, 160))
        glow.setBlurRadius(26)
        self.start_btn.setGraphicsEffect(glow)

        # Decorative animation clock — repaints only the animated
        # header/footer bands, not the whole card.
        self._deco_timer = QTimer(self)
        self._deco_timer.setInterval(50)
        self._deco_timer.timeout.connect(self._deco_tick)
        self._deco_timer.start()

        # Center on screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )

    # --- start / signals ---

    def _on_start(self):
        visuals = {key: row.switch.isChecked()
                   for key, row in self._rows.items()}
        self.started.emit(self.color_select.color(),
                          self.strength_select.value(), visuals)
        self.hide()

    # --- entrance animation ---

    def showEvent(self, event):
        super().showEvent(event)
        self._deco_timer.start()
        if self._entrance_done:
            return
        self._entrance_done = True
        target = self.pos()
        self.move(target + QPoint(0, 26))
        self.setWindowOpacity(0.0)

        self._fade_in = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_in.setDuration(480)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_in.start()

        self._slide_in = QPropertyAnimation(self, b"pos", self)
        self._slide_in.setDuration(480)
        self._slide_in.setStartValue(self.pos())
        self._slide_in.setEndValue(target)
        self._slide_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._slide_in.start()

    # --- painting ---

    def _deco_tick(self):
        """Repaint only the animated header and footer bands."""
        w, h = self.width(), self.height()
        self.update(QRect(0, 0, w, 175))
        self.update(QRect(0, h - self.FOOTER_H, w, self.FOOTER_H))

    def _ensure_bg_cache(self):
        """Render the static card background (gradient + border) once."""
        if self._bg_cache is not None:
            return
        w, h = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        pm = QPixmap(int(w * dpr), int(h * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 18, 18)
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, BG_TOP)
        grad.setColorAt(1, BG_BOTTOM)
        p.fillPath(path, QBrush(grad))
        p.setPen(QPen(_with_alpha(ACCENT, 70), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 18, 18)
        p.end()
        self._bg_cache = pm

    def _particle_font(self, size: int) -> QFont:
        font = self._particle_fonts.get(size)
        if font is None:
            font = QFont("Arial", size)
            self._particle_fonts[size] = font
        return font

    def paintEvent(self, event):
        self._ensure_bg_cache()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = time.time() - self._t0
        w, h = self.width(), self.height()
        dirty = event.rect()

        p.drawPixmap(0, 0, self._bg_cache)

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 18, 18)
        p.setClipPath(clip)

        header = QRect(0, 0, w, 175)
        footer = QRect(0, h - self.FOOTER_H, w, self.FOOTER_H)

        # Drifting chess pieces (confined to header/footer bands)
        for pc in self._pieces:
            band_top, band_bot = pc["band"]
            span = band_bot - band_top
            yfrac = band_top + ((pc["y"] - t * pc["speed"] / span) % 1.0) * span
            y = yfrac * h
            strip = header if band_bot <= 0.5 else footer
            if not dirty.intersects(strip):
                continue
            x = pc["x"] * w + math.sin(t * 0.7 + pc["phase"]) * pc["sway"]
            p.setFont(self._particle_font(int(pc["size"])))
            p.setPen(QColor(200, 215, 240, pc["alpha"]))
            p.drawText(QPointF(x, y), pc["ch"])

        if dirty.intersects(header):
            # Soft accent glow behind the header emblem
            pulse = 0.7 + 0.3 * math.sin(t * 1.6)
            glow = QRadialGradient(QPointF(w / 2, 64), 95)
            glow.setColorAt(0, _with_alpha(ACCENT, int(30 * pulse)))
            glow.setColorAt(1, _with_alpha(ACCENT, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawRect(QRectF(0, 0, w, 170))

            # Knight emblem with glow pulse
            self._knight_font.setPointSize(int(40 + 2 * math.sin(t * 1.6)))
            p.setFont(self._knight_font)
            p.setPen(_with_alpha(ACCENT, 80))
            p.drawText(QRectF(0, 14, w, 64), Qt.AlignmentFlag.AlignCenter, "♞")
            p.setPen(QColor(232, 241, 255))
            p.drawText(QRectF(0, 12, w, 64), Qt.AlignmentFlag.AlignCenter, "♞")

            # Title with a moving shimmer
            p.setFont(self._title_font)
            p.setPen(QPen(QBrush(self._shimmer(
                w, t, QColor(214, 226, 243), QColor(255, 255, 255), ACCENT,
            )), 0))
            p.drawText(QRectF(0, 84, w, 34), Qt.AlignmentFlag.AlignCenter,
                       "CHESS VISION")

            # Subtitle
            p.setFont(self._sub_font)
            p.setPen(TEXT_DIM)
            p.drawText(QRectF(0, 118, w, 20), Qt.AlignmentFlag.AlignCenter,
                       "Real-time move intelligence · Stockfish inside")

        if dirty.intersects(footer):
            fy = h - self.FOOTER_H

            sep = QLinearGradient(0, 0, w, 0)
            sep.setColorAt(0, _with_alpha(ACCENT, 0))
            sep.setColorAt(0.5, _with_alpha(ACCENT, 90))
            sep.setColorAt(1, _with_alpha(ACCENT, 0))
            p.setPen(QPen(QBrush(sep), 1))
            p.drawLine(QPointF(30, fy + 6), QPointF(w - 30, fy + 6))

            p.setFont(self._tiny_font)
            p.setPen(QColor(107, 120, 144))
            p.drawText(QRectF(0, fy + 14, w, 14), Qt.AlignmentFlag.AlignCenter,
                       "DEVELOPED & OWNED BY")

            p.setFont(self._brand_font)
            p.setPen(QPen(QBrush(self._shimmer(
                w, t * 0.8, GOLD, QColor(255, 232, 160), GOLD,
            )), 0))
            p.drawText(QRectF(0, fy + 30, w, 22), Qt.AlignmentFlag.AlignCenter,
                       "KRYSKATA-C")

            p.setFont(self._copy_font)
            p.setPen(QColor(85, 97, 122))
            p.drawText(QRectF(0, fy + 56, w, 14), Qt.AlignmentFlag.AlignCenter,
                       "© 2026 Kryskata-C · All rights reserved")

        p.end()

    @staticmethod
    def _shimmer(width: int, t: float, base: QColor, hi: QColor,
                 tint: QColor) -> QLinearGradient:
        """Horizontal gradient with a highlight band sweeping across."""
        grad = QLinearGradient(0, 0, width, 0)
        pos = (t * 0.35) % 1.4 - 0.2
        grad.setColorAt(0, base)
        for offset, color in ((-0.13, base), (0.0, hi), (0.06, tint),
                              (0.13, base)):
            sp = pos + offset
            if 0.0 < sp < 1.0:
                grad.setColorAt(sp, color)
        grad.setColorAt(1, base)
        return grad

    # --- window interactions ---

    def hideEvent(self, event):
        super().hideEvent(event)
        self._deco_timer.stop()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
