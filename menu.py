"""Animated startup menu for Chess Vision.

Developed & owned by Kryskata-C.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QPointF, QRectF, pyqtSignal, pyqtProperty,
    QPropertyAnimation, QEasingCurve,
)
from PyQt6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPainterPath,
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
    """Segmented White/Black picker with a sliding indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = "w"
        self._sel = 0.0  # 0 = white side, 1 = black side
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
        color = "w" if event.position().x() < self.width() / 2 else "b"
        if color == self._color:
            return
        self._color = color
        self._anim.stop()
        self._anim.setStartValue(self._sel)
        self._anim.setEndValue(0.0 if color == "w" else 1.0)
        self._anim.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        p.setBrush(QColor(22, 29, 43))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 12, 12)

        # Sliding indicator
        half = (w - 8) / 2
        ix = 4 + self._sel * half
        grad = QLinearGradient(ix, 0, ix + half, 0)
        grad.setColorAt(0, ACCENT_DEEP)
        grad.setColorAt(1, ACCENT)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(ix, 4, half, h - 8), 9, 9)

        # Labels
        font = QFont("Helvetica Neue", 13, QFont.Weight.DemiBold)
        p.setFont(font)
        white_on = 1.0 - self._sel
        p.setPen(_lerp_color(TEXT_DIM, QColor(255, 255, 255), white_on))
        p.drawText(QRectF(0, 0, w / 2, h), Qt.AlignmentFlag.AlignCenter,
                   "♔  White")
        p.setPen(_lerp_color(TEXT_DIM, QColor(255, 255, 255), self._sel))
        p.drawText(QRectF(w / 2, 0, w / 2, h), Qt.AlignmentFlag.AlignCenter,
                   "♚  Black")
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

    started = pyqtSignal(str, dict)  # (color "w"/"b", visuals dict)

    HEADER_H = 148
    FOOTER_H = 84

    def __init__(self):
        super().__init__()
        self._drag_pos: Optional[QPoint] = None
        self._t0 = time.time()
        self._entrance_done = False

        self.setWindowTitle("Chess Vision")
        self.setFixedSize(400, 676)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Drifting background chess pieces
        self._pieces = [{
            "ch": random.choice("♞♜♝♛♚♟"
                                "♘♖♗♕♔♙"),
            "x": random.random(),
            "y": random.random(),
            "size": random.uniform(13, 30),
            "speed": random.uniform(0.010, 0.032),
            "sway": random.uniform(6, 18),
            "phase": random.uniform(0, math.tau),
            "alpha": random.randint(14, 36),
        } for _ in range(16)]

        # --- interactive content ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, self.HEADER_H, 26, self.FOOTER_H)
        layout.setSpacing(0)

        self.color_select = ColorSelect()
        layout.addWidget(self.color_select)
        layout.addSpacing(14)

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

        # Pulsing glow on the start button
        glow = QGraphicsDropShadowEffect(self.start_btn)
        glow.setOffset(0, 0)
        glow.setColor(_with_alpha(ACCENT, 160))
        glow.setBlurRadius(18)
        self.start_btn.setGraphicsEffect(glow)
        self._glow_anim = QPropertyAnimation(glow, b"blurRadius", self)
        self._glow_anim.setDuration(1200)
        self._glow_anim.setStartValue(14.0)
        self._glow_anim.setKeyValueAt(0.5, 34.0)
        self._glow_anim.setEndValue(14.0)
        self._glow_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._glow_anim.setLoopCount(-1)
        self._glow_anim.start()

        # Decorative animation clock
        self._deco_timer = QTimer(self)
        self._deco_timer.setInterval(33)
        self._deco_timer.timeout.connect(self.update)
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
        self.started.emit(self.color_select.color(), visuals)
        self._deco_timer.stop()
        self._glow_anim.stop()
        self.hide()

    # --- entrance animation ---

    def showEvent(self, event):
        super().showEvent(event)
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

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = time.time() - self._t0
        w, h = self.width(), self.height()

        # Card background with rounded corners
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 18, 18)
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, BG_TOP)
        grad.setColorAt(1, BG_BOTTOM)
        p.fillPath(path, QBrush(grad))
        p.setClipPath(path)

        # Soft accent glow behind the header emblem
        pulse = 0.7 + 0.3 * math.sin(t * 1.6)
        glow = QRadialGradient(QPointF(w / 2, 64), 95)
        glow.setColorAt(0, _with_alpha(ACCENT, int(30 * pulse)))
        glow.setColorAt(1, _with_alpha(ACCENT, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawRect(QRectF(0, 0, w, 170))

        # Drifting chess pieces
        for pc in self._pieces:
            y = ((pc["y"] - t * pc["speed"]) % 1.15) - 0.075
            x = pc["x"] * w + math.sin(t * 0.7 + pc["phase"]) * pc["sway"]
            p.setFont(QFont("Arial", int(pc["size"])))
            p.setPen(QColor(200, 215, 240, pc["alpha"]))
            p.drawText(QPointF(x, y * h), pc["ch"])

        # Knight emblem with glow pulse
        knight_size = 40 + 2 * math.sin(t * 1.6)
        p.setFont(QFont("Arial", int(knight_size)))
        p.setPen(_with_alpha(ACCENT, 80))
        p.drawText(QRectF(0, 14, w, 64), Qt.AlignmentFlag.AlignCenter, "♞")
        p.setPen(QColor(232, 241, 255))
        p.drawText(QRectF(0, 12, w, 64), Qt.AlignmentFlag.AlignCenter, "♞")

        # Title with a moving shimmer
        title_font = QFont("Helvetica Neue", 23, QFont.Weight.Black)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 5)
        p.setFont(title_font)
        p.setPen(QPen(QBrush(self._shimmer(w, t, QColor(214, 226, 243),
                                           QColor(255, 255, 255), ACCENT)), 0))
        p.drawText(QRectF(0, 84, w, 34), Qt.AlignmentFlag.AlignCenter,
                   "CHESS VISION")

        # Subtitle
        sub_font = QFont("Helvetica Neue", 10)
        sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        p.setFont(sub_font)
        p.setPen(TEXT_DIM)
        p.drawText(QRectF(0, 118, w, 20), Qt.AlignmentFlag.AlignCenter,
                   "Real-time move intelligence · Stockfish inside")

        # --- footer ---
        fy = h - self.FOOTER_H

        sep = QLinearGradient(0, 0, w, 0)
        sep.setColorAt(0, _with_alpha(ACCENT, 0))
        sep.setColorAt(0.5, _with_alpha(ACCENT, 90))
        sep.setColorAt(1, _with_alpha(ACCENT, 0))
        p.setPen(QPen(QBrush(sep), 1))
        p.drawLine(QPointF(30, fy + 6), QPointF(w - 30, fy + 6))

        tiny = QFont("Helvetica Neue", 8)
        tiny.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(tiny)
        p.setPen(QColor(107, 120, 144))
        p.drawText(QRectF(0, fy + 14, w, 14), Qt.AlignmentFlag.AlignCenter,
                   "DEVELOPED & OWNED BY")

        brand_font = QFont("Helvetica Neue", 14, QFont.Weight.Black)
        brand_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        p.setFont(brand_font)
        p.setPen(QPen(QBrush(self._shimmer(w, t * 0.8, GOLD,
                                           QColor(255, 232, 160), GOLD)), 0))
        p.drawText(QRectF(0, fy + 30, w, 22), Qt.AlignmentFlag.AlignCenter,
                   "KRYSKATA-C")

        p.setFont(QFont("Helvetica Neue", 8))
        p.setPen(QColor(85, 97, 122))
        p.drawText(QRectF(0, fy + 56, w, 14), Qt.AlignmentFlag.AlignCenter,
                   "© 2026 Kryskata-C · All rights reserved")

        # Card border
        p.setClipping(False)
        p.setPen(QPen(_with_alpha(ACCENT, 70), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 18, 18)
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
