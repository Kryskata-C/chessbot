"""PyQt6 transparent click-through overlay window."""

from __future__ import annotations

import sys
import ctypes
import ctypes.util
from typing import Optional
from PyQt6.QtCore import Qt, QRect, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QWidget, QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel,
)


class MenuWindow(QDialog):
    """Startup menu for selecting player color."""

    color_selected = pyqtSignal(str)  # emits "w" or "b"

    def __init__(self):
        super().__init__()
        self._drag_pos: Optional[QPoint] = None
        self.setWindowTitle("Chess Vision")
        self.setFixedSize(260, 160)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )

        # Dark theme stylesheet
        self.setStyleSheet("""
            QDialog {
                background: #1e1e1e;
                border: 1px solid #444;
                border-radius: 10px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 16px;
                font-weight: bold;
            }
            QComboBox {
                background: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 6px 12px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #2d2d2d;
                color: #e0e0e0;
                selection-background-color: #3a7bd5;
            }
            QPushButton {
                background: #3a7bd5;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a8be5;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Chess Vision")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        row = QHBoxLayout()
        label = QLabel("Play as:")
        label.setStyleSheet("font-size: 14px; font-weight: normal;")
        self.combo = QComboBox()
        self.combo.addItems(["White", "Black"])
        row.addWidget(label)
        row.addWidget(self.combo)
        layout.addLayout(row)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)

        # Center on screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )

    def _on_start(self):
        color = "w" if self.combo.currentIndex() == 0 else "b"
        self.color_selected.emit(color)
        self.hide()

    # --- draggable window ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


PIECE_UNICODE = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f",
}


class DebugBoardWindow(QWidget):
    """Small always-on-top window showing what pieces the scanner detects."""

    SQUARE_PX = 40  # size of each square in the debug board

    def __init__(self):
        super().__init__()
        self._drag_pos: Optional[QPoint] = None
        self.positions: list[list[str | None]] = [[None] * 8 for _ in range(8)]
        self.white_on_bottom: bool = True
        self.turn: str = "w"
        self.piece_count: int = 0
        self.estimated_elo: int | None = None
        self.opponent_acpl: float | None = None
        self.bot_accuracy: float | None = None
        self.bot_cpl: float | None = None

        size = self.SQUARE_PX * 8 + 40  # board + margins for labels
        self.setFixedSize(size, size + 68)  # extra space for info text + ELO + accuracy
        self.setWindowTitle("Debug Board")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setStyleSheet("background: #1e1e1e;")

        # Position in top-right area of screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.move(geo.width() - self.width() - 20, 80)

    def set_positions(self, positions: list[list[str | None]],
                      white_on_bottom: bool, turn: str, piece_count: int,
                      estimated_elo: int | None = None,
                      opponent_acpl: float | None = None,
                      bot_accuracy: float | None = None,
                      bot_cpl: float | None = None):
        self.positions = positions
        self.white_on_bottom = white_on_bottom
        self.turn = turn
        self.piece_count = piece_count
        self.estimated_elo = estimated_elo
        self.opponent_acpl = opponent_acpl
        self.bot_accuracy = bot_accuracy
        self.bot_cpl = bot_cpl
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        sq = self.SQUARE_PX
        margin = 20  # left/top margin for rank/file labels

        light = QColor(238, 238, 210)  # chess.com beige
        dark = QColor(118, 150, 86)    # chess.com green

        ranks = "87654321" if self.white_on_bottom else "12345678"
        files = "abcdefgh" if self.white_on_bottom else "hgfedcba"

        # Draw rank labels
        label_font = QFont("Helvetica Neue", 10)
        painter.setFont(label_font)
        painter.setPen(QColor(160, 160, 160))
        for i in range(8):
            y = margin + i * sq + sq // 2 + 5
            painter.drawText(2, y, ranks[i])

        # Draw file labels
        for i in range(8):
            x = margin + i * sq + sq // 2 - 4
            painter.drawText(x, margin + 8 * sq + 15, files[i])

        # Draw board squares and pieces
        piece_font = QFont("Arial", sq - 12)
        for row in range(8):
            for col in range(8):
                x = margin + col * sq
                y = margin + row * sq

                # Square color
                is_light = (row + col) % 2 == 0
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(light if is_light else dark)
                painter.drawRect(x, y, sq, sq)

                # Piece
                p = self.positions[row][col]
                if p:
                    painter.setFont(piece_font)
                    sym = PIECE_UNICODE.get(p, "?")
                    painter.setPen(QColor(0, 0, 0) if p.isupper() else QColor(40, 40, 40))
                    painter.drawText(
                        QRect(x, y, sq, sq),
                        Qt.AlignmentFlag.AlignCenter,
                        sym,
                    )

        # Info bar at bottom
        info_font = QFont("Helvetica Neue", 10, QFont.Weight.Bold)
        painter.setFont(info_font)
        turn_name = "White" if self.turn == "w" else "Black"
        info = f"{self.piece_count} pieces | {turn_name} to move"
        painter.setPen(QColor(200, 200, 200))
        info_y = margin + 8 * sq + 24
        painter.drawText(margin, info_y, info)

        # ELO estimate line
        elo_font = QFont("Helvetica Neue", 9)
        painter.setFont(elo_font)
        painter.setPen(QColor(230, 168, 23))  # gold/amber
        elo_y = info_y + 18
        if self.estimated_elo is not None:
            acpl_str = f" (ACPL: {self.opponent_acpl:.0f})" if self.opponent_acpl is not None else ""
            elo_text = f"Est. opponent ELO: ~{self.estimated_elo}{acpl_str}"
        else:
            elo_text = "Est. ELO: analyzing..."
        painter.drawText(margin, elo_y, elo_text)

        # Bot accuracy line
        acc_y = elo_y + 16
        painter.setPen(QColor(120, 180, 230))  # soft blue
        if self.bot_accuracy is not None:
            cpl_str = f" | CPL: {self.bot_cpl:.0f}" if self.bot_cpl is not None else ""
            acc_text = f"Bot accuracy: {self.bot_accuracy:.0f}%{cpl_str}"
        else:
            acc_text = "Bot accuracy: --"
        painter.drawText(margin, acc_y, acc_text)

        painter.end()

    # --- draggable ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


class OverlayWindow(QWidget):
    """Transparent always-on-top overlay that draws move highlights."""

    def __init__(self):
        super().__init__()
        self.highlights: list[dict] = []  # list of {x, y, w, h}
        self.status_text: str = ""
        self.status_color: QColor = QColor(0, 120, 255)  # blue default
        self._status_timer: Optional[QTimer] = None
        self._native_setup_done = False

        # Frameless, always-on-top, transparent, click-through
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)

        # Cover entire screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.setGeometry(geo)

    def showEvent(self, event):
        """After the window is shown, pin it at the OS level."""
        super().showEvent(event)
        if not self._native_setup_done:
            self._native_setup_done = True
            # Delay slightly so the NSWindow is fully created
            QTimer.singleShot(100, self._setup_macos_overlay)

    def _setup_macos_overlay(self):
        """Use Cocoa APIs to make the overlay truly pinned and invisible to clicks."""
        try:
            lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))

            lib.sel_registerName.restype = ctypes.c_void_p
            lib.sel_registerName.argtypes = [ctypes.c_char_p]

            # Typed wrappers for objc_msgSend (required for arm64 ABI)
            send = ctypes.cast(
                lib.objc_msgSend,
                ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
            )
            send_long = ctypes.cast(
                lib.objc_msgSend,
                ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long),
            )
            send_bool = ctypes.cast(
                lib.objc_msgSend,
                ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool),
            )

            nsview = int(self.winId())
            nswindow = send(nsview, lib.sel_registerName(b"window"))
            if not nswindow:
                return

            # Window level above everything (NSScreenSaverWindowLevel = 1000)
            send_long(nswindow, lib.sel_registerName(b"setLevel:"), 1000)

            # Truly ignore all mouse events at the OS level
            send_bool(nswindow, lib.sel_registerName(b"setIgnoresMouseEvents:"), True)

            # Show on all desktops/spaces and stay visible during Expose
            # canJoinAllSpaces (1<<0) | stationary (1<<4) | fullScreenAuxiliary (1<<8)
            send_long(
                nswindow,
                lib.sel_registerName(b"setCollectionBehavior:"),
                (1 << 0) | (1 << 4) | (1 << 8),
            )

            print("macOS overlay: pinned above all windows, click-through enabled")
        except Exception as e:
            print(f"macOS overlay setup warning: {e}")

    def set_status(self, text: str, color: Optional[QColor] = None, duration_ms: int = 0):
        """Show a status banner at the top of the screen.

        Args:
            text: Status message to display.
            color: Background color for the banner.
            duration_ms: If > 0, auto-clear after this many ms.
        """
        self.status_text = text
        if color:
            self.status_color = color
        self.update()

        # Cancel any previous auto-clear timer
        if self._status_timer:
            self._status_timer.stop()
            self._status_timer = None

        if duration_ms > 0:
            self._status_timer = QTimer()
            self._status_timer.setSingleShot(True)
            self._status_timer.timeout.connect(self._clear_status)
            self._status_timer.start(duration_ms)

    def _clear_status(self):
        self.status_text = ""
        self.update()

    def set_highlights(self, rects: list[dict]):
        """Update the highlighted squares and repaint.

        Args:
            rects: List of dicts with keys x, y, w, h (screen coordinates).
        """
        # Clamp rectangles to screen bounds so edge squares aren't clipped
        geo = self.geometry()
        clamped = []
        for r in rects:
            x = max(0, r["x"])
            y = max(0, r["y"])
            w = min(r["w"], geo.width() - x)
            h = min(r["h"], geo.height() - y)
            clamped.append({"x": x, "y": y, "w": w, "h": h})
        self.highlights = clamped
        self.update()

    def clear_highlights(self):
        self.highlights = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw status banner at top of screen
        if self.status_text:
            font = QFont("Helvetica Neue", 16, QFont.Weight.Bold)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(self.status_text)
            text_height = metrics.height()

            pad_x, pad_y = 24, 12
            banner_w = text_width + pad_x * 2
            banner_h = text_height + pad_y * 2
            banner_x = (self.width() - banner_w) // 2
            banner_y = 30

            # Background pill
            bg = QColor(self.status_color)
            bg.setAlpha(220)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(banner_x, banner_y, banner_w, banner_h, 12, 12)

            # Text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                QRect(banner_x, banner_y, banner_w, banner_h),
                Qt.AlignmentFlag.AlignCenter,
                self.status_text,
            )

        # Draw square highlights
        for i, rect in enumerate(self.highlights):
            # "From" square: red, "To" square: green
            if i == 0:
                painter.setBrush(QColor(255, 0, 0, 80))
                painter.setPen(QPen(QColor(255, 0, 0, 200), 3))
            else:
                painter.setBrush(QColor(0, 255, 0, 80))
                painter.setPen(QPen(QColor(0, 255, 0, 200), 3))

            r = QRect(rect["x"], rect["y"], rect["w"], rect["h"])
            painter.drawRect(r)

        painter.end()
