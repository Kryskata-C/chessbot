"""One visual system for every window: the website's palette (warm black,
bone type, one phosphor-green accent) so the app looks like the thing the
customer bought. Frameless, draggable cards with a rounded background."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint, QRectF, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QFont, QCursor
from PyQt6.QtWidgets import QDialog, QLabel, QPushButton, QHBoxLayout, QWidget, QApplication

BG, PANEL, PANEL2, LINE = "#0f0e0c", "#171511", "#1f1c17", "#2a2620"
INK, MUTED, ACC, ACC_INK, RED, AMBER = "#efe9dd", "#9b9384", "#b9f24a", "#0e0d0b", "#ef6b5a", "#f0b64a"
SERIF = "Georgia, Iowan Old Style, Times New Roman"
SANS = "Helvetica Neue, Helvetica, Arial"
MONO = "SF Mono, Menlo, Monaco"

STYLE = f"""
QWidget {{ color: {INK}; font-family: {SANS}; font-size: 13px; background: transparent; }}
QLabel#wordmark {{ font-family: {SERIF}; font-size: 28px; letter-spacing: -0.5px; }}
QLabel#h {{ color: {MUTED}; font-family: {MONO}; font-size: 10px; letter-spacing: 2px; }}
QLabel#dim {{ color: {MUTED}; font-size: 12px; }}
QLabel#err {{ color: {RED}; font-size: 12px; }}
QLabel#ok {{ color: {ACC}; font-size: 12px; }}
QLabel#big {{ font-family: {MONO}; font-size: 40px; font-weight: 500; }}
QLineEdit {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 10px; padding: 11px 13px;
             font-size: 14px; selection-background-color: {ACC}; selection-color: {ACC_INK}; }}
QLineEdit:focus {{ border-color: {ACC}; }}
QPushButton {{ background: {ACC}; color: {ACC_INK}; border: none; border-radius: 12px; padding: 13px 18px;
               font-weight: 700; font-size: 14px; }}
QPushButton:hover {{ background: #c9ff5c; }}
QPushButton:pressed {{ background: #a6dd3c; }}
QPushButton:disabled {{ background: {PANEL2}; color: {MUTED}; }}
QPushButton#ghost {{ background: transparent; color: {INK}; border: 1px solid {LINE}; font-weight: 500; }}
QPushButton#ghost:hover {{ border-color: {MUTED}; }}
QPushButton#link {{ background: transparent; color: {MUTED}; border: none; padding: 6px; font-weight: 500; font-size: 12px; }}
QPushButton#link:hover {{ color: {INK}; }}
QPushButton#close {{ background: transparent; color: {MUTED}; border: none; font-size: 16px; padding: 2px 8px; font-weight: 400; }}
QPushButton#close:hover {{ color: {INK}; }}
QPushButton#tile {{ background: {PANEL2}; color: {MUTED}; border: 1px solid {LINE}; border-radius: 10px;
                    padding: 12px 6px; font-weight: 600; font-size: 13px; }}
QPushButton#tile:hover {{ color: {INK}; }}
QPushButton#tile:checked {{ color: {ACC}; border-color: {ACC}; background: #1b2012; }}
QPushButton#pill {{ background: {PANEL2}; color: {MUTED}; border: 1px solid {LINE}; border-radius: 15px;
                    padding: 6px 11px; font-weight: 500; font-size: 12px; }}
QPushButton#pill:checked {{ color: {ACC_INK}; background: {ACC}; border-color: {ACC}; }}
QCheckBox {{ color: {MUTED}; font-size: 12px; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {LINE}; background: {PANEL2}; }}
QCheckBox::indicator:checked {{ background: {ACC}; border-color: {ACC}; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACC}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; background: {INK}; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QTableWidget {{ background: {PANEL}; alternate-background-color: {PANEL2}; gridline-color: {LINE};
                border: 1px solid {LINE}; border-radius: 10px; font-size: 12.5px; }}
QTableWidget::item {{ padding: 6px; border: none; }}
QTableWidget::item:selected {{ background: #1b2012; color: {INK}; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; border-bottom: 1px solid {LINE};
                        padding: 8px 6px; font-family: {MONO}; font-size: 10px; letter-spacing: 1px; }}
QComboBox {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 8px; padding: 5px 10px; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; selection-background-color: {ACC}; selection-color: {ACC_INK}; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 30px; }}
QToolTip {{ background: {PANEL2}; color: {INK}; border: 1px solid {LINE}; }}
"""


class Card(QDialog):
    """Frameless, always-on-top, draggable rounded card."""

    def __init__(self, width: int, closable: bool = True, on_close=None):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(STYLE)
        self.setFixedWidth(width)
        self._drag: QPoint | None = None
        self._on_close = on_close
        self._closable = closable

    def header(self, title: str = "Chess Vision") -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        mark = QLabel(title); mark.setObjectName("wordmark")
        h.addWidget(mark); h.addStretch()
        if self._closable:
            x = QPushButton("×"); x.setObjectName("close"); x.setCursor(Qt.CursorShape.PointingHandCursor)
            x.clicked.connect(self._close_clicked); h.addWidget(x)
        return w

    def _close_clicked(self):
        if self._on_close:
            self._on_close()
        else:
            self.close()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath(); path.addRoundedRect(r, 18, 18)
        p.fillPath(path, QColor(BG))
        p.setPen(QPen(QColor(255, 255, 255, 22), 1)); p.drawPath(path)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def center_on_screen(self):
        """Centre on the display the cursor is on. A move() is only honoured
        once macOS has mapped the window, and moving too early lands the
        card off-screen on multi-display setups, so place, check, retry."""
        # The display under the cursor: that is where the user is working
        # (chess.com on the external display), and join_all_spaces() lets
        # the card sit over a fullscreen app there. Primary as a fallback.
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        self.adjustSize()
        target = QPoint(g.center().x() - self.width() // 2,
                        g.center().y() - self.height() // 2)
        self._place(target, 8)

    def join_all_spaces(self):
        """Show on every Space, including other apps' fullscreen Spaces
        (canJoinAllSpaces | fullScreenAuxiliary), at floating level. Same
        Cocoa trick the overlay uses; without it the card hides behind
        whatever is fullscreen on that display."""
        if QApplication.platformName() != "cocoa":
            return  # offscreen/xcb: winId() is not an NSView, objc calls would crash
        try:
            import ctypes, ctypes.util
            lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            lib.sel_registerName.restype = ctypes.c_void_p
            lib.sel_registerName.argtypes = [ctypes.c_char_p]
            send = ctypes.cast(lib.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))
            send_long = ctypes.cast(lib.objc_msgSend, ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long))
            nswindow = send(int(self.winId()), lib.sel_registerName(b"window"))
            if not nswindow:
                return
            send_long(nswindow, lib.sel_registerName(b"setCollectionBehavior:"), (1 << 0) | (1 << 8))
            send_long(nswindow, lib.sel_registerName(b"setLevel:"), 3)  # NSFloatingWindowLevel
            send(nswindow, lib.sel_registerName(b"orderFrontRegardless"))
        except Exception as e:
            print(f"card space pinning warning: {e}")

    def showEvent(self, e):
        super().showEvent(e)
        self.center_on_screen()
        # Re-place once the window can join fullscreen Spaces: a move onto
        # such a display before that is bounced back to the primary one.
        QTimer.singleShot(60, lambda: (self.join_all_spaces(), self.center_on_screen()))

    def _place(self, target: QPoint, tries: int):
        self.move(target)
        if tries <= 0:
            return
        QTimer.singleShot(120, lambda: (
            None if self.geometry().topLeft() == target else self._place(target, tries - 1)))


def section(text: str) -> QLabel:
    l = QLabel(text.upper()); l.setObjectName("h"); return l
