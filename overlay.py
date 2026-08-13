"""PyQt6 transparent click-through overlay window."""

from __future__ import annotations

import sys
import math
import time
import ctypes
import ctypes.util
from typing import Optional
from PyQt6.QtCore import Qt, QRect, QRectF, QPointF, QTimer, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt6.QtWidgets import QWidget, QApplication


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


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


# Visual palette
ARROW_GREEN = QColor(60, 200, 90)
ARROW_RED = QColor(235, 70, 60)
ARROW_GOLD = QColor(240, 190, 60)
REPLY_ORANGE = QColor(255, 130, 45)
CANDIDATE_BLUE = QColor(120, 180, 255)
THREAT_RED = QColor(255, 65, 55)
TRAIL_ORANGE = QColor(255, 150, 60)

# Animation timing
ARROW_DRAW_S = 0.26      # main arrow draw-on duration
REPLY_DELAY_S = 0.28     # pause before enemy reply arrow appears
PLY_STAGGER_S = 0.30     # stagger between successive PV plies
GHOST_PERIOD_S = 1.7     # ghost piece slide loop length (speed knob)
TRAIL_DURATION_S = 2.6   # enemy move trail fade-out


class OverlayWindow(QWidget):
    """Transparent always-on-top overlay that draws animated move visuals."""

    def __init__(self):
        super().__init__()
        self.status_text: str = ""
        self.status_color: QColor = QColor(0, 120, 255)  # blue default
        self._status_timer: Optional[QTimer] = None
        self._native_setup_done = False

        # Which visual effects are enabled (menu toggles override these)
        self.visuals: dict = {
            "arrow": True, "ghost": True, "reply": True, "pv": True,
            "candidates": True, "threats": True, "trail": True,
            "evalbar": True,
        }

        # Board geometry in screen coords: {x, y, sq, wob}
        self._board_geo: Optional[dict] = None
        # Current suggestion being visualized
        self._suggestion: Optional[dict] = None
        # Player pieces in danger (square names like "e4")
        self._threats: list[str] = []
        # Opponent's last move trail: {move, t0}
        self._trail: Optional[dict] = None
        # Eval bar state (white's winning fraction, 0..1)
        self._eval_target: float = 0.5
        self._eval_shown: float = 0.5
        self._eval_cp: int = 0
        self._has_eval: bool = False

        self._anim = QTimer(self)
        self._anim.setInterval(33)
        self._anim.timeout.connect(self._tick)

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_visual_config(self, visuals: dict):
        """Enable/disable individual visual effects (from menu toggles)."""
        self.visuals.update(visuals)

    def set_board_geometry(self, board: dict, white_on_bottom: bool):
        """Cache detected board position so visuals can map squares to pixels."""
        self._board_geo = {
            "x": float(board["x"]),
            "y": float(board["y"]),
            "sq": float(board["square_size"]),
            "wob": bool(white_on_bottom),
        }

    def show_suggestion(
        self,
        move_uci: str,
        piece_symbol: str | None = None,
        crit: float = 0.0,
        is_capture: bool = False,
        is_check: bool = False,
        pv: list[str] | None = None,
        candidates: list[str] | None = None,
    ):
        """Visualize a suggested move with animated arrow + ghost piece.

        Args:
            move_uci: The suggested move, e.g. "e2e4".
            piece_symbol: FEN symbol of the moving piece ("N", "p", ...).
            crit: 0-1 criticality; shifts arrow color green -> red.
            is_capture: Draw a pulsing capture ring on the target square.
            is_check: Style the arrow gold (checking move).
            pv: Predicted continuation after the move (enemy reply first).
            candidates: Alternative good moves to ghost in faintly.
        """
        self._suggestion = {
            "move": move_uci,
            "piece": piece_symbol,
            "crit": max(0.0, min(1.0, float(crit))),
            "capture": is_capture,
            "check": is_check,
            "pv": list(pv or []),
            "candidates": list(candidates or []),
            "t0": time.time(),
        }
        self._ensure_anim()
        self.update()

    def clear_highlights(self):
        self._suggestion = None
        self.update()

    def set_threats(self, squares: list[str]):
        """Squares (e.g. ["c3", "f7"]) holding player pieces in danger."""
        self._threats = list(squares)
        if self._threats:
            self._ensure_anim()
        self.update()

    def flash_enemy_move(self, move_uci: str):
        """Show a fading trail on the opponent's last move."""
        self._trail = {"move": move_uci, "t0": time.time()}
        self._ensure_anim()
        self.update()

    def set_eval(self, cp_white_pov: int):
        """Update the eval bar target (centipawns from White's POV)."""
        self._eval_cp = cp_white_pov
        # Map centipawns to a win fraction with a logistic curve
        x = max(-2000, min(2000, cp_white_pov))
        self._eval_target = 1.0 / (1.0 + math.exp(-x / 280.0))
        if abs(cp_white_pov) >= 90000:
            self._eval_target = 0.995 if cp_white_pov > 0 else 0.005
        self._has_eval = True
        self._ensure_anim()
        self.update()

    def reset_board_visuals(self):
        """Clear everything tied to the board (lost board / new game)."""
        self._suggestion = None
        self._threats = []
        self._trail = None
        self._has_eval = False
        self.update()

    # ------------------------------------------------------------------
    # Animation clock
    # ------------------------------------------------------------------

    def _ensure_anim(self):
        if not self._anim.isActive():
            self._anim.start()

    def _anim_needed(self) -> bool:
        if self._suggestion is not None or self._trail is not None:
            return True
        if self._threats and self.visuals.get("threats", True):
            return True
        if self._has_eval and abs(self._eval_shown - self._eval_target) > 0.002:
            return True
        return False

    def _tick(self):
        if self._trail and time.time() - self._trail["t0"] > TRAIL_DURATION_S:
            self._trail = None
        if not self._anim_needed():
            self._anim.stop()
        self.update()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _square_rect(self, name: str) -> Optional[QRectF]:
        """Screen rect of a square by name ("e4"), or None if no board."""
        geo = self._board_geo
        if geo is None or len(name) < 2:
            return None
        file = ord(name[0]) - ord("a")
        rank = int(name[1]) - 1
        if not (0 <= file <= 7 and 0 <= rank <= 7):
            return None
        if geo["wob"]:
            col, row = file, 7 - rank
        else:
            col, row = 7 - file, rank
        sq = geo["sq"]
        return QRectF(geo["x"] + col * sq, geo["y"] + row * sq, sq, sq)

    def _move_rects(self, uci: str) -> tuple[Optional[QRectF], Optional[QRectF]]:
        return self._square_rect(uci[0:2]), self._square_rect(uci[2:4])

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        now = time.time()

        # Smoothly chase the eval target
        self._eval_shown += (self._eval_target - self._eval_shown) * 0.15

        self._draw_status(painter)

        if self._board_geo is not None:
            if self._has_eval and self.visuals.get("evalbar", True):
                self._draw_eval_bar(painter)
            if self._threats and self.visuals.get("threats", True):
                self._draw_threats(painter, now)
            if self._trail is not None and self.visuals.get("trail", True):
                self._draw_trail(painter, now)
            if self._suggestion is not None:
                self._draw_suggestion(painter, now)

        painter.end()

    def _draw_status(self, painter: QPainter):
        if not self.status_text:
            return
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

        bg = QColor(self.status_color)
        bg.setAlpha(220)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(banner_x, banner_y, banner_w, banner_h, 12, 12)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            QRect(banner_x, banner_y, banner_w, banner_h),
            Qt.AlignmentFlag.AlignCenter,
            self.status_text,
        )

    def _draw_arrow(
        self,
        painter: QPainter,
        a: QPointF,
        b: QPointF,
        frac: float,
        color: QColor,
        width: float,
        alpha: int = 230,
        dashed: bool = False,
        glow: bool = False,
    ):
        """Draw an arrow from a to b, drawn-on up to `frac` of its length."""
        if frac <= 0.02:
            return
        dx, dy = b.x() - a.x(), b.y() - a.y()
        dist = math.hypot(dx, dy)
        if dist < 2:
            return
        ux, uy = dx / dist, dy / dist
        tip = QPointF(a.x() + dx * frac, a.y() + dy * frac)
        head = max(width * 2.2, 10.0)
        base = QPointF(tip.x() - ux * head, tip.y() - uy * head)

        c = QColor(color)
        c.setAlpha(alpha)

        if glow:
            gc = QColor(color)
            gc.setAlpha(int(alpha * 0.25))
            pen = QPen(gc, width * 2.1)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(a, base)

        pen = QPen(c, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(a, base)

        # Arrowhead
        px, py = -uy, ux
        hw = head * 0.6
        poly = QPolygonF([
            tip,
            QPointF(base.x() + px * hw, base.y() + py * hw),
            QPointF(base.x() - px * hw, base.y() - py * hw),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(c)
        painter.drawPolygon(poly)

    def _draw_suggestion(self, painter: QPainter, now: float):
        s = self._suggestion
        fr, to = self._move_rects(s["move"])
        if fr is None or to is None:
            return
        geo = self._board_geo
        sq = geo["sq"]
        t = now - s["t0"]
        a, b = fr.center(), to.center()

        crit = s["crit"]
        color = _lerp_color(ARROW_GREEN, ARROW_RED, crit)
        if s["check"]:
            color = ARROW_GOLD

        # Faint candidate arrows behind everything else
        if self.visuals.get("candidates", True):
            for cand in s["candidates"]:
                cf, ct = self._move_rects(cand)
                if cf is not None and ct is not None:
                    self._draw_arrow(
                        painter, cf.center(), ct.center(), 1.0,
                        CANDIDATE_BLUE, sq * 0.07, alpha=70,
                    )

        if self.visuals.get("arrow", True):
            prog = _ease_out_cubic(min(1.0, t / ARROW_DRAW_S))
            pulse = 1.0 + 0.14 * crit * math.sin(now * 2 * math.pi / 0.9)
            self._draw_arrow(
                painter, a, b, prog, color, sq * 0.15 * pulse,
                alpha=235, glow=True,
            )
        else:
            # Classic fallback: colored from/to squares
            painter.setBrush(QColor(255, 0, 0, 80))
            painter.setPen(QPen(QColor(255, 0, 0, 200), 3))
            painter.drawRect(fr)
            painter.setBrush(QColor(0, 255, 0, 80))
            painter.setPen(QPen(QColor(0, 255, 0, 200), 3))
            painter.drawRect(to)

        # Pulsing capture ring on the target square
        if s["capture"]:
            k = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 0.8)
            rad = sq * (0.40 + 0.05 * k)
            pen = QPen(QColor(
                REPLY_ORANGE.red(), REPLY_ORANGE.green(), REPLY_ORANGE.blue(),
                int(150 + 80 * k),
            ), 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(b, rad, rad)

        # Predicted continuation: enemy reply (dashed) + our follow-up
        pv = s["pv"]
        if pv and (self.visuals.get("reply", True) or self.visuals.get("pv", True)):
            t_reply = t - (ARROW_DRAW_S + REPLY_DELAY_S)
            if t_reply > 0:
                rf, rt = self._move_rects(pv[0])
                if rf is not None and rt is not None:
                    prog = _ease_out_cubic(min(1.0, t_reply / 0.25))
                    self._draw_arrow(
                        painter, rf.center(), rt.center(), prog,
                        REPLY_ORANGE, sq * 0.11, alpha=200, dashed=True,
                    )
            if len(pv) > 1 and self.visuals.get("pv", True):
                t3 = t - (ARROW_DRAW_S + REPLY_DELAY_S + PLY_STAGGER_S)
                if t3 > 0:
                    ff, ft = self._move_rects(pv[1])
                    if ff is not None and ft is not None:
                        prog = _ease_out_cubic(min(1.0, t3 / 0.25))
                        self._draw_arrow(
                            painter, ff.center(), ft.center(), prog,
                            ARROW_GREEN, sq * 0.09, alpha=110,
                        )

        # Ghost piece gliding along the move
        if self.visuals.get("ghost", True) and s["piece"]:
            self._draw_ghost(painter, s, a, b, sq, t)

    def _draw_ghost(self, painter: QPainter, s: dict, a: QPointF, b: QPointF,
                    sq: float, t: float):
        if t < ARROW_DRAW_S:
            return  # let the arrow draw on first
        sym = PIECE_UNICODE.get(s["piece"])
        if sym is None:
            return

        ct = ((t - ARROW_DRAW_S) % GHOST_PERIOD_S) / GHOST_PERIOD_S
        if ct < 0.62:
            k = _ease_in_out(ct / 0.62)
            alpha = 200.0
        elif ct < 0.85:
            k = 1.0
            alpha = 200.0
        else:
            k = 1.0
            alpha = 200.0 * (1.0 - (ct - 0.85) / 0.15)
        if ct < 0.08:
            alpha = min(alpha, 200.0 * ct / 0.08)
        alpha = int(max(0.0, alpha))

        pos = QPointF(a.x() + (b.x() - a.x()) * k, a.y() + (b.y() - a.y()) * k)
        rect = QRectF(pos.x() - sq / 2, pos.y() - sq / 2, sq, sq)

        font = QFont("Arial", int(sq * 0.72))
        painter.setFont(font)

        is_white = s["piece"].isupper()
        fill = QColor(250, 250, 250, alpha) if is_white else QColor(30, 30, 30, alpha)
        outline = QColor(20, 20, 20, int(alpha * 0.8)) if is_white \
            else QColor(240, 240, 240, int(alpha * 0.8))

        painter.setPen(outline)
        for ox, oy in ((-1.5, 0), (1.5, 0), (0, -1.5), (0, 1.5)):
            painter.drawText(
                rect.translated(ox, oy), Qt.AlignmentFlag.AlignCenter, sym
            )
        painter.setPen(fill)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, sym)

    def _draw_threats(self, painter: QPainter, now: float):
        for i, name in enumerate(self._threats):
            r = self._square_rect(name)
            if r is None:
                continue
            k = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 1.25 + i * 0.9)
            fill = QColor(
                THREAT_RED.red(), THREAT_RED.green(), THREAT_RED.blue(),
                int(28 + 52 * k),
            )
            pen = QPen(QColor(
                THREAT_RED.red(), THREAT_RED.green(), THREAT_RED.blue(),
                int(120 + 90 * k),
            ), 2.5)
            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawRoundedRect(r.adjusted(2, 2, -2, -2), 6, 6)

    def _draw_trail(self, painter: QPainter, now: float):
        tr = self._trail
        age = now - tr["t0"]
        if age >= TRAIL_DURATION_S:
            return
        k = (1.0 - age / TRAIL_DURATION_S) ** 1.5
        fr, to = self._move_rects(tr["move"])
        if fr is None or to is None:
            return

        fill = QColor(
            TRAIL_ORANGE.red(), TRAIL_ORANGE.green(), TRAIL_ORANGE.blue(),
            int(85 * k),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(fr.adjusted(2, 2, -2, -2), 6, 6)
        painter.drawRoundedRect(to.adjusted(2, 2, -2, -2), 6, 6)

        pen = QPen(QColor(
            TRAIL_ORANGE.red(), TRAIL_ORANGE.green(), TRAIL_ORANGE.blue(),
            int(140 * k),
        ), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(fr.center(), to.center())

    def _draw_eval_bar(self, painter: QPainter):
        geo = self._board_geo
        sq = geo["sq"]
        h = sq * 8
        bw = 9.0
        x = geo["x"] - 16
        if x < 4:
            x = geo["x"] + sq * 8 + 7
        y = geo["y"]

        painter.setPen(QPen(QColor(0, 0, 0, 140), 1))
        painter.setBrush(QColor(24, 26, 30, 215))
        painter.drawRoundedRect(QRectF(x, y, bw, h), 4, 4)

        frac = max(0.02, min(0.98, self._eval_shown))
        wh = h * frac
        if geo["wob"]:
            white_rect = QRectF(x, y + h - wh, bw, wh)
        else:
            white_rect = QRectF(x, y, bw, wh)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(245, 245, 245, 230))
        painter.drawRoundedRect(white_rect, 3, 3)

        # Midline marker
        my = y + h / 2
        painter.setPen(QPen(QColor(150, 150, 150, 120), 1))
        painter.drawLine(QPointF(x, my), QPointF(x + bw, my))

        # Eval text chip above the bar
        cp = self._eval_cp
        if abs(cp) >= 90000:
            text = "M+" if cp > 0 else "M-"
        else:
            text = f"{cp / 100:+.1f}"
        font = QFont("Helvetica Neue", 9, QFont.Weight.Bold)
        painter.setFont(font)
        chip = QRectF(x - 15, y - 22, 40, 17)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(24, 26, 30, 210))
        painter.drawRoundedRect(chip, 5, 5)
        painter.setPen(QColor(235, 235, 235))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
