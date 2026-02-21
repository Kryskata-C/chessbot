"""Chess Vision - Screen Chess Analyzer with Overlay.

Captures the screen, detects a chess.com board, recognizes pieces,
runs Stockfish for the best move, and highlights it on a transparent overlay.
"""

from __future__ import annotations

import sys
import signal

import chess

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QShortcut, QKeySequence, QColor
from PyQt6.QtWidgets import QApplication

from capture import capture_screen
from board_detector import detect_board
from piece_recognizer import (
    recognize_board,
    detect_orientation,
    positions_to_fen,
    get_templates,
    reload_templates,
)
from engine import ChessEngine
from overlay import OverlayWindow, MenuWindow

SCAN_INTERVAL_MS = 200

# Status colors
BLUE = QColor(0, 120, 255)
GREEN = QColor(30, 180, 60)
ORANGE = QColor(230, 140, 0)
RED = QColor(220, 40, 40)


class ChessVision:
    def __init__(self):
        self.engine = ChessEngine(depth=12, threads=2)
        self.overlay = OverlayWindow()
        self.menu = MenuWindow()
        self.last_fen_position: str | None = None
        self.last_move: str | None = None
        self.player_color: str | None = None  # "w" or "b"
        self.current_turn: str = "w"  # white always moves first
        self.running = True
        self.has_templates = len(get_templates()) > 0
        self.scan_timer: QTimer | None = None
        # Position stability: require same FEN for 2 scans before accepting
        self._pending_fen: str | None = None
        self._pending_count: int = 0
        self._STABLE_SCANS = 2

        # Wire up menu → start
        self.menu.color_selected.connect(self._on_color_selected)

    def _on_color_selected(self, color: str):
        """Called when the user picks a color and clicks Start."""
        self.player_color = color
        self.current_turn = "w"  # white always moves first
        color_name = "White" if color == "w" else "Black"
        print(f"Playing as: {color_name}")

        if self.has_templates:
            self.overlay.set_status(
                f"Playing as {color_name}  ({len(get_templates())} templates)",
                GREEN, duration_ms=3000,
            )
        else:
            self.overlay.set_status(
                "No templates — looking for board to auto-calibrate...",
                ORANGE,
            )

        # Show overlay and start scanning now
        self.overlay.show()
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.scan)
        self.scan_timer.start(SCAN_INTERVAL_MS)

    def auto_calibrate(self, screenshot, board):
        """Try to extract templates from a starting-position board."""
        import os, cv2
        from calibrate import STARTING_POSITION, TEMPLATE_DIR, TEMPLATE_SIZE

        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        sq = board["square_size"]
        saved = 0
        seen_light = set()
        seen_dark = set()

        for row in range(8):
            for col in range(8):
                piece_name = STARTING_POSITION[row][col]
                if piece_name is None:
                    continue
                is_light = (row + col) % 2 == 0
                seen = seen_light if is_light else seen_dark
                if piece_name in seen:
                    continue
                seen.add(piece_name)

                x = int(board["x"] + col * sq)
                y = int(board["y"] + row * sq)
                w, h = int(sq), int(sq)
                square_img = screenshot[y : y + h, x : x + w]
                if square_img.size == 0:
                    continue
                template = cv2.resize(square_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
                suffix = "light" if is_light else "dark"
                path = os.path.join(TEMPLATE_DIR, f"{piece_name}_{suffix}.png")
                cv2.imwrite(path, template)
                saved += 1

        if saved > 0:
            reload_templates()
            self.has_templates = True
            print(f"Auto-calibrated: saved {saved} templates")
            self.overlay.set_status(
                f"Auto-calibrated {saved} piece templates!", GREEN, duration_ms=3000
            )
        return saved

    def _is_valid_for_turn(self, fen_position: str, turn: str) -> bool:
        """Check if a position is valid for the given turn."""
        try:
            b = chess.Board(f"{fen_position} {turn} KQkq - 0 1")
            return b.is_valid() and bool(list(b.legal_moves))
        except Exception:
            return False

    def _infer_current_turn(self, old_fen_pos: str, new_fen_pos: str) -> str | None:
        """Infer whose turn it is by comparing old and new piece positions.

        Looks at which color's pieces appeared on new squares to determine
        who just moved. Returns the color whose turn it is NOW.
        """
        old_rows = old_fen_pos.split("/")
        new_rows = new_fen_pos.split("/")
        if len(old_rows) != 8 or len(new_rows) != 8:
            return None

        def expand(row_str):
            out = []
            for ch in row_str:
                if ch.isdigit():
                    out.extend([None] * int(ch))
                else:
                    out.append(ch)
            return (out + [None] * 8)[:8]

        changed = 0
        white_arrived = 0
        black_arrived = 0

        for r in range(8):
            old_rank = expand(old_rows[r])
            new_rank = expand(new_rows[r])
            for c in range(8):
                if old_rank[c] != new_rank[c]:
                    changed += 1
                    if new_rank[c] is not None:
                        if new_rank[c].isupper():
                            white_arrived += 1
                        else:
                            black_arrived += 1

        # Too many changes = recognition noise, not a real move
        if changed > 6:
            return None

        if white_arrived > black_arrived:
            return "b"  # white moved -> black's turn
        elif black_arrived > white_arrived:
            return "w"  # black moved -> white's turn

        return None

    def scan(self):
        """One scan cycle: capture -> detect -> recognize -> analyze -> highlight."""
        if not self.running:
            return

        try:
            screenshot = capture_screen()
            board = detect_board(screenshot)

            if board is None:
                self.overlay.clear_highlights()
                self.overlay.set_status("Scanning... no board found", BLUE)
                return

            if not self.has_templates:
                self.overlay.set_status("Board found — calibrating...", ORANGE)
                self.auto_calibrate(screenshot, board)
                if not self.has_templates:
                    self.overlay.set_status(
                        "Board found but calibration failed — check starting position",
                        RED,
                    )
                return

            positions = recognize_board(screenshot, board)
            white_on_bottom = detect_orientation(positions)

            piece_count = sum(
                1 for row in positions for p in row if p is not None
            )

            fen_position = positions_to_fen(positions, "w").split(" ")[0]

            # No change from accepted position — nothing to do
            if fen_position == self.last_fen_position:
                self._pending_fen = None
                self._pending_count = 0
                return

            # Position changed — immediately clear highlights for instant feedback
            self.overlay.clear_highlights()

            # Require position to be stable before accepting
            if fen_position == self._pending_fen:
                self._pending_count += 1
            else:
                self._pending_fen = fen_position
                self._pending_count = 1

            if self._pending_count < self._STABLE_SCANS:
                return  # wait for position to stabilize

            # Position is stable — accept it
            self._pending_fen = None
            self._pending_count = 0

            # Determine whose turn it is by analyzing what changed
            if self.last_fen_position is not None:
                inferred = self._infer_current_turn(
                    self.last_fen_position, fen_position
                )
                if inferred is not None:
                    self.current_turn = inferred
                else:
                    # Fallback: validate both turns with chess rules
                    toggled = "b" if self.current_turn == "w" else "w"
                    picked = toggled
                    for candidate in [toggled, self.current_turn]:
                        try:
                            b = chess.Board(
                                f"{fen_position} {candidate} KQkq - 0 1"
                            )
                            if b.is_valid():
                                picked = candidate
                                break
                        except Exception:
                            pass
                    self.current_turn = picked

            self.last_fen_position = fen_position

            if piece_count < 4:
                self.overlay.set_status(
                    f"Board found — only {piece_count} pieces detected",
                    ORANGE,
                )
                self.overlay.clear_highlights()
                return

            # Skip analysis on opponent's turn
            if self.current_turn != self.player_color:
                color_name = "White" if self.player_color == "w" else "Black"
                self.overlay.set_status(
                    f"Opponent's turn (you are {color_name})", BLUE
                )
                self.overlay.clear_highlights()
                return

            # Analyze for the player's turn
            fen = f"{fen_position} {self.player_color} KQkq - 0 1"
            print(f"FEN: {fen}  ({piece_count} pieces)")

            best_move = self.engine.get_best_move(fen)
            if best_move is None:
                self.overlay.set_status("No legal moves found", ORANGE)
                self.overlay.clear_highlights()
                return

            self.last_move = best_move
            print(f"Best move: {best_move}")

            from_rect, to_rect = self.engine.move_to_screen_coords(
                best_move, board, white_on_bottom
            )
            self.overlay.set_highlights([from_rect, to_rect])
            self.overlay.set_status(
                f"Best move: {best_move}", GREEN, duration_ms=4000
            )

        except Exception as e:
            print(f"Scan error: {e}")
            self.overlay.set_status(f"Error: {e}", RED, duration_ms=5000)

    def stop(self):
        self.running = False
        self.overlay.clear_highlights()
        QApplication.quit()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    vision = ChessVision()

    vision.menu.show()  # show menu first; overlay + scanning starts after color is chosen

    quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), vision.overlay)
    quit_shortcut.activated.connect(vision.stop)

    print("Chess Vision — select your color and click Start.")
    print("Press Ctrl+Q to quit, or Ctrl+C in terminal.")
    print()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
