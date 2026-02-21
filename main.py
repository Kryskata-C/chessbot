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
from overlay import OverlayWindow

SCAN_INTERVAL_MS = 500

# Status colors
BLUE = QColor(0, 120, 255)
GREEN = QColor(30, 180, 60)
ORANGE = QColor(230, 140, 0)
RED = QColor(220, 40, 40)


class ChessVision:
    def __init__(self):
        self.engine = ChessEngine(depth=12, threads=2)
        self.overlay = OverlayWindow()
        self.last_fen_position: str | None = None
        self.last_move: str | None = None
        self.player_color: str | None = None  # "w" or "b"
        self.running = True
        self.has_templates = len(get_templates()) > 0

        if self.has_templates:
            self.overlay.set_status(
                f"Chess Vision ready  ({len(get_templates())} templates loaded)",
                GREEN, duration_ms=3000,
            )
        else:
            self.overlay.set_status(
                "No templates — looking for board to auto-calibrate...",
                ORANGE,
            )

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

            # Detect player color once from board orientation (your pieces on bottom)
            if self.player_color is None:
                self.player_color = "w" if white_on_bottom else "b"
                color_name = "White" if self.player_color == "w" else "Black"
                print(f"Playing as: {color_name}")
                self.overlay.set_status(
                    f"Playing as {color_name}", GREEN, duration_ms=3000
                )

            piece_count = sum(
                1 for row in positions for p in row if p is not None
            )

            fen_position = positions_to_fen(positions, "w").split(" ")[0]

            # Only re-analyze if position changed
            if fen_position == self.last_fen_position:
                return

            self.last_fen_position = fen_position

            if piece_count < 4:
                self.overlay.set_status(
                    f"Board found — only {piece_count} pieces detected",
                    ORANGE,
                )
                self.overlay.clear_highlights()
                return

            # Always try to analyze for the player's color
            fen = f"{fen_position} {self.player_color} KQkq - 0 1"
            print(f"FEN: {fen}  ({piece_count} pieces)")

            best_move = self.engine.get_best_move(fen)
            if best_move is None:
                # Can't find a move for player — probably opponent's turn
                color_name = "White" if self.player_color == "w" else "Black"
                self.overlay.set_status(
                    f"Opponent's turn (you are {color_name})", BLUE
                )
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

    vision.overlay.show()

    timer = QTimer()
    timer.timeout.connect(vision.scan)
    timer.start(SCAN_INTERVAL_MS)

    quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), vision.overlay)
    quit_shortcut.activated.connect(vision.stop)

    print("Chess Vision started!")
    print(f"Scanning every {SCAN_INTERVAL_MS/1000:.1f}s")
    print("Press Ctrl+Q to quit, or Ctrl+C in terminal.")
    print()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
