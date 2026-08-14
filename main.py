"""Chess Vision - Screen Chess Analyzer with Overlay.

Captures the screen, detects a chess.com board, recognizes pieces,
runs Stockfish for the best move, and highlights it on a transparent overlay.
"""

from __future__ import annotations

import sys
import signal
import threading
import time

import chess
import cv2
import numpy as np

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence, QColor
from PyQt6.QtWidgets import QApplication

from capture import capture_screen
from board_detector import detect_board
from piece_recognizer import (
    recognize_board,
    positions_to_fen,
    get_templates,
    reload_templates,
)
from engine import ChessEngine
from elo_estimator import EloEstimator
from move_selector import HumanMoveSelector
from overlay import OverlayWindow, DebugBoardWindow
from menu import MenuWindow

SCAN_INTERVAL_MS = 400

STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# Status colors
BLUE = QColor(0, 120, 255)
GREEN = QColor(30, 180, 60)
ORANGE = QColor(230, 140, 0)
RED = QColor(220, 40, 40)


def infer_castling(fen_position: str) -> str:
    """Infer castling rights from piece placement.

    Only grant castling if the king and rook are on their starting squares.
    """
    rows = fen_position.split("/")
    if len(rows) != 8:
        return "-"

    def expand(row_str):
        out = []
        for ch in row_str:
            if ch.isdigit():
                out.extend([None] * int(ch))
            else:
                out.append(ch)
        return (out + [None] * 8)[:8]

    rank1 = expand(rows[7])  # white's back rank
    rank8 = expand(rows[0])  # black's back rank

    rights = ""
    # White king on e1 (col 4)
    if rank1[4] == "K":
        if rank1[7] == "R":  # h1 rook
            rights += "K"
        if rank1[0] == "R":  # a1 rook
            rights += "Q"
    # Black king on e8 (col 4)
    if rank8[4] == "k":
        if rank8[7] == "r":  # h8 rook
            rights += "k"
        if rank8[0] == "r":  # a8 rook
            rights += "q"

    return rights if rights else "-"


class ChessVision(QObject):
    """Scan/analysis pipeline split across threads.

    Screen capture must stay on the GUI thread (macOS CoreGraphics display
    enumeration fails from background threads), but it's cheap. The heavy
    work — recognition and every engine call — runs on a worker thread that
    consumes the latest captured frame, so overlay animations never stutter.
    GUI updates are marshaled back through the sig_gui queued signal.
    """

    sig_gui = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self.engine = ChessEngine(depth=12, threads=2)
        self.move_selector = HumanMoveSelector(self.engine)
        self.overlay = OverlayWindow()
        self.debug_board = DebugBoardWindow()
        self.menu = MenuWindow()
        self.last_fen_position: str | None = None
        self.last_move: str | None = None
        self.player_color: str | None = None  # "w" or "b"
        self.current_turn: str = "w"  # white always moves first
        self.running = True
        self.has_templates = len(get_templates()) > 0
        self.elo_estimator = EloEstimator()
        self._scan_thread: threading.Thread | None = None
        self._capture_timer: QTimer | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_offset: tuple[int, int] = (0, 0)
        self._frame_ready = threading.Event()
        # Once the board is located, capture only its neighborhood
        # (full-screen Retina grabs are ~5x more expensive)
        self._capture_region: dict | None = None
        self._monitor_bounds: dict | None = None
        # Per-square brightness fingerprint of the last scanned board —
        # lets us skip 200ms of template matching when nothing changed
        self._last_cells: np.ndarray | None = None
        # Position stability: require same FEN for 2 scans before accepting
        self._pending_fen: str | None = None
        self._pending_count: int = 0
        self._STABLE_SCANS = 3
        # Game boundary tracking
        self._game_over: bool = False
        self._en_passant: str = "-"
        # Board coordinate cache — reuse if detection is close to last known
        self._cached_board: dict | None = None
        self._BOARD_DRIFT_THRESHOLD = 5  # max pixel drift before re-caching
        # Piece-lifted guard: don't accept boards where pieces only
        # disappeared (someone is dragging), with a stale-out safety cap
        self._lift_skips: int = 0
        self._MAX_LIFT_SKIPS = 6
        # King-missing guard: if a king stays invisible, our own overlay
        # visuals may be covering it — clear them once to break the loop
        self._king_miss_scans: int = 0
        self._KING_MISS_RESET = 3
        # Tracked game state: recognition identifies which legal move was
        # played; turn/castling/en-passant then come from real game state
        # instead of being inferred per scan.
        self.game_board: chess.Board | None = None
        self._last_analyzed_fen: str | None = None
        # Visual effect toggles (menu can override) and enemy-move capture
        self.visuals: dict = {}
        self._new_enemy_move: chess.Move | None = None

        # Wire up menu → start, and worker thread → GUI updates
        self.menu.started.connect(self._on_started)
        self.sig_gui.connect(self._apply_gui)

    def _gui(self, op: str, payload: dict | None = None):
        """Queue a GUI update from the scan thread."""
        self.sig_gui.emit(op, payload or {})

    def _status(self, text: str, color: QColor | None = None,
                duration_ms: int = 0):
        """Queue a status banner update from the scan thread."""
        self._gui("status", {
            "text": text, "color": color, "duration_ms": duration_ms,
        })

    def _apply_gui(self, op: str, payload: dict):
        """Runs on the GUI thread: apply a queued overlay/debug update."""
        if op == "status":
            self.overlay.set_status(
                payload["text"], payload.get("color"),
                payload.get("duration_ms", 0),
            )
        elif op == "geo":
            self.overlay.set_board_geometry(payload["board"], payload["wob"])
        elif op == "suggestion":
            self.overlay.show_suggestion(**payload)
        elif op == "threats":
            self.overlay.set_threats(payload["squares"])
        elif op == "trail":
            self.overlay.flash_enemy_move(payload["move"])
        elif op == "eval":
            self.overlay.set_eval(payload["cp"])
        elif op == "clear":
            self.overlay.clear_highlights()
        elif op == "reset_visuals":
            self.overlay.reset_board_visuals()
        elif op == "debug":
            self.debug_board.set_positions(**payload)

    def _on_started(self, color: str, visuals: dict):
        """Called when the user picks a color + visuals and clicks Start."""
        self.visuals = visuals
        self.overlay.set_visual_config(visuals)
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

        # Show overlay and debug board, then start the pipeline: a GUI-side
        # timer captures frames, the worker thread does recognition + engine.
        self.overlay.show()
        self.debug_board.show()
        try:
            from capture import get_monitor_info
            self._monitor_bounds = get_monitor_info(1)
        except Exception:
            self._monitor_bounds = None
        self._capture_timer = QTimer(self)
        self._capture_timer.timeout.connect(self._capture_tick)
        self._capture_timer.start(SCAN_INTERVAL_MS)
        self._scan_thread = threading.Thread(
            target=self._scan_loop, daemon=True
        )
        self._scan_thread.start()

    def _capture_tick(self):
        """GUI thread: grab a frame (board region when known) for the worker."""
        if not self.running:
            return
        region = self._capture_region
        try:
            frame = capture_screen(region=region)
        except Exception as e:
            print(f"Capture error: {e}")
            self._capture_region = None
            return
        offset = (region["left"], region["top"]) if region else (0, 0)
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_offset = offset
        self._frame_ready.set()

    def _scan_loop(self):
        """Worker thread: process the newest frame as fast as it can keep up.

        If analysis takes longer than the capture interval, intermediate
        frames are simply replaced — the worker always sees the latest.
        """
        while self.running:
            if not self._frame_ready.wait(timeout=1.0):
                continue
            self._frame_ready.clear()
            with self._frame_lock:
                frame = self._latest_frame
                offset = self._latest_offset
            if frame is not None:
                self.scan(frame, offset)

    def auto_calibrate(self, screenshot, board):
        """Try to extract templates from a starting-position board."""
        import os, cv2
        from calibrate import STARTING_POSITION, TEMPLATE_DIR, TEMPLATE_SIZE

        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        sq = board["square_size"]
        img_h, img_w = screenshot.shape[:2]
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

                x = round(board["x"] + col * sq)
                y = round(board["y"] + row * sq)
                w, h = round(sq), round(sq)
                # Clamp to image bounds
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = min(w, img_w - x)
                h = min(h, img_h - y)
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
            self._status(
                f"Auto-calibrated {saved} piece templates!", GREEN,
                duration_ms=3000,
            )
        return saved

    def _is_valid_for_turn(self, fen_position: str, turn: str) -> bool:
        """Check if a position is valid for the given turn."""
        try:
            castling = infer_castling(fen_position)
            b = chess.Board(f"{fen_position} {turn} {castling} - 0 1")
            return b.is_valid() and bool(list(b.legal_moves))
        except Exception:
            return False

    @staticmethod
    def _diff_stats(old_fen_pos: str, new_fen_pos: str) -> tuple[int, int, int] | None:
        """Compare two piece placements.

        Returns (changed_squares, white_arrivals, black_arrivals),
        or None if either placement is malformed.
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

        return changed, white_arrived, black_arrived

    def _infer_current_turn(self, old_fen_pos: str, new_fen_pos: str) -> str | None:
        """Infer whose turn it is by comparing old and new piece positions.

        Looks at which color's pieces appeared on new squares to determine
        who just moved. Returns the color whose turn it is NOW, "same" if
        the side to move is unchanged (an even number of half-moves merged
        into one update), or None if the diff is too noisy to trust.
        """
        stats = self._diff_stats(old_fen_pos, new_fen_pos)
        if stats is None:
            return None
        changed, white_arrived, black_arrived = stats

        # Too many changes = recognition noise, not a real move
        if changed > 6:
            return None

        if white_arrived > black_arrived:
            return "b"  # white moved -> black's turn
        elif black_arrived > white_arrived:
            return "w"  # black moved -> white's turn

        # No arrivals at all: a piece was lifted mid-drag or occluded —
        # not a completed move. Let the caller's validity fallback decide.
        if white_arrived == 0:
            return None

        # Equal non-zero arrivals: a fast move + reply (premove) merged
        # into one update. Every completed half-move lands a piece
        # somewhere, so an equal count means an even number of half-moves
        # — the side to move hasn't changed.
        return "same"

    def _reset_game_state(self):
        """Reset all per-game state for a new game."""
        self.move_selector.reset()
        self.elo_estimator.reset()
        self.last_fen_position = None
        self.current_turn = "w"
        self._game_over = False
        self._en_passant = "-"
        self._lift_skips = 0
        self._king_miss_scans = 0
        self.game_board = None
        self._last_analyzed_fen = None
        self._new_enemy_move = None
        self._gui("reset_visuals")
        print("Game state reset for new game.")

    # ------------------------------------------------------------------
    # Game tracking: keep a real chess.Board and use recognition to
    # identify which legal move(s) were played. Turn, castling, and
    # en passant then follow exactly from game state, so a single misread
    # square can't desync the turn tracker.
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_placement(placement: str) -> list[str | None]:
        """Expand a FEN placement to a flat 64-list (a8..h1 order)."""
        out: list[str | None] = []
        for row in placement.split("/"):
            for ch in row:
                if ch.isdigit():
                    out.extend([None] * int(ch))
                else:
                    out.append(ch)
        return out

    @classmethod
    def _count_mismatches(cls, a: str, b: str) -> int:
        ea, eb = cls._expand_placement(a), cls._expand_placement(b)
        if len(ea) != 64 or len(eb) != 64:
            return 99
        return sum(1 for x, y in zip(ea, eb) if x != y)

    @classmethod
    def _changed_squares(cls, a: str, b: str) -> set[int]:
        """chess.Square indices that differ between two placements."""
        ea, eb = cls._expand_placement(a), cls._expand_placement(b)
        changed: set[int] = set()
        if len(ea) != 64 or len(eb) != 64:
            return changed
        for i in range(64):
            if ea[i] != eb[i]:
                row, col = divmod(i, 8)  # row 0 = rank 8
                changed.add(chess.square(col, 7 - row))
        return changed

    def _match_moves(self, target: str) -> list[chess.Move] | None:
        """Find the legal move sequence (1 or 2 plies) that turns the
        tracked position into the recognized placement.

        The 2-ply search covers a fast move + reply merging into one scan
        update. The fuzzy pass tolerates one misread square (e.g. wrong
        piece color under chess.com's move highlight), trusting the move
        whose outcome is uniquely closest to what was recognized.
        """
        b = self.game_board

        # Exact single move
        for mv in b.legal_moves:
            b.push(mv)
            match = b.board_fen() == target
            b.pop()
            if match:
                return [mv]

        # Exact move + reply. The first move must touch a changed square
        # (its origin square always ends up changed).
        changed = self._changed_squares(b.board_fen(), target)
        for m1 in list(b.legal_moves):
            if m1.from_square not in changed and m1.to_square not in changed:
                continue
            b.push(m1)
            found = None
            for m2 in b.legal_moves:
                b.push(m2)
                if b.board_fen() == target:
                    found = m2
                b.pop()
                if found:
                    break
            b.pop()
            if found:
                return [m1, found]

        # Fuzzy single move: allow one misread square, require uniqueness
        best, best_n, second_n = None, 99, 99
        for mv in b.legal_moves:
            b.push(mv)
            n = self._count_mismatches(b.board_fen(), target)
            b.pop()
            if n < best_n:
                best, second_n, best_n = mv, best_n, n
            elif n < second_n:
                second_n = n
        if best is not None and best_n <= 1 and second_n > best_n:
            print(f"Fuzzy-matched {best.uci()} ({best_n} misread square)")
            return [best]

        return None

    def _init_game_state(self, fen_position: str):
        """(Re)build the tracked game from a raw recognized placement."""
        if fen_position == STARTING_PLACEMENT:
            turn = "w"
        elif self.last_fen_position is not None:
            inferred = self._infer_current_turn(
                self.last_fen_position, fen_position
            )
            if inferred == "same":
                turn = self.current_turn
            elif inferred is not None:
                turn = inferred
            else:
                # Fallback: validate both turns with chess rules
                turn = "b" if self.current_turn == "w" else "w"
                for candidate in [turn, self.current_turn]:
                    try:
                        b = chess.Board(
                            f"{fen_position} {candidate} "
                            f"{infer_castling(fen_position)} - 0 1"
                        )
                        if b.is_valid():
                            turn = candidate
                            break
                    except Exception:
                        pass
        else:
            turn = self.current_turn

        self.current_turn = turn
        try:
            self.game_board = chess.Board(
                f"{fen_position} {turn} {infer_castling(fen_position)} - 0 1"
            )
        except Exception:
            self.game_board = None
        print(f"Game tracking (re)synced: turn={turn}")

    def _track_position(self, fen_position: str):
        """Advance the tracked game to match a newly recognized placement."""
        b = self.game_board
        if b.board_fen() == fen_position:
            return
        # A placement differing on a single square can't be a completed
        # move (every move changes at least two squares) — recognition
        # noise; keep the tracked state.
        if self._count_mismatches(b.board_fen(), fen_position) <= 1:
            return
        moves = self._match_moves(fen_position)
        if moves is not None:
            for mv in moves:
                mover = "w" if b.turn == chess.WHITE else "b"
                b.push(mv)
                if mover != self.player_color:
                    self._new_enemy_move = mv
            print(f"Tracked: {' '.join(m.uci() for m in moves)}")
            self.current_turn = "w" if b.turn == chess.WHITE else "b"
            return

        # Takeback? (user pressed undo) — walk back through our own history
        tmp = b.copy()
        for k in range(1, min(6, len(b.move_stack)) + 1):
            tmp.pop()
            if tmp.board_fen() == fen_position:
                for _ in range(k):
                    b.pop()
                print(f"Takeback detected — rewound {k} half-move(s)")
                self.current_turn = "w" if b.turn == chess.WHITE else "b"
                return

        print("Recognized position doesn't follow legally — resyncing.")
        self._init_game_state(fen_position)

    _PIECE_VALUES = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9,
    }

    def _compute_threats(self) -> list[str]:
        """Squares holding player pieces that are hanging or under-defended.

        A piece counts as threatened when it's attacked and either has no
        defender or is attacked by something cheaper than itself. The king
        square is included when in check.
        """
        b = self.game_board
        if b is None or self.player_color is None:
            return []
        us = chess.WHITE if self.player_color == "w" else chess.BLACK
        out: list[str] = []
        for sq, piece in b.piece_map().items():
            if piece.color != us:
                continue
            if piece.piece_type == chess.KING:
                if b.turn == us and b.is_check():
                    out.append(chess.square_name(sq))
                continue
            attackers = b.attackers(not us, sq)
            if not attackers:
                continue
            defenders = b.attackers(us, sq)
            cheapest = min(
                self._PIECE_VALUES.get(b.piece_at(a).piece_type, 99)
                for a in attackers
            )
            if not defenders or cheapest < self._PIECE_VALUES.get(piece.piece_type, 0):
                out.append(chess.square_name(sq))
        return out

    def _infer_en_passant(self, old_fen_pos: str, new_fen_pos: str) -> str:
        """Infer en passant target square from a pawn double-push."""
        old_rows = old_fen_pos.split("/")
        new_rows = new_fen_pos.split("/")
        if len(old_rows) != 8 or len(new_rows) != 8:
            return "-"

        def expand(row_str):
            out = []
            for ch in row_str:
                if ch.isdigit():
                    out.extend([None] * int(ch))
                else:
                    out.append(ch)
            return (out + [None] * 8)[:8]

        col_letters = "abcdefgh"

        for c in range(8):
            # White pawn double-push: from row 6 (rank 2) to row 4 (rank 4)
            old_r6 = expand(old_rows[6])
            new_r6 = expand(new_rows[6])
            old_r4 = expand(old_rows[4])
            new_r4 = expand(new_rows[4])
            if (old_r6[c] == "P" and new_r6[c] is None
                    and new_r4[c] == "P" and old_r4[c] is None):
                return f"{col_letters[c]}3"  # EP square behind the pawn

            # Black pawn double-push: from row 1 (rank 7) to row 3 (rank 5)
            old_r1 = expand(old_rows[1])
            new_r1 = expand(new_rows[1])
            old_r3 = expand(old_rows[3])
            new_r3 = expand(new_rows[3])
            if (old_r1[c] == "p" and new_r1[c] is None
                    and new_r3[c] == "p" and old_r3[c] is None):
                return f"{col_letters[c]}6"  # EP square behind the pawn

        return "-"

    def _update_capture_region(self, board_abs: dict):
        """Ask the capture side for just the board's neighborhood."""
        pad = int(board_abs["square_size"])
        x = int(board_abs["x"] - pad)
        y = int(board_abs["y"] - pad)
        w = int(board_abs["width"] + 2 * pad)
        h = int(board_abs.get("height", board_abs["width"]) + 2 * pad)
        mb = self._monitor_bounds
        if mb:
            x = max(mb["left"], x)
            y = max(mb["top"], y)
            w = min(w, mb["left"] + mb["width"] - x)
            h = min(h, mb["top"] + mb["height"] - y)
        else:
            x, y = max(0, x), max(0, y)
        if w > 0 and h > 0:
            self._capture_region = {
                "left": x, "top": y, "width": w, "height": h,
            }

    def scan(self, screenshot, offset: tuple[int, int] = (0, 0)):
        """One scan cycle: detect -> recognize -> analyze -> highlight.

        `screenshot` may be a region crop; `offset` maps its origin back
        to absolute screen coordinates.
        """
        if not self.running:
            return

        try:
            ox, oy = offset
            board = detect_board(screenshot)

            if board is None:
                self._cached_board = None
                self._capture_region = None  # fall back to full-screen grabs
                self._last_cells = None
                self._gui("reset_visuals")
                self._status("Scanning... no board found", BLUE)
                return

            # Work in absolute screen coordinates for caching/overlay
            board_abs = {**board, "x": board["x"] + ox, "y": board["y"] + oy}

            # Stabilize board coordinates: if the new detection is very close
            # to the cached one, reuse the cached coords to prevent jitter
            # that causes edge-square recognition noise.
            if self._cached_board is not None:
                dx = abs(board_abs["x"] - self._cached_board["x"])
                dy = abs(board_abs["y"] - self._cached_board["y"])
                ds = abs(board_abs["width"] - self._cached_board["width"])
                if dx <= self._BOARD_DRIFT_THRESHOLD and \
                   dy <= self._BOARD_DRIFT_THRESHOLD and \
                   ds <= self._BOARD_DRIFT_THRESHOLD:
                    board_abs = self._cached_board
                else:
                    self._cached_board = board_abs
            else:
                self._cached_board = board_abs
            self._update_capture_region(board_abs)

            # Frame-local coordinates for recognition on this screenshot
            board = {**board_abs, "x": board_abs["x"] - ox,
                     "y": board_abs["y"] - oy}

            if not self.has_templates:
                self._status("Board found — calibrating...", ORANGE)
                self.auto_calibrate(screenshot, board)
                if not self.has_templates:
                    self._status(
                        "Board found but calibration failed — check starting position",
                        RED,
                    )
                return

            # Cheap change gate: compare per-square brightness against the
            # previous scan and skip the expensive template matching
            # (~200ms) when the board hasn't visibly changed. Disabled
            # while a pending position is stabilizing or a king is missing
            # (the missing-king guard must keep rescanning to fire).
            bx, by = int(board["x"]), int(board["y"])
            bs = int(board["width"])
            crop = screenshot[max(0, by):by + bs, max(0, bx):bx + bs]
            if crop.size:
                gray = cv2.cvtColor(cv2.resize(crop, (64, 64)),
                                    cv2.COLOR_BGR2GRAY)
                cells = gray.reshape(8, 8, 8, 8).mean(axis=(1, 3))
                if (self._last_cells is not None
                        and self._pending_fen is None
                        and self._king_miss_scans == 0
                        and float(np.abs(cells - self._last_cells).max()) < 5.0):
                    self._last_cells = cells
                    return  # board visually unchanged — nothing to do
                self._last_cells = cells

            positions = recognize_board(screenshot, board)
            # The player's pieces are always at the bottom on chess.com, so
            # orientation follows from the chosen color. Never guess it from
            # piece placement — that flips in endgames when pieces advance.
            white_on_bottom = self.player_color == "w"
            self._gui("geo", {"board": board_abs, "wob": white_on_bottom})

            piece_count = sum(
                1 for row in positions for p in row if p is not None
            )

            fen_position = positions_to_fen(
                positions, "w", white_on_bottom
            ).split(" ")[0]

            # A king can never leave the board. If one is missing even after
            # recovery, this scan is recognition noise (piece mid-drag,
            # cursor/glow overlay) — Stockfish crashes on king-less FENs,
            # so never accept or analyze such a position.
            if "K" not in fen_position or "k" not in fen_position:
                self._king_miss_scans += 1
                if self._king_miss_scans == self._KING_MISS_RESET:
                    self._gui("reset_visuals")
                self._status("Scan unclear — king not visible", ORANGE)
                return
            self._king_miss_scans = 0

            # New game detection: piece count jumps back near 32
            if self._game_over and piece_count >= 30 and fen_position == STARTING_PLACEMENT:
                self._reset_game_state()
                self._status("New game detected!", GREEN, duration_ms=3000)
                print("New game detected — state reset.")
                return

            # In game-over state, keep scanning but skip analysis
            if self._game_over:
                self._status("Game over — waiting for new game...", ORANGE)
                return

            # No change from accepted position — nothing to do
            if fen_position == self.last_fen_position:
                self._pending_fen = None
                self._pending_count = 0
                return

            # Position changed — keep old highlights visible until new ones
            # are ready, instead of clearing them eagerly which causes flicker.

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

            # A board where pieces only disappeared is not a position —
            # someone is holding a piece mid-drag. Accepting it poisons the
            # diff baseline and desyncs turn tracking, so wait for the
            # piece to land (with a stale-out cap in case a piece is
            # genuinely lost to recognition).
            if self.last_fen_position is not None:
                stats = self._diff_stats(self.last_fen_position, fen_position)
                if stats is not None:
                    changed, w_arr, b_arr = stats
                    if (changed > 0 and w_arr == 0 and b_arr == 0
                            and self._lift_skips < self._MAX_LIFT_SKIPS):
                        self._lift_skips += 1
                        self._status("Piece lifted — waiting for the move", BLUE)
                        return
            self._lift_skips = 0

            # Track the game: identify which legal move(s) explain the new
            # placement; falls back to a hard resync when nothing does.
            if self.game_board is None:
                self._init_game_state(fen_position)
            else:
                self._track_position(fen_position)

            # Feed the overlay: opponent's last move trail + threat radar
            if self._new_enemy_move is not None:
                self._gui("trail", {"move": self._new_enemy_move.uci()})
                self._new_enemy_move = None
            self._gui("threats", {"squares": self._compute_threats()})

            # Infer en passant from pawn double-pushes (only used by the
            # string-FEN fallback when game tracking is unavailable)
            if self.last_fen_position is not None:
                self._en_passant = self._infer_en_passant(
                    self.last_fen_position, fen_position
                )
            else:
                self._en_passant = "-"

            # Check for game over
            game_over_board = self.game_board
            if game_over_board is None:
                try:
                    cb = chess.Board(
                        f"{fen_position} {self.current_turn} "
                        f"{infer_castling(fen_position)} {self._en_passant} 0 1"
                    )
                    if cb.is_valid():
                        game_over_board = cb
                except Exception:
                    pass
            if game_over_board is not None and game_over_board.is_game_over():
                self._game_over = True
                result = game_over_board.result()
                self._status(
                    f"Game over ({result}) — waiting for new game...", ORANGE
                )
                self._gui("reset_visuals")
                self.last_fen_position = fen_position
                print(f"Game over detected: {result}")
                return

            # Estimate opponent ELO when opponent just moved
            if (self.last_fen_position is not None
                    and self.current_turn == self.player_color):
                # Opponent just moved: evaluate before and after
                opp_color = "b" if self.player_color == "w" else "w"
                old_castling = infer_castling(self.last_fen_position)
                new_castling = infer_castling(fen_position)
                old_fen = f"{self.last_fen_position} {opp_color} {old_castling} {self._en_passant} 0 1"
                new_fen = f"{fen_position} {self.player_color} {new_castling} {self._en_passant} 0 1"
                try:
                    eval_before = self.engine.get_evaluation(old_fen)
                    eval_after = self.engine.get_evaluation(new_fen)
                    # Both evals are from side-to-move POV:
                    # eval_before = how good it was for opponent
                    # eval_after = how good it is for player (negative = bad for opponent)
                    # CPL = eval_before + eval_after (opponent's loss)
                    cpl = eval_before + eval_after
                    self.elo_estimator.record_move(cpl)
                    # eval_after is from the player's POV — flip for White POV
                    white_cp = eval_after if self.player_color == "w" else -eval_after
                    self._gui("eval", {"cp": white_cp})
                except Exception as e:
                    print(f"ELO estimation error: {e}")

            self.last_fen_position = fen_position

            # Feed opponent ELO estimate to move selector for adaptive play
            elo_est = self.elo_estimator.get_estimate()
            acpl = self.elo_estimator.get_acpl()
            self.move_selector.set_opponent_elo(elo_est)

            # Update debug board GUI
            self._gui("debug", {
                "positions": positions,
                "white_on_bottom": white_on_bottom,
                "turn": self.current_turn,
                "piece_count": piece_count,
                "estimated_elo": elo_est,
                "opponent_acpl": acpl,
                "bot_accuracy": self.move_selector.get_accuracy(),
                "bot_cpl": self.move_selector.get_avg_cpl(),
            })

            if piece_count < 4:
                self._status(
                    f"Board found — only {piece_count} pieces detected",
                    ORANGE,
                )
                self._gui("clear")
                return

            # Skip analysis on opponent's turn
            if self.current_turn != self.player_color:
                color_name = "White" if self.player_color == "w" else "Black"
                self._status(
                    f"Opponent's turn (you are {color_name})", BLUE
                )
                self._gui("clear")
                return

            # Analyze for the player's turn. Prefer the tracked game state:
            # exact castling/en-passant, and immune to one-square misreads.
            if self.game_board is not None:
                fen = self.game_board.fen()
            else:
                castling = infer_castling(fen_position)
                fen = f"{fen_position} {self.player_color} {castling} {self._en_passant} 0 1"

            if fen == self._last_analyzed_fen:
                return  # already suggested a move for this exact position
            self._last_analyzed_fen = fen
            print(f"FEN: {fen}  ({piece_count} pieces)")

            chosen_move = self.move_selector.select_move(fen, piece_count)
            if chosen_move is None:
                self._status("No legal moves found", ORANGE)
                self._gui("clear")
                return

            self.last_move = chosen_move
            print(f"Suggested move: {chosen_move}")

            self._show_suggestion(chosen_move, fen)
            self._status(f"Move: {chosen_move}", GREEN, duration_ms=4000)

        except Exception as e:
            print(f"Scan error: {e}")
            self._status(f"Error: {e}", RED, duration_ms=5000)

    def _visual_on(self, key: str) -> bool:
        return self.visuals.get(key, True)

    def _show_suggestion(self, chosen_move: str, fen: str):
        """Send the suggested move to the overlay with all visual context:
        moving piece, criticality, capture/check flags, predicted
        continuation, and alternative candidates."""
        piece_sym = None
        is_capture = False
        is_check = False
        pv: list[str] = []

        try:
            b = self.game_board.copy() if self.game_board is not None \
                else chess.Board(fen)
            mv = chess.Move.from_uci(chosen_move)
            piece = b.piece_at(mv.from_square)
            piece_sym = piece.symbol() if piece else None
            is_capture = b.is_capture(mv)
            is_check = b.gives_check(mv)

            # Shallow lookahead for the predicted continuation
            if self._visual_on("reply") or self._visual_on("pv"):
                b.push(mv)
                reply = self.engine.get_best_move(b.fen(), depth=8)
                if reply:
                    pv.append(reply)
                    if self._visual_on("pv"):
                        b.push(chess.Move.from_uci(reply))
                        follow_up = self.engine.get_best_move(b.fen(), depth=8)
                        if follow_up:
                            pv.append(follow_up)
        except Exception as e:
            print(f"Suggestion visual context error: {e}")

        candidates = []
        if self._visual_on("candidates"):
            candidates = [
                m["move"] for m in self.move_selector.last_top_moves[:3]
                if m["move"] != chosen_move
            ]

        self._gui("suggestion", {
            "move_uci": chosen_move,
            "piece_symbol": piece_sym,
            "crit": self.move_selector.last_criticality,
            "is_capture": is_capture,
            "is_check": is_check,
            "pv": pv,
            "candidates": candidates,
        })

        # Eval bar from the fresh analysis (player POV -> White POV)
        best_eval = self.move_selector.last_best_eval
        if best_eval is not None:
            white_cp = best_eval if self.player_color == "w" else -best_eval
            self._gui("eval", {"cp": white_cp})

    def stop(self):
        self.running = False
        self._frame_ready.set()  # unblock the worker so it can exit
        if self._capture_timer is not None:
            self._capture_timer.stop()
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
