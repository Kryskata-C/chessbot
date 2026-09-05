"""Identify chess pieces on each square using template matching."""

from __future__ import annotations

import os
import cv2
import numpy as np
import chess

from paths import TEMPLATE_DIR
TEMPLATE_SIZE = 80
MATCH_THRESHOLD = 0.55
# Relaxed threshold used only when a king is missing from the board.
# Overlays like chess.com's red check-glow can drag the king's normal
# match score below MATCH_THRESHOLD, making the king "vanish".
KING_RECOVERY_THRESHOLD = 0.25

# Base piece names -> FEN symbols
PIECE_NAMES = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}


def _load_templates() -> list[tuple[str, str, np.ndarray]]:
    """Load all piece template images.

    Returns:
        List of (base_name, fen_symbol, image) tuples.
        Multiple entries per piece if light/dark variants exist.
    """
    templates = []
    shaded: dict[str, np.ndarray] = {}
    SYNTHESIZED.clear()
    if not os.path.isdir(TEMPLATE_DIR):
        return templates
    for fname in os.listdir(TEMPLATE_DIR):
        if not fname.endswith(".png"):
            continue
        stem = fname[:-4]  # remove .png
        # Match against known piece names (with optional _light/_dark suffix)
        base = stem.replace("_light", "").replace("_dark", "")
        if base not in PIECE_NAMES:
            continue
        path = os.path.join(TEMPLATE_DIR, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            templates.append((base, PIECE_NAMES[base], img))
            shaded[stem] = img
    templates.extend(_synthesize_missing_shades(shaded))
    return templates


def _border_color(img: np.ndarray) -> np.ndarray:
    """Median BGR of a template's 4px frame — the square colour under it."""
    edge = np.concatenate([
        img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3),
        img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3),
    ])
    return np.median(edge, axis=0)


def _repaint_background(img: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Recolour the square behind a piece template from `src` to `dst`."""
    f = img.astype(np.float32)
    # 0 where the pixel is the square colour, 1 where it is piece
    alpha = np.clip(np.linalg.norm(f - src, axis=2) / 40.0, 0, 1)[..., None]
    repainted = alpha * f + (1 - alpha) * (f - src + dst)
    return repainted.clip(0, 255).astype(np.uint8)


_highlight_sets: dict[tuple[int, int, int], list[tuple[str, str, np.ndarray]]] = {}


def _templates_for_background(bg: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """Every piece template repainted onto the given square colour, built
    once per distinct colour. Used for chess.com's last-move highlight
    squares, whose yellow tint otherwise drags scores toward the threshold
    (the calibration never sees a piece on a highlighted square)."""
    key = tuple(int(v) // 6 for v in bg)
    cached = _highlight_sets.get(key)
    if cached is not None:
        return cached
    made = []
    for base, sym, tmpl in get_templates():
        made.append((base, sym, _repaint_background(tmpl, _border_color(tmpl), bg)))
    _highlight_sets[key] = made
    return made


def _synthesize_missing_shades(
    shaded: dict[str, np.ndarray],
) -> list[tuple[str, str, np.ndarray]]:
    """Fill in templates calibration can't capture.

    Calibration cuts templates from the starting position, where each
    king and queen sits on exactly one square colour. On the other colour
    the match drops to the ~0.55 threshold (a white king on e2 scored
    0.55 vs 0.97), so the king "vanishes" and the scan stalls. Build the
    missing variant by repainting the square colour behind the piece.
    """
    light = [t for n, t in shaded.items() if n.endswith("_light")]
    dark = [t for n, t in shaded.items() if n.endswith("_dark")]
    if not light or not dark:
        return []
    light_bg = np.median([_border_color(t) for t in light], axis=0)
    dark_bg = np.median([_border_color(t) for t in dark], axis=0)
    made = []
    for stem, img in shaded.items():
        for shade, other in (("_light", "_dark"), ("_dark", "_light")):
            if not stem.endswith(shade):
                continue
            other_stem = stem[: -len(shade)] + other
            if other_stem in shaded:
                continue
            src, dst = (light_bg, dark_bg) if shade == "_light" else (dark_bg, light_bg)
            base = other_stem.replace("_light", "").replace("_dark", "")
            SYNTHESIZED.add(other_stem)
            made.append((base, PIECE_NAMES[base],
                         _repaint_background(img, src, dst)))
    if made:
        print(f"Synthesized {len(made)} missing square-colour templates: "
              + ", ".join(n for n, _, _ in made))
    return made


_templates: list[tuple[str, str, np.ndarray]] | None = None
# Stems (e.g. "white_king_light") that only exist as repainted guesses;
# the live game replaces them with real crops as the pieces visit those
# squares (see harvest_templates).
SYNTHESIZED: set[str] = set()


def harvest_templates(screenshot: np.ndarray, board: dict,
                      positions: list[list[str | None]],
                      skip: set[tuple[int, int]] = frozenset()) -> int:
    """Replace synthesized square-colour templates with real crops from a
    placement the game tracker has confirmed. Returns how many were saved."""
    if not SYNTHESIZED:
        return 0
    sym_to_name = {v: k for k, v in PIECE_NAMES.items()}
    saved = 0
    for row in range(8):
        for col in range(8):
            sym = positions[row][col]
            if sym is None or (row, col) in skip:
                continue
            shade = "light" if (row + col) % 2 == 0 else "dark"
            stem = f"{sym_to_name[sym]}_{shade}"
            if stem not in SYNTHESIZED:
                continue
            img = _square_image(screenshot, board, row, col)
            if img is None or img.shape[0] < 8 or img.shape[1] < 8:
                continue
            os.makedirs(TEMPLATE_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(TEMPLATE_DIR, stem + ".png"),
                        cv2.resize(img, (TEMPLATE_SIZE, TEMPLATE_SIZE)))
            SYNTHESIZED.discard(stem)
            saved += 1
    if saved:
        print(f"Learned {saved} real square-colour template(s) from play")
        reload_templates()
    return saved


def get_templates() -> list[tuple[str, str, np.ndarray]]:
    global _templates
    if _templates is None:
        _templates = _load_templates()
    return _templates


def reload_templates():
    """Force reload templates from disk."""
    global _templates
    _templates = None
    return get_templates()


def recognize_square(square_img: np.ndarray,
                     templates: list[tuple[str, str, np.ndarray]] | None = None,
                     ) -> str | None:
    """Identify the piece on a single square image.

    Returns:
        FEN piece character (e.g., 'K', 'p') or None for empty.
    """
    if templates is None:
        templates = get_templates()
    if not templates:
        return None

    square_resized = cv2.resize(square_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))

    best_score = -1
    best_fen = None

    for _base, fen_sym, tmpl in templates:
        result = cv2.matchTemplate(square_resized, tmpl, cv2.TM_CCOEFF_NORMED)
        score = result.max()
        if score > best_score:
            best_score = score
            best_fen = fen_sym

    if best_score >= MATCH_THRESHOLD and best_fen is not None:
        return best_fen
    return None


def _square_image(screenshot: np.ndarray, board: dict, row: int, col: int) -> np.ndarray | None:
    """Crop one square from the screenshot, clamped to image bounds."""
    sq = board["square_size"]
    img_h, img_w = screenshot.shape[:2]
    x = round(board["x"] + col * sq)
    y = round(board["y"] + row * sq)
    w = round(sq)
    h = round(sq)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    square_img = screenshot[y : y + h, x : x + w]
    return square_img if square_img.size > 0 else None


def _recover_missing_king(
    screenshot: np.ndarray, board: dict,
    positions: list[list[str | None]], king_sym: str,
) -> bool:
    """Place a missing king on the best-matching empty-looking square.

    A king can never actually leave the board, so if recognition lost one
    (check-glow, hover effects), rematch king templates over the squares
    that read as empty using a relaxed threshold. Squares already holding
    a recognized piece are never overridden.
    """
    king_templates = [t for t in get_templates() if t[1] == king_sym]
    if not king_templates:
        return False

    best_score = KING_RECOVERY_THRESHOLD
    best_rc: tuple[int, int] | None = None
    for row in range(8):
        for col in range(8):
            if positions[row][col] is not None:
                continue
            square_img = _square_image(screenshot, board, row, col)
            if square_img is None:
                continue
            square_resized = cv2.resize(square_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            for _base, _fen, tmpl in king_templates:
                score = cv2.matchTemplate(
                    square_resized, tmpl, cv2.TM_CCOEFF_NORMED
                ).max()
                if score > best_score:
                    best_score = score
                    best_rc = (row, col)

    if best_rc is not None:
        positions[best_rc[0]][best_rc[1]] = king_sym
        print(f"Recovered missing {king_sym!r} at row={best_rc[0]} col={best_rc[1]} (score {best_score:.2f})")
        return True
    return False


HIGHLIGHT_YELLOWNESS = 70.0


def _ring_color(img: np.ndarray) -> np.ndarray:
    """Median BGR of a square's outer ring — its background even with a
    piece in the middle."""
    h, w = img.shape[:2]
    k = max(2, int(min(h, w) * 0.12))
    ring = np.concatenate([
        img[:k].reshape(-1, 3), img[-k:].reshape(-1, 3),
        img[:, :k].reshape(-1, 3), img[:, -k:].reshape(-1, 3),
    ]).astype(np.float32)
    return np.median(ring, axis=0)


def last_move_highlight(screenshot: np.ndarray, board: dict) -> list[tuple[int, int]]:
    """Squares carrying chess.com's yellow last-move highlight.

    Sampled from the outer ring of each square so a piece in the middle
    doesn't matter. Yellowness = mean(R, G) - B: plain squares score
    under ~45 on the default green board, highlighted ones over ~90.
    Returns (row, col) pairs in screen orientation.
    """
    found = []
    for row in range(8):
        for col in range(8):
            img = _square_image(screenshot, board, row, col)
            if img is None or min(img.shape[:2]) < 8:
                continue
            b, g, r = _ring_color(img)
            if (r + g) / 2 - b > HIGHLIGHT_YELLOWNESS:
                found.append((row, col))
    return found


def recognize_board(screenshot: np.ndarray, board: dict) -> list[list[str | None]]:
    """Recognize all pieces on the board.

    Args:
        screenshot: Full screen BGR image.
        board: Board detection result dict.

    Returns:
        8x8 list, rows top-to-bottom, cols left-to-right.
        Each cell is a FEN piece char or None.
    """
    highlighted = set(last_move_highlight(screenshot, board))
    positions = []
    for row in range(8):
        rank = []
        for col in range(8):
            square_img = _square_image(screenshot, board, row, col)
            if square_img is None:
                rank.append(None)
                continue
            templates = None
            if (row, col) in highlighted:
                templates = _templates_for_background(_ring_color(square_img))
            rank.append(recognize_square(square_img, templates))
        positions.append(rank)

    # Kings can never be captured — if one wasn't recognized, try to
    # recover it with a relaxed match before the scan is used.
    flat = [p for rank in positions for p in rank]
    for king_sym in ("K", "k"):
        if king_sym not in flat:
            _recover_missing_king(screenshot, board, positions, king_sym)

    return positions


def board_to_fen(positions: list[list[str | None]], white_on_bottom: bool = True) -> str:
    """Convert 8x8 position array to FEN string."""
    fen_rows = []
    for row in positions:
        fen_row = ""
        empty = 0
        for piece in row:
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


def detect_orientation(positions: list[list[str | None]]) -> bool:
    """Detect whether white is on the bottom of the screen."""
    white_bottom = 0
    white_top = 0
    for col in range(8):
        for row in [6, 7]:
            p = positions[row][col]
            if p and p.isupper():
                white_bottom += 1
        for row in [0, 1]:
            p = positions[row][col]
            if p and p.isupper():
                white_top += 1
    return white_bottom >= white_top


def detect_player_color(positions: list[list[str | None]]) -> str | None:
    """Guess which color the player is from screen-bottom piece majority.

    On chess.com the player's pieces always start at the bottom, so a
    clear majority of one color in the bottom two rows (with the other
    color on top) identifies the player. Returns "w"/"b" only when both
    halves agree unambiguously, None otherwise. Only trustworthy near
    the start of a game — pieces invade the far side in endgames, so
    the result must be locked in once and never re-guessed mid-game.
    """
    w_bottom = b_bottom = w_top = b_top = 0
    for col in range(8):
        for row in (6, 7):
            p = positions[row][col]
            if p:
                if p.isupper():
                    w_bottom += 1
                else:
                    b_bottom += 1
        for row in (0, 1):
            p = positions[row][col]
            if p:
                if p.isupper():
                    w_top += 1
                else:
                    b_top += 1
    if w_bottom - b_bottom >= 4 and b_top - w_top >= 4:
        return "w"
    if b_bottom - w_bottom >= 4 and w_top - b_top >= 4:
        return "b"
    return None


def positions_to_fen(
    positions: list[list[str | None]],
    turn: str = "w",
    white_on_bottom: bool | None = None,
) -> str:
    """Convert recognized positions to a full FEN string.

    Pass `white_on_bottom` explicitly when the orientation is known (the
    player's pieces are always at the bottom on chess.com) — guessing it
    from piece placement flips in endgames when pieces invade the far side.
    """
    white_bottom = (
        detect_orientation(positions) if white_on_bottom is None
        else white_on_bottom
    )
    piece_placement = board_to_fen(positions, white_bottom)

    if not white_bottom:
        rows = piece_placement.split("/")
        rows = [row[::-1] for row in reversed(rows)]
        piece_placement = "/".join(rows)

    return f"{piece_placement} {turn} KQkq - 0 1"
