"""A small human opening repertoire.

Engines pick openings fresh every game; people play the same handful of
lines for years. The bot follows this book for its first moves while the
position stays in it, then hands over to the engine + human layer.

Each installation gets its own favourites: line weights are multiplied by
a stable per-user affinity (seeded from a file in the home directory), so
one user is an Italian-and-Caro-Kann player and another lives in the
Sicilian and the London. Set OPENINGS_SEED to pin it (self-play does).
"""

from __future__ import annotations

import os
import random

import chess

# (SAN line, weight). White lines and Black lines share the tree: at every
# node the side to move draws from the continuations the lines provide.
LINES: list[tuple[str, float]] = [
    # --- as White: 1.e4 ---
    ("e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O", 3),
    ("e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 c3 d6 O-O", 3),
    ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O", 2),
    ("e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3", 1),
    ("e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6 Nc3", 1),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2", 2),
    ("e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5", 2),
    ("e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3", 2),
    ("e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6", 1),
    ("e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3", 2),
    ("e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3", 1),
    ("e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 a3", 1),
    ("e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3", 2),
    ("e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3", 2),
    ("e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 Bf5 Bd2", 1),
    ("e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O", 1),
    ("e4 g6 d4 Bg7 Nc3 d6 Nf3 Nf6 Be2", 1),
    ("e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2", 1),
    # --- as White: 1.d4 ---
    ("d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4", 2),
    ("d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3", 1),
    ("d4 d5 c4 dxc4 e3 Nf6 Bxc4 e6 Nf3 c5 O-O", 1),
    ("d4 d5 Nf3 Nf6 Bf4 e6 e3 c5 c3 Nc6 Nbd2 Bd6 Bg3", 2),
    ("d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3", 1),
    ("d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O", 1),
    ("d4 Nf6 Nf3 e6 Bf4 d5 e3 c5 c3 Nc6 Nbd2", 2),
    ("d4 Nf6 Nf3 g6 Bf4 Bg7 e3 O-O Be2 d6 h3", 1),
    # --- as Black vs 1.e4 ---
    ("e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7", 1),
    ("e4 e5 Nc3 Nf6 Bc4 Nxe4 Qh5 Nd6 Bb3 Nc6 Nb5 g6", 1),
    ("e4 e5 Bc4 Nf6 d3 c6 Nf3 d5", 1),
    ("e4 e5 f4 exf4 Nf3 d6 d4 g5 h4 g4", 1),
    ("e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4", 1),
    ("e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6 c3 Nf6 Bf4 Bg4", 1),
    ("e4 c6 Nc3 d5 Nf3 Bg4 h3 Bxf3 Qxf3 e6", 1),
    # --- as Black vs 1.d4 and flank openings ---
    ("d4 d5 c4 e6 Nf3 Nf6 Nc3 Be7 Bg5 O-O e3 h6", 1),
    ("d4 d5 Nf3 Nf6 c4 e6 Nc3 Be7 Bf4 O-O e3 c5", 1),
    ("d4 d5 Bf4 Nf6 e3 e6 Nf3 c5 c3 Nc6 Nbd2 Bd6", 1),
    ("d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O", 1),
    ("c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6", 1),
    ("c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 e6", 1),
    ("Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O", 1),
    ("Nf3 Nf6 c4 e6 Nc3 d5 d4 Be7", 1),
]

MAX_BOOK_PLIES = 16
FOLLOW_PROBABILITY = 0.92   # occasionally a player forgets the line


def _affinity_rng() -> random.Random:
    seed = os.environ.get("OPENINGS_SEED")
    if seed is None:
        from paths import REPERTOIRE_FILE as path
        try:
            with open(path) as f:
                seed = f.read().strip()
        except OSError:
            seed = str(random.getrandbits(48))
            try:
                with open(path, "w") as f:
                    f.write(seed)
            except OSError:
                pass
    return random.Random(seed)


def _build() -> dict[str, list[tuple[str, float]]]:
    """placement+turn -> [(uci, weight)], accumulated over every line."""
    rng = _affinity_rng()
    book: dict[str, dict[str, float]] = {}
    for line, weight in LINES:
        weight *= rng.uniform(0.4, 2.0)  # this installation's taste
        board = chess.Board()
        for san in line.split():
            key = f"{board.board_fen()} {'w' if board.turn else 'b'}"
            try:
                move = board.parse_san(san)
            except ValueError as e:
                raise ValueError(f"bad book line {line!r} at {san}: {e}") from e
            node = book.setdefault(key, {})
            node[move.uci()] = node.get(move.uci(), 0.0) + weight
            board.push(move)
    return {k: list(v.items()) for k, v in book.items()}


_BOOK = _build()


def _key(board: chess.Board) -> str:
    return f"{board.board_fen()} {'w' if board.turn else 'b'}"


def in_book(board: chess.Board) -> bool:
    return board.ply() < MAX_BOOK_PLIES and _key(board) in _BOOK


def book_move(board: chess.Board, rng: random.Random | None = None) -> str | None:
    """A repertoire move for this position, or None when out of book."""
    if not in_book(board):
        return None
    r = rng or random
    if r.random() > FOLLOW_PROBABILITY:
        return None
    options = _BOOK[_key(board)]
    moves, weights = zip(*options)
    return r.choices(moves, weights=weights, k=1)[0]
