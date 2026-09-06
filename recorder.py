"""Records every live game for later analysis.

One JSON line per event (suggestion, tracked move, resync, result) is
appended as it happens, so a crash loses nothing, and a PGN is written
when the game ends. Files land in live_games/<timestamp>_<colour>.{jsonl,pgn}.
"""

from __future__ import annotations

import json
import os
import time

import chess
import chess.pgn

from paths import LIVE_GAMES_DIR as LIVE_DIR


class GameRecorder:
    def __init__(self):
        self._f = None
        self._path: str | None = None
        self._moves: list[str] = []
        self._meta: dict = {}
        self._resyncs = 0
        self._pgn: str | None = None
        # Called with summary() when a game ends (main.py hooks the uploader
        # and adds the move selector's per-game stats through `stats_fn`).
        self.on_finish = None
        self.stats_fn = None

    def start(self, player_color: str | None, target_elo: int) -> None:
        self.finish(None)
        os.makedirs(LIVE_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        colour = {"w": "white", "b": "black"}.get(player_color or "", "auto")
        self._path = os.path.join(LIVE_DIR, f"{stamp}_{colour}")
        self._f = open(self._path + ".jsonl", "a")
        self._moves = []
        self._resyncs = 0
        self._meta = {"started": time.time(), "color": player_color,
                      "target_elo": target_elo, "opp_rating": None}
        self._emit(ev="start", **self._meta)

    def _emit(self, **ev) -> None:
        if self._f is None:
            return
        ev["t"] = round(time.time(), 3)
        self._f.write(json.dumps(ev) + "\n")
        self._f.flush()

    def set_color(self, color: str) -> None:
        self._meta["color"] = color
        self._emit(ev="color", color=color)

    def opponent_rating(self, rating: int) -> None:
        self._meta["opp_rating"] = rating
        self._emit(ev="opp_rating", rating=rating)

    def suggestion(self, fen: str, decision: dict, think_s: float | None) -> None:
        self._emit(ev="suggest", fen=fen, think=think_s, **decision)

    def moves(self, board: chess.Board, moves: list[chess.Move],
              player_color: str | None) -> None:
        """Moves just tracked (board is already advanced past them)."""
        tmp = board.copy()
        for _ in moves:
            tmp.pop()
        for mv in moves:
            mover = "w" if tmp.turn == chess.WHITE else "b"
            san = tmp.san(mv)
            tmp.push(mv)
            self._moves.append(mv.uci())
            self._emit(ev="move", uci=mv.uci(), san=san, by=mover,
                       mine=(mover == player_color), fen=tmp.fen())

    def resync(self, fen_position: str, turn: str) -> None:
        self._resyncs += 1
        self._emit(ev="resync", placement=fen_position, turn=turn)

    def finish(self, result: str | None) -> None:
        if self._f is None:
            return
        self._emit(ev="end", result=result, moves=len(self._moves),
                   resyncs=self._resyncs)
        self._f.close()
        self._f = None
        self._meta["result"] = result
        self._meta["ended"] = time.time()
        self._pgn = None
        try:
            self._pgn = self._write_pgn(result)
        except Exception as e:
            print(f"PGN write failed: {e}")
        if self.on_finish is not None:
            try:
                self.on_finish(self.summary())
            except Exception as e:
                print(f"game summary hook failed: {e}")

    def summary(self) -> dict:
        """Everything the website shows about a game (see stats.py)."""
        colour = self._meta.get("color")
        result = self._meta.get("result")
        score = None
        if result in ("1-0", "0-1") and colour:
            score = 1.0 if (result == "1-0") == (colour == "w") else 0.0
        elif result == "1/2-1/2":
            score = 0.5
        out = {**self._meta, "score": score, "plies": len(self._moves),
               "resyncs": self._resyncs, "pgn": self._pgn}
        if self.stats_fn is not None:
            try:
                out.update(self.stats_fn() or {})
            except Exception as e:
                print(f"game stats failed: {e}")
        return out

    def _write_pgn(self, result: str | None) -> str:
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        for uci in self._moves:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                break  # a resync happened; the PGN stops where continuity did
            node = node.add_variation(mv)
            board.push(mv)
        colour = self._meta.get("color")
        me = f"Me (bot {self._meta.get('target_elo')})"
        opp = f"Opponent ({self._meta.get('opp_rating') or '?'})"
        game.headers["Event"] = "Chess Vision live game"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["White"] = me if colour == "w" else opp
        game.headers["Black"] = opp if colour == "w" else me
        game.headers["Result"] = result or "*"
        if self._resyncs:
            game.headers["Annotator"] = f"{self._resyncs} resync(s); moves may be incomplete"
        text = str(game)
        with open(self._path + ".pgn", "w") as f:
            f.write(text + "\n")
        print(f"Game saved: {self._path}.pgn")
        return text
