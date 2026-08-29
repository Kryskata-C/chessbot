"""Headless self-play testing: the bot vs a strength-limited Stockfish.

Runs full games with the real HumanMoveSelector (same code path the live
bot uses) against a Stockfish opponent capped at a chosen ELO, without any
screen capture or chess.com. Reports results, ACPL, realized ELO, and —
the reason this exists — how many material sacrifices were declined vs
played, so brilliancy leaks show up here instead of in a live game.

Usage:
    python selfplay.py --games 2 --target-elo 1600 --opp-elo 1600
    python selfplay.py --games 4 --opp-elo 1800 --move-time 100

PGNs are saved to selfplay_pgns/ for pasting into chess.com analysis.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import chess
import chess.pgn
from stockfish import Stockfish

from engine import ChessEngine, find_stockfish
from move_selector import HumanMoveSelector, _SAC_NET


def make_opponent(elo: int, threads: int = 1) -> Stockfish:
    """Stockfish capped at a UCI ELO (min 1320 in modern builds)."""
    return Stockfish(
        path=find_stockfish(),
        depth=12,
        parameters={
            "Threads": threads,
            "Hash": 64,
            "UCI_LimitStrength": True,
            "UCI_Elo": max(1320, min(3190, elo)),
        },
    )


def play_game(
    selector: HumanMoveSelector,
    opponent: Stockfish,
    bot_is_white: bool,
    move_time_ms: int,
    max_plies: int,
) -> dict:
    """Play one full game; returns stats plus the finished board."""
    board = chess.Board()
    selector.reset()

    # Count sacrifice declines by watching the filter shrink its input.
    stats = {"declined": 0, "sacs_played": [], "plies": 0}
    orig_filter = selector._filter_sacrifices

    def counting_filter(b, top):
        out = orig_filter(b, top)
        if len(out) < len(top):
            stats["declined"] += 1
        return out

    selector._filter_sacrifices = counting_filter
    try:
        while not board.is_game_over(claim_draw=True) and stats["plies"] < max_plies:
            if board.turn == (chess.WHITE if bot_is_white else chess.BLACK):
                piece_count = len(board.piece_map())
                uci = selector.select_move(board.fen(), piece_count)
                if uci is None:
                    break
                move = chess.Move.from_uci(uci)
                if selector._move_material_net(board, move) <= _SAC_NET:
                    stats["sacs_played"].append(
                        f"{board.fullmove_number}. {board.san(move)}"
                    )
            else:
                opponent.set_fen_position(board.fen())
                uci = opponent.get_best_move_time(move_time_ms)
                if uci is None:
                    break
                move = chess.Move.from_uci(uci)
            board.push(move)
            stats["plies"] += 1
    finally:
        selector._filter_sacrifices = orig_filter

    stats["board"] = board
    stats["result"] = board.result(claim_draw=True)
    return stats


def bot_score(result: str, bot_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    won_as_white = result == "1-0"
    return 1.0 if won_as_white == bot_is_white else 0.0


def save_pgn(board: chess.Board, path: str, bot_is_white: bool,
             target_elo: int, opp_elo: int) -> None:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Selfplay test"
    game.headers["White"] = f"Bot({target_elo})" if bot_is_white else f"SF({opp_elo})"
    game.headers["Black"] = f"SF({opp_elo})" if bot_is_white else f"Bot({target_elo})"
    game.headers["Result"] = board.result(claim_draw=True)
    with open(path, "w") as f:
        print(game, file=f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--target-elo", type=int, default=1600)
    ap.add_argument("--opp-elo", type=int, default=1600)
    ap.add_argument("--depth", type=int, default=12,
                    help="bot engine search depth (12 = live default)")
    ap.add_argument("--move-time", type=int, default=150,
                    help="opponent think time per move (ms)")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--pgn-dir", default="selfplay_pgns")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    os.makedirs(args.pgn_dir, exist_ok=True)

    engine = ChessEngine(depth=args.depth, threads=2)
    selector = HumanMoveSelector(engine)
    selector.set_target_elo(args.target_elo)
    opponent = make_opponent(args.opp_elo)

    total, declined, sacs = 0.0, 0, []
    for i in range(args.games):
        bot_is_white = i % 2 == 0
        color = "White" if bot_is_white else "Black"
        print(f"\n=== Game {i + 1}/{args.games}: bot is {color}, "
              f"target {args.target_elo} vs SF {args.opp_elo} ===")
        # The live bot learns the opponent over the game; here we tell it.
        selector.set_opponent_elo(args.opp_elo, moves=12)
        t0 = time.time()
        g = play_game(selector, opponent, bot_is_white,
                      args.move_time, args.max_plies)
        score = bot_score(g["result"], bot_is_white)
        total += score
        declined += g["declined"]
        sacs.extend(g["sacs_played"])
        pgn_path = os.path.join(args.pgn_dir, f"selfplay_{int(time.time())}_{i+1}.pgn")
        save_pgn(g["board"], pgn_path, bot_is_white,
                 args.target_elo, args.opp_elo)
        acpl = selector.get_avg_cpl()
        acc = selector.get_accuracy()
        realized = selector.get_realized_elo()
        print(f"Result: {g['result']}  (bot {'won' if score == 1 else 'drew' if score == 0.5 else 'lost'})  "
              f"{g['plies']} plies in {time.time() - t0:.0f}s")
        print(f"  ACPL {acpl:.0f}  accuracy {acc:.0f}%  realized ELO {realized}  "
              f"sacs declined {g['declined']}  sacs played {g['sacs_played'] or 'none'}")
        print(f"  PGN: {pgn_path}")

    print(f"\n=== Summary: bot scored {total}/{args.games} vs SF {args.opp_elo}  "
          f"| sacrifices declined {declined}, played {len(sacs)} "
          f"{sacs if sacs else ''}")


if __name__ == "__main__":
    main()
