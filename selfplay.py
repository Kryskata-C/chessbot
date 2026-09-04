"""Self-play tuning harness: the human-imitation bot vs a rating-capped
Stockfish. Prints a per-move decision line and a per-game error profile,
saves PGNs to selfplay_pgns/, and streams JSON events to selfplay_runs/
for the live dashboard (dashboard.py).

    ./venv/bin/python selfplay.py --games 10 --target-elo 1600 --opp-elo 1600

By default the bot learns the opponent's strength from their moves, seeded
with a printed-rating prior exactly like the live app (--opp-prior overrides
the prior, --no-adapt pins the opponent rating instead).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import chess
import chess.pgn
from stockfish import Stockfish

# Self-play uses one fixed repertoire taste so runs are comparable.
os.environ.setdefault("OPENINGS_SEED", "selfplay")

from engine import ChessEngine, find_stockfish
from elo_estimator import EloEstimator, blend_opponent_elo
from move_selector import HumanMoveSelector, _SAC_NET
from session import SessionGovernor


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


class EventLog:
    """Append-only JSON lines the dashboard tails."""

    def __init__(self, path: str | None):
        self.path = path
        self._f = open(path, "a") if path else None

    def emit(self, **ev) -> None:
        if self._f is None:
            return
        ev["t"] = round(time.time(), 3)
        self._f.write(json.dumps(ev) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()


def white_pov(cp: int, side_is_white: bool) -> int:
    return cp if side_is_white else -cp


def play_game(
    selector: HumanMoveSelector,
    opponent: Stockfish,
    engine: ChessEngine,
    bot_is_white: bool,
    move_time_ms: int,
    max_plies: int,
    events: EventLog,
    game_index: int,
    adapt: bool,
    prior: int | None,
    start_fen: str | None = None,
    blunder_p: float = 0.0,
) -> dict:
    """Play one full game; returns stats plus the finished board."""
    board = chess.Board(start_fen) if start_fen else chess.Board()
    selector.reset()
    estimator = EloEstimator()
    if adapt:
        selector.set_opponent_elo(prior, 12 if prior is not None else 0)

    # Count sacrifice declines by watching the filter shrink its input,
    # and collect per-move (loss, best_eval) pairs for real ACPL numbers
    # (the selector's own EMA over-weights the endgame).
    stats = {"declined": 0, "sacs_played": [], "plies": 0, "losses": [],
             "opp_losses": []}
    orig_filter = selector._filter_sacrifices
    orig_record = selector._record

    def counting_filter(b, top):
        out = orig_filter(b, top)
        if len(out) < len(top):
            stats["declined"] += 1
        return out

    def recording_record(top_moves, chosen, loss):
        best = selector.last_best_eval
        stats["losses"].append((loss, best if best is not None else 0))
        orig_record(top_moves, chosen, loss)

    selector._filter_sacrifices = counting_filter
    selector._record = recording_record
    last_bot_eval: int | None = None  # bot POV, after the bot's move
    try:
        while not board.is_game_over(claim_draw=True) and stats["plies"] < max_plies:
            bot_to_move = board.turn == (chess.WHITE if bot_is_white else chess.BLACK)
            if bot_to_move:
                piece_count = len(board.piece_map())
                uci = selector.select_move(board.fen(), piece_count)
                if uci is None:
                    break
                move = chess.Move.from_uci(uci)
                d = dict(selector.last_decision) if selector.last_decision else {}
                sac = None
                if selector._move_material_net(board, move) <= _SAC_NET:
                    ev = selector.last_best_eval
                    forced = (selector.last_criticality >= 0.6
                              and d.get("best") == uci)
                    kind = ("conversion" if ev is not None and ev >= 400
                            else "forced" if forced else "SUSPICIOUS")
                    sac = f"{board.fullmove_number}. {board.san(move)} [{kind}, eval {ev}]"
                    stats["sacs_played"].append(sac)
                think = selector.suggest_think_time(uci, piece_count)
                san = board.san(move)
                board.push(move)
                stats["plies"] += 1
                last_bot_eval = d.get("chosen_eval", d.get("best_eval"))
                events.emit(
                    ev="move", g=game_index, ply=stats["plies"], by="bot",
                    san=san, uci=uci, fen=board.fen(),
                    eval_w=white_pov(last_bot_eval, bot_is_white) if last_bot_eval is not None else None,
                    think=round(think, 1), sac=sac, **{k: v for k, v in d.items()},
                )
            else:
                opponent.set_fen_position(board.fen())
                uci = opponent.get_best_move_time(move_time_ms)
                if uci is None:
                    break
                if blunder_p > 0 and random.random() < blunder_p:
                    # A human-style lapse: any legal move, hung pieces included
                    uci = random.choice(list(board.legal_moves)).uci()
                move = chess.Move.from_uci(uci)
                san = board.san(move)
                # Opponent's centipawn loss, the way the live app measures it
                if last_bot_eval is not None:
                    eval_before = -last_bot_eval  # opponent POV before their move
                else:
                    eval_before = engine.get_evaluation(board.fen(), depth=10)
                board.push(move)
                stats["plies"] += 1
                eval_after = engine.get_evaluation(board.fen(), depth=10)  # bot POV
                cpl = max(0, eval_before + eval_after)
                stats["opp_losses"].append(cpl)
                estimator.record_move(cpl, eval_before)
                opp_est = None
                if adapt:
                    observed = estimator.get_estimate()
                    n = estimator.get_move_count()
                    opp_est = blend_opponent_elo(prior, observed, n)
                    selector.set_opponent_elo(opp_est, max(n, 12) if prior is not None else n)
                events.emit(
                    ev="move", g=game_index, ply=stats["plies"], by="opp",
                    san=san, uci=uci, fen=board.fen(),
                    eval_w=white_pov(eval_after, bot_is_white),
                    loss=int(cpl), opp_est=opp_est,
                )
                last_bot_eval = None
    finally:
        selector._filter_sacrifices = orig_filter
        selector._record = orig_record

    stats["board"] = board
    stats["result"] = board.result(claim_draw=True)
    stats["opp_est"] = selector._opponent_elo if adapt else None
    return stats


def loss_profile(losses: list[tuple[int, int]]) -> str:
    """Human-readable error profile: overall and contested-phase ACPL,
    plus how many real mistakes (>=100cp) were made."""
    if not losses:
        return "no data"
    s = profile_stats(losses)
    return (f"ACPL {s['acpl']:.0f} (contested {s['c_acpl']:.0f} over "
            f"{s['contested']} moves), mistakes>=100cp: {s['mistakes']}, "
            f"worst {s['worst']}cp")


def profile_stats(losses: list[tuple[int, int]]) -> dict:
    all_l = [l for l, _ in losses]
    contested = [l for l, best in losses if abs(best) < 300]
    return {
        "acpl": sum(all_l) / len(all_l) if all_l else 0.0,
        "c_acpl": sum(contested) / len(contested) if contested else 0.0,
        "contested": len(contested),
        "mistakes": sum(1 for l in all_l if l >= 100),
        "worst": max(all_l) if all_l else 0,
        "moves": len(all_l),
    }


def bot_score(result: str, bot_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    won_as_white = result == "1-0"
    return 1.0 if won_as_white == bot_is_white else 0.0


def save_pgn(board: chess.Board, path: str, bot_is_white: bool,
             target_elo: int, opp_elo: int) -> None:
    game = chess.pgn.Game.from_board(board)  # keeps a FEN header for set-up starts
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
    ap.add_argument("--opp-prior", type=int, default=None,
                    help="rating 'printed next to the opponent' (default: --opp-elo)")
    ap.add_argument("--no-adapt", action="store_true",
                    help="pin the opponent rating instead of learning it")
    ap.add_argument("--depth", type=int, default=12,
                    help="bot's Stockfish search depth")
    ap.add_argument("--move-time", type=int, default=150,
                    help="opponent think time per move (ms)")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--pgn-dir", default="selfplay_pgns")
    ap.add_argument("--runs-dir", default="selfplay_runs")
    ap.add_argument("--label", default="", help="name shown on the dashboard")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--session", action="store_true",
                    help="enable the cross-game governor (throwaway state per run)")
    ap.add_argument("--no-book", action="store_true", help="disable the opening book")
    ap.add_argument("--opp-blunder", type=float, default=0.0,
                    help="probability per opponent move of a random legal move (a blundering human)")
    ap.add_argument("--start-fens", default=None,
                    help="file of FENs, one per line; game i starts from FEN i (cycling), bot = side to move")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    os.makedirs(args.pgn_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    adapt = not args.no_adapt
    prior = args.opp_prior if args.opp_prior is not None else args.opp_elo
    run_id = f"run_{int(time.time() * 1000)}_{os.getpid()}"
    events = EventLog(os.path.join(args.runs_dir, run_id + ".jsonl"))
    events.emit(ev="run", id=run_id, label=args.label or run_id,
                target=args.target_elo, opp=args.opp_elo, prior=prior,
                adapt=adapt, games=args.games, depth=args.depth)

    engine = ChessEngine(depth=args.depth, threads=2)
    selector = HumanMoveSelector(engine)
    selector.set_target_elo(args.target_elo)
    selector.use_book = not args.no_book
    governor = SessionGovernor(
        os.path.join(args.runs_dir, run_id + ".session.json") if args.session else None)
    opponent = make_opponent(args.opp_elo)

    start_fens = None
    if args.start_fens:
        with open(args.start_fens) as f:
            start_fens = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    total, declined, sacs = 0.0, 0, []
    for i in range(args.games):
        bot_is_white = i % 2 == 0
        start_fen = None
        if start_fens:
            start_fen = start_fens[i % len(start_fens)]
            bot_is_white = chess.Board(start_fen).turn == chess.WHITE
        color = "White" if bot_is_white else "Black"
        print(f"\n=== Game {i + 1}/{args.games}: bot is {color}, "
              f"target {args.target_elo} vs SF {args.opp_elo} ===")
        if not adapt:
            selector.set_opponent_elo(args.opp_elo, moves=12)
        selector.session_temp_mult = governor.temp_mult
        selector.session_edge_shift = governor.edge_shift
        events.emit(ev="game_start", g=i, bot_white=bot_is_white,
                    temp_mult=governor.temp_mult, edge_shift=governor.edge_shift)
        t0 = time.time()
        g = play_game(selector, opponent, engine, bot_is_white,
                      args.move_time, args.max_plies, events, i, adapt, prior,
                      start_fen, args.opp_blunder)
        score = bot_score(g["result"], bot_is_white)
        total += score
        declined += g["declined"]
        sacs.extend(g["sacs_played"])
        pgn_path = os.path.join(args.pgn_dir, f"selfplay_{int(time.time())}_{i+1}.pgn")
        save_pgn(g["board"], pgn_path, bot_is_white,
                 args.target_elo, args.opp_elo)
        acc = selector.get_accuracy() or 0.0
        ps = profile_stats(g["losses"])
        print(f"Result: {g['result']}  (bot {'won' if score == 1 else 'drew' if score == 0.5 else 'lost'})  "
              f"{g['plies']} plies in {time.time() - t0:.0f}s")
        print(f"  {loss_profile(g['losses'])}  best-move {acc:.0f}%  "
              f"sacs declined {g['declined']}  sacs played {g['sacs_played'] or 'none'}")
        print(f"  PGN: {pgn_path}")
        gov_line = governor.record_game(acc, selector.get_contested_cpl(), score)
        if governor.enabled:
            print(f"  {gov_line}")
        events.emit(
            ev="game", g=i, bot_white=bot_is_white, result=g["result"],
            score=score, plies=g["plies"], secs=round(time.time() - t0, 1),
            best_pct=round(acc, 1), sacs=g["sacs_played"], declined=g["declined"],
            opp_est=g["opp_est"], eff=selector.get_effective_elo(),
            realized=selector.get_realized_elo(), pgn=pgn_path,
            governor=gov_line,
            losses=[l for l, _ in g["losses"]], opp_acpl=(
                sum(g["opp_losses"]) / len(g["opp_losses"]) if g["opp_losses"] else None),
            **ps,
        )

    print(f"\n=== Summary: bot scored {total}/{args.games} vs SF {args.opp_elo}  "
          f"| sacrifices declined {declined}, played {len(sacs)} "
          f"{sacs if sacs else ''}")
    events.emit(ev="done", score=total, games=args.games,
                declined=declined, sacs=sacs)
    events.close()


if __name__ == "__main__":
    main()
