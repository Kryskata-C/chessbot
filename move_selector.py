"""Human-like move selection driven by a target ELO.

The user sets a target strength before the game. Everything here works to
make the *stream of suggested moves* look like a real player of that
strength — realistic mistakes, no robotic perfection, and above all no
40-move domination when a normal human would just win a normal game.

Four coupled mechanisms:

1. **ELO -> error budget.** Target ELO maps to a target average centipawn
   loss (ACPL). That sets the base "temperature" of a softmax over engine
   candidates, plus how many candidates we even consider (weak players
   weigh more, worse moves; strong players tunnel on the best few).

2. **Human-plausibility prior.** Every candidate is re-weighted by how a
   human would *feel* about it: captures / checks / forward moves are
   tempting; quiet moves, retreats, and moves that give up material (the
   engine's brilliant sacrifices, judged by static exchange) are
   under-selected. Scaled by weakness, so the inserted error lands exactly
   where a human misses — never as an obvious out-of-nowhere blunder.
   On top of that, a sacrifice filter removes engine brilliancies from the
   candidate list outright — before the criticality machinery can declare
   them "forced" — unless the imitated level would plausibly see them or
   the sacrifice is the only move that keeps the game alive.

3. **Anti-domination governor.** Each game samples a target winning margin
   (so results vary: sometimes close, sometimes comfortable). When we are
   crushing beyond it, we ease off — pick sound-but-not-crushing moves that
   keep the win while letting the margin breathe — with a hard floor so a
   won game is never thrown.

4. **Closed-loop ELO controller.** The bot's own move losses feed a live
   "realized ELO". A slow controller nudges the error budget until realized
   ELO matches the target the user asked for.

5. **Opponent adaptation.** The menu ELO is only a prior. As the opponent's
   own moves reveal their strength, the *effective* ELO we imitate drifts
   toward "a bit better than them" (the edge is sampled per game, so some
   games are close and some comfortable). On top of that, every move asks
   the human question — "do I have to play a good move here, or am I ahead
   enough against *this* opponent to afford a realistic small slip?" — via
   a risk appetite built from the eval cushion and the opponent's strength.
   The goal is to beat the opponent the way a slightly better human would,
   not the way an engine would.
"""

from __future__ import annotations

import math
import random

import chess

from engine import ChessEngine
from elo_estimator import effective_loss, decided_factor, EloEstimator, elo_to_acpl, acpl_to_elo
from openings import book_move

# Piece values in centipawns, for judging captures and apparent hangs.
_PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Mate scores arrive as 100000 - 100*N (N = moves to mate), so anything
# at or above this threshold is "mate is on the board".
_MATE_CP = 90000
_WINNING_CP = 150   # eval above this = "we are winning"
_SAC_NET = -160        # net material loss (cp) that marks a move as a sacrifice
_SAC_KEEP_FLOOR = -80  # declining a sacrifice must leave at least this eval


class HumanMoveSelector:
    """Selects moves that mimic a human of a chosen ELO while staying sound."""

    DEFAULT_TARGET_ELO = 1400
    NUM_CANDIDATES_MAX = 16
    TREND_WINDOW = 6

    def __init__(self, engine: ChessEngine):
        self.engine = engine
        self._target_elo: int = self.DEFAULT_TARGET_ELO
        self._opponent_elo: int | None = None
        self._opponent_moves: int = 0
        # How far above the opponent we aim to play this game (ELO). Sampled
        # per game so results vary like a real slightly-better player's.
        self._opp_edge: int = self._sample_opp_edge()
        # Smoothed effective ELO: our read on the opponent firms up over
        # moves rather than lurching with every noisy estimate update.
        self._form: int = 0
        self._eff_elo: float = float(self.DEFAULT_TARGET_ELO)

        # Per-game state
        self._eval_history: list[int] = []
        self._move_number: int = 0
        self._consecutive_best: int = 0
        # The bot's own recent moves, for move-to-move coherence (humans
        # don't shuffle a rook out and back, or re-move one piece aimlessly).
        self._recent_moves: list[chess.Move] = []
        # Placements already reached on our turns: a player with an edge
        # avoids repeating positions (repetition = draw = losing the edge).
        self._seen_placements: dict[str, int] = {}

        # Anti-domination governor: the winning margin (cp) we try to
        # plateau at this game. Resampled each game for variety.
        self._target_margin: int = 250
        # Conversion clock: consecutive own moves spent in a winning
        # position. A human plateaus for a while, then converts — the bot
        # used to coast at +3 for 50 moves and grind K+B+P endgames.
        self._won_for: int = 0
        self._mate_misses: int = 0  # consecutive "didn't see the mate" moves

        # Closed-loop controller: multiplies the base error budget so the
        # realized ELO converges to the target.
        self._temp_gain: float = 1.0
        self._realized = EloEstimator()

        # Display / accuracy tracking
        self._total_moves: int = 0
        self._best_move_hits: int = 0
        self._total_cpl: float = 0.0
        self._contested_moves: int = 0
        self._contested_cpl: float = 0.0

        # Last-analysis data exposed to the overlay visualizer
        self.last_top_moves: list[dict] = []
        self.last_criticality: float = 0.0
        self.last_best_eval: int | None = None
        self.last_cushion: int = 0
        self.last_decision: dict = {}
        self._piece_count: int = 32
        # Session governor levers (see session.py); neutral by default
        self.session_temp_mult: float = 1.0
        self.session_edge_shift: int = 0
        self.use_book: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target_elo(self, elo: int) -> None:
        """Set the strength the suggestions should imitate."""
        self._target_elo = int(max(300, min(3000, elo)))
        self._eff_elo = float(self._target_elo)

    def get_target_elo(self) -> int:
        return self._target_elo

    def set_opponent_elo(self, elo: int | None, moves: int | None = None) -> None:
        """Update estimated opponent strength and how many of their moves it
        rests on (confidence). Drives the effective ELO and risk appetite."""
        self._opponent_elo = elo
        if moves is not None:
            self._opponent_moves = moves
        elif elo is not None:
            self._opponent_moves = max(self._opponent_moves, 12)  # trust it

    def get_effective_elo(self) -> int:
        """The strength actually being imitated right now (target adapted
        toward the opponent)."""
        return self._effective_elo()

    def get_realized_elo(self) -> int | None:
        """The bot's own live ELO, estimated from the moves it has picked."""
        return self._realized.get_estimate()

    def select_move(self, fen: str, piece_count: int) -> str | None:
        """Pick a human-like move. Returns a UCI string, or None if none."""
        # Thin boards get a deeper search: at depth 12 a won +4 endgame
        # shows no plan and the bot shuffles for 30 moves playing "best"
        # moves; at depth 18-22 the conversion (and the short mate) is
        # visible, and few pieces make it cheap.
        # (Depth 22 with 10 candidate lines took 40s in K+R vs K — MultiPV
        # keeps searching every line to depth; 18 is ~0.1-0.7s there.)
        depth = 18 if piece_count <= 8 else 16 if piece_count <= 14 else None
        top_moves = self.engine.get_top_moves(fen, self._num_candidates(), depth=depth)
        if not top_moves:
            self.last_top_moves = []
            self.last_criticality = 0.0
            self.last_best_eval = None
            return self.engine.get_best_move(fen)

        self._piece_count = piece_count
        board = self._safe_board(fen)
        booked = self._book_choice(board, top_moves, piece_count)
        if booked is not None:
            return booked
        top_moves = self._filter_sacrifices(board, top_moves)
        self.last_top_moves = top_moves

        best_eval = top_moves[0]["eval"]
        self.last_best_eval = best_eval
        self.last_criticality = self._compute_criticality(top_moves)
        self._update_effective_elo()

        if len(top_moves) == 1:
            self._record(top_moves, top_moves[0]["move"], 0)
            return top_moves[0]["move"]

        self._eval_history.append(best_eval)
        if best_eval >= _WINNING_CP:
            self._won_for += 1
        elif best_eval < 100:
            self._won_for = 0

        # Mate on the board: a human who sees it plays it. Only mating
        # moves are considered, the shortest strongly preferred (a mate in
        # 3 instead of 2 is human; a rook shuffle with mate available is
        # not). Loss is recorded as 0: a slower mate is not an error.
        if best_eval >= _MATE_CP:
            mating = [m for m in top_moves if m["eval"] >= _MATE_CP]
            mate_in = max(1, round((100000 - best_eval) / 100))
            pool = list(mating)
            # Q+R vs K took 16 moves from mate-in-13 with a flat temp of
            # 120: short mates are found and played sharply, long ones may
            # still be "missed" — but not move after move (a player who
            # has been looking for the mate for two moves finds it).
            mate_temp = 60.0 if mate_in <= 6 else 120.0
            if mate_in >= 4 and self._mate_misses < 2:
                # A club player sees mate-in-1/2 every time; a longer mate
                # is often "missed" for a crushing ordinary move (chess.com
                # marks it a miss, exactly the human texture we want).
                # Such moves enter the softmax with a pseudo-loss that grows
                # with playing strength and shrinks with mate length.
                miss = (150.0 + 30.0 * mate_in + 0.3 * max(0.0, self._effective_elo() - 1500)
                        + 120.0 * self._mate_misses)
                for m in top_moves:
                    if m["eval"] < _MATE_CP and m["eval"] >= 700:
                        pool.append({**m, "eval": int(best_eval - miss), "_miss": m["eval"]})
            chosen = self._weighted_select(
                board, pool, best_eval, mate_temp, False, 400.0, raw_loss=True
            )
            cushion = self._cushion(best_eval)
            self.last_cushion = cushion
            if any(m["move"] == chosen and "_miss" in m for m in pool):
                self._mate_misses += 1
                print(f"  [miss] mate in {mate_in} not seen — played {chosen} instead")
            else:
                self._mate_misses = 0
            self._record(top_moves, chosen, 0)
            self._run_controller()
            self._log(chosen, top_moves, 0, mate_temp, False, cushion)
            return chosen

        # Decisive material (a piece up in a bare endgame, or +15): every
        # human converts this, and the engine's line is the human line.
        # Near-best play with a little slack — never Bh1-Be4 shuffles at
        # +50 waiting for something to happen.
        if best_eval >= 2500 or (best_eval >= 900 and piece_count <= 8):
            pool = [m for m in top_moves if best_eval - m["eval"] <= 150] or top_moves[:1]
            chosen = self._weighted_select(
                board, pool, best_eval, 60.0, False, 150.0, raw_loss=True
            )
            cushion = self._cushion(best_eval)
            self.last_cushion = cushion
            chosen_eval = next((m["eval"] for m in pool if m["move"] == chosen), best_eval)
            loss = max(0, best_eval - chosen_eval)
            self._record(top_moves, chosen, loss)
            self._run_controller()
            self._log(chosen, top_moves, loss, 60.0, False, cushion)
            return chosen

        # How far this position is even allowed to deviate. Forcing positions
        # (recaptures, tactics) collapse this toward zero so the obvious move
        # gets played; quiet positions leave room for human imprecision.
        ceiling = self._loss_ceiling(self.last_criticality)
        # ...then ask the human question: is there room for a slip against
        # this opponent, or does this position need a good move?
        cushion = self._cushion(best_eval)
        self.last_cushion = cushion
        # Attention lapse: real games are decided by the occasional genuine
        # blunder in an equal position — without a heavy tail the error
        # stream is suspiciously uniform (six games of never losing more
        # than ~150cp on a move). During a lapse the guardrails open up.
        lapse = self._roll_lapse(best_eval)
        if lapse:
            # A slip's size is bounded by what the lead can absorb: in an
            # equal game a lapse is an inaccuracy, not a hung piece — the
            # game must stay winnable-looking, not thrown.
            affordable = max(90.0, cushion + 90.0)
            ceiling = max(ceiling, min(self._lapse_ceiling(), affordable))
        else:
            ceiling = self._bound_ceiling_by_cushion(ceiling, cushion)

        # Optionally surface a tempting-but-bad move (the classic human
        # blunder) so weak play isn't capped at the engine's Nth-best.
        pool = self._augment_with_temptations(
            fen, top_moves, best_eval, ceiling, force=lapse
        )

        temperature, coasting = self._compute_temperature(best_eval, piece_count)
        if lapse:
            temperature *= 2.5

        chosen = self._weighted_select(
            board, pool, best_eval, temperature, coasting, ceiling
        )

        # A lapse that finds a plausible-looking mistake commits to it:
        # the whole point of the heavy tail is that sometimes the human
        # plays the move without seeing the refutation. Selection among
        # the lemons is prior-driven (huge temperature mutes the loss
        # term), so the mistake still looks like one a person would make.
        if lapse:
            lemons = [m for m in pool
                      if 60 <= best_eval - m["eval"] <= ceiling]
            if lemons and random.random() < 0.6:
                chosen = self._weighted_select(
                    board, lemons, best_eval, 1e9, False, ceiling
                )
                print("  [lapse] attention slip — played "
                      f"{chosen} without seeing the refutation")

        chosen_eval = next(
            (m["eval"] for m in pool if m["move"] == chosen), best_eval
        )
        loss = max(0, best_eval - chosen_eval)
        if best_eval >= _MATE_CP and chosen_eval >= _MATE_CP:
            loss = 0  # a slower mate is still a mate, not an error
        self._record(top_moves, chosen, loss)
        self._run_controller()
        self._log(chosen, top_moves, loss, temperature, coasting, cushion)
        # Remember the position our move creates, so later turns can steer
        # away from recreating it (repetition = draw = wasted edge).
        if board is not None:
            try:
                mv = chess.Move.from_uci(chosen)
                if mv in board.legal_moves:
                    board.push(mv)
                    key = board.board_fen()
                    board.pop()
                    self._seen_placements[key] = \
                        self._seen_placements.get(key, 0) + 1
            except ValueError:
                pass
        return chosen

    def _book_choice(self, board: chess.Board | None, top_moves: list[dict],
                     piece_count: int) -> str | None:
        """Play the repertoire while the position is in it. The engine's
        candidates are still fetched so the move is scored and logged like
        any other; a book move the engine really dislikes (a line the
        opponent has refuted) is skipped and play falls through."""
        if not self.use_book or board is None:
            return None
        uci = book_move(board)
        if uci is None:
            return None
        try:
            if chess.Move.from_uci(uci) not in board.legal_moves:
                return None
        except ValueError:
            return None
        best_eval = top_moves[0]["eval"]
        chosen_eval = next((m["eval"] for m in top_moves if m["move"] == uci), None)
        if chosen_eval is None:
            board.push(chess.Move.from_uci(uci))
            chosen_eval = -self.engine.get_evaluation(board.fen(), depth=10)
            board.pop()
        loss = max(0, best_eval - chosen_eval)
        if loss > 80:
            return None
        self.last_top_moves = top_moves
        self.last_best_eval = best_eval
        self.last_criticality = self._compute_criticality(top_moves)
        self._update_effective_elo()
        self._eval_history.append(best_eval)
        cushion = self._cushion(best_eval)
        self.last_cushion = cushion
        self._record(top_moves, uci, loss)
        self._log(uci, top_moves, loss, self._base_temperature(), False, cushion)
        self.last_decision["book"] = True
        return uci

    def suggest_think_time(self, chosen: str, piece_count: int) -> float:
        """How long a human would plausibly think before playing `chosen`.

        Timing is a tell chess.com looks at as much as move quality:
        recaptures and only-moves come instantly, real decisions take a
        while, worse positions get more thought, slips happen when moving
        fast. Seconds, log-normally jittered, for the overlay countdown.
        """
        top = self.last_top_moves
        best = top[0]["eval"] if top else 0
        chosen_eval = next((m["eval"] for m in top if m["move"] == chosen), best)
        loss = max(0, best - chosen_eval)
        crit = self.last_criticality
        if self._move_number < 8:
            base = 3.0                      # opening: familiar territory
        elif crit >= 0.6:
            base = 2.5                      # one obvious move
        elif loss >= 60:
            base = 4.0                      # slips happen when moving fast
        elif crit < 0.15:
            base = 11.0                     # several playable moves: a real decision
        else:
            base = 7.0
        if self.last_cushion < 0:
            base *= 1.4                     # under pressure, people think longer
        if piece_count <= 10:
            base *= 0.7                     # simple endgames go quicker
        if len(top) >= 2 and abs(top[0]["eval"] - top[1]["eval"]) < 15 and crit < 0.3:
            base *= 1.2                     # two near-equal options
        seconds = base * math.exp(random.gauss(0.0, 0.35))
        return max(1.0, min(30.0, seconds))

    def get_accuracy(self) -> float | None:
        if self._total_moves == 0:
            return None
        return (self._best_move_hits / self._total_moves) * 100

    def get_avg_cpl(self) -> float | None:
        if self._total_moves == 0:
            return None
        return self._total_cpl / self._total_moves

    def get_contested_cpl(self) -> float | None:
        """Average loss on moves played from live positions (|eval| < 3)."""
        if self._contested_moves == 0:
            return None
        return self._contested_cpl / self._contested_moves

    def reset(self) -> None:
        """Clear per-game state and sample a fresh game script."""
        self._eval_history.clear()
        self._move_number = 0
        self._consecutive_best = 0
        self._recent_moves.clear()
        self._seen_placements.clear()
        self._opponent_elo = None
        self._opponent_moves = 0
        self._opp_edge = int(max(-10, min(400, self._sample_opp_edge()
                                          + self.session_edge_shift)))
        # Form: a real player is a different strength every day. The
        # rating imitated this game is the target shifted by a sampled
        # offset, before any adaptation to the opponent.
        self._form = int(max(-180, min(180, random.gauss(0, 80))))
        self._eff_elo = float(self._target_elo + self._form)
        self._temp_gain = 1.0
        self._realized.reset()
        self._total_moves = 0
        self._best_move_hits = 0
        self._total_cpl = 0.0
        self._contested_moves = 0
        self._contested_cpl = 0.0
        self.last_top_moves = []
        self.last_criticality = 0.0
        self.last_best_eval = None
        self._target_margin = self._sample_target_margin()
        self._won_for = 0
        self._mate_misses = 0

    def _press(self) -> float:
        """0 = happy to plateau, 1 = converting with intent. Grows with the
        moves spent winning (a club player sits on +3 for a handful of
        moves, then trades down and finishes), with a thinner board (won
        endgames are technique) and with a crushing eval."""
        press = min(1.0, max(0.0, (self._won_for - 6) / 12.0))
        if self._piece_count <= 16:
            press = max(press, 0.6)
        if self._piece_count <= 10:
            press = max(press, 0.85)
        if self.last_best_eval is not None and self.last_best_eval >= 700:
            press = max(press, 0.5)
        return press

    # ------------------------------------------------------------------
    # ELO -> error budget
    # ------------------------------------------------------------------

    def _base_temperature(self) -> float:
        """Softmax temperature (cp) seeded from the target ELO's ACPL.

        The closed-loop controller fine-tunes the constant, so this only
        needs to be monotonic in ELO. Weaker -> hotter -> more spread.

        The temp->realized-ACPL relationship is strongly sublinear (we only
        draw from top-N candidates, whose losses are bounded), so weak play
        needs a temperature well above its target ACPL to actually spread.
        """
        acpl = elo_to_acpl(self._effective_elo())
        weakness = self._naturalness_strength()
        return max(6.0, min(800.0, acpl * (1.9 + 3.4 * weakness))) * self.session_temp_mult

    def _num_candidates(self) -> int:
        """Weak players consider more (and worse) moves than strong ones."""
        # A floor of 8: with only the engine's top 5 in a quiet position
        # every candidate is within ~20cp of best, so the temperature has
        # nothing to spend and the controller saturates trying.
        n = round(8 + (1900 - self._effective_elo()) / 110)
        return int(max(6, min(self.NUM_CANDIDATES_MAX, n)))

    def _loss_ceiling(self, criticality: float) -> float:
        """The largest single-move error (cp) this position may deviate by.

        This is the guard that stops a 1600 from ever hanging a piece for
        nothing, and — crucially — forces the obvious move when the position
        is forcing. Two inputs:

        - ELO: the worse the player, the bigger a plausible one-move error.
        - Criticality: when there is one clearly best move (a recapture, a
          forced tactic), even a modest player finds it, so the ceiling
          collapses toward "best move only". This is what a human does when
          the position screams for a move — and what a bot fails to imitate
          when it wanders off into a superficial pawn grab.
        """
        # A catastrophe guard, not the primary error shaper: set it a few
        # multiples above the level's typical loss so the normal human spread
        # passes through untouched and only the disaster tail is cut. The
        # softmax temperature does the fine shaping.
        acpl = elo_to_acpl(self._effective_elo())
        ceiling = acpl * 5.0 * (1.0 - 0.85 * criticality)
        # Opening discipline: humans play near-book early, so hug the best
        # moves for the first few plies instead of drifting into a passive,
        # bot-looking setup. (This also disables temptation blunders early.)
        if self._move_number < 6:
            ceiling *= 0.5
        return max(12.0, ceiling)

    def _naturalness_strength(self) -> float:
        """How much superficial move features sway the choice (0..1).

        Strong at low ELO, fades to ~0 by ~1900 where players calculate
        rather than react to how a move looks.
        """
        return max(0.0, min(1.0, (1900 - self._effective_elo()) / 1300))

    # ------------------------------------------------------------------
    # Opponent adaptation: effective ELO and per-move risk appetite
    # ------------------------------------------------------------------

    def _sample_opp_edge(self) -> int:
        """How much stronger than the opponent we try to be this game.

        Centred on a solid edge (a clearly better human who reliably wins
        with ordinary moves); varies between comfortable and dominant, but
        never dips to parity — the point is to win, just believably.
        """
        # Scaled to the level: a 1200 who beats 1200s is a 1330, not a
        # 1500. Roughly a tenth of the rating, with game-to-game spread.
        mean = 0.11 * self._target_elo + 20
        edge = random.gauss(mean, 55)
        return int(max(30, min(300, edge)))

    def _opponent_confidence(self) -> float:
        """0 = no idea who we're playing, 1 = estimate is trustworthy."""
        if self._opponent_elo is None:
            return 0.0
        return max(0.0, min(1.0, (self._opponent_moves - 3) / 9.0))

    def _wanted_elo(self) -> float:
        """Where the effective ELO should head: the menu target as a prior,
        pulled toward (opponent + edge) as the opponent's strength becomes
        clear. Bounded to +-400 around the target so the user's choice still
        means something (a '1600' bot never turns into a 2400 or a 900)."""
        base = float(self._target_elo + self._form)
        conf = self._opponent_confidence()
        if conf <= 0.0:
            return base
        wanted = self._opponent_elo + self._opp_edge
        wanted = max(base - 400, min(base + 400, wanted))
        # Keep a little pull toward the menu target even at full confidence.
        return base + conf * 0.85 * (wanted - base)

    def _update_effective_elo(self) -> None:
        """Drift the effective ELO toward the wanted value (once per move)."""
        self._eff_elo += 0.3 * (self._wanted_elo() - self._eff_elo)

    def _effective_elo(self) -> int:
        """The ELO we imitate right now (smoothed, opponent-adapted)."""
        return int(round(self._eff_elo))

    def _opponent_gap(self) -> float:
        """Opponent ELO minus ours (positive = they are the stronger side),
        scaled by confidence. 0 when unknown."""
        conf = self._opponent_confidence()
        if conf <= 0.0:
            return 0.0
        return conf * (self._opponent_elo - self._effective_elo())

    def _cushion_floor(self) -> int:
        """The eval we would rather not sink below against this opponent.

        A stronger opponent punishes slips, so we keep more in hand; a
        weaker one gives things back, so equality is an acceptable place
        to drift to.
        """
        floor = 40 + 0.35 * self._opponent_gap()
        return int(max(-40, min(220, floor)))

    def _cushion(self, best_eval: int) -> int:
        """How much eval we can spend before dipping under the floor.
        Negative = we are already below where we want to be."""
        if best_eval >= _MATE_CP:
            return 2000
        if best_eval <= -_MATE_CP:
            return -2000
        return best_eval - self._cushion_floor()

    def _risk_appetite(self, cushion: int) -> float:
        """Temperature multiplier: relax when there is room for a slip,
        bear down when the position needs a good move.

        Asymmetric on purpose — we relax gently when ahead but tighten
        hard when behind, which is also how a human who wants to win
        plays: careless when comfortable, focused when in trouble.
        """
        if cushion >= 0:
            return 1.0 + 0.5 * min(1.0, cushion / 250.0)
        return max(0.4, 1.0 + cushion / 400.0)

    def _bound_ceiling_by_cushion(self, ceiling: float, cushion: int) -> float:
        """Shrink the allowed one-move error to what we can afford.

        Ahead: never spend more than a slice of the lead in one move — a
        realistic *small* mistake, not one that hands the game back.
        Behind: the ceiling contracts so we stop making things worse.
        The floor of the bound scales with ELO so a weak level still gets
        its characteristic imprecision in equal positions.
        """
        acpl = elo_to_acpl(self._effective_elo())
        if cushion >= 0:
            # Slice of the lead that may go in one move: a thin edge is
            # nursed (a good player does not let +1 slip to 0.00 in one
            # careless move), a big one can be spent freely.
            share = 0.35 + 0.25 * min(1.0, cushion / 400.0)
            share *= 1.0 - 0.6 * self._press()  # converting: nurse the lead
            spendable = acpl * 1.6 + share * cushion
            return max(12.0, min(ceiling, spendable))
        shrink = max(0.5, 1.0 + cushion / 500.0)
        return max(12.0, ceiling * shrink)

    # ------------------------------------------------------------------
    # Attention lapses: the heavy tail of the human error distribution
    # ------------------------------------------------------------------

    def _roll_lapse(self, best_eval: int) -> bool:
        """Occasionally a human just misses something. Never in clearly
        forced positions (a 1600 still recaptures) and only when there is
        an edge to spend — slips in level positions compound into losses,
        while slips when comfortably better are pure human texture."""
        if best_eval < 60 or best_eval >= _MATE_CP:
            return False
        if self.last_criticality >= 0.75:
            return False
        p = 0.015 + 0.09 * self._naturalness_strength()
        return random.random() < p

    def _lapse_ceiling(self) -> float:
        """How big a mistake a lapse may produce (cp). A hung pawn or a
        missed tactic at club level, up to a hung piece for weak levels."""
        return 140.0 + 380.0 * self._naturalness_strength()

    # ------------------------------------------------------------------
    # Temperature: combine ELO budget, phase, danger, and the governor
    # ------------------------------------------------------------------

    def _compute_temperature(
        self, best_eval: int, piece_count: int
    ) -> tuple[float, bool]:
        """Return (temperature, coasting) for this position."""
        # The controller's loosening (gain > 1) is only honored when we
        # have a cushion to spend it from. Equal or worse positions get
        # tight play: humans concentrate when the game is on the line,
        # and slow 20cp-per-move leaks in balanced middlegames are what
        # turn winnable games into losses.
        gain = self._temp_gain
        cushion_now = self._cushion(best_eval)
        if cushion_now <= 0:
            gain = min(gain, 1.0)
        elif gain > 1.0:
            # A small edge is guarded, not spent: honour the loosening in
            # proportion to how much lead there is to spend it from (a
            # +1.3 game blown by three 50cp moves is how draws happen).
            gain = 1.0 + (gain - 1.0) * min(1.0, cushion_now / 400.0)
        press = self._press() if best_eval >= _WINNING_CP else 0.0
        if piece_count <= 12:
            # Won endgames are technique: nobody relaxes into random rook
            # moves with a pawn to push. Loosening is capped here.
            gain = min(gain, 1.5)
        elif best_eval >= _WINNING_CP:
            # Coasting already relaxes below; letting the controller pile
            # its full gain on top produced 250cp leaks every move and a
            # +7.5 game traded down into a dead draw.
            gain = min(gain, 2.0)
        if press > 0:
            gain = min(gain, 2.0 - press)  # converting: no loosening on top
        temp = self._base_temperature() * gain

        # Opening book: humans play memorized theory early -> tighter.
        if self._move_number < 6:
            temp *= 0.5 + (self._move_number / 6) * 0.45

        # Endgames are more forcing; humans (and we) get more precise.
        if piece_count <= 12:
            temp *= 0.7

        # Long games: a human trying to win stops drifting and presses.
        if self._move_number > 30:
            temp *= max(0.55, 1.0 - (self._move_number - 30) * 0.012)

        # An obvious best move gets found even by weaker players.
        temp *= 1.0 - 0.6 * self.last_criticality

        # Risk appetite: room for a slip against this opponent -> relax;
        # position needs a good move -> bear down. Asymmetric by design.
        temp *= self._risk_appetite(self._cushion(best_eval))
        temp *= 1.0 - 0.4 * self._trend_urgency()

        # Anti-domination governor: crushing beyond our target margin ->
        # ease off (coast) instead of pressing for the fastest kill.
        coasting = False
        if best_eval >= _WINNING_CP and best_eval < _MATE_CP:
            over = best_eval - self._target_margin
            if over > 0:
                coasting = True  # keeps the win-floor guard on
                # Plateau early, then convert: the relaxation fades as the
                # conversion clock runs, and the whole temperature tightens.
                temp *= 1.0 + min(over / max(self._target_margin, 80), 0.8) * (1.0 - press)
        if press > 0:
            temp *= 1.0 - 0.35 * press

        # Winning big is where humans get sloppy, not sharp: whatever the
        # criticality and controller did above, keep a floor of looseness
        # that scales with the cushion (the ceiling and win floor still
        # forbid throwing the game). Not in mate/decisive modes (handled
        # before we get here).
        cushion_now = self._cushion(best_eval)
        if best_eval >= 400 and best_eval < _MATE_CP:
            temp = max(temp, min(160.0, 40.0 + 0.06 * cushion_now))

        # Avoid a suspiciously perfect streak.
        if self._consecutive_best >= 6:
            temp *= min(1.0 + 0.08 * (self._consecutive_best - 5), 1.5)

        floor = 3.0 if best_eval <= -250 else 6.0
        return max(floor, temp), coasting

    def _trend_urgency(self) -> float:
        """0 = stable/improving, ->1 = our eval is sliding downhill."""
        if len(self._eval_history) < 3:
            return 0.0
        window = self._eval_history[-self.TREND_WINDOW:]
        n = len(window)
        x_mean = (n - 1) / 2
        y_mean = sum(window) / n
        numer = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(window))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        if denom == 0:
            return 0.0
        slope = numer / denom
        return 0.0 if slope >= 0 else min(1.0, abs(slope) / 100)

    def _compute_criticality(self, top_moves: list[dict]) -> float:
        """0 = many good options, 1 = one clearly forced move."""
        if len(top_moves) < 2:
            return 1.0
        gap = top_moves[0]["eval"] - top_moves[1]["eval"]
        # A gap only forces the move when the alternative is actually bad.
        # At +9 the second-best move is still +7: nobody is "forced", and
        # treating it so made the bot play a 15-move engine-perfect finish
        # (92% accuracy, game rating 2200 at target 1700).
        second = top_moves[1]["eval"]
        if 0 < second < _MATE_CP:
            # Alternative keeps a small edge (<=100): fully forcing. Keeps a
            # clear win (>=250): mostly a matter of taste (x0.35), fading
            # further as the position becomes decided.
            if second <= 100:
                f = 1.0
            elif second <= 250:
                f = 1.0 - 0.65 * (second - 100) / 150.0
            else:
                f = 0.35 * max(0.25, decided_factor(second)) / 1.0
                f = max(0.1, f)
            gap *= f
        if gap <= 30:
            return 0.0
        return min(1.0, (gap - 30) / 170)

    # ------------------------------------------------------------------
    # Anti-domination governor
    # ------------------------------------------------------------------

    def _sample_target_margin(self) -> int:
        """The winning margin (cp) we try to plateau at this game.

        Skewed toward modest, human-looking wins with an occasional
        comfortable game — but rarely a total rout.
        """
        margin = random.gauss(260, 110)
        return int(max(90, min(650, margin)))

    def _win_floor(self) -> int:
        """Never coast into a move that drops eval below this — the win is
        protected. Scales with how much of a cushion we sampled, and with
        the opponent: a stronger one gets less rope."""
        floor = self._target_margin * 0.65 + 0.3 * self._opponent_gap()
        return int(max(90, min(380, floor)))

    # ------------------------------------------------------------------
    # Brilliancy suppression: sub-elite humans do not find engine
    # sacrifices, so the imitation must not play them either
    # ------------------------------------------------------------------

    def _sacrifice_vision(self) -> float:
        """Probability this level spots and trusts a material sacrifice."""
        return max(0.0, min(1.0, (self._effective_elo() - 1500) / 900.0))

    def _see_capture_value(self, board: chess.Board, to_sq: chess.Square) -> int:
        """Best material (cp) the side to move can net by capturing on to_sq.

        Static exchange: capture with the least valuable attacker, let the
        opponent do the same, and never continue a losing sequence. Using
        legal moves makes (absolutely) pinned pieces sit the exchange out.
        """
        occupant = board.piece_at(to_sq)
        if occupant is None:
            return 0
        caps = [m for m in board.legal_moves
                if m.to_square == to_sq and board.is_capture(m)]
        if not caps:
            return 0
        cap = min(caps, key=lambda m: _PIECE_VALUE.get(
            board.piece_at(m.from_square).piece_type, 0))
        gain = _PIECE_VALUE.get(occupant.piece_type, 0)
        board.push(cap)
        net = gain - self._see_capture_value(board, to_sq)
        board.pop()
        return max(0, net)

    def _move_material_net(self, board: chess.Board, move: chess.Move) -> int:
        """Net material (cp) a move stands to gain or give up on its landing
        square once exchanges resolve. Clearly negative = a sacrifice."""
        if move.promotion:
            return 0  # promotions have their own bonus; never call them sacs
        gain = 0
        if board.is_capture(move):
            if board.is_en_passant(move):
                gain = 100
            else:
                victim = board.piece_at(move.to_square)
                gain = _PIECE_VALUE.get(victim.piece_type, 0) if victim else 0
        board.push(move)
        lost = self._see_capture_value(board, move.to_square)
        board.pop()
        return gain - lost

    def _filter_sacrifices(
        self, board: chess.Board | None, top_moves: list[dict]
    ) -> list[dict]:
        """Drop engine sacrifices ("brilliancies") the imitated level would
        not find, when a sound ordinary move exists.

        The criticality/ceiling machinery treats a big eval gap as "obvious,
        must-play" — true for a recapture, false for a deep sacrifice that
        is obvious only to the engine. Removing the sacrifice *before* that
        machinery runs makes the bot play the human move: the best line it
        never saw simply doesn't exist for it.
        """
        if board is None or len(top_moves) < 2:
            return top_moves
        # Technique mode: in a clearly won position (or with mate on the
        # board) humans of every level happily give material back to
        # convert — and refusing every tactical shot stalls won games into
        # draws. These sacs read as "best move", not as brilliancies.
        if top_moves[0]["eval"] >= 450:
            return top_moves
        if random.random() < self._sacrifice_vision():
            return top_moves  # this level spots it this time

        nets: dict[str, int] = {}
        for m in top_moves:
            try:
                mv = chess.Move.from_uci(m["move"])
            except ValueError:
                continue
            if mv in board.legal_moves:
                nets[m["move"]] = self._move_material_net(board, mv)
        sacs = [m for m in top_moves if nets.get(m["move"], 0) <= _SAC_NET]
        if not sacs:
            return top_moves
        non_sac = [m for m in top_moves if nets.get(m["move"], 0) > _SAC_NET]
        if not non_sac:
            return top_moves  # every option gives up material — play on

        # Declining must not wreck the game: the best ordinary move has to
        # keep a playable position, or cost only a human-sized slip. When
        # the sacrifice is the lone save, even a human digs in and finds it.
        best_eval = top_moves[0]["eval"]
        alt_eval = non_sac[0]["eval"]
        budget = max(120.0, 2.5 * elo_to_acpl(self._effective_elo()))
        if alt_eval < _SAC_KEEP_FLOOR and best_eval - alt_eval > budget:
            return top_moves
        if sacs[0] is top_moves[0]:
            print(f"  [sac] declined {top_moves[0]['move']} "
                  f"(net {nets[top_moves[0]['move']]}cp material, "
                  f"decline costs {best_eval - alt_eval}cp)")
        return non_sac

    # ------------------------------------------------------------------
    # Human-plausibility prior
    # ------------------------------------------------------------------

    def _human_prior(self, board: chess.Board | None, uci: str) -> float:
        """Log-weight bonus for how tempting a move looks to a human.

        Positive = a human is drawn to it; negative = a human tends to
        overlook it. Callers scale this by naturalness before applying.
        """
        if board is None:
            return 0.0
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return 0.0
        if move not in board.legal_moves:
            return 0.0

        bonus = 0.0
        mover = board.piece_at(move.from_square)
        if mover is None:
            return 0.0

        # Checks grab attention.
        if board.gives_check(move):
            bonus += 0.6

        # Material on the landing square once exchanges resolve. A winning
        # or equal capture is magnetic; a move that *gives up* material —
        # the engine sacrifice, capture or quiet — is what a human refuses.
        net = self._move_material_net(board, move)
        if board.is_capture(move):
            if net >= 0:
                bonus += 0.7
            elif net > _SAC_NET:
                bonus += 0.15
            else:
                bonus -= 0.9
        elif net <= _SAC_NET:
            bonus -= 0.9

        # Promotions are hard to miss.
        if move.promotion == chess.QUEEN:
            bonus += 0.8

        # Retreats (toward our own back rank) are psychologically avoided.
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        forward = to_rank - from_rank if board.turn == chess.WHITE else from_rank - to_rank
        if forward < 0 and not board.is_capture(move) and not board.gives_check(move):
            bonus -= 0.3

        # Early queen sortie: bringing the queen out in the opening to a
        # loose square is a classic way to get it chased or trapped, and a
        # principle most players above beginner level follow. (This exact
        # pattern — Qa4 then a lost queen — cost a real game.)
        if (mover.piece_type == chess.QUEEN and self._move_number < 10
                and not board.is_capture(move)):
            home_rank = 0 if board.turn == chess.WHITE else 7
            if abs(chess.square_rank(move.to_square) - home_rank) >= 2:
                bonus -= 0.7

        return bonus

    def _progress_prior(self, board: chess.Board | None, uci: str,
                        best_eval: int) -> float:
        """In a won position a human follows a plan: trade pieces, push
        the passed pawn, promote, bring the king up, keep checking. Rewards
        those and penalizes purposeless moves of any piece. Unscaled by
        naturalness — this is technique, not talent. Zero unless winning
        and the board has thinned enough for plans to be concrete."""
        if board is None or self._piece_count > 20 or best_eval < 250:
            return 0.0
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return 0.0
        mover = board.piece_at(move.from_square)
        if mover is None or move not in board.legal_moves:
            return 0.0
        scale = 1.0 if self._piece_count <= 12 else 0.6
        crushing = 1.5 if best_eval >= 800 else 1.0
        bonus = 0.0
        capture = board.is_capture(move)
        check = board.gives_check(move)
        if move.promotion is not None:
            bonus += 1.5 if move.promotion == chess.QUEEN else 0.3
        elif capture:
            victim = board.piece_at(move.to_square)
            # Trading when ahead is the first thing every coach teaches.
            bonus += (0.5 if victim is not None and victim.piece_type != chess.PAWN else 0.2) * scale
        elif mover.piece_type == chess.PAWN:
            if self._is_passed_pawn(board, move.from_square):
                bonus += 0.9
        elif mover.piece_type == chess.KING:
            # King marching toward the action (the enemy king) is progress.
            enemy_king = board.king(not board.turn)
            if enemy_king is not None:
                before = chess.square_distance(move.from_square, enemy_king)
                after = chess.square_distance(move.to_square, enemy_king)
                bonus += 0.4 if after < before else -0.2 * crushing
        elif not check:
            # A quiet piece move that neither trades, checks nor advances a
            # pawn: shuffling. Rook shuffles killed one live game, bishop
            # shuffles at +50 another.
            if mover.piece_type in (chess.ROOK, chess.QUEEN):
                bonus -= 0.6 * crushing
            else:
                bonus -= 0.35 * scale * crushing
        if check and not capture:
            bonus += 0.15
        return bonus

    @staticmethod
    def _is_passed_pawn(board: chess.Board, square: chess.Square) -> bool:
        pawn = board.piece_at(square)
        if pawn is None or pawn.piece_type != chess.PAWN:
            return False
        file, rank = chess.square_file(square), chess.square_rank(square)
        ahead = range(rank + 1, 8) if pawn.color == chess.WHITE else range(rank - 1, -1, -1)
        for r in ahead:
            for f in (file - 1, file, file + 1):
                if 0 <= f <= 7:
                    p = board.piece_at(chess.square(f, r))
                    if p is not None and p.piece_type == chess.PAWN and p.color != pawn.color:
                        return False
        return True

    def _coherence_penalty(self, uci: str) -> float:
        """Penalize move-to-move incoherence — the un-human tells that a
        per-position engine produces: shuffling a piece out and back, or
        re-moving the same piece aimlessly. Applied at every ELO (nobody
        good plays Ra2 then Ra1); the softmax loss term still lets a genuine
        best-move reversal through when it is clearly forced.
        """
        if not self._recent_moves:
            return 0.0
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return 0.0

        penalty = 0.0
        last = self._recent_moves[-1]
        # Immediate reversal: move the same piece straight back where it came.
        if move.from_square == last.to_square and move.to_square == last.from_square:
            penalty -= 1.6
        # Return to a square this piece vacated in its last couple of moves.
        elif move.to_square in {m.from_square for m in self._recent_moves[-2:]}:
            penalty -= 0.7
        # Re-moving the piece that just moved (no fresh development/plan).
        if move.from_square == last.to_square and penalty == 0.0:
            penalty -= 0.35
        return penalty

    def _repetition_penalty(
        self, board: chess.Board | None, uci: str, best_eval: int
    ) -> float:
        """Steer away from recreating positions we've already had when we
        stand better — a player with an edge doesn't drift into a
        repetition draw. In truly equal positions repetition stays a
        legitimate (human) outcome, so no penalty there."""
        if board is None or best_eval < 60 or not self._seen_placements:
            return 0.0
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return 0.0
        if move not in board.legal_moves:
            return 0.0
        board.push(move)
        seen = self._seen_placements.get(board.board_fen(), 0)
        board.pop()
        return -1.5 * seen

    # ------------------------------------------------------------------
    # Tempting-but-bad move injection (realistic blunders at low ELO)
    # ------------------------------------------------------------------

    def _augment_with_temptations(
        self, fen: str, top_moves: list[dict], best_eval: int, ceiling: float,
        force: bool = False,
    ) -> list[dict]:
        """Occasionally add a superficially tempting losing move to the pool.

        The engine's top-N are all "reasonable"; real weak-player blunders
        live further down (grabbing a poisoned pawn, a premature attack).
        We surface a couple, evaluate them, and let the human prior decide.
        Guarded by ELO and probability to bound engine cost, and never in a
        forcing position (a tempting pawn grab must not override a recapture).
        """
        strength = self._naturalness_strength()
        if strength <= 0.05:
            return top_moves
        if not force:
            # A tiny ceiling means the position is forcing — do not tempt.
            if ceiling < 40 or self.last_criticality >= 0.6:
                return top_moves
            if random.random() > 0.15 + 0.55 * strength:
                return top_moves

        board = self._safe_board(fen)
        if board is None:
            return top_moves

        known = {m["move"] for m in top_moves}
        tempting: list[chess.Move] = []
        quiet: list[chess.Move] = []
        for move in board.legal_moves:
            uci = move.uci()
            if uci in known:
                continue
            if board.is_capture(move) or board.gives_check(move):
                tempting.append(move)
            elif force:
                # A lapse blunder is usually a quiet move that misses the
                # opponent's idea — captures alone can't model that.
                quiet.append(move)
        if not tempting and not quiet:
            return top_moves

        random.shuffle(tempting)
        random.shuffle(quiet)
        pool = list(top_moves)
        n_consider = 3 + int(round(2 * strength))
        for move in tempting[:n_consider] + quiet[:4]:
            board.push(move)
            reply_eval = self.engine.get_evaluation(board.fen(), depth=8)
            board.pop()
            # reply_eval is from the opponent's POV after our move.
            our_eval = -reply_eval
            # Only worth surfacing if it is a mistake this level would
            # plausibly make — bounded by the per-position ceiling so we
            # never inject an error too big for the target ELO.
            loss = best_eval - our_eval
            if 60 <= loss <= ceiling:
                pool.append({"move": move.uci(), "eval": our_eval})
        return pool

    # ------------------------------------------------------------------
    # Weighted selection: softmax over losses, biased by the human prior
    # ------------------------------------------------------------------

    def _weighted_select(
        self,
        board: chess.Board | None,
        pool: list[dict],
        best_eval: int,
        temperature: float,
        coasting: bool,
        ceiling: float,
        raw_loss: bool = False,
    ) -> str:
        strength = self._naturalness_strength()
        endgame = self._piece_count <= 12

        # Losses are measured by how much they matter: in a decided
        # position (say +9) a 200cp "slip" changes nothing, and insisting
        # on the engine's fastest win there is the least human thing the
        # bot can do. The coast floor below still guards the result.
        def cost(m: dict) -> float:
            loss = best_eval - m["eval"]
            if raw_loss:
                return float(loss)
            # A decided position discounts losses, but never below half:
            # "relaxed" must mean slower, not sloppy.
            factor = max(decided_factor(best_eval), 0.5)
            return loss * factor

        # Hard error ceiling: discard any move worse than this level would
        # plausibly play. This is what forces recaptures/tactics (tiny
        # ceiling) and forbids oversized blunders — always keep the best.
        candidates = [m for m in pool if cost(m) <= ceiling]
        if not candidates:
            candidates = [pool[0]]

        # In coast mode, protect the win: only consider moves that keep the
        # eval above the floor. Never coast a won game into equality.
        if coasting:
            floor = self._win_floor()
            kept = [m for m in candidates if m["eval"] >= floor]
            if kept:
                candidates = kept

        moves, weights = [], []
        for m in candidates:
            loss = cost(m)
            exponent = -loss / max(temperature, 1.0)
            exponent += strength * self._human_prior(board, m["move"])
            exponent += self._progress_prior(board, m["move"], best_eval)
            exponent += self._coherence_penalty(m["move"])
            exponent += self._repetition_penalty(board, m["move"], best_eval)
            exponent = max(-30.0, min(30.0, exponent))
            moves.append(m["move"])
            weights.append(math.exp(exponent))

        total = sum(weights)
        if total <= 0:
            return pool[0]["move"]
        return random.choices(moves, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Closed-loop controller
    # ------------------------------------------------------------------

    def _run_controller(self) -> None:
        """Nudge the error budget so realized ELO tracks the target.

        Realized above target -> playing too strong -> loosen (raise gain).
        Realized below target -> playing too weak -> tighten (lower gain).
        """
        realized = self._realized.get_estimate()
        if realized is None:
            return
        error = realized - self._effective_elo()  # +ve = too strong
        step = max(-0.15, min(0.15, error / 900.0))
        self._temp_gain *= math.exp(step)
        self._temp_gain = max(0.3, min(6.0, self._temp_gain))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _record(self, top_moves: list[dict], chosen: str, loss: int) -> None:
        if chosen == top_moves[0]["move"]:
            self._consecutive_best += 1
            self._best_move_hits += 1
        else:
            self._consecutive_best = 0
        self._move_number += 1
        self._total_moves += 1
        self._total_cpl += loss
        if self.last_best_eval is not None and abs(self.last_best_eval) < 300:
            self._contested_moves += 1
            self._contested_cpl += loss
        self._realized.record_move(loss, self.last_best_eval)
        try:
            self._recent_moves.append(chess.Move.from_uci(chosen))
            self._recent_moves = self._recent_moves[-4:]
        except ValueError:
            pass

    def _safe_board(self, fen: str) -> chess.Board | None:
        try:
            return chess.Board(fen)
        except ValueError:
            return None

    def _log(self, chosen, top_moves, loss, temperature, coasting,
             cushion) -> None:
        tag = "*" if chosen == top_moves[0]["move"] else " "
        chosen_eval = next(
            (m["eval"] for m in top_moves if m["move"] == chosen),
            top_moves[0]["eval"] - loss,
        )
        self.last_decision = {
            "move": chosen, "best": top_moves[0]["move"], "loss": int(loss),
            "best_eval": int(top_moves[0]["eval"]), "chosen_eval": int(chosen_eval),
            "temp": round(float(temperature), 1), "target": self._target_elo,
            "eff": self._effective_elo(), "opp": self._opponent_elo,
            "edge": self._opp_edge, "form": self._form, "realized": self.get_realized_elo(),
            "gain": round(self._temp_gain, 2), "margin": self._target_margin,
            "cushion": int(cushion), "crit": round(self.last_criticality, 2),
            "coast": bool(coasting), "n_cand": len(top_moves),
        }
        realized = self.get_realized_elo()
        realized_s = f"{realized}" if realized is not None else "--"
        opp_s = f"{self._opponent_elo}" if self._opponent_elo is not None else "--"
        print(
            f"  [{tag}] move={chosen}  loss={loss}cp  temp={temperature:.0f}  "
            f"target={self._target_elo}  eff={self._effective_elo()}  "
            f"opp={opp_s}(+{self._opp_edge})  realized={realized_s}  "
            f"gain={self._temp_gain:.2f}  margin={self._target_margin}  "
            f"cushion={cushion}  crit={self.last_criticality:.2f}  "
            f"coast={int(coasting)}"
        )
