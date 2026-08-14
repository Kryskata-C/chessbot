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
   tempting; quiet moves, retreats, and moves that hang a piece (the
   engine's brilliant sacrifices) are under-selected. Scaled by weakness,
   so the inserted error lands exactly where a human misses — never as an
   obvious out-of-nowhere blunder.

3. **Anti-domination governor.** Each game samples a target winning margin
   (so results vary: sometimes close, sometimes comfortable). When we are
   crushing beyond it, we ease off — pick sound-but-not-crushing moves that
   keep the win while letting the margin breathe — with a hard floor so a
   won game is never thrown.

4. **Closed-loop ELO controller.** The bot's own move losses feed a live
   "realized ELO". A slow controller nudges the error budget until realized
   ELO matches the target the user asked for.
"""

from __future__ import annotations

import math
import random

import chess

from engine import ChessEngine
from elo_estimator import EloEstimator, elo_to_acpl, acpl_to_elo

# Piece values in centipawns, for judging captures and apparent hangs.
_PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

_MATE_CP = 100000
_WINNING_CP = 150   # eval above this = "we are winning"


class HumanMoveSelector:
    """Selects moves that mimic a human of a chosen ELO while staying sound."""

    DEFAULT_TARGET_ELO = 1400
    NUM_CANDIDATES_MAX = 16
    TREND_WINDOW = 6

    def __init__(self, engine: ChessEngine):
        self.engine = engine
        self._target_elo: int = self.DEFAULT_TARGET_ELO
        self._opponent_elo: int | None = None

        # Per-game state
        self._eval_history: list[int] = []
        self._move_number: int = 0
        self._consecutive_best: int = 0
        # The bot's own recent moves, for move-to-move coherence (humans
        # don't shuffle a rook out and back, or re-move one piece aimlessly).
        self._recent_moves: list[chess.Move] = []

        # Anti-domination governor: the winning margin (cp) we try to
        # plateau at this game. Resampled each game for variety.
        self._target_margin: int = 250

        # Closed-loop controller: multiplies the base error budget so the
        # realized ELO converges to the target.
        self._temp_gain: float = 1.0
        self._realized = EloEstimator()

        # Display / accuracy tracking
        self._total_moves: int = 0
        self._best_move_hits: int = 0
        self._total_cpl: float = 0.0

        # Last-analysis data exposed to the overlay visualizer
        self.last_top_moves: list[dict] = []
        self.last_criticality: float = 0.0
        self.last_best_eval: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target_elo(self, elo: int) -> None:
        """Set the strength the suggestions should imitate."""
        self._target_elo = int(max(300, min(3000, elo)))

    def get_target_elo(self) -> int:
        return self._target_elo

    def set_opponent_elo(self, elo: int | None) -> None:
        """Update estimated opponent strength (used to bound risk)."""
        self._opponent_elo = elo

    def get_realized_elo(self) -> int | None:
        """The bot's own live ELO, estimated from the moves it has picked."""
        return self._realized.get_estimate()

    def select_move(self, fen: str, piece_count: int) -> str | None:
        """Pick a human-like move. Returns a UCI string, or None if none."""
        top_moves = self.engine.get_top_moves(fen, self._num_candidates())
        self.last_top_moves = top_moves or []
        if not top_moves:
            self.last_criticality = 0.0
            self.last_best_eval = None
            return self.engine.get_best_move(fen)

        best_eval = top_moves[0]["eval"]
        self.last_best_eval = best_eval
        self.last_criticality = self._compute_criticality(top_moves)

        if len(top_moves) == 1:
            self._record(top_moves, top_moves[0]["move"], 0)
            return top_moves[0]["move"]

        self._eval_history.append(best_eval)

        # How far this position is even allowed to deviate. Forcing positions
        # (recaptures, tactics) collapse this toward zero so the obvious move
        # gets played; quiet positions leave room for human imprecision.
        ceiling = self._loss_ceiling(self.last_criticality)

        # Optionally surface a tempting-but-bad move (the classic human
        # blunder) so weak play isn't capped at the engine's Nth-best.
        pool = self._augment_with_temptations(fen, top_moves, best_eval, ceiling)

        board = self._safe_board(fen)
        temperature, coasting = self._compute_temperature(best_eval, piece_count)

        chosen = self._weighted_select(
            board, pool, best_eval, temperature, coasting, ceiling
        )

        chosen_eval = next(
            (m["eval"] for m in pool if m["move"] == chosen), best_eval
        )
        loss = max(0, best_eval - chosen_eval)
        self._record(top_moves, chosen, loss)
        self._run_controller()
        self._log(chosen, top_moves, loss, temperature, coasting)
        return chosen

    def get_accuracy(self) -> float | None:
        if self._total_moves == 0:
            return None
        return (self._best_move_hits / self._total_moves) * 100

    def get_avg_cpl(self) -> float | None:
        if self._total_moves == 0:
            return None
        return self._total_cpl / self._total_moves

    def reset(self) -> None:
        """Clear per-game state and sample a fresh game script."""
        self._eval_history.clear()
        self._move_number = 0
        self._consecutive_best = 0
        self._recent_moves.clear()
        self._opponent_elo = None
        self._temp_gain = 1.0
        self._realized.reset()
        self._total_moves = 0
        self._best_move_hits = 0
        self._total_cpl = 0.0
        self.last_top_moves = []
        self.last_criticality = 0.0
        self.last_best_eval = None
        self._target_margin = self._sample_target_margin()

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
        acpl = elo_to_acpl(self._target_elo)
        weakness = self._naturalness_strength()
        return max(6.0, min(800.0, acpl * (1.9 + 3.4 * weakness)))

    def _num_candidates(self) -> int:
        """Weak players consider more (and worse) moves than strong ones."""
        n = round(5 + (1900 - self._target_elo) / 110)
        return int(max(4, min(self.NUM_CANDIDATES_MAX, n)))

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
        acpl = elo_to_acpl(self._target_elo)
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
        return max(0.0, min(1.0, (1900 - self._target_elo) / 1300))

    # ------------------------------------------------------------------
    # Temperature: combine ELO budget, phase, danger, and the governor
    # ------------------------------------------------------------------

    def _compute_temperature(
        self, best_eval: int, piece_count: int
    ) -> tuple[float, bool]:
        """Return (temperature, coasting) for this position."""
        temp = self._base_temperature() * self._temp_gain

        # Opening book: humans play memorized theory early -> tighter.
        if self._move_number < 6:
            temp *= 0.5 + (self._move_number / 6) * 0.45

        # Endgames are more forcing; humans (and we) get more precise.
        if piece_count <= 12:
            temp *= 0.7

        # An obvious best move gets found even by weaker players.
        temp *= 1.0 - 0.6 * self.last_criticality

        # Danger: when losing, fight harder (tighten). This is asymmetric —
        # we relax when ahead but bear down when behind.
        if best_eval <= -250:
            temp *= 0.4
        elif best_eval <= -80:
            temp *= 0.7
        temp *= 1.0 - 0.4 * self._trend_urgency()

        # Anti-domination governor: crushing beyond our target margin ->
        # ease off (coast) instead of pressing for the fastest kill.
        coasting = False
        if best_eval >= _WINNING_CP and best_eval < _MATE_CP:
            over = best_eval - self._target_margin
            if over > 0:
                coasting = True
                temp *= 1.0 + min(over / max(self._target_margin, 80), 1.5)

        # Avoid a suspiciously perfect streak.
        if self._consecutive_best >= 6:
            temp *= min(1.0 + 0.06 * (self._consecutive_best - 5), 1.3)

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
        protected. Scales with how much of a cushion we sampled."""
        return max(90, int(self._target_margin * 0.5))

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

        # Checks and captures grab attention.
        if board.gives_check(move):
            bonus += 0.6
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            victim_val = _PIECE_VALUE.get(victim.piece_type, 0) if victim else 100
            attacker_val = _PIECE_VALUE.get(mover.piece_type, 0)
            # Winning/equal captures are magnetic; losing captures less so.
            bonus += 0.7 if victim_val >= attacker_val else 0.15

        # Promotions are hard to miss.
        if move.promotion == chess.QUEEN:
            bonus += 0.8

        # Apparent hang: landing where a cheaper enemy piece can take, with
        # no friendly defender. This is exactly the engine sacrifice a weak
        # human refuses — so we under-select it.
        if not board.is_capture(move):
            enemy = not board.turn
            attackers = board.attackers(enemy, move.to_square)
            if attackers:
                cheapest = min(
                    _PIECE_VALUE.get(board.piece_at(sq).piece_type, 0)
                    for sq in attackers
                )
                mover_val = _PIECE_VALUE.get(mover.piece_type, 0)
                board.push(move)
                defended = bool(board.attackers(not enemy, move.to_square))
                board.pop()
                if cheapest < mover_val and not defended:
                    bonus -= 0.9

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

    # ------------------------------------------------------------------
    # Tempting-but-bad move injection (realistic blunders at low ELO)
    # ------------------------------------------------------------------

    def _augment_with_temptations(
        self, fen: str, top_moves: list[dict], best_eval: int, ceiling: float
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
        for move in board.legal_moves:
            uci = move.uci()
            if uci in known:
                continue
            if board.is_capture(move) or board.gives_check(move):
                tempting.append(move)
        if not tempting:
            return top_moves

        random.shuffle(tempting)
        pool = list(top_moves)
        n_consider = 3 + int(round(2 * strength))
        for move in tempting[:n_consider]:
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
    ) -> str:
        strength = self._naturalness_strength()

        # Hard error ceiling: discard any move worse than this level would
        # plausibly play. This is what forces recaptures/tactics (tiny
        # ceiling) and forbids oversized blunders — always keep the best.
        candidates = [m for m in pool if best_eval - m["eval"] <= ceiling]
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
            loss = best_eval - m["eval"]
            exponent = -loss / max(temperature, 1.0)
            exponent += strength * self._human_prior(board, m["move"])
            exponent += self._coherence_penalty(m["move"])
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
        error = realized - self._target_elo  # +ve = too strong
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
        self._realized.record_move(loss)
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

    def _log(self, chosen, top_moves, loss, temperature, coasting) -> None:
        tag = "*" if chosen == top_moves[0]["move"] else " "
        realized = self.get_realized_elo()
        realized_s = f"{realized}" if realized is not None else "--"
        print(
            f"  [{tag}] move={chosen}  loss={loss}cp  temp={temperature:.0f}  "
            f"target={self._target_elo}  realized={realized_s}  "
            f"gain={self._temp_gain:.2f}  margin={self._target_margin}  "
            f"crit={self.last_criticality:.2f}  coast={int(coasting)}"
        )
