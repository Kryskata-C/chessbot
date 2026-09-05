<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Kryskata-C/chessbot@main/assets/banner.svg" alt="Chess Vision" width="100%"/>
</p>

<h3 align="center">👁️ It sees the board. 🧠 It thinks like a human. ⚡ It plays to <em>your</em> opponent.</h3>

<p align="center">
  <strong>A real-time chess.com move assistant that reads your screen with computer vision, runs Stockfish underneath,<br/>then deliberately <em>de-optimizes</em> the engine through a mathematical model of human play —<br/>so the moves it hands you look like a slightly-better-than-your-opponent human, not a 3500-rated machine.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/macOS-Sonoma%2B-000000?style=for-the-badge&logo=apple&logoColor=white"/>
  <img src="https://img.shields.io/badge/Engine-Stockfish-47a341?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Overlay-PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Proprietary-f7b731?style=for-the-badge"/>
</p>

<p align="center">
  <code>screen → HSV mask → contours → template match → FEN → Stockfish MultiPV → softmax over regret → human prior → opponent-adaptive risk → glowing overlay</code>
</p>

---

## 🎯 What This Actually Is

Most "chess assistants" are a screenshot and an arrow. This is a **five-stage pipeline** where the interesting part is *after* the engine:

| Stage | What happens | Math involved |
|---|---|---|
| 👁️ **Vision** | Finds the board, reads all 64 squares | HSV thresholding, morphology, normalized cross-correlation |
| 🧭 **State** | Turns pixels into a legal game state, tracks whose turn it is | Diff-gating, legal-move matching, orientation inference |
| ⚙️ **Engine** | Stockfish MultiPV, depth 12, top-N candidates | Centipawn evals, mate scoring |
| 🧠 **Human layer** | Picks a move a human of ELO $E$ would play *against this opponent* | Softmax over regret, ELO↔ACPL curve, closed-loop control, risk appetite |
| ✨ **Overlay** | Draws it — arrows, ghost pieces, threat radar, eval bar | Cocoa window levels, click-through, ~2% CPU |

The bot is not built to win every game. It is built to **beat the opponent the way a slightly better human would** — with real, small, well-placed mistakes — and to constantly ask, move by move:

> *"Do I need a good move here, or am I far enough ahead against this opponent to afford a realistic slip?"*

---

## ⚡ The Pipeline

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Kryskata-C/chessbot@main/assets/architecture.svg" alt="Architecture" width="100%"/>
</p>

### 👁️ Stage 1 — Vision

**Board detection** (`board_detector.py`). Every frame is converted to HSV and masked for chess.com's two square colors:

$$
M = \mathbb{1}\big[H \in [30,90],\, S \in [40,255],\, V \in [80,200]\big] \;\lor\; \mathbb{1}\big[H \in [20,45],\, S \in [10,80],\, V \in [180,255]\big]
$$

The mask is cleaned with a $5\times5$ morphological **close** ($\times 3$) then **open** ($\times 2$) — fills the gaps between squares, kills specks — and the largest external contour with area $> 10^4$ px and aspect $0.8 < w/h < 1.2$ wins. It's snapped to a perfect square, $s = \min(w,h)$, so each square is $s/8$ px.

**Piece recognition** (`piece_recognizer.py`). Each square is resized to a fixed template size and matched against 20 templates (six piece types × two colors, on light *and* dark squares — because a black knight on green is not the same pixels as a black knight on beige). Scoring is normalized cross-correlation:

$$
R(x,y) = \frac{\sum_{x',y'} \big(T'(x',y')\cdot I'(x+x',y+y')\big)}{\sqrt{\sum T'(x',y')^2 \cdot \sum I'(x+x',y+y')^2}}, \qquad T' = T - \bar T,\; I' = I - \bar I
$$

A square is a piece if $\max_{\text{templates}} R \geq 0.55$. Kings get special treatment: when a king "vanishes" (check-glow and last-move highlights change the square's background), a recovery pass rescans empty squares with a relaxed $R \geq 0.25$ — because a position without a king is not a position.

Templates are auto-extracted from the starting position by `calibrate.py`, so they match **your** display and **your** theme pixel-for-pixel.

### 🧭 Stage 2 — State

Raw recognition is noisy: mid-animation frames, hover highlights, premove arrows. So the app never trusts a single frame blindly:

- **Diff gate.** Frames are compared as placement strings; a change touching one square can't be a completed move (pieces don't teleport), so it's discarded. Multi-square changes are decomposed into *white arrivals* and *black arrivals* to infer whose move just happened.
- **Legal-move matching.** A `python-chess` board shadows the real game. Each new placement is matched against every legal move (and legal 2-move sequences, for the case where a frame was skipped) — the placement is only accepted if the diff *is* a legal move. Zero hallucinated positions.
- **Orientation.** Your color is inferred once from which side's pieces sit on the bottom ranks.
- **Turn tracking** falls out of the above: if a legal *opponent* move explains the diff, it's your turn.

### ⚙️ Stage 3 — Engine

Stockfish (depth 12, 2 threads, 128 MB hash) in **MultiPV** mode returns the top-$N$ candidates with centipawn evaluations; mates are mapped to $\pm 10^5$. The human layer decides how many candidates it even wants to see:

$$
N = \mathrm{clamp}\!\Big(\big\lfloor 5 + \tfrac{1900 - E}{110} \big\rceil,\; 4,\; 16\Big)
$$

A 2000 plays from ~4 moves. An 800 weighs ~15 (and most of them are bad).

---

## 🧠 Stage 4 — The Human Layer

This is the core of the project. `move_selector.py` is a **stochastic policy over engine candidates**, parametrized by an ELO $E$ and shaped by five coupled mechanisms.

### 4.1 — ELO ⇄ Accuracy

Everything is anchored to one empirical curve linking rating to *average centipawn loss* (ACPL):

$$
E(\text{acpl}) = 4034 - 667\,\ln(\text{acpl}), \qquad \text{acpl}(E) = e^{(4034 - E)/667}
$$

| ELO | Target ACPL |
|---|---|
| 2500 | ~10 cp |
| 2000 | ~21 cp |
| 1600 | ~38 cp |
| 1200 | ~70 cp |
| 800 | ~127 cp |

The same curve runs **both directions**: it estimates the opponent from *their* moves, and it estimates the bot from *its own* chosen moves (the "realized ELO").

### 4.2 — Softmax over regret

Given candidates $m$ with evals $v_m$, best $v^\star$, define each move's regret $\Delta_m = v^\star - v_m$. The move is sampled from

$$
P(m) \;\propto\; \exp\!\Big( -\frac{\Delta_m}{T} \;+\; w\,\pi(m) \;+\; \kappa(m) \Big), \qquad \Delta_m \le L_{\max}
$$

Three terms, three ideas:

- **$T$ — temperature.** How much regret we're willing to spend. Hot = weak. Built from a stack of multipliers (below).
- **$\pi(m)$ — human prior.** How *tempting* the move looks to a human, scaled by weakness $w = \mathrm{clamp}\!\big(\tfrac{1900 - E}{1300},0,1\big)$. Weak players react to how a move *looks*; strong players calculate.
- **$\kappa(m)$ — coherence.** Penalizes un-human sequences (Ra2 then Ra1).
- **$L_{\max}$ — the ceiling.** A hard cutoff on single-move error. This is what makes a 1600 *never* hang a piece for nothing.

#### The human prior $\pi(m)$

| Feature | Log-weight | Why |
|---|---|---|
| Gives check | **+0.6** | Checks grab attention |
| Winning/equal capture | **+0.7** | Magnetic |
| Losing capture | +0.15 | Still tempting, less so |
| Queen promotion | **+0.8** | Impossible to miss |
| Apparent hang (lands en prise, undefended, cheaper attacker) | **−0.9** | The engine sacrifice a human refuses |
| Quiet retreat | −0.3 | Psychologically avoided |
| Early queen sortie to a loose square (< move 10) | **−0.7** | Qa4?! then a lost queen — cost a real game |

#### The coherence penalty $\kappa(m)$

| Pattern | Log-weight |
|---|---|
| Immediate reversal (piece straight back) | **−1.6** |
| Return to a square vacated in last 2 own moves | −0.7 |
| Re-moving the piece that just moved | −0.35 |

Applied at every ELO — nobody good shuffles. Because it's additive in the exponent, a *genuinely forced* reversal (huge $\Delta$ on everything else) still wins.

### 4.3 — Temperature: the multiplier stack

$$
T = \underbrace{\mathrm{clamp}\big(\text{acpl}(E)\,(1.9 + 3.4w),\,6,\,800\big)}_{T_{\text{base}}}
\cdot g
\cdot f_{\text{open}}
\cdot f_{\text{end}}
\cdot (1 - 0.6\,c)
\cdot A(C)
\cdot (1 - 0.4\,u)
\cdot f_{\text{coast}}
\cdot f_{\text{streak}}
$$

| Factor | Formula | Effect |
|---|---|---|
| $g$ | closed-loop gain (§4.5) | tunes realized ELO onto target |
| $f_{\text{open}}$ | $0.5 + 0.45\cdot\tfrac{n}{6}$ for plies $n<6$ | near-book early |
| $f_{\text{end}}$ | $0.7$ if $\le 12$ pieces | endgames are forcing |
| $c$ — criticality | $\mathrm{clamp}\!\big(\tfrac{(v_1 - v_2) - 30}{170},0,1\big)$ | one obvious move → everyone finds it |
| $A(C)$ — risk appetite | §4.6 | relax when ahead, bear down when behind |
| $u$ — trend urgency | OLS slope of last 6 evals; $u = \min(1, \lvert\beta\rvert/100)$ if $\beta<0$ | sliding downhill → focus |
| $f_{\text{coast}}$ | $1 + \min\!\big(\tfrac{v^\star - M}{M}, 1.5\big)$ when $v^\star > M$ | anti-domination governor |
| $f_{\text{streak}}$ | $\min(1 + 0.06(k-5), 1.3)$ after $k\ge6$ best moves in a row | no suspicious perfection |

And the ceiling:

$$
L_{\max} = 5\cdot\text{acpl}(E)\cdot(1 - 0.85\,c) \quad (\times 0.5 \text{ in the first 6 plies}),\qquad L_{\max}\ge 12
$$

When the position screams for one move ($c \to 1$), $L_{\max}$ collapses toward "best move only" — a recapture gets recaptured, a mate-in-one gets played. When five moves are all fine ($c = 0$), a 1600 has ~190 cp of room to be human.

### 4.4 — Temptation injection

The engine's top-$N$ are all *reasonable*. Real weak-player blunders live further down: the poisoned pawn, the premature attack. So with probability $p = 0.15 + 0.55w$ (never in forcing positions), a few captures/checks *outside* the top-$N$ are evaluated at shallow depth and injected into the pool if their loss sits in the plausible window $60 \le \Delta \le L_{\max}$. Then the human prior decides — and the human prior *loves* captures. Blunders land where a human's would.

### 4.5 — Anti-domination governor + closed-loop controller

Each game samples a **target margin** $M \sim \mathcal{N}(260, 110)$ clamped to $[90, 650]$ cp. Below it, play normally. Above it, *coast* — the temperature rises and only moves keeping the eval above a **win floor** are allowed:

$$
\text{floor}_{\text{win}} = \mathrm{clamp}\big(0.5M + 0.3\,\gamma,\; 70,\; 320\big)
$$

so a won game is never thrown, but it's also never a 40-move rout. Some games are close, some comfortable — like a real player's.

Meanwhile a slow **proportional controller** measures the bot's own realized ELO $E_r$ (its chosen-move losses through the ACPL curve) and nudges the gain:

$$
g \leftarrow g\cdot\exp\!\Big(\mathrm{clamp}\!\big(\tfrac{E_r - E_{\text{eff}}}{900},\,-0.15,\,0.15\big)\Big), \qquad g\in[0.3, 6]
$$

Playing too strong → loosen. Too weak → tighten. It converges within a game.

### 4.6 — Opponent adaptation ⭐ *(new)*

The menu ELO is only a **prior**. The bot watches the opponent's moves, estimates their rating $E_o$ from their ACPL, and drifts the ELO it imitates toward *"a bit better than them"*:

$$
\text{conf} = \mathrm{clamp}\!\Big(\tfrac{n_o - 3}{9},0,1\Big),\qquad
E_{\text{want}} = E_t + 0.85\cdot\text{conf}\cdot\Big(\mathrm{clamp}(E_o + \varepsilon,\; E_t \pm 300) - E_t\Big)
$$

$$
E_{\text{eff}} \leftarrow E_{\text{eff}} + 0.3\,(E_{\text{want}} - E_{\text{eff}})
$$

where the **edge** $\varepsilon \sim \mathcal{N}(90, 55)$ clamped to $[-10, 200]$ is sampled per game — sometimes near parity (a genuinely close game, which we may lose), sometimes comfortable. Every ELO-driven knob above ($T_{\text{base}}$, $N$, $L_{\max}$, $w$, the controller setpoint) reads $E_{\text{eff}}$.

Then, **every single move**, the risk question. Let $\gamma = \text{conf}\cdot(E_o - E_{\text{eff}})$ be the strength gap (positive = they're better). The **cushion** is how much eval we can spend before dipping under an opponent-aware floor:

$$
C = v^\star - \mathrm{clamp}(40 + 0.35\gamma,\,-40,\,220)
$$

A stronger opponent punishes slips → keep more in hand. A weaker one gives it back → equality is fine to drift to. The cushion drives both the temperature and the ceiling:

$$
A(C) = \begin{cases} 1 + 0.5\min(1, C/250) & C \ge 0 \\ \max(0.4,\; 1 + C/400) & C < 0 \end{cases}
\qquad
L_{\max} \leftarrow \begin{cases} \min\big(L_{\max},\; 1.8\,\text{acpl} + 0.6\,C\big) & C \ge 0 \\ L_{\max}\cdot\max(0.5,\; 1 + C/500) & C < 0 \end{cases}
$$

Read it as: **ahead → relax, but never spend more than a slice of the lead in one move** (a realistic *small* mistake, not one that hands the game back). **Equal or behind → focus.** Asymmetric on purpose. That's how a human who wants to win actually plays.

### 4.7 — Estimator warm-up

Both ELO estimators use an EMA on capped CPL with a running-mean warm-up so one early move can't dominate:

$$
\alpha_n = \max\!\big(0.15,\; \tfrac{1}{n}\big), \qquad \overline{\text{cpl}}_n = \alpha_n\,\text{cpl}_n + (1-\alpha_n)\,\overline{\text{cpl}}_{n-1}
$$

### 📊 Calibration (self-play, v1)

| Target | Realized | Note |
|---|---|---|
| 800 | ~945 | structural floor — can't average huge loss without hanging pieces every move |
| 1200 | ~1400 | |
| 1600 | ~1700 | |
| 2000 | ~2080 | |
| 2400 | ~2640 | |

Real-game tuning is ongoing. The console log prints every decision:

```
[ ] move=d1a4  loss=82cp  temp=326  target=1600  eff=1671  opp=1618(+90)  realized=1694  gain=1.86  margin=157  cushion=68  crit=0.17  coast=0
```

---

## ✨ Stage 5 — The Overlay

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Kryskata-C/chessbot@main/assets/features.svg" alt="Features" width="100%"/>
</p>

A transparent PyQt6 window pinned above **everything** via native Cocoa calls (`ctypes`): `NSScreenSaverWindowLevel` (level 1000), `setIgnoresMouseEvents:YES` for full click-through, `canJoinAllSpaces` so it follows you across desktops, and it excludes itself from screen capture so it never sees its own arrows.

Toggle every visual from the animated startup menu:

| Visual | What it does |
|---|---|
| ➤ **Best-move arrow** | Animated glowing arrow for the chosen move |
| ♞ **Ghost piece slide** | Translucent piece glides along the move |
| ⚔ **Enemy reply arrow** | Predicted response, dashed orange |
| ☰ **Line preview** | Chains the next plies of the engine line |
| ≡ **Candidate moves** | Faint arrows for alternatives |
| ☠ **Threat radar** | Pulsing red glow on your pieces under attack |
| ✦ **Enemy move trail** | Fading trail of the opponent's last move |
| ▮ **Live eval bar** | Animated eval bar beside the board |

Plus a **debug board** window showing what the vision layer sees, opponent ACPL/ELO, and `Bot ELO: 1600 target · 1676 eff · 1650 realized`.

**Performance:** capture stays on the GUI thread (mss requirement), recognition runs on a worker, frames are captured only for the board region once found, and a placement-diff gate skips the engine entirely when nothing changed. Idle cost ≈ **2% CPU**.

---

## 🚀 Setup

**Requirements:** macOS (Cocoa overlay), Python 3.10+, Stockfish.

```bash
brew install stockfish
git clone https://github.com/Kryskata-C/chessbot.git
cd chessbot
pip install -r requirements.txt
```

Grant screen recording: **System Settings → Privacy & Security → Screen Recording** → enable your terminal.

### Calibrate once

Open chess.com at the **starting position** (default green/beige theme):

```bash
python3 calibrate.py
```

Extracts the 20 piece templates from *your* screen into `templates/`. (Skip it and the app auto-calibrates the first time it sees a starting position — from either side of the board — and recalibrates by itself when it later sees a starting position that the current templates can't read, i.e. you changed piece set or theme.)

### Play

```bash
python3 main.py
```

Pick your color, pick the bot strength (Novice → Master, 400–2800), toggle visuals, hit start.

| Key | Action |
|---|---|
| `Ctrl+Q` | Quit |
| `Ctrl+C` | Quit (terminal) |

### Ship it as an app

`packaging/build_app.sh` turns the checkout into a double-clickable **Chess Vision.app** (PyInstaller) with Stockfish bundled inside — no terminal, no Homebrew, no Python on the user's Mac. The opponent-rating OCR uses macOS's own Vision framework, so nothing else needs installing.

```bash
pip install -r requirements.txt pyinstaller
packaging/build_app.sh                                   # dist/Chess Vision.app + dist/ChessVision-<ver>-<arch>.zip
CV_SIGN_IDENTITY="Developer ID Application: …" packaging/build_app.sh   # hardened-runtime signed, notarisable
packaging/notarize.sh dist/ChessVision-1.0.0-arm64.zip   # after `xcrun notarytool store-credentials chessvision`
```

* Builds for the CPU of the Mac it runs on (Apple Silicon or Intel); build on each to ship both.
* Writable state lives in `~/Library/Application Support/Chess Vision/` (templates, live game logs, `chess-vision.log` with everything the terminal would have shown).
* Without a Developer ID the app is ad-hoc signed: other Macs need right-click → Open once (or `xattr -dr com.apple.quarantine`).
* First launch asks for **Screen Recording** permission for "Chess Vision" (System Settings → Privacy & Security); restart the app after granting.

---

## 🗂 Project Structure

```
chessbot/
├── main.py              # Scan loop, state machine, turn tracking, GUI dispatch
├── move_selector.py     # 🧠 The human layer — the math above lives here
├── elo_estimator.py     # ELO ⇄ ACPL curve, EMA estimators
├── engine.py            # Stockfish wrapper (MultiPV, eval, recovery)
├── board_detector.py    # HSV mask → morphology → contour → board rect
├── piece_recognizer.py  # Template matching, king recovery, FEN builder
├── capture.py           # mss screen/region capture
├── overlay.py           # PyQt6 overlay + debug board window
├── menu.py              # Animated startup menu (color, ELO, visuals)
├── auth.py / account_ui.py / config.py  # Supabase accounts: login, licence gate, admin panel
├── recorder.py          # Records every live game to live_games/ (JSONL + PGN)
├── openings.py          # Human opening repertoire (per-user favourites)
├── session.py           # Cross-game governor: keeps best-move % / ACPL human over a session
├── opponent_rating.py   # OCR of the opponent's printed rating (prior for adaptation)
├── calibrate.py         # Template extraction
├── recalibrate.py       # Quick non-interactive recalibration
├── selfplay.py          # Tuning harness: bot vs rating-capped Stockfish, streams JSON events
├── dashboard.py         # Live training dashboard (serves dashboard/index.html)
├── runstats.py          # One-line summary per self-play run
└── templates/           # Generated piece templates
```

Knobs: `SCAN_INTERVAL_MS` in `main.py` (default 400), `ChessEngine(depth=12, threads=2)`.

| Depth | ~Strength | Latency |
|---|---|---|
| 8 | ~2200 | instant |
| 12 | ~2800 | ~50 ms |
| 18 | ~3200 | 2–5 s |

---

## 🔐 Accounts (login, subscriptions, admin)

The app opens with a sign-in window; the menu only appears for a licensed user or an admin.
Accounts live in a Supabase project (free tier): email + password auth plus a `profiles` table
(`role` user|admin, `active`, `expires_at`) protected by row-level security, so the publishable key
shipped in `config.py` can't read anyone else's row. Sessions are remembered in the macOS keychain.

- New accounts are **inactive** until an admin enables them (admin panel button after sign-in:
  toggle active, set expiry, +30d, change role).
- Set `CHESS_VISION_SUPABASE_URL` (or edit `config.py`) to point at the project. The secret /
  service-role key is never used by the app.
- Schema + policies: see the SQL in the project notes (`profiles`, `is_admin()`, `handle_new_user` trigger).

---

## 🧪 Tuning harness & live dashboard

Every strength change is judged by self-play, never by feel:

```bash
./venv/bin/python selfplay.py --games 10 --target-elo 1600 --opp-elo 1600 --label "what changed"
./venv/bin/python selfplay.py --games 10 --opp-elo 1320 --opp-prior 1600   # a "1600" that plays like 1320
./venv/bin/python selfplay.py --games 20 --session                          # with the cross-game governor
./venv/bin/python runstats.py -v                                            # compare runs
```

By default the bot learns the opponent the way it does live (printed-rating prior + observed loss);
`--no-adapt` pins it, `--no-book` disables the repertoire. Each run writes PGNs to `selfplay_pgns/`
and one JSON line per move/game to `selfplay_runs/`.

```bash
./venv/bin/python dashboard.py     # then open http://localhost:8765
```

The dashboard follows the newest run: live board with the eval bar, the decision line (effective /
realized / opponent rating, temperature, cushion, criticality, think time), rating and eval charts for
the current game, the loss-per-move histogram for the whole run (all moves vs contested positions),
a per-game table, and an iteration table across runs. What "good" looks like: score ≥ 9/10 vs an
equal opponent, best-move 30–50 %, contested ACPL 22–40, zero SUSPICIOUS sacrifices.

---

## 🔧 Troubleshooting

**"No board found"** — board fully visible on any monitor (each display is scanned in turn), default green/beige theme, screen recording permitted.

**Poor recognition** — show the starting position (new game) with nothing covering the board: the app re-cuts its templates when the position on screen is plainly the start but the templates disagree. From a checkout you can also re-run `python3 calibrate.py`.

**Packaged app does nothing / quits** — read `~/Library/Application Support/Chess Vision/chess-vision.log`; the usual cause is Screen Recording permission not granted yet.

**Wrong color** — color is inferred on first scan; restart between games if you switch.

---

## 🔭 Roadmap — where the math goes next

The v1 human layer is a hand-tuned generative model. The plan is to make each piece **learned or principled** rather than tuned:

**📖 Opening book, properly.** Replace "engine + tight temperature" in the opening with a Polyglot book and popularity-weighted sampling, $P(m) \propto n_m^{1/\tau(E)}$, where $n_m$ is how often humans at rating $E$ actually play $m$. Lower ELO → hotter $\tau$ → more sidelines. Kills the last "why did it play a3" tells.

**⏱ Thinking-time model.** Move latency conditioned on the position: $\log t \sim \mathcal{N}\big(\mu(c, \Delta, \text{phase}),\, \sigma^2\big)$ — fast in forcing positions, slow when criticality is low and the eval just swung. Currently moves are suggested instantly; timing is the biggest remaining tell for anyone watching a clock.

**📈 Bayesian opponent model.** The EMA on ACPL is a point estimate. Replace it with a posterior over $E_o$ — a normal-normal update per move with per-move variance from position complexity — so `conf` becomes real posterior width, and the edge $\varepsilon$ can be chosen against uncertainty rather than a fixed ramp.

**🎯 Blunder hazard, not just temperature.** Real human error is bimodal: mostly small imprecision, occasionally one big miss. Model the big miss explicitly as a per-move hazard $h = \sigma\big(\beta_0 + \beta_1 (E_{\text{eff}} - E^\star) + \beta_2\,\text{complexity} + \beta_3\,\text{clock}\big)$ instead of stretching the softmax tail; fit $\beta$ from real games.

**🧬 Learned prior.** Swap the hand-written $\pi(m)$ table for a Maia-style rating-conditioned policy — "what would a 1500 *actually* click here" — as the prior, keeping the softmax-over-regret as the safety envelope.

**🎛 Calibration from real games.** Regress realized ACPL against target ACPL over logged games and fit the $T_{\text{base}}$ scale and $L_{\max}$ multiplier per rating band, replacing the self-play table above.

**🎨 Style vectors.** Per-game sampled traits — aggression, simplification tendency, exchange appetite — added as extra terms in $\pi(m)$, so consecutive games don't share a fingerprint. Simplify against tacticians, press against passive players.

**🖥 Windows / Linux overlays.**

---

## 🧰 Tech Stack

[mss](https://github.com/BoboTiG/python-mss) · [OpenCV](https://opencv.org/) · [python-chess](https://python-chess.readthedocs.io/) · [Stockfish](https://stockfishchess.org/) via [stockfish](https://pypi.org/project/stockfish/) · [PyQt6](https://www.riverbankcomputing.com/software/pyqt/)

---

## 📜 License

Proprietary — all rights reserved. Source code is visible for educational purposes only. See [LICENSE](LICENSE).
