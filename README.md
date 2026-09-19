<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Kryskata-C/chessbot@main/assets/banner.svg" alt="Chess Vision" width="100%"/>
</p>

<h3 align="center">The first chess bot that plays like a human.</h3>

<p align="center">
  <strong>Chess Vision watches your chess.com board, runs Stockfish underneath, and then deliberately <em>de-optimises</em> the engine through a model of human play, so the move it suggests is the one a slightly-better-than-your-opponent person would find, not a 3500-rated machine.</strong>
</p>

<p align="center">
  <a href="https://chessvision.cc"><img src="https://img.shields.io/badge/chessvision.cc-%E2%82%AC10.99%20%2F%20month-b9f24a?style=for-the-badge"/></a>
  <img src="https://img.shields.io/badge/macOS-12%2B%20Apple%20Silicon-000000?style=for-the-badge&logo=apple&logoColor=white"/>
  <img src="https://img.shields.io/badge/Windows-10%20%2F%2011-0078d4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Engine-Stockfish-47a341?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Proprietary-f7b731?style=for-the-badge"/>
</p>

<p align="center">
  <code>screen → board → pieces → legal game state → Stockfish MultiPV → softmax over regret → human prior → opponent-adaptive risk → think timer → overlay</code>
</p>

---

## What it is

A desktop app for Mac and Windows, sold as a monthly subscription at [chessvision.cc](https://chessvision.cc). You open chess.com in your browser, start the app, pick a strength, and it draws the move it would play straight onto your board: a glowing arrow, a ghost piece, the opponent's likely reply, an eval bar, threat warnings. You still make every move yourself.

What makes it different is everything *after* the engine. Perfect moves in a 1500 game stand out a mile. Chess Vision plays a bit better than the person across the board, with small, natural slips in the right places, and on every move it asks one question:

> *Is this a moment to be sharp, or am I far enough ahead against this opponent to play something relaxed?*

This repository is the whole product: the app, the packaging and release pipeline, the self-play tuning harness, and the Supabase side of accounts and billing. The website lives in [chess-vision-site](https://github.com/Kryskata-C/chess-vision-site).

---

## The pipeline

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Kryskata-C/chessbot@main/assets/architecture.svg" alt="Architecture" width="100%"/>
</p>

| Stage | What happens | Where |
|---|---|---|
| **Vision** | Finds the board on any display, reads all 64 squares | `board_detector.py`, `piece_recognizer.py`, `capture.py` |
| **State** | Turns noisy frames into a legal game, tracks whose turn it is | `main.py` |
| **Engine** | Stockfish MultiPV, depth 12, top-N candidates | `engine.py` |
| **Human layer** | Picks the move a human of rating *E* would play against *this* opponent | `move_selector.py`, `elo_estimator.py`, `openings.py`, `session.py` |
| **Timing** | Decides how long a human would think, paced to the clock | `think_time.py` |
| **Overlay** | Draws it, click-through, above everything, invisible to its own capture | `overlay.py`, `native.py` |

### Vision

The board is found by masking chess.com's two default square colours in HSV, closing and opening the mask, and taking the largest roughly square contour. Pieces are read by normalised cross-correlation against 20 templates (six piece types, two colours, on light and dark squares): a square holds a piece when the best match scores at least 0.55. Kings get a recovery pass at 0.25, because chess.com's check glow drags their score down and a position without a king is not a position.

Templates are cut from the screen the first time the app sees a starting position, from either side of the board, and re-cut whenever a starting position is plainly on screen but the templates disagree. That's why the app only knows chess.com's **Green board with Neo pieces**, the defaults, and why it shows a reminder card about them after sign-in.

### State

No single frame is trusted. A change touching one square can't be a completed move, so it's dropped. Multi-square changes are decomposed into arrivals per colour and matched against every legal move, and legal two-move sequences for when a frame was skipped, on a `python-chess` board that shadows the real game. A placement is accepted only if the diff *is* a legal move, and a fuzzy match or resync has to be seen stable twice. Zero hallucinated positions; a promotion pop-up or a piece caught mid-animation can't rewrite the game.

### Engine

Stockfish in MultiPV mode returns the top-N candidates with centipawn evals; mates map to ±100000. The human layer decides how many candidates it wants to see, `N = clamp(round(5 + (1900 − E) / 110), 4, 16)`: a 2000 chooses from about 4 moves, an 800 weighs about 15, most of them bad.

---

## The human layer

`move_selector.py` is a stochastic policy over engine candidates, parametrised by a rating *E* and shaped by a handful of coupled mechanisms. Every constant below was set by self-play, never by feel.

**Rating ⇄ accuracy.** One empirical curve anchors everything, linking rating to average centipawn loss:

$$E(\text{acpl}) = 4034 - 667\ln(\text{acpl}), \qquad \text{acpl}(E) = e^{(4034 - E)/667}$$

A 2000 loses about 21 cp a move, a 1200 about 70, an 800 about 127. The same curve runs both ways: it estimates the opponent from their moves and the bot from its own.

**Softmax over regret.** With candidate evals $v_m$, best $v^\star$, and regret $\Delta_m = v^\star - v_m$, the move is sampled from

$$P(m) \propto \exp\!\Big(-\frac{\Delta_m}{T} + w\,\pi(m) + \kappa(m)\Big), \qquad \Delta_m \le L_{\max}$$

- $T$ is the temperature: how much regret we're willing to spend. It's a base value from the rating curve multiplied by a stack of situational factors: near-book in the first plies, tighter in endgames, tighter when one move is obviously forced, looser with a comfortable lead, tighter when the eval is sliding, looser after a suspicious run of perfect moves.
- $\pi(m)$ is a human prior: how tempting a move *looks*. Checks, captures and promotions attract; quiet retreats, early queen sorties and moves that land a piece en prise repel. It's scaled by weakness $w$, because weak players react to how a move looks and strong players calculate.
- $\kappa(m)$ is a coherence penalty on un-human sequences: shuffling a piece straight back, returning to a square just vacated.
- $L_{\max}$ is a hard ceiling on single-move error. It collapses toward "best move only" when the position screams for one move, and it's what makes a 1600 never hang a piece for nothing.

**Temptation injection.** The engine's top-N are all reasonable. Real weak-player blunders live further down: the poisoned pawn, the premature attack. With a rating-dependent probability, a few captures and checks outside the top-N are evaluated at shallow depth and injected if their loss sits in a plausible window, and the prior, which loves captures, does the rest. Blunders land where a human's would.

**Anti-domination governor.** Each game samples a target margin. Above it the bot coasts, letting the temperature rise while only allowing moves that keep the eval above a win floor. A won game is never thrown, and it's never a 40-move rout either. Some games are close, some comfortable, like a real player's.

**Closed-loop controller.** A slow proportional controller measures the bot's realised rating from its own move losses and nudges the temperature gain so the realised rating converges on the target within a game.

**Opponent adaptation.** The menu rating is only a prior. The bot reads the opponent's printed rating off chess.com with OCR, watches their moves, estimates their rating from their loss, and drifts the rating it imitates toward "a bit better than them". The edge is sampled per game, so sometimes it's near parity and the game is genuinely close. Then on every move it computes a cushion, how much eval it can spend before dipping under an opponent-aware floor, and lets that drive both temperature and ceiling. Ahead: relax, but never spend more than a slice of the lead in one move. Equal or behind: focus. Asymmetric on purpose.

**Opening repertoire** (`openings.py`). Engines pick openings fresh every game; people play the same handful of lines for years. The bot follows a small human book for its first moves, with per-installation favourites, then hands over to the engine.

**Session governor** (`session.py`). Per-game randomness makes single games vary the way a person's do, but a best-move rate sitting at 55% for a month is a tell no single game shows. The governor watches the last few games and keeps the averages human too.

**Think time** (`think_time.py`). The overlay's timer is paced to the clock: the player's clock is read with OCR each turn, the time control is chosen in the menu or inferred from the clock at move one, and the remaining time is split over the moves still expected, minus a reserve. Each move's share is bent by the position, obvious moves fast, real decisions slow, time trouble flattening everything toward premove speed. Simulated over thousands of games it never flags.

Every decision is printed, so a game can be read back move by move:

```
[ ] move=d1a4  loss=82cp  temp=326  target=1600  eff=1671  opp=1618(+90)  realized=1694  gain=1.86  margin=157  cushion=68  crit=0.17  coast=0
```

---

## The overlay

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Kryskata-C/chessbot@main/assets/features.svg" alt="Features" width="100%"/>
</p>

A transparent PyQt6 window pinned above everything and fully click-through, which excludes itself from screen capture so it never sees its own arrows. Every visual is a toggle on the setup card: move arrow, ghost piece, their reply, line preview, alternatives, threats, move trail, eval bar, think timer. A debug board shows what the vision layer sees and the live rating estimates.

Capture stays on the GUI thread, recognition runs on a worker, only the board region is grabbed once found, and a placement-diff gate skips the engine entirely when nothing changed. Idle cost is about 2% CPU.

---

## Running from source

Python 3.11 or newer, Stockfish, and on macOS a terminal with Screen Recording permission.

```bash
brew install stockfish                      # Windows: any stockfish.exe on PATH
git clone https://github.com/Kryskata-C/chessbot.git && cd chessbot
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python main.py
```

The app opens with a sign-in window (see Accounts), then the board reminder, then the setup card: colour, strength 400 to 2800, time control, visuals, start. `Ctrl+Q` quits.

`config.py` reads `CHESS_VISION_SUPABASE_URL`, `CHESS_VISION_SUPABASE_KEY` and `CHESS_VISION_SITE_URL` from the environment, with the production values as defaults. `SCAN_INTERVAL_MS` in `main.py` (400) and `ChessEngine(depth=12, threads=2)` are the two knobs worth knowing.

---

## Releases

One command cuts a release for both platforms:

```bash
./venv/bin/python packaging/release.py 1.2.0 --push \
  -n "new: Headline: detail" -n "improved: Headline: detail" -n "fixed: Headline: detail"
./venv/bin/python packaging/release.py 1.2.0 --refresh     # once CI has attached the Windows zip
```

It bumps `version.py`, builds the Mac app with `packaging/build_app.sh` (PyInstaller, Stockfish bundled, signed with whatever `CV_SIGN_IDENTITY` names, ad-hoc otherwise), prepends the release to the website's `releases.js` with notes, tags `vX`, pushes both repos and creates the GitHub release with the Mac zip. The tag triggers `.github/workflows/windows-build.yml`, which builds `ChessVision-X-windows-x64.zip` with `packaging/build_win.ps1` and attaches it to the same release. `--refresh` then writes the Windows link into the site.

The app checks `releases.js` after sign-in and shows a download link when a newer build exists (`updates.py`).

Platform differences sit behind one set of functions in `native.py`:

| | macOS | Windows |
|---|---|---|
| Overlay pinned, click-through, hidden from capture | Cocoa window level, `setIgnoresMouseEvents:`, `NSWindowSharingNone` | `WS_EX_TRANSPARENT` + `WS_EX_TOPMOST`, `SetWindowDisplayAffinity` |
| Screen-recording permission | TCC prompt on launch, relaunch after granting | none needed |
| Remembered session | Keychain | Credential Manager |
| Opponent-rating and clock OCR | Apple Vision | Windows.Media.Ocr (`requirements-win-ocr.txt`), else tesseract |
| Data directory | `~/Library/Application Support/Chess Vision` | `%LOCALAPPDATA%\Chess Vision` |

Writable state (templates, live game logs, `chess-vision.log`, `prefs.json`) lives in the data directory. The Windows build has only been exercised in CI so far, not on a real PC.

---

## Accounts and billing

The app and the website share one Supabase project: email and password auth plus a `profiles` table (`role` user | friend | admin, `active`, `expires_at`, Stripe ids) behind row-level security, so the publishable key shipped in `config.py` can't read anyone else's row. Licence rules are identical in `auth.py` and the site's `sb.js`:

- **user** needs an active subscription
- **friend** has unlimited access and no admin powers
- **admin** has everything, including the accounts panel (in the app after sign-in and at `/admin` on the site)

Billing is Stripe with no custom checkout. The site's Subscribe button is a Stripe Payment Link carrying the user's id; the dashboard's Manage subscription link is the Customer Portal. `supabase/functions/stripe-webhook` is a Supabase Edge Function that verifies Stripe's signature and flips `active` and `expires_at` on checkout, renewal, failed payment and cancellation, with a day of grace and an idempotency table. Its README has the full setup and testing checklist.

SQL to run once in a fresh project, all in `supabase/`: `friend_role.sql`, `games.sql`, `stripe.sql`. Deploy the function with `supabase functions deploy stripe-webhook` after `supabase link`; `config.toml` already disables JWT verification for it.

**Game history.** Every finished live game is uploaded to `public.games` by `stats.py`, and the website dashboard shows results, accuracy and a recent-games table. Checkmate, stalemate and material draws come from the board; resignations, flags, agreed draws and aborts are read off chess.com's game-over dialog by `result_reader.py`.

---

## Tuning

Every strength change is judged by self-play against rating-capped Stockfish:

```bash
./venv/bin/python selfplay.py --games 10 --target-elo 1600 --opp-elo 1600 --label "what changed"
./venv/bin/python selfplay.py --games 10 --opp-elo 1320 --opp-prior 1600   # a "1600" that plays like 1320
./venv/bin/python selfplay.py --games 20 --session                          # with the cross-game governor
./venv/bin/python runstats.py -v                                            # compare runs
./venv/bin/python dashboard.py                                              # live dashboard on :8765
```

Runs write PGNs to `selfplay_pgns/` and one JSON line per move to `selfplay_runs/`. The dashboard follows the newest run: live board, the decision line, rating and eval charts, loss histograms, per-game and per-iteration tables. `export_training.py` snapshots the runs into the website's admin-only Training tab.

What good looks like against an equal opponent: score of 9 out of 10 or better, best-move rate between 30 and 50 percent, contested ACPL between 22 and 40, zero suspicious sacrifices. Tuning is closed as of 1.1; the current constants ship.

---

## Project structure

```
chessbot/
├── main.py                 # scan loop, state machine, turn tracking, app entry
├── move_selector.py        # the human layer
├── elo_estimator.py        # rating ⇄ ACPL curve, estimators
├── openings.py             # human opening repertoire
├── session.py              # cross-game governor
├── think_time.py           # clock OCR + think timer
├── engine.py               # Stockfish wrapper (MultiPV, mates, recovery)
├── board_detector.py       # HSV mask → contour → board rect
├── piece_recognizer.py     # template matching, king recovery, auto-calibration
├── capture.py              # mss screen/region capture
├── result_reader.py        # game-over dialog OCR
├── opponent_rating.py      # OCR of the opponent's printed rating
├── overlay.py              # overlay + debug board
├── native.py               # macOS / Windows specifics behind one API
├── menu.py                 # setup card
├── board_setup.py          # board-theme reminder after sign-in
├── account_ui.py, auth.py  # sign-in, licence gate, admin panel (Supabase)
├── permissions.py          # Screen Recording flow (macOS)
├── updates.py              # newer-build check against the website
├── stats.py                # uploads finished games
├── recorder.py             # live game logs (JSONL + PGN)
├── ui_theme.py             # the app's visual system
├── config.py, paths.py, version.py
├── selfplay.py, runstats.py, dashboard.py, export_training.py   # tuning
├── calibrate.py, debug_*.py                                     # dev tools
├── packaging/              # build_app.sh, build_win.ps1, release.py, notarize.sh
├── supabase/               # SQL migrations, stripe-webhook Edge Function
└── .github/workflows/      # Windows build on tags
```

---

## Troubleshooting

**No board found.** The board must be fully visible on some display, on chess.com's Green theme, with screen recording allowed. Each display is scanned in turn.

**Pieces misread.** Show a starting position with nothing covering the board and the app re-cuts its templates. From source you can also run `calibrate.py`.

**Packaged app quits or does nothing.** Read `chess-vision.log` in the data directory. On a Mac the usual cause is Screen Recording not granted, and macOS forgets the grant after an update: switch Chess Vision off and on in that list.

**Wrong colour.** Auto-detect infers it on the first scan. Pick White or Black on the setup card if you switch sides mid-session.

---

## Roadmap

- **Any board theme.** Theme-independent board finding, so the reminder card can go.
- **Bayesian opponent model.** A posterior over the opponent's rating instead of an EMA point estimate, so confidence is real posterior width.
- **Blunder hazard.** Human error is bimodal: mostly small imprecision, occasionally one big miss. Model the big miss as an explicit per-move hazard fitted from real games, rather than stretching the softmax tail.
- **Learned prior.** A rating-conditioned policy in the style of Maia as the prior, with softmax-over-regret kept as the safety envelope.
- **Style vectors.** Per-game traits, aggression, simplification, exchange appetite, so consecutive games don't share a fingerprint.
- **Windows on real hardware.** Overlay click-through, capture exclusion and 125/150% display scaling all still need a real PC.

---

## Tech

[mss](https://github.com/BoboTiG/python-mss) · [OpenCV](https://opencv.org/) · [python-chess](https://python-chess.readthedocs.io/) · [Stockfish](https://stockfishchess.org/) · [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) · [Supabase](https://supabase.com/) · [Stripe](https://stripe.com/) · [Netlify](https://www.netlify.com/)

## License

Proprietary, all rights reserved. The source is public for reading only: no copying, reuse, modification, redistribution, or commercial use without written permission. See [LICENSE](LICENSE).
