"""Print a one-line summary per self-play run (and the per-game opponent
estimates), for judging tuning iterations without the dashboard."""
import glob, json, os, sys
files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfplay_runs", "run_*.jsonl")))
args = [a for a in sys.argv[1:] if a != "-v"]
if args:
    files = files[-int(args[0]):]
for f in files:
    evs = [json.loads(l) for l in open(f) if l.strip()]
    head = evs[0]; games = [e for e in evs if e.get("ev") == "game"]; n = len(games)
    done = any(e.get("ev") == "done" for e in evs)
    if not n:
        print(f"{head['label'][:46]:46s} (no games yet)"); continue
    mean = lambda k: sum(g.get(k, 0) for g in games) / n
    susp = sum(1 for g in games for s in g.get("sacs", []) if "SUSPICIOUS" in s)
    wdl = f"W{sum(g['score']==1 for g in games)}D{sum(g['score']==0.5 for g in games)}L{sum(g['score']==0 for g in games)}"
    bp = [g.get("best_pct", 0) for g in games]
    spread = (sum((b - mean("best_pct")) ** 2 for b in bp) / n) ** 0.5
    print(f"{head['label'][:46]:46s} {sum(g['score'] for g in games):4.1f}/{head['games']} {wdl} best {mean('best_pct'):4.1f}% (sd {spread:4.1f}, {min(bp):.0f}-{max(bp):.0f}) acpl {mean('acpl'):4.1f} contested {mean('c_acpl'):4.1f} worst {max(g['worst'] for g in games):3d} mist {mean('mistakes'):.1f} susp {susp} {'' if done else '(running)'}")
    conv = []
    for g in games:
        moves = [e for e in evs if e.get("ev") == "move" and e.get("g") == g["g"]]
        sign = 1 if g.get("bot_white") else -1
        won_at = next((m["ply"] for m in moves if m.get("eval_w") is not None and sign * m["eval_w"] >= 500
                       and sum(c.isalpha() for c in m["fen"].split()[0]) <= 12), None)
        if won_at is not None and g["score"] == 1:
            conv.append(g["plies"] - won_at)
    if conv:
        print(f"   won-endgame conversion: {sum(conv)/len(conv):.0f} plies avg over {len(conv)} games (max {max(conv)})")
    if "-v" in sys.argv:
        print("   opp est:", [g.get("opp_est") for g in games], " eff:", [g.get("eff") for g in games], " plies:", [g["plies"] for g in games])
