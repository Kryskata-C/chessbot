"""Live training dashboard: serves dashboard/index.html and the self-play
event streams in selfplay_runs/ as JSON.

    ./venv/bin/python dashboard.py            # http://localhost:8765
"""

from __future__ import annotations

import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "selfplay_runs")
STATIC = os.path.join(ROOT, "dashboard")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


def _read_events(path: str, start: int = 0) -> list[dict]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line still being written
    return out


def _summaries() -> list[dict]:
    runs = []
    if not os.path.isdir(RUNS):
        return runs
    for name in sorted(os.listdir(RUNS)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(RUNS, name)
        evs = _read_events(path)
        if not evs or evs[0].get("ev") != "run":
            continue
        head = evs[0]
        games = [e for e in evs if e.get("ev") == "game"]
        done = any(e.get("ev") == "done" for e in evs)
        n = len(games)
        runs.append({
            "id": name[:-6], "label": head.get("label"), "target": head.get("target"),
            "opp": head.get("opp"), "prior": head.get("prior"), "adapt": head.get("adapt"),
            "games": head.get("games"), "played": n, "done": done,
            "score": sum(g.get("score", 0) for g in games),
            "best_pct": round(sum(g.get("best_pct", 0) for g in games) / n, 1) if n else None,
            "acpl": round(sum(g.get("acpl", 0) for g in games) / n, 1) if n else None,
            "c_acpl": round(sum(g.get("c_acpl", 0) for g in games) / n, 1) if n else None,
            "worst": max((g.get("worst", 0) for g in games), default=0),
            "wins": sum(1 for g in games if g.get("score") == 1),
            "draws": sum(1 for g in games if g.get("score") == 0.5),
            "losses": sum(1 for g in games if g.get("score") == 0),
            "suspicious": sum(1 for g in games for s in g.get("sacs", []) if "SUSPICIOUS" in s),
            "started": head.get("t"), "mtime": os.path.getmtime(path),
        })
    return runs


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=STATIC, **kw)

    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/runs":
            return self._json(_summaries())
        if u.path == "/api/run":
            q = parse_qs(u.query)
            run_id = q.get("id", [""])[0]
            start = int(q.get("from", ["0"])[0])
            path = os.path.join(RUNS, os.path.basename(run_id) + ".jsonl")
            if not os.path.isfile(path):
                return self._json({"events": [], "next": start})
            evs = _read_events(path, start)
            return self._json({"events": evs, "next": start + len(evs)})
        if u.path == "/":
            self.path = "/index.html"
        return super().do_GET()


if __name__ == "__main__":
    print(f"Chess Vision training dashboard: http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
