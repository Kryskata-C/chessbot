"""Publish the training dashboard into the website (chess-vision-site).

    ./venv/bin/python export_training.py [path/to/chess-vision-site]

Writes a static snapshot of every self-play run in selfplay_runs/ to
<site>/training/data/ (runs.json + one <run>.json per run) and regenerates
<site>/training.html from dashboard/index.html, wrapped in the site's nav and
gated to admin accounts. The page uses the local dashboard.py (port 8765)
live when it is running, otherwise the snapshot.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

from dashboard import RUNS, _read_events, _summaries

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "chess-vision-site"))
DATA = os.path.join(SITE, "training", "data")
SRC = os.path.join(ROOT, "dashboard", "index.html")

NAV = """<div style="position: relative; background: #0e0d0b; color: #efe9dd; min-height: 100vh;" id="page">
  <div class="grain"></div>
  <div class="wrap" style="display: flex; align-items: center; justify-content: space-between; height: 84px;">
    <a href="index.html" style="display: inline-flex; align-items: center; gap: 12px; font-family: var(--serif); font-size: 24px; letter-spacing: -0.01em;"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="oklch(0.88 0.21 128)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z"></path><circle cx="12" cy="12" r="3"></circle></svg><span>Chess Vision</span></a>
    <div class="nav-links" style="display: flex; gap: 34px; font-size: 15px; color: #9b9384;">
      <a href="index.html#why">Why it feels human</a>
      <a href="index.html#how">How it works</a>
      <a href="index.html#pricing">Pricing</a>
    </div>
    <span id="nav-auth"><a href="account.html" class="chip"><span class="avatar">·</span><span class="chip-text">Signing in…</span></a></span>
  </div>
"""

# Replaces the dashboard's fetch layer: live server when reachable, snapshot otherwise.
DATA_LAYER = """
/* ---------- data source: live dashboard.py if it is running, else the exported snapshot ---------- */
const LIVE_BASE = "http://127.0.0.1:8765";
// https page -> local http server: Chrome needs the target address space declared (Local Network Access)
const LIVE_OPTS = { cache: "no-store", targetAddressSpace: "loopback" };
const SNAPSHOT = "training/data";
const src = { live: null, cache: {} };
async function detectLive() {
  try { const r = await fetch(LIVE_BASE + "/api/runs", LIVE_OPTS); if (r.ok) { src.live = true; return; } } catch (e) {}
  src.live = false;
}
async function apiRuns() {
  if (src.live === null) await detectLive();
  if (src.live) return (await fetch(LIVE_BASE + "/api/runs", LIVE_OPTS)).json();
  if (!src.cache.runs) src.cache.runs = (await fetch(SNAPSHOT + "/runs.json")).json();
  return src.cache.runs;
}
async function apiRun(id, from) {
  if (src.live) return (await fetch(`${LIVE_BASE}/api/run?id=${encodeURIComponent(id)}&from=${from}`, LIVE_OPTS)).json();
  if (!src.cache[id]) src.cache[id] = (await fetch(`${SNAPSHOT}/${encodeURIComponent(id)}.json`)).json();
  const all = (await src.cache[id]).events;
  return { events: all.slice(from), next: all.length };
}
function markSource() {
  const pill = $("source"); if (!pill) return;
  pill.textContent = src.live ? "live · dashboard.py" : "snapshot · " + (src.cache.exported || "");
  pill.style.color = src.live ? "var(--acc)" : "";
}
"""


def export_data() -> int:
    os.makedirs(DATA, exist_ok=True)
    for old in os.listdir(DATA):
        if old.endswith(".json"):
            os.remove(os.path.join(DATA, old))
    runs = _summaries()
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(DATA, "runs.json"), "w") as f:
        json.dump(runs, f, separators=(",", ":"))
    for r in runs:
        evs = _read_events(os.path.join(RUNS, r["id"] + ".jsonl"))
        with open(os.path.join(DATA, r["id"] + ".json"), "w") as f:
            json.dump({"events": evs, "next": len(evs)}, f, separators=(",", ":"))
    with open(os.path.join(DATA, "meta.json"), "w") as f:
        json.dump({"exported": stamp, "runs": len(runs)}, f)
    return len(runs)


def scope_css(css: str) -> str:
    """Prefix every selector with #tr so the dashboard's styles stay identical
    without leaking into (or being overridden by) the site's styles.css."""
    res, depth, buf, in_at = [], 0, "", False
    j = 0
    while j < len(css):
        ch = css[j]
        if ch == "{":
            if depth == 0:
                sel = buf.strip(); buf = ""
                if sel.startswith("@media"):
                    in_at = True; res.append(sel + " {")
                elif sel.startswith("@keyframes"):
                    in_at = "kf"; res.append(sel + " {")
                else:
                    res.append(_prefix(sel) + " {")
            elif in_at == "kf":
                res.append(buf + "{"); buf = ""
            else:  # inside @media
                res.append(_prefix(buf.strip()) + " {"); buf = ""
            depth += 1
        elif ch == "}":
            depth -= 1
            res.append(buf + "}"); buf = ""
            if depth == 0:
                in_at = False
        else:
            buf += ch
        j += 1
    return "".join(res)


def _prefix(sel: str) -> str:
    parts = []
    for s in sel.split(","):
        s = s.strip()
        if not s:
            continue
        if s == ":root":
            parts.append("#tr")
        elif s in ("html, body", "html", "body"):
            parts.append("#tr")
        elif s == "*":
            parts.append("#tr *")
        else:
            parts.append("#tr " + s)
    return ", ".join(parts)


def build_page() -> None:
    html = open(SRC).read()
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    body = re.search(r"<body>(.*?)<script>", html, re.S).group(1)
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    css = css.replace("html, body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); }\n  body { padding: 20px 24px 40px; }",
                      "body { color: var(--ink); font-family: var(--sans); padding: 20px 0 40px; }")
    # keyframe names must not collide with the site's (styles.css has its own `rise`)
    css = re.sub(r"@keyframes (\w+)", r"@keyframes tr-\1", css)
    css = re.sub(r"animation: (\w+)", r"animation: tr-\1", css)
    css = scope_css(css)
    # classes the site's styles.css also styles (.stat, .btn, .bar …) are reverted inside #tr
    # so the dashboard's own rules apply on a clean slate
    site_css = open(os.path.join(SITE, "styles.css")).read()
    site_classes = set(re.findall(r"\.([a-zA-Z][\w-]*)", site_css))
    dash_classes = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))
    clash = sorted((site_classes & dash_classes) - {"mono"})
    if clash:
        css = css.replace("}#tr * {", "}" + ", ".join(f"#tr .{c}" for c in clash) + " { all: revert; box-sizing: border-box; }#tr * {", 1)
        assert "all: revert" in css
    body = body.replace('<span class="pill mono" id="params"></span>',
                        '<span class="pill mono" id="params"></span>\n  <span class="pill mono" id="source"></span>')
    js = js.replace('const r = await fetch(`/api/run?id=${encodeURIComponent(state.sel)}&from=${state.next}`);\n    const j = await r.json();',
                    'const j = await apiRun(state.sel, state.next);')
    js = js.replace('const r = await fetch("/api/runs"); const runs = await r.json();',
                    'const runs = await apiRuns(); markSource();')
    js = js.replace('try { const r = await fetch(`/api/run?id=${encodeURIComponent(id)}&from=0`); evs = (await r.json()).events; } catch (e) {}',
                    'try { evs = (await apiRun(id, 0)).events; } catch (e) {}')
    js = js.replace('refreshRuns();\nsetInterval(poll, 1000);\nsetInterval(refreshRuns, 4000);',
                    'async function start() {\n  const gate = await CV.requireAdmin(); if (!gate) return;\n  CV.navAuth();\n  $("tr").hidden = false;\n  try { src.cache.exported = (await (await fetch(SNAPSHOT + "/meta.json")).json()).exported; } catch (e) {}\n  await refreshRuns();\n  setInterval(() => { if (src.live) poll(); }, 1000);\n  setInterval(() => { if (src.live) refreshRuns(); }, 4000);\n}\nstart();')
    for needle in ("apiRun(state.sel, state.next)", "apiRuns(); markSource();", "evs = (await apiRun(id, 0)).events", "start();"):
        assert needle in js, f"dashboard/index.html changed shape; update export_training.py ({needle})"
    page = f"""<!doctype html>
<html lang="en" class="no-js">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chess Vision — training</title>
  <meta name="robots" content="noindex">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Instrument+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
  <link rel="stylesheet" href="styles.css">
  <!-- GENERATED by chessbot/export_training.py from chessbot/dashboard/index.html — edit there, then re-run. -->
  <style>{css}</style>
</head>
<body>
{NAV}
  <div id="tr" style="width: calc(100% - 48px); margin: 0 auto;" hidden>{body}</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.49.4/dist/umd/supabase.min.js"></script>
<script src="sb.js"></script>
<script>{DATA_LAYER}{js}</script>
</body>
</html>
"""
    with open(os.path.join(SITE, "training.html"), "w") as f:
        f.write(page)


if __name__ == "__main__":
    if not os.path.isdir(SITE):
        sys.exit(f"site folder not found: {SITE}")
    n = export_data()
    build_page()
    print(f"exported {n} runs to {DATA} and wrote {os.path.join(SITE, 'training.html')}")
