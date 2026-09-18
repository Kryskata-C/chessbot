"""Update check: read the website's releases.js (written by
packaging/release.py) and say whether a newer build than this one exists.
Network failures are silent — the check must never get in the way."""

from __future__ import annotations

import json
import re
import sys

from config import SITE_URL
from version import __version__


def _key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def is_newer(candidate: str, current: str = __version__) -> bool:
    try:
        return _key(candidate) > _key(current)
    except ValueError:
        return False


def fetch_latest(timeout: float = 6.0) -> dict | None:
    """Newest release entry {version, date, mac, win, notes} or None."""
    try:
        import httpx  # bundled with the supabase client; ships its own CA bundle
        r = httpx.get(f"{SITE_URL}/releases.js", timeout=timeout,
                      headers={"Cache-Control": "no-cache"})
        r.raise_for_status()
        m = re.search(r"window\.CV_RELEASES\s*=\s*(\[.*?\]);", r.text, re.S)
        releases = json.loads(m.group(1)) if m else []
        return releases[0] if releases else None
    except Exception as e:
        print(f"update check skipped: {e}")
        return None


def check() -> dict | None:
    """{version, url} when a newer build is published, else None."""
    latest = fetch_latest()
    if not latest or not is_newer(str(latest.get("version", ""))):
        return None
    url = latest.get("mac") if sys.platform == "darwin" else latest.get("win")
    return {"version": latest["version"], "url": url or f"{SITE_URL}/changelog.html"}
