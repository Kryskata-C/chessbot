#!/usr/bin/env python3
"""Cut a release: bump version.py, build the Mac app, publish the zip and a
changelog entry to the website, commit and tag.

    packaging/release.py 1.1.0                       # notes = commit subjects since the last tag
    packaging/release.py 1.1.0 -n "new: Headline: the detail" -n "improved: Headline: detail" -n "fixed: ..."
    packaging/release.py 1.1.0 --no-build            # site entry + tag only (zip built elsewhere)
    packaging/release.py 1.1.0 --push                # also push both repos (tags included)

What it does, in order:
  1. writes version.py, commits "Version X" in chessbot
  2. packaging/build_app.sh (unless --no-build) -> dist/ChessVision-X-arm64.zip
  3. prepends the release to <site>/releases.js (version, date, download
     URLs on GitHub Releases, notes)
  4. commits the site, tags chessbot vX (the tag also triggers the Windows
     CI build, which attaches its zip to the same GitHub release)
  5. with --push: pushes both repos and creates the GitHub release with the Mac zip

    packaging/release.py 1.1.0 --refresh            # later: pick up the Windows zip's URL
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
RELEASE_BASE = "https://github.com/Kryskata-C/chessbot/releases/download"


def sh(*cmd, cwd=ROOT, capture=False) -> str:
    r = subprocess.run(cmd, cwd=cwd, check=True, text=True,
                       capture_output=capture)
    return (r.stdout or "").strip() if capture else ""


def current_version() -> str:
    ns = {}
    with open(os.path.join(ROOT, "version.py")) as f:
        exec(f.read(), ns)
    return ns["__version__"]


def last_tag() -> str | None:
    try:
        return sh("git", "describe", "--tags", "--abbrev=0", "--match", "v*", capture=True)
    except subprocess.CalledProcessError:
        return None


def parse_note(text: str) -> dict:
    """'kind: Title: detail' -> {kind, title, detail}. kind defaults to new,
    detail may be empty. The site shows kind as a label, title bold."""
    kind = "new"
    m = re.match(r"^(new|improved|fixed)\s*:\s*(.+)$", text, re.I)
    if m:
        kind, text = m.group(1).lower(), m.group(2)
    title, _, detail = text.partition(": ")
    return {"kind": kind, "title": title.strip(), "detail": detail.strip()}


def notes_from_git(since: str | None) -> list[str]:
    rng = f"{since}..HEAD" if since else "HEAD"
    out = sh("git", "log", "--format=%s", rng, capture=True)
    notes = []
    for line in out.splitlines():
        line = line.strip()
        # Housekeeping commits are not release notes.
        if not line or re.match(r"^(README|Version|Merge|Bump|WIP)\b", line, re.I):
            continue
        notes.append(line.split(";")[0].rstrip("."))
    return notes


def update_releases_js(path: str, entry: dict) -> None:
    """releases.js holds `window.CV_RELEASES = [ ... ];` newest first."""
    releases = []
    if os.path.exists(path):
        text = open(path).read()
        m = re.search(r"window\.CV_RELEASES\s*=\s*(\[.*?\]);", text, re.S)
        if m:
            releases = json.loads(m.group(1))
    releases = [r for r in releases if r["version"] != entry["version"]]
    releases.insert(0, entry)
    body = json.dumps(releases, indent=2, ensure_ascii=False)
    with open(path, "w") as f:
        f.write("/* Release history, newest first. Written by chessbot/packaging/release.py;\n"
                "   read by sb.js (download links, current version) and changelog.html.\n"
                "   mac / win are file names inside downloads/ (null = no build for that platform). */\n"
                f"window.CV_RELEASES = {body};\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("-n", "--note", action="append", default=[], help="release note (repeatable)")
    ap.add_argument("--site", default=os.path.join(ROOT, "..", "chess-vision-site"))
    ap.add_argument("--no-build", action="store_true", help="skip build_app.sh; use an existing dist zip if any")
    ap.add_argument("--push", action="store_true", help="push both repos and the tag")
    ap.add_argument("--refresh", action="store_true", help="only re-read the GitHub release's assets into releases.js")
    a = ap.parse_args()

    version = a.version.lstrip("v")
    if a.refresh:
        return refresh(version, os.path.abspath(a.site))
    if not SEMVER.match(version):
        sys.exit(f"version must look like 1.2.3, got {a.version}")
    site = os.path.abspath(a.site)
    if not os.path.isdir(os.path.join(site, "downloads")):
        sys.exit(f"site not found at {site} (use --site)")
    if sh("git", "status", "--porcelain", capture=True):
        sys.exit("chessbot has uncommitted changes — commit them first")
    if f"v{version}" in sh("git", "tag", capture=True).split():
        sys.exit(f"tag v{version} already exists")

    prev = last_tag()
    notes = a.note or notes_from_git(prev)
    if not notes:
        sys.exit("no release notes: pass -n ... (nothing committed since the last tag)")
    print(f"Release {version} (previous tag: {prev or 'none'})")
    for n in notes:
        print(f"  - {n}")

    # 1. version bump
    if current_version() != version:
        vp = os.path.join(ROOT, "version.py")
        src = open(vp).read()
        open(vp, "w").write(re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', src))
        sh("git", "add", "version.py")
        sh("git", "commit", "-q", "-m", f"Version {version}")
        print(f"version.py -> {version}")

    # 2. build
    mac_zip = os.path.join(ROOT, "dist", f"ChessVision-{version}-arm64.zip")
    if not a.no_build:
        sh(os.path.join(ROOT, "packaging", "build_app.sh"))
    if not os.path.exists(mac_zip):
        cands = glob.glob(os.path.join(ROOT, "dist", f"ChessVision-{version}-*.zip"))
        mac_zip = cands[0] if cands else None

    # 3. publish the zip to GitHub Releases (the site links to it; the repo
    #    is public so the asset URL needs no sign-in). The Windows CI build
    #    attaches its zip to the same release from the v-tag.
    mac_url = None
    if mac_zip:
        mac_url = f"{RELEASE_BASE}/v{version}/{os.path.basename(mac_zip)}"
        print(f"Mac zip: {os.path.basename(mac_zip)} ({os.path.getsize(mac_zip) / 1e6:.0f} MB) -> {mac_url}")
    else:
        print("no Mac zip for this version (site entry will have no Mac download)")
    win_url = None  # filled in by `release.py refresh VERSION` once CI has uploaded it

    # 4. changelog entry
    entry = {"version": version, "date": dt.date.today().isoformat(),
             "mac": mac_url, "win": win_url, "notes": [parse_note(n) for n in notes]}
    update_releases_js(os.path.join(site, "releases.js"), entry)
    print("releases.js updated")

    # 5. commit + tag
    sh("git", "add", "releases.js", cwd=site)
    if sh("git", "status", "--porcelain", "releases.js", cwd=site, capture=True):
        sh("git", "commit", "-q", "-m", f"Release {version}", cwd=site)
    sh("git", "tag", "-a", f"v{version}", "-m", f"Chess Vision {version}\n\n"
       + "\n".join(f"- {parse_note(n)['title']}" for n in notes))
    print(f"tagged v{version}")
    if a.push:
        sh("git", "push", "-q", "origin", "main", "--follow-tags")
        sh("git", "push", "-q", "origin", "main", cwd=site)
        print("pushed both repos")
        if mac_zip:
            sh("gh", "release", "create", f"v{version}", mac_zip, "--title", f"Chess Vision {version}",
               "--notes", "\n".join(f"- {parse_note(n)['title']}" for n in notes))
            print(f"GitHub release v{version} created with the Mac zip")
    else:
        print("next: git push origin main --follow-tags, push the site, then\n"
              f"      gh release create v{version} {mac_zip} --title 'Chess Vision {version}' --notes-from-tag")
    return 0


def refresh(version: str, site: str) -> int:
    """Point releases.js at whatever assets the GitHub release has now
    (run after the Windows CI build attached its zip)."""
    out = sh("gh", "release", "view", f"v{version}", "--json", "assets", capture=True)
    names = [a_["name"] for a_ in json.loads(out)["assets"]]
    path = os.path.join(site, "releases.js")
    text = open(path).read()
    m = re.search(r"window\.CV_RELEASES\s*=\s*(\[.*?\]);", text, re.S)
    releases = json.loads(m.group(1))
    for r in releases:
        if r["version"] != version:
            continue
        for key, pat in (("mac", "arm64"), ("win", "windows")):
            hit = next((n for n in names if pat in n), None)
            r[key] = f"{RELEASE_BASE}/v{version}/{hit}" if hit else None
            print(f"{key}: {r[key] or 'no asset'}")
    text = text[:m.start(1)] + json.dumps(releases, indent=2, ensure_ascii=False) + text[m.end(1):]
    open(path, "w").write(text)
    sh("git", "add", "releases.js", cwd=site)
    if sh("git", "status", "--porcelain", "releases.js", cwd=site, capture=True):
        sh("git", "commit", "-q", "-m", f"Release {version}: download links", cwd=site)
        sh("git", "push", "-q", "origin", "main", cwd=site)
        print("site updated and pushed")
    else:
        print("releases.js already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
