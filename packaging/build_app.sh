#!/bin/zsh
# Build dist/Chess Vision.app (and a zip) for the CPU this Mac runs on.
#
#   packaging/build_app.sh                 # ad-hoc signed: fine for your own Macs
#   CV_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" packaging/build_app.sh
#
# With a Developer ID the app is signed with the hardened runtime so it can be
# notarised (see packaging/notarize.sh). Without one, other Macs will refuse to
# open it until the user right-clicks > Open, or runs:
#   xattr -dr com.apple.quarantine "/Applications/Chess Vision.app"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
VERSION="${CV_VERSION:-1.0.0}"
ARCH="$(uname -m)"

# 1. Stage Stockfish. Prefer an explicit CV_STOCKFISH, else Homebrew's.
mkdir -p packaging/stage
SF="${CV_STOCKFISH:-$(command -v stockfish || true)}"
[[ -n "$SF" && -x "$SF" ]] || { echo "Stockfish not found: brew install stockfish, or set CV_STOCKFISH"; exit 1; }
cp -f "$SF" packaging/stage/stockfish
chmod +x packaging/stage/stockfish
echo "Stockfish: $SF ($(file -b packaging/stage/stockfish | cut -d, -f1))"

# 2. Icon (regenerated so it always matches assets/).
QT_QPA_PLATFORM=offscreen $PY packaging/make_icon.py >/dev/null

# 3. Build.
rm -rf "build/Chess Vision" "dist/Chess Vision" "dist/Chess Vision.app"
CV_VERSION="$VERSION" $PY -m PyInstaller --noconfirm --clean ChessVision.spec
APP="dist/Chess Vision.app"

# 4. Sign. Every Mach-O inside gets signed; ad-hoc when no identity is given.
IDENTITY="${CV_SIGN_IDENTITY:-}"
if [[ -z "$IDENTITY" ]] && security find-identity -v -p codesigning | grep -q "Chess Vision Dev"; then
  IDENTITY="Chess Vision Dev"   # local dev identity (packaging/make_dev_identity.sh)
fi
IDENTITY="${IDENTITY:--}"
ENT=packaging/entitlements.plist
if [[ "$IDENTITY" == "-" ]]; then
  codesign --force --deep --sign - "$APP"
elif [[ "$IDENTITY" == "Chess Vision Dev" ]]; then
  codesign --force --deep --sign "$IDENTITY" "$APP"   # self-signed: no timestamp/hardened runtime
else
  codesign --force --deep --options runtime --timestamp --entitlements "$ENT" --sign "$IDENTITY" "$APP"
fi
codesign --verify --deep --strict "$APP" && echo "signed ($IDENTITY)"

# 5. Zip for distribution (ditto keeps the bundle's metadata intact).
ZIP="dist/ChessVision-${VERSION}-${ARCH}.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "built: $APP"
echo "zip:   $ZIP ($(du -h "$ZIP" | cut -f1))"
