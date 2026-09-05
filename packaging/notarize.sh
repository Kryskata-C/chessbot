#!/bin/zsh
# Notarise a Developer-ID-signed build so Gatekeeper opens it on any Mac.
# One-time: xcrun notarytool store-credentials chessvision --apple-id YOU@EMAIL --team-id TEAMID
# (asks for an app-specific password from appleid.apple.com)
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP="${1:?usage: packaging/notarize.sh dist/ChessVision-VERSION-ARCH.zip}"
xcrun notarytool submit "$ZIP" --keychain-profile chessvision --wait
APP="dist/Chess Vision.app"
xcrun stapler staple "$APP"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "stapled + re-zipped: $ZIP"
