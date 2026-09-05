#!/bin/zsh
# One-time, on the development Mac: create a local self-signed code-signing
# identity called "Chess Vision Dev". build_app.sh signs with it when present,
# and because the signature then stays the same across rebuilds, macOS keeps
# the Screen Recording grant instead of forgetting it after every build
# (ad-hoc signatures change each time). Customers get a Developer ID build.
#
# Run it yourself: it touches your login keychain and macOS will ask for
# your password once to trust the certificate for code signing.
set -euo pipefail
NAME="Chess Vision Dev"
if security find-identity -v -p codesigning | grep -q "$NAME"; then
  echo "identity '$NAME' already exists"; exit 0
fi
T=$(mktemp -d)
openssl req -x509 -newkey rsa:2048 -keyout "$T/key.pem" -out "$T/cert.pem" -days 3650 -nodes \
  -subj "/CN=$NAME" -addext "keyUsage=digitalSignature" -addext "extendedKeyUsage=codeSigning"
openssl pkcs12 -export -out "$T/id.p12" -inkey "$T/key.pem" -in "$T/cert.pem" -passout pass:dev -legacy 2>/dev/null \
  || openssl pkcs12 -export -out "$T/id.p12" -inkey "$T/key.pem" -in "$T/cert.pem" -passout pass:dev
security import "$T/id.p12" -k ~/Library/Keychains/login.keychain-db -P dev -T /usr/bin/codesign
# Trust it for code signing (macOS asks for your password once).
security add-trusted-cert -r trustRoot -p codeSign -k ~/Library/Keychains/login.keychain-db "$T/cert.pem"
rm -rf "$T"
security find-identity -v -p codesigning | grep "$NAME" && echo "done — rebuild with packaging/build_app.sh"
