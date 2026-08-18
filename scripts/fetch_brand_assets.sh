#!/usr/bin/env bash
# Vendor the Qdrant brand assets: fonts and logos.
#
# Everything is committed to the repo and served locally. The demo must run with no network
# at runtime -- venue wifi is expected to fail -- so a CDN font import would silently fall
# back to a system font on stage, which nobody notices until it is projected.
#
# This script exists so the committed binaries are reproducible rather than mystery files.
# Re-run it to refresh them; it prints checksums so you can diff against what is committed.
#
# Fonts are SIL Open Font License 1.1 (see the OFL files it downloads alongside them).
# OFL permits redistribution inside another repo as long as the license travels with the
# files. The repo is Apache-2.0; the fonts remain OFL-1.1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATIC="$ROOT/vector_taste/ui/static"
FONTS="$STATIC/fonts"
IMG="$STATIC/img"

mkdir -p "$FONTS" "$IMG"

MONA_RAW="https://raw.githubusercontent.com/github/mona-sans/main"
GEIST_RAW="https://raw.githubusercontent.com/vercel/geist-font/main"
LOGOS="https://qdrant.tech/img/brand-resources-logos"

get() { # url dest
  printf '  %-42s' "$(basename "$2")"
  if curl -sSfL --retry 3 -o "$2" "$1"; then
    printf '%8s bytes\n' "$(wc -c < "$2" | tr -d ' ')"
  else
    echo "FAILED: $1" >&2
    return 1
  fi
}

echo "fonts -> $FONTS"
# Unmodified upstream variable fonts, NOT self-subset. Mona Sans carries a Reserved Font
# Name ("Mona Sans" / "Mona"); under a strict OFL reading a subset is a modified version
# that may not keep the name. 204KB total is not worth that question in a company repo.
get "$MONA_RAW/fonts/webfonts/variable/MonaSansVF%5Bopsz%2Cwght%5D.woff2" \
    "$FONTS/MonaSansVF.woff2"
get "$GEIST_RAW/fonts/GeistMono/webfonts/GeistMono%5Bwght%5D.woff2" \
    "$FONTS/GeistMono.woff2"
get "$MONA_RAW/OFL.txt"  "$FONTS/OFL-mona-sans.txt"
get "$GEIST_RAW/OFL.txt" "$FONTS/OFL-geist.txt"

echo
echo "logos -> $IMG"
# Two lockups: they swap by theme in CSS rather than by rewriting src in JS, so there is
# no flash on load.
get "$LOGOS/qdrant-logo-red-white.svg" "$IMG/qdrant-logo-red-white.svg"
get "$LOGOS/qdrant-logo-red-black.svg" "$IMG/qdrant-logo-red-black.svg"
# Qdrant publishes no full-color SVG favicon (/favicon.svg 404s; only a monochrome
# safari-pinned-tab.svg exists), so the brandmark serves as one.
get "$LOGOS/qdrant-brandmark-red.svg"  "$IMG/qdrant-brandmark-red.svg"

echo
echo "checksums:"
(cd "$STATIC" && shasum -a 256 fonts/* img/* | sed 's/^/  /')

# A truncated download still exits 0 from curl in some proxy setups; catch it here rather
# than discovering a broken font on stage.
for f in "$FONTS/MonaSansVF.woff2" "$FONTS/GeistMono.woff2"; do
  size=$(wc -c < "$f" | tr -d ' ')
  if [ "$size" -lt 20000 ]; then
    echo "error: $(basename "$f") is only ${size} bytes -- truncated download" >&2
    exit 1
  fi
done

echo
echo "done. Fonts are OFL-1.1; see $FONTS/OFL-*.txt"
