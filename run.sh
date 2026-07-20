#!/usr/bin/env bash
# Run the bf16 attention kernel under QEMU user-mode. Build it first with ./build.sh.
# No root required: uses the locally-extracted qemu-aarch64-static.
# Any arguments are forwarded to the binary, e.g.  ./run.sh --bk 256 --prepack-v
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
QEMU="$HERE/qemu_pkg/extracted/usr/bin/qemu-aarch64-static"

if [[ ! -x "$QEMU" ]]; then
  echo "qemu-aarch64-static not found at $QEMU"
  echo "Fetch it without root:  (cd $HERE/qemu_pkg && apt-get download qemu-user-static && dpkg-deb -x qemu-user-static_*.deb ./extracted)"
  exit 1
fi
if [[ ! -x "$HERE/flash" ]]; then
  echo "flash binary not found at $HERE/flash -- build it first:  ./build.sh"
  exit 1
fi

echo "# qemu    : $($QEMU --version | head -1)"
echo "== run flash (short suite, default bk) =="
"$QEMU" -cpu max "$HERE/flash" "$@"
echo
echo "== more =="
echo "  ./run.sh --bk 256      # same suite, another key block, NO REBUILD"
echo "  ./run.sh --prepack-v   # transpose V with a SEPARATE operator (packv phase -> 0)"
echo "  ./run.sh --long        # + Qwen3 1k/2k/4k prefill sweep (~19 min)"
