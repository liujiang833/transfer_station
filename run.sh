#!/usr/bin/env bash
# Build the bf16 attention kernel for AArch64 and run it under QEMU user-mode.
# No root required: uses the locally-extracted qemu-aarch64-static.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
QEMU="$HERE/qemu_pkg/extracted/usr/bin/qemu-aarch64-static"
CC=${CC:-aarch64-linux-gnu-gcc}
MARCH=${MARCH:-armv8.6-a+sve+bf16}

if [[ ! -x "$QEMU" ]]; then
  echo "qemu-aarch64-static not found at $QEMU"
  echo "Fetch it without root:  (cd $HERE/qemu_pkg && apt-get download qemu-user-static && dpkg-deb -x qemu-user-static_*.deb ./extracted)"
  exit 1
fi

echo "# compiler: $($CC --version | head -1)"
echo "# qemu    : $($QEMU --version | head -1)"
echo "# march   : $MARCH"

for src in sanity attn flash; do
  echo "== build $src =="
  "$CC" -O2 -static -march="$MARCH" "$HERE/$src.c" -lm -o "$HERE/$src"
  echo "== run $src =="
  "$QEMU" -cpu max "$HERE/$src"
  echo
done
