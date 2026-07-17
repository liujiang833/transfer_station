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

echo "== build flash =="
"$CC" -O2 -static -march="$MARCH" -Wall -Wextra "$HERE/flash.c" -lm -o "$HERE/flash"
echo "== run flash (short suite, default bk) =="
"$QEMU" -cpu max "$HERE/flash"
echo
echo "== self-test attn_flash_pick_bk's analytical cache model =="
"$QEMU" -cpu max "$HERE/flash" --check-pick-bk
echo
echo "== more (not run here) =="
echo "  $QEMU -cpu max $HERE/flash --bk 256      # same suite, another key block, NO REBUILD"
echo "  $QEMU -cpu max $HERE/flash --sweep-bk    # tune bk (~20s; QEMU models no cache -- see README)"
echo "  $QEMU -cpu max $HERE/flash --long        # + Qwen3 1k/2k/4k prefill sweep (~19 min)"
