#!/usr/bin/env bash
# Build the bf16 attention kernel (flash.c) for AArch64. No root, no QEMU needed here.
# Run it afterwards with ./run.sh (which does NOT rebuild).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CC=${CC:-aarch64-linux-gnu-gcc}
MARCH=${MARCH:-armv8.6-a+sve+bf16}

echo "# compiler: $($CC --version | head -1)"
echo "# march   : $MARCH"
echo "== build flash =="
"$CC" -O2 -static -march="$MARCH" -Wall -Wextra "$HERE/flash.c" -lm -o "$HERE/flash"
echo "built: $HERE/flash"
