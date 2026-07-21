#!/usr/bin/env bash
# Build the bf16 attention kernel (flash.c) for AArch64. No root, no QEMU needed here.
# Run it afterwards with ./run.sh (which does NOT rebuild).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CC=${CC:-aarch64-linux-gnu-gcc}
CXX=${CXX:-${CC/gcc/g++}} # C++ compiler for bfdot_peak; derived from CC (gcc->g++), override freely
MARCH=${MARCH:-armv8.6-a+sve+bf16}

echo "# compiler: $($CC --version | head -1)"
echo "# march   : $MARCH"
echo "== build flash =="
# -fno-omit-frame-pointer keeps a frame pointer in every function so perf can walk the
# stack without DWARF -- readable flame graphs of the FORCE_NOINLINE phases (pack_v/qk/softmax/pv).
"$CC" -O2 -fno-omit-frame-pointer -static -march="$MARCH" -Wall -Wextra "$HERE/flash.c" -lm -o "$HERE/flash"
echo "built: $HERE/flash"

# bfdot_peak: SVE BFDOT peak-throughput microbenchmark (C++). Diagnostic/secondary, and native-only
# (its numbers are meaningless under QEMU), so it is NOT -static and a missing C++ compiler warns and
# skips rather than failing the whole build -- flash is the primary artifact.
echo "== build bfdot_peak =="
if command -v "$CXX" >/dev/null 2>&1; then
    "$CXX" -O3 -march="$MARCH" -Wall -Wextra "$HERE/bfdot_peak.cpp" -o "$HERE/bfdot_peak"
    echo "built: $HERE/bfdot_peak"
else
    echo "skip: C++ compiler '$CXX' not found (set CXX=..., e.g. CXX=g++ or CXX=clang++ on the ARM target)"
fi
