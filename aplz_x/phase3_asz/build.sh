#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
P2="$DIR/../phase2_hetero"

echo "==> Ensuring Phase 2 Metal shader library is up to date"
xcrun -sdk macosx metal -std=metal3.0 -O3 -c "$P2/hetero_kernels.metal" -o "$DIR/hetero_kernels.air"
xcrun -sdk macosx metallib "$DIR/hetero_kernels.air" -o "$DIR/hetero_kernels.metallib"
rm -f "$DIR/hetero_kernels.air"

echo "==> Compiling Phase 3 CLI: asz (-O3 -flto)"
clang++ -std=c++17 -O3 -flto -march=armv8.5-a+crc -Wall -Wextra \
    -fobjc-arc \
    -framework Foundation -framework Metal -framework QuartzCore \
    "$DIR/asz_main.mm" -o "$DIR/asz"

echo "==> Compiling Phase 3 test suite: test_phase3 (-O3)"
clang++ -std=c++17 -O3 -march=armv8.5-a+crc -Wall -Wextra \
    -fobjc-arc \
    -framework Foundation -framework Metal -framework QuartzCore \
    "$DIR/test_phase3.mm" -o "$DIR/test_phase3"

echo "==> Running Phase 3 automated verification suite"
"$DIR/test_phase3" "$DIR"
