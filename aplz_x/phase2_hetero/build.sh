#!/usr/bin/env bash
# build.sh — APLZ-X Phase 2 (UMA 異種協調エンジン) ビルド・検証・競合比較スクリプト
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"
SDK="$(xcrun --sdk macosx --show-sdk-path)"

echo "==> Compiling Metal shader: hetero_kernels.metal"
xcrun -sdk macosx metal \
    -std=metal3.0 \
    -O2 \
    -Wall \
    -Wno-unused-variable \
    -c "${DIR}/hetero_kernels.metal" \
    -o "${DIR}/hetero_kernels.air"

xcrun -sdk macosx metallib \
    "${DIR}/hetero_kernels.air" \
    -o "${DIR}/hetero_kernels.metallib"

echo "==> Compiling host engine & benchmark: bench_phase2.mm (-O3 -flto)"
clang++ \
    -std=c++17 \
    -O3 \
    -flto \
    -DNDEBUG \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -isysroot "${SDK}" \
    -framework Metal \
    -framework Foundation \
    -framework QuartzCore \
    -o "${DIR}/bench_phase2" \
    "${DIR}/bench_phase2.mm"

echo "==> Running Phase 2 correctness suite & 64 MiB in-memory benchmark"
"${DIR}/bench_phase2" "${DIR}" "${ROOT}"
