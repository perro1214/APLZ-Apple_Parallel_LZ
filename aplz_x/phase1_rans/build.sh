#!/usr/bin/env bash
# build.sh — APLZ-X Phase 1 (32-way Coalesced rANS) ビルド & 検証スクリプト
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SDK="$(xcrun --sdk macosx --show-sdk-path)"

echo "==> Compiling Metal shader: rans32.metal"
xcrun -sdk macosx metal \
    -std=metal3.0 \
    -O2 \
    -Wall \
    -Wno-unused-variable \
    -c "${DIR}/rans32.metal" \
    -o "${DIR}/rans32.air"

xcrun -sdk macosx metallib \
    "${DIR}/rans32.air" \
    -o "${DIR}/rans32.metallib"

echo "==> Compiling host benchmark & verifier: bench_rans32.mm (-O3 -flto)"
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
    -o "${DIR}/bench_rans32" \
    "${DIR}/bench_rans32.mm"

echo "==> Running correctness suite & 64 MiB benchmark"
"${DIR}/bench_rans32" "${DIR}"
