#!/usr/bin/env bash
# build.sh — APLZ-X (.asz) 統合ビルド & テストスクリプト (macOS / Apple Silicon)
#
# 使い方:
#   ./build.sh           # asz CLI + Metal カーネル + 全テスト／ベンチバイナリを build/ へビルド
#   ./build.sh test      # ビルド後に自動検証スイートを実行
#   ./build.sh clean     # build/ を削除
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf "$BUILD_DIR"
    echo "==> Cleaned $BUILD_DIR"
    exit 0
fi

mkdir -p "$BUILD_DIR"

echo "==> [1/4] Compiling Metal 3.0 shaders (rans32.metallib / hetero_kernels.metallib)..."
xcrun -sdk macosx metal -std=metal3.0 -O3 -c "$ROOT_DIR/src/rans/rans32.metal" -o "$BUILD_DIR/rans32.air"
xcrun -sdk macosx metallib "$BUILD_DIR/rans32.air" -o "$BUILD_DIR/rans32.metallib"

xcrun -sdk macosx metal -std=metal3.0 -O3 -c "$ROOT_DIR/src/engine/hetero_kernels.metal" -o "$BUILD_DIR/hetero_kernels.air"
xcrun -sdk macosx metallib "$BUILD_DIR/hetero_kernels.air" -o "$BUILD_DIR/hetero_kernels.metallib"

COMMON_FLAGS=(
    -std=c++17 -O3 -flto -march=armv8.5-a+crc -Wall -Wextra
    -fobjc-arc
    -framework Foundation -framework Metal -framework QuartzCore
)

echo "==> [2/4] Building asz CLI..."
clang++ "${COMMON_FLAGS[@]}" "$ROOT_DIR/src/asz/asz_main.mm" -o "$BUILD_DIR/asz"

echo "==> [3/4] Building test_phase3..."
clang++ "${COMMON_FLAGS[@]}" "$ROOT_DIR/tests/test_phase3.mm" -o "$BUILD_DIR/test_phase3"

echo "==> [4/4] Building benchmarks (bench_rans32, bench_phase2)..."
clang++ "${COMMON_FLAGS[@]}" "$ROOT_DIR/tests/bench_rans32.mm" -o "$BUILD_DIR/bench_rans32"
clang++ "${COMMON_FLAGS[@]}" "$ROOT_DIR/tests/bench_phase2.mm" -o "$BUILD_DIR/bench_phase2"

echo "==> Build complete:"
echo "    $BUILD_DIR/asz"
echo "    $BUILD_DIR/hetero_kernels.metallib"
echo "    $BUILD_DIR/test_phase3"
echo "    $BUILD_DIR/bench_rans32"
echo "    $BUILD_DIR/bench_phase2"

if [[ "${1:-}" == "test" ]]; then
    echo ""
    echo "==> Running Phase 3 automated test suite..."
    "$BUILD_DIR/test_phase3" "$BUILD_DIR"
    echo ""
    echo "==> Running CLI version & smoke check..."
    "$BUILD_DIR/asz" --version
fi
