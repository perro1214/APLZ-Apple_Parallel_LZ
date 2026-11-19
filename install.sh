#!/usr/bin/env bash
# install.sh — APLZ v4 (.aplz) ビルド & インストーラ (Apple Silicon / macOS)
#
# 使い方:
#   ./install.sh                     # ~/.local/bin/aplz へインストール (sudo 不要)
#   ./install.sh --system            # /usr/local/bin/aplz へシステムインストール
#   ./install.sh --prefix /opt/aplz  # 任意プレフィックスへインストール
#   ./install.sh --uninstall         # アンインストール
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
UNINSTALL=0
USE_SUDO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"; shift 2 ;;
        --system)
            PREFIX="/usr/local"; shift ;;
        --uninstall|uninstall)
            UNINSTALL=1; shift ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "[ERR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

BIN_DIR="$PREFIX/bin"
SHARE_DIR="$PREFIX/share/aplz"

CHECK_DIR="$PREFIX"
while [[ ! -e "$CHECK_DIR" && "$CHECK_DIR" != "/" ]]; do
    CHECK_DIR="$(dirname "$CHECK_DIR")"
done
if [[ ! -w "$CHECK_DIR" ]]; then
    USE_SUDO="sudo"
fi

if [[ "$UNINSTALL" -eq 1 ]]; then
    echo "==> Uninstalling aplz from $PREFIX"
    $USE_SUDO rm -f "$BIN_DIR/aplz" "$SHARE_DIR/hetero_kernels.metallib"
    $USE_SUDO rmdir "$SHARE_DIR" 2>/dev/null || true
    echo "==> Done."
    exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[ERR] APLZ requires macOS with Apple Metal." >&2
    exit 1
fi

TMP_BUILD="$(mktemp -d /tmp/aplz_install_XXXXXX)"
trap 'rm -rf "$TMP_BUILD"' EXIT

echo "==> [1/3] Compiling Metal 3.0 compute kernels (hetero_kernels.metallib)..."
xcrun -sdk macosx metal -std=metal3.0 -O3 -c "$ROOT_DIR/src/engine/hetero_kernels.metal" -o "$TMP_BUILD/hetero_kernels.air"
xcrun -sdk macosx metallib "$TMP_BUILD/hetero_kernels.air" -o "$TMP_BUILD/hetero_kernels.metallib"

echo "==> [2/3] Compiling aplz CLI (-O3 -flto -march=armv8.5-a+crc)..."
clang++ -std=c++17 -O3 -flto -march=armv8.5-a+crc -Wall -Wextra \
    -fobjc-arc \
    -framework Foundation -framework Metal -framework QuartzCore \
    "$ROOT_DIR/src/archive/aplz_main.mm" -o "$TMP_BUILD/aplz"

echo "==> [3/3] Installing to $PREFIX..."
$USE_SUDO mkdir -p "$BIN_DIR" "$SHARE_DIR"
$USE_SUDO install -m 755 "$TMP_BUILD/aplz" "$BIN_DIR/aplz"
$USE_SUDO install -m 644 "$TMP_BUILD/hetero_kernels.metallib" "$SHARE_DIR/hetero_kernels.metallib"

# 旧バージョンの残存バイナリを掃除
$USE_SUDO rm -f "$BIN_DIR/gpu_zip" "$BIN_DIR/compression.metal" "$BIN_DIR/asz" "$PREFIX/share/asz/hetero_kernels.metallib" 2>/dev/null || true
$USE_SUDO rmdir "$PREFIX/share/asz" 2>/dev/null || true

# Smoke test
echo "==> Running installation smoke test..."
"$BIN_DIR/aplz" --version
"$BIN_DIR/aplz" c -3 -q -o "$TMP_BUILD/smoke.aplz" "$ROOT_DIR/SPEC.md"
"$BIN_DIR/aplz" t "$TMP_BUILD/smoke.aplz" >/dev/null
"$BIN_DIR/aplz" d "$TMP_BUILD/smoke.aplz" "$TMP_BUILD/SPEC.out" -q
cmp -s "$ROOT_DIR/SPEC.md" "$TMP_BUILD/SPEC.out"
echo "==> Installed successfully: $BIN_DIR/aplz"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo ""
        echo "[NOTE] $BIN_DIR is not in your PATH. Add this to ~/.zshrc:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac
