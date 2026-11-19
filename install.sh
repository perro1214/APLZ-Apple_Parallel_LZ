#!/usr/bin/env bash
# install.sh — APLZ-X (.asz) ビルド & インストーラ (Apple Silicon / macOS)
#
# 使い方:
#   ./install.sh                     # ~/.local/bin/asz へインストール (sudo 不要)
#   ./install.sh --system            # /usr/local/bin/asz へシステムインストール
#   ./install.sh --prefix /opt/asz   # 任意プレフィックスへインストール
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
SHARE_DIR="$PREFIX/share/asz"

CHECK_DIR="$PREFIX"
while [[ ! -e "$CHECK_DIR" && "$CHECK_DIR" != "/" ]]; do
    CHECK_DIR="$(dirname "$CHECK_DIR")"
done
if [[ ! -w "$CHECK_DIR" ]]; then
    USE_SUDO="sudo"
fi

if [[ "$UNINSTALL" -eq 1 ]]; then
    echo "==> Uninstalling asz from $PREFIX"
    $USE_SUDO rm -f "$BIN_DIR/asz" "$SHARE_DIR/hetero_kernels.metallib"
    $USE_SUDO rmdir "$SHARE_DIR" 2>/dev/null || true
    echo "==> Done."
    exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[ERR] APLZ-X (asz) requires macOS with Apple Metal." >&2
    exit 1
fi

TMP_BUILD="$(mktemp -d /tmp/asz_install_XXXXXX)"
trap 'rm -rf "$TMP_BUILD"' EXIT

echo "==> [1/3] Compiling Metal 3.0 compute kernels (hetero_kernels.metallib)..."
xcrun -sdk macosx metal -std=metal3.0 -O3 -c "$ROOT_DIR/src/engine/hetero_kernels.metal" -o "$TMP_BUILD/hetero_kernels.air"
xcrun -sdk macosx metallib "$TMP_BUILD/hetero_kernels.air" -o "$TMP_BUILD/hetero_kernels.metallib"

echo "==> [2/3] Compiling asz CLI (-O3 -flto -march=armv8.5-a+crc)..."
clang++ -std=c++17 -O3 -flto -march=armv8.5-a+crc -Wall -Wextra \
    -fobjc-arc \
    -framework Foundation -framework Metal -framework QuartzCore \
    "$ROOT_DIR/src/asz/asz_main.mm" -o "$TMP_BUILD/asz"

echo "==> [3/3] Installing to $PREFIX..."
$USE_SUDO mkdir -p "$BIN_DIR" "$SHARE_DIR"
$USE_SUDO install -m 755 "$TMP_BUILD/asz" "$BIN_DIR/asz"
$USE_SUDO install -m 644 "$TMP_BUILD/hetero_kernels.metallib" "$SHARE_DIR/hetero_kernels.metallib"

# Smoke test
echo "==> Running installation smoke test..."
"$BIN_DIR/asz" --version
"$BIN_DIR/asz" c -3 -q -o "$TMP_BUILD/smoke.asz" "$ROOT_DIR/SPEC.md"
"$BIN_DIR/asz" t "$TMP_BUILD/smoke.asz" >/dev/null
"$BIN_DIR/asz" d "$TMP_BUILD/smoke.asz" "$TMP_BUILD/SPEC.out" -q
cmp -s "$ROOT_DIR/SPEC.md" "$TMP_BUILD/SPEC.out"
echo "==> Installed successfully: $BIN_DIR/asz"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo ""
        echo "[NOTE] $BIN_DIR is not in your PATH. Add this to ~/.zshrc:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac
