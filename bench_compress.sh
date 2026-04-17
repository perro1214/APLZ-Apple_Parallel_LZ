#!/usr/bin/env bash
# bench_compress.sh — APLZ / gzip / zip / zstd / lz4 / xz 圧縮ベンチマーク v2
#
# Usage:
#   ./bench_compress.sh [SIZE_MB]         テストデータを生成してベンチマーク (デフォルト: 1024 MB)
#   ./bench_compress.sh <file>            既存ファイルでベンチマーク
#   ./bench_compress.sh [SIZE_MB] --debug APLZ デバッグ出力を stderr に表示
#
# 環境変数:
#   BENCH_RUNS=3    各ツール実行回数 (ベスト採用, デフォルト: 3)
#   APLZ_DIR        gpu_zip バイナリのディレクトリ
#   SKIP_XZ=1       xz をスキップ (大ファイルで非常に遅い)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_RUNS="${BENCH_RUNS:-3}"
SKIP_XZ="${SKIP_XZ:-0}"
DEBUG_BENCH=false   # --debug フラグ
APLZ_DBG_FLAG=""    # "" or "--debug" (unquoted 展開で引数なし/あり を切り替え)
TMPDIR_BENCH="/tmp/aplz_bench_$$"
SUMMARY_FILE="$TMPDIR_BENCH/summary.tsv"

# ─── APLZ バイナリ検出 ────────────────────────────────────────────────────────
APLZ_DIR="${APLZ_DIR:-$SCRIPT_DIR}"
GPU_ZIP="$APLZ_DIR/gpu_zip"
SHADER="$APLZ_DIR/compression.metal"
if [[ ! -x "$GPU_ZIP" ]]; then
    GPU_ZIP="$(command -v gpu_zip 2>/dev/null || true)"
    [[ -n "$GPU_ZIP" ]] && SHADER="$(dirname "$GPU_ZIP")/compression.metal"
fi
if [[ ! -x "$GPU_ZIP" ]]; then
    echo "Error: gpu_zip not found. Set APLZ_DIR or install APLZ." >&2; exit 1
fi

# ─── クリーンアップ ──────────────────────────────────────────────────────────
cleanup() { rm -rf "$TMPDIR_BENCH"; }
trap cleanup EXIT INT TERM
mkdir -p "$TMPDIR_BENCH"
printf "label\ttool\ttype\tout_bytes\tratio_pct\tspeed_mbs\ttime_s\n" > "$SUMMARY_FILE"

# ─── sudo / purge ────────────────────────────────────────────────────────────
CAN_PURGE=false
if command -v purge &>/dev/null; then
    printf "==> Requesting sudo for cache flush (purge)... "
    if sudo -v 2>/dev/null; then CAN_PURGE=true; echo "OK"
    else echo "skipped (no sudo)"; fi
fi
flush_cache() { $CAN_PURGE && sudo purge 2>/dev/null || true; }

# ─── ツール検出 ──────────────────────────────────────────────────────────────
HAS_GZIP=false; HAS_ZIP=false; HAS_ZSTD=false; HAS_LZ4=false; HAS_XZ=false

_detect() {
    local name="$1" var="$2"
    if command -v "$name" &>/dev/null; then
        printf "  %-10s %s\n" "$name:" "$(command -v "$name")"
        eval "$var=true"
    else
        printf "  %-10s not found (skipped)\n" "$name:"
    fi
}

echo "==> Tools:"
printf "  %-10s %s\n" "aplz:" "$GPU_ZIP"
_detect gzip HAS_GZIP
_detect zip  HAS_ZIP
_detect zstd HAS_ZSTD
_detect lz4  HAS_LZ4
if [[ "$SKIP_XZ" == "1" ]]; then
    printf "  %-10s skipped (SKIP_XZ=1)\n" "xz:"
else
    _detect xz HAS_XZ
fi
echo ""

# ─── ハードウェア情報 ────────────────────────────────────────────────────────
echo "==> System:"
if command -v system_profiler &>/dev/null; then
    _hw=$(system_profiler SPHardwareDataType 2>/dev/null)
    _chip=$(printf '%s' "$_hw" | awk -F': ' '/Chip:/{gsub(/^ +/,"",$2); print $2}' | head -1)
    _ram=$( printf '%s' "$_hw" | awk -F': ' '/Memory:/{gsub(/^ +/,"",$2); print $2}' | head -1)
    printf "  %-10s %s\n" "Chip:"  "${_chip:-unknown}"
    printf "  %-10s %s\n" "RAM:"   "${_ram:-unknown}"
fi
printf "  %-10s %s\n" "Date:"   "$(date '+%Y-%m-%d %H:%M:%S')"
printf "  %-10s %d (ベスト採用)\n" "Runs:"   "$BENCH_RUNS"
$DEBUG_BENCH && printf "  %-10s %s\n" "Debug:"  "ON (APLZ stderr を表示)" || true
echo ""

# ─── ユーティリティ ──────────────────────────────────────────────────────────
numfmt_size() {
    awk -v b="$1" 'BEGIN{
        if(b>=1073741824) printf "%.2fGB", b/1073741824
        else if(b>=1048576) printf "%.1fMB", b/1048576
        else if(b>=1024) printf "%.1fKB", b/1024
        else printf "%dB", b
    }'
}

file_size() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }

# bash の time ビルトインで計測 (python3 不要) — BENCH_RUNS 回実行しベストを _BEST_TIME にセット
# _BENCH_DBG=true のとき: stdout は /dev/null、stderr (デバッグ出力) はターミナルに表示
_BEST_TIME="0.001"
_BENCH_DBG=false
time_best() {
    local _best="" _t _i _tf
    for (( _i=0; _i<BENCH_RUNS; _i++ )); do
        flush_cache
        if $_BENCH_DBG; then
            # time 出力と コマンド stderr が混在するので temp ファイルで分離
            # time 出力: TIMEFORMAT='%R' → "0.123" (純粋な小数のみの行)
            # デバッグ出力: "[APLZ DBG] ..." → それ以外の行
            _tf="$TMPDIR_BENCH/.time_${RANDOM}"
            { TIMEFORMAT='%R'; time "$@" >/dev/null; } 2>"$_tf"
            _t=$(grep -E '^[0-9]+\.[0-9]+$' "$_tf" | tail -1)
            grep -vE '^[0-9]+\.[0-9]+$' "$_tf" >&2 || true
            rm -f "$_tf"
        else
            _t=$( { TIMEFORMAT='%R'; time "$@" >/dev/null 2>&1; } 2>&1 )
        fi
        if [[ -z "$_best" ]] || awk "BEGIN{exit !(${_t:-0}+0 < ${_best}+0)}"; then
            _best="${_t:-0.001}"
        fi
    done
    _BEST_TIME="${_best:-0.001}"
}

# ─── テストデータ生成 ────────────────────────────────────────────────────────
generate_test_data() {
    local size_mb="$1"
    echo "==> Generating test data (${size_mb} MB each)..."

    # テキスト: 繰り返しパターン (高圧縮率)
    python3 -c "
import sys
block = b'The quick brown fox jumps over the lazy dog. ' * 1000
total = ${size_mb} * 1024 * 1024
written = 0
while written < total:
    chunk = block[:min(len(block), total - written)]
    sys.stdout.buffer.write(chunk)
    written += len(chunk)
" > "$TMPDIR_BENCH/test_text.bin"

    # ランダム: 非圧縮データ
    dd if=/dev/urandom of="$TMPDIR_BENCH/test_rand.bin" \
        bs=1048576 count="$size_mb" 2>/dev/null

    # ミックス: 前半テキスト + 後半ランダム
    local half=$(( size_mb / 2 ))
    {
        dd if="$TMPDIR_BENCH/test_text.bin" bs=1048576 count="$half" 2>/dev/null
        dd if="$TMPDIR_BENCH/test_rand.bin" bs=1048576 count="$half" 2>/dev/null
    } > "$TMPDIR_BENCH/test_mix.bin"

    printf "  Text  : %s\n" "$(numfmt_size "$(file_size "$TMPDIR_BENCH/test_text.bin")")"
    printf "  Random: %s\n" "$(numfmt_size "$(file_size "$TMPDIR_BENCH/test_rand.bin")")"
    printf "  Mixed : %s\n" "$(numfmt_size "$(file_size "$TMPDIR_BENCH/test_mix.bin")")"
    echo ""
}

# ─── APLZ GPU ウォームアップ ──────────────────────────────────────────────────
warmup_aplz() {
    local input="$1"
    printf "==> Warming up APLZ (Metal compile + GPU init)... "
    if $DEBUG_BENCH; then
        echo ""
        # デバッグモード: stderr をターミナルに出力
        "$GPU_ZIP" -c "$input" "$TMPDIR_BENCH/warmup.aplz" "$SHADER" $APLZ_DBG_FLAG \
            >/dev/null || true
    else
        "$GPU_ZIP" -c "$input" "$TMPDIR_BENCH/warmup.aplz" "$SHADER" \
            >/dev/null 2>&1 || true
        echo "done"
    fi
    rm -f "$TMPDIR_BENCH/warmup.aplz"
    echo ""
}

# ─── ベンチマーク関数 ────────────────────────────────────────────────────────
bench_compress() {
    local label="$1" tool="$2" input="$3" output="$4"
    shift 4
    local in_sz
    in_sz=$(file_size "$input")
    rm -f "$output"

    printf "  %-13s " "$tool"
    time_best "$@"
    local t="$_BEST_TIME"

    local out_sz=0
    [[ -f "$output" ]] && out_sz=$(file_size "$output")

    local ratio speed
    ratio=$(awk "BEGIN{printf \"%.1f\", 100.0*$out_sz/$in_sz}")
    speed=$(awk "BEGIN{printf \"%.1f\", $in_sz/1048576/$t}")

    printf "%10s  %6s%%  %10s MB/s  %8ss\n" \
        "$(numfmt_size "$out_sz")" "$ratio" "$speed" "$t"

    printf "%s\t%s\tcompress\t%d\t%s\t%s\t%s\n" \
        "$label" "$tool" "$out_sz" "$ratio" "$speed" "$t" >> "$SUMMARY_FILE"
}

bench_decompress() {
    local label="$1" tool="$2" input="$3" output="$4" original="$5"
    shift 5
    local orig_sz
    orig_sz=$(file_size "$original")
    rm -f "$output"

    printf "  %-13s " "$tool"
    time_best "$@"
    local t="$_BEST_TIME"

    local speed
    speed=$(awk "BEGIN{printf \"%.1f\", $orig_sz/1048576/$t}")

    local verdict="OK"
    if [[ -f "$output" ]]; then
        cmp -s "$original" "$output" 2>/dev/null || verdict="MISMATCH!"
    else
        verdict="NO_OUTPUT!"
    fi

    printf "%10s MB/s  %8ss  [%s]\n" "$speed" "$t" "$verdict"

    printf "%s\t%s\tdecompress\t-\t-\t%s\t%s\n" \
        "$label" "$tool" "$speed" "$t" >> "$SUMMARY_FILE"
}

# ─── メインベンチマーク ──────────────────────────────────────────────────────
run_benchmark() {
    local input_file="$1" label="$2"
    local in_sz in_human pfx
    in_sz=$(file_size "$input_file")
    in_human=$(numfmt_size "$in_sz")
    pfx="$TMPDIR_BENCH/${label}"

    echo "━━━ $label ($in_human) ━━━"
    echo ""

    # ── 圧縮 ────────────────────────────────────────────────────────────────
    printf "  [圧縮] (%d runs, ベスト)\n" "$BENCH_RUNS"
    printf "  %-13s %10s  %7s  %13s  %9s\n" "Tool" "Output" "Ratio" "Speed" "Time"
    printf "  %-13s %10s  %7s  %13s  %9s\n" "────" "──────" "─────" "─────" "────"

    _BENCH_DBG=$DEBUG_BENCH
    bench_compress "$label" "APLZ" "$input_file" "${pfx}.aplz" \
        "$GPU_ZIP" -c "$input_file" "${pfx}.aplz" "$SHADER" $APLZ_DBG_FLAG
    _BENCH_DBG=false

    if $HAS_GZIP; then
        # gzip -c で標準出力に書いて in-place 変更を回避
        bench_compress "$label" "gzip" "$input_file" "${pfx}.gz" \
            bash -c 'gzip -c "$1" > "$2"' -- "$input_file" "${pfx}.gz"
    fi

    if $HAS_ZIP; then
        bench_compress "$label" "ZIP" "$input_file" "${pfx}.zip" \
            zip -j -q "${pfx}.zip" "$input_file"
    fi

    if $HAS_ZSTD; then
        bench_compress "$label" "Zstandard" "$input_file" "${pfx}.zst" \
            zstd -f -q "$input_file" -o "${pfx}.zst"
    fi

    if $HAS_LZ4; then
        bench_compress "$label" "lz4" "$input_file" "${pfx}.lz4" \
            lz4 -f -q "$input_file" "${pfx}.lz4"
    fi

    if $HAS_XZ; then
        # xz は出力ファイル指定に --stdout を使う
        bench_compress "$label" "xz" "$input_file" "${pfx}.xz" \
            bash -c 'xz -k --stdout "$1" > "$2"' -- "$input_file" "${pfx}.xz"
    fi

    echo ""

    # ── 解凍 ────────────────────────────────────────────────────────────────
    printf "  [解凍] (%d runs, ベスト)\n" "$BENCH_RUNS"
    printf "  %-13s %13s  %9s  %s\n" "Tool" "Speed" "Time" "Verify"
    printf "  %-13s %13s  %9s  %s\n" "────" "─────" "────" "──────"

    _BENCH_DBG=$DEBUG_BENCH
    [[ -f "${pfx}.aplz" ]] && bench_decompress "$label" "APLZ" \
        "${pfx}.aplz" "${pfx}_dec_aplz.bin" "$input_file" \
        "$GPU_ZIP" -d "${pfx}.aplz" "${pfx}_dec_aplz.bin" "$SHADER" $APLZ_DBG_FLAG
    _BENCH_DBG=false

    if $HAS_GZIP && [[ -f "${pfx}.gz" ]]; then
        bench_decompress "$label" "gzip" "${pfx}.gz" "${pfx}_dec_gz.bin" "$input_file" \
            bash -c 'gzip -d -c "$1" > "$2"' -- "${pfx}.gz" "${pfx}_dec_gz.bin"
    fi

    if $HAS_ZIP && [[ -f "${pfx}.zip" ]]; then
        local zip_dir="${pfx}_unzip"
        local zip_out="$zip_dir/$(basename "$input_file")"
        rm -rf "$zip_dir"
        bench_decompress "$label" "ZIP" "${pfx}.zip" "$zip_out" "$input_file" \
            unzip -o -q -d "$zip_dir" "${pfx}.zip"
    fi

    if $HAS_ZSTD && [[ -f "${pfx}.zst" ]]; then
        bench_decompress "$label" "Zstandard" "${pfx}.zst" "${pfx}_dec_zst.bin" "$input_file" \
            zstd -d -f -q "${pfx}.zst" -o "${pfx}_dec_zst.bin"
    fi

    if $HAS_LZ4 && [[ -f "${pfx}.lz4" ]]; then
        bench_decompress "$label" "lz4" "${pfx}.lz4" "${pfx}_dec_lz4.bin" "$input_file" \
            lz4 -d -f -q "${pfx}.lz4" "${pfx}_dec_lz4.bin"
    fi

    if $HAS_XZ && [[ -f "${pfx}.xz" ]]; then
        bench_decompress "$label" "xz" "${pfx}.xz" "${pfx}_dec_xz.bin" "$input_file" \
            bash -c 'xz -d --stdout "$1" > "$2"' -- "${pfx}.xz" "${pfx}_dec_xz.bin"
    fi

    echo ""
}

# ─── サマリー表示 ────────────────────────────────────────────────────────────
print_summary() {
    echo "============================================================"
    echo " Summary"
    echo "============================================================"
    echo ""

    local labels tools
    labels=$(tail -n +2 "$SUMMARY_FILE" | awk -F'\t' '{print $1}' | awk '!seen[$0]++')
    tools=$(tail -n +2  "$SUMMARY_FILE" | awk -F'\t' '{print $2}' | awk '!seen[$0]++')

    _summary_table() {
        local type="$1" col="$2" unit="$3"
        printf "  [%s]\n" "$unit"
        printf "  %-13s" "Tool"
        while IFS= read -r lbl; do printf " %10s" "$lbl"; done <<< "$labels"
        echo ""
        while IFS= read -r tool; do
            printf "  %-13s" "$tool"
            while IFS= read -r lbl; do
                local val
                val=$(awk -F'\t' -v l="$lbl" -v t="$tool" -v tp="$type" -v c="$col" \
                    '$1==l && $2==t && $3==tp {print $c}' "$SUMMARY_FILE")
                printf " %10s" "${val:--}"
            done <<< "$labels"
            echo ""
        done <<< "$tools"
        echo ""
    }

    _summary_table "compress"   5 "圧縮比 % (小さいほど良い)"
    _summary_table "compress"   6 "圧縮速度 MB/s (大きいほど良い)"
    _summary_table "decompress" 6 "解凍速度 MB/s (大きいほど良い)"
}

# ─── エントリポイント ─────────────────────────────────────────────────────────
# --debug を全引数からスキャンし、残りを位置引数として再セット
_POS=()
for _a in "$@"; do
    if [[ "$_a" == "--debug" ]]; then
        DEBUG_BENCH=true
        APLZ_DBG_FLAG="--debug"
    else
        _POS+=("$_a")
    fi
done
set -- "${_POS[@]+"${_POS[@]}"}"
ARG="${1:-1024}"

echo "============================================================"
echo " APLZ Compression Benchmark v2"
echo "============================================================"
echo ""

if [[ -f "$ARG" ]]; then
    warmup_aplz "$ARG"
    run_benchmark "$ARG" "$(basename "$ARG")"
else
    SIZE_MB="$ARG"
    [[ "$SIZE_MB" =~ ^[0-9]+$ ]] || { echo "Error: argument must be a size in MB or an existing file." >&2; exit 1; }
    generate_test_data "$SIZE_MB"
    warmup_aplz "$TMPDIR_BENCH/test_text.bin"
    run_benchmark "$TMPDIR_BENCH/test_text.bin" "Text"
    run_benchmark "$TMPDIR_BENCH/test_rand.bin" "Random"
    run_benchmark "$TMPDIR_BENCH/test_mix.bin"  "Mixed"
fi

print_summary

echo "============================================================"
echo " Benchmark complete."
echo "============================================================"
