#!/usr/bin/env bash
# setup_finder.sh — APLZ v4 (.aplz) macOS Finder サービス / クイックアクション自動セットアップ
#
# Finder の右クリックメニュー（「サービス」または「クイックアクション」）に以下を登録します:
#   - 「APLZ で圧縮」: 選択したファイル／フォルダを .aplz 形式で高速圧縮
#   - 「APLZ で解凍」: 選択した .aplz ファイルを解凍（全層 CRC32C 検証つき）
#
# 特長:
#   - 同名ファイル／フォルダが存在する場合は自動リネーム（例: foo 2.aplz / foo 2）し上書き防止
#   - 同一ボリューム上での一時作業＋アトミック移動により外付け SSD でも二重コピーゼロ
#   - 直接読み込みに失敗するクラウドドライブ（Google Drive / iCloud 等）では自動リトライ退避
#   - 日本語・スペース・引用符を含むファイル名および通知メッセージに完全対応
#
# 使い方:
#   ./setup_finder.sh              # インストール / 更新
#   ./setup_finder.sh --uninstall  # アンインストール

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICES_DIR="$HOME/Library/Services"

if [[ "${1:-}" == "--uninstall" || "${1:-}" == "uninstall" ]]; then
    echo "==> Removing Finder services (APLZ で圧縮 / APLZ で解凍)..."
    rm -rf "${SERVICES_DIR}/APLZ で圧縮.workflow"
    rm -rf "${SERVICES_DIR}/APLZ で解凍.workflow"
    /System/Library/CoreServices/pbs -flush 2>/dev/null || true
    echo "==> Removed."
    exit 0
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,16p' "$0" | sed 's/^# \?//'
    exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[ERR] Finder services require macOS." >&2
    exit 1
fi

find_aplz() {
    local candidates=(
        "$HOME/.local/bin/aplz"
        "/usr/local/bin/aplz"
        "/opt/homebrew/bin/aplz"
        "$ROOT_DIR/build/aplz"
    )
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && echo "$c" && return
    done
    echo ""
}

APLZ_PATH="$(find_aplz)"
if [[ -z "$APLZ_PATH" ]]; then
    echo "[ERR] aplz command not found. Run ./install.sh first." >&2
    exit 1
fi

echo "==> Using aplz: ${APLZ_PATH}"

# ─── Automator Quick Action / Service ワークフロー生成 ──────────────────────────
create_workflow() {
    local name="$1"
    local shell_script="$2"
    local wf_root="${SERVICES_DIR}/${name}.workflow"
    local wf_dir="${wf_root}/Contents"

    rm -rf "$wf_root"
    mkdir -p "$wf_dir"

    /usr/bin/python3 - "$name" "$wf_dir" "$shell_script" "$APLZ_PATH" <<'PYEOF'
import os, sys, uuid
from xml.sax.saxutils import escape

name, wf_dir, script, aplz_path = sys.argv[1:5]
script = script.replace("__APLZ_FALLBACK__", aplz_path)

u1, u2, u3 = str(uuid.uuid4()).upper(), str(uuid.uuid4()).upper(), str(uuid.uuid4()).upper()
esc_name = escape(name, {'"': '&quot;'})
esc_script = escape(script, {'"': '&quot;'})

info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
\t<key>NSServices</key>
\t<array>
\t\t<dict>
\t\t\t<key>NSMenuItem</key>
\t\t\t<dict>
\t\t\t\t<key>default</key>
\t\t\t\t<string>{esc_name}</string>
\t\t\t</dict>
\t\t\t<key>NSMessage</key>
\t\t\t<string>runWorkflowAsService</string>
\t\t\t<key>NSRequiredContext</key>
\t\t\t<dict>
\t\t\t\t<key>NSApplicationIdentifier</key>
\t\t\t\t<string>com.apple.finder</string>
\t\t\t</dict>
\t\t\t<key>NSSendFileTypes</key>
\t\t\t<array>
\t\t\t\t<string>public.item</string>
\t\t\t</array>
\t\t</dict>
\t</array>
</dict>
</plist>
'''

doc_wflow = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
\t<key>AMApplicationBuild</key>
\t<string>528</string>
\t<key>AMApplicationVersion</key>
\t<string>2.10</string>
\t<key>AMDocumentVersion</key>
\t<string>2</string>
\t<key>actions</key>
\t<array>
\t\t<dict>
\t\t\t<key>action</key>
\t\t\t<dict>
\t\t\t\t<key>AMAccepts</key>
\t\t\t\t<dict>
\t\t\t\t\t<key>Container</key>
\t\t\t\t\t<string>List</string>
\t\t\t\t\t<key>Optional</key>
\t\t\t\t\t<true/>
\t\t\t\t\t<key>Types</key>
\t\t\t\t\t<array>
\t\t\t\t\t\t<string>com.apple.cocoa.string</string>
\t\t\t\t\t</array>
\t\t\t\t</dict>
\t\t\t\t<key>AMActionVersion</key>
\t\t\t\t<string>2.0.3</string>
\t\t\t\t<key>AMApplication</key>
\t\t\t\t<array>
\t\t\t\t\t<string>Automator</string>
\t\t\t\t</array>
\t\t\t\t<key>AMParameterProperties</key>
\t\t\t\t<dict>
\t\t\t\t\t<key>COMMAND_STRING</key>
\t\t\t\t\t<dict/>
\t\t\t\t\t<key>CheckedForUserDefaultShell</key>
\t\t\t\t\t<dict/>
\t\t\t\t\t<key>inputMethod</key>
\t\t\t\t\t<dict/>
\t\t\t\t\t<key>shell</key>
\t\t\t\t\t<dict/>
\t\t\t\t\t<key>source</key>
\t\t\t\t\t<dict/>
\t\t\t\t</dict>
\t\t\t\t<key>AMProvides</key>
\t\t\t\t<dict>
\t\t\t\t\t<key>Container</key>
\t\t\t\t\t<string>List</string>
\t\t\t\t\t<key>Types</key>
\t\t\t\t\t<array>
\t\t\t\t\t\t<string>com.apple.cocoa.string</string>
\t\t\t\t\t</array>
\t\t\t\t</dict>
\t\t\t\t<key>ActionBundlePath</key>
\t\t\t\t<string>/System/Library/Automator/Run Shell Script.action</string>
\t\t\t\t<key>ActionName</key>
\t\t\t\t<string>シェルスクリプトを実行</string>
\t\t\t\t<key>ActionParameters</key>
\t\t\t\t<dict>
\t\t\t\t\t<key>COMMAND_STRING</key>
\t\t\t\t\t<string>{esc_script}</string>
\t\t\t\t\t<key>CheckedForUserDefaultShell</key>
\t\t\t\t\t<true/>
\t\t\t\t\t<key>inputMethod</key>
\t\t\t\t\t<integer>1</integer>
\t\t\t\t\t<key>shell</key>
\t\t\t\t\t<string>/bin/bash</string>
\t\t\t\t\t<key>source</key>
\t\t\t\t\t<string></string>
\t\t\t\t</dict>
\t\t\t\t<key>BundleIdentifier</key>
\t\t\t\t<string>com.apple.RunShellScript</string>
\t\t\t\t<key>CFBundleVersion</key>
\t\t\t\t<string>2.0.3</string>
\t\t\t\t<key>CanShowSelectedItemsWhenRun</key>
\t\t\t\t<false/>
\t\t\t\t<key>CanShowWhenRun</key>
\t\t\t\t<true/>
\t\t\t\t<key>Category</key>
\t\t\t\t<array>
\t\t\t\t\t<string>AMCategoryUtilities</string>
\t\t\t\t</array>
\t\t\t\t<key>Class Name</key>
\t\t\t\t<string>RunShellScriptAction</string>
\t\t\t\t<key>InputUUID</key>
\t\t\t\t<string>{u1}</string>
\t\t\t\t<key>Keywords</key>
\t\t\t\t<array>
\t\t\t\t\t<string>Shell</string>
\t\t\t\t\t<string>Script</string>
\t\t\t\t</array>
\t\t\t\t<key>OutputUUID</key>
\t\t\t\t<string>{u2}</string>
\t\t\t\t<key>UUID</key>
\t\t\t\t<string>{u3}</string>
\t\t\t\t<key>UnlocalizedApplications</key>
\t\t\t\t<array>
\t\t\t\t\t<string>Automator</string>
\t\t\t\t</array>
\t\t\t\t<key>arguments</key>
\t\t\t\t<dict>
\t\t\t\t\t<key>0</key>
\t\t\t\t\t<dict>
\t\t\t\t\t\t<key>default value</key>
\t\t\t\t\t\t<string>for f in &quot;$@&quot;; do echo &quot;$f&quot;; done</string>
\t\t\t\t\t\t<key>name</key>
\t\t\t\t\t\t<string>COMMAND_STRING</string>
\t\t\t\t\t\t<key>required</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>type</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>uuid</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t</dict>
\t\t\t\t\t<key>1</key>
\t\t\t\t\t<dict>
\t\t\t\t\t\t<key>default value</key>
\t\t\t\t\t\t<string>/bin/bash</string>
\t\t\t\t\t\t<key>name</key>
\t\t\t\t\t\t<string>shell</string>
\t\t\t\t\t\t<key>required</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>type</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>uuid</key>
\t\t\t\t\t\t<string>1</string>
\t\t\t\t\t</dict>
\t\t\t\t\t<key>2</key>
\t\t\t\t\t<dict>
\t\t\t\t\t\t<key>default value</key>
\t\t\t\t\t\t<integer>1</integer>
\t\t\t\t\t\t<key>name</key>
\t\t\t\t\t\t<string>inputMethod</string>
\t\t\t\t\t\t<key>required</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>type</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>uuid</key>
\t\t\t\t\t\t<string>2</string>
\t\t\t\t\t</dict>
\t\t\t\t\t<key>3</key>
\t\t\t\t\t<dict>
\t\t\t\t\t\t<key>default value</key>
\t\t\t\t\t\t<true/>
\t\t\t\t\t\t<key>name</key>
\t\t\t\t\t\t<string>CheckedForUserDefaultShell</string>
\t\t\t\t\t\t<key>required</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>type</key>
\t\t\t\t\t\t<string>0</string>
\t\t\t\t\t\t<key>uuid</key>
\t\t\t\t\t\t<string>3</string>
\t\t\t\t\t</dict>
\t\t\t\t</dict>
\t\t\t\t<key>conversionLabel</key>
\t\t\t\t<integer>0</integer>
\t\t\t\t<key>isViewVisible</key>
\t\t\t\t<integer>1</integer>
\t\t\t\t<key>location</key>
\t\t\t\t<string>301.500000:253.000000</string>
\t\t\t\t<key>nibPath</key>
\t\t\t\t<string>/System/Library/Automator/Run Shell Script.action/Contents/Resources/Base.lproj/main.nib</string>
\t\t\t</dict>
\t\t\t<key>isViewVisible</key>
\t\t\t<integer>1</integer>
\t\t</dict>
\t</array>
\t<key>connectors</key>
\t<dict/>
\t<key>workflowMetaData</key>
\t<dict>
\t\t<key>applicationBundleID</key>
\t\t<string>com.apple.finder</string>
\t\t<key>applicationBundleIDsByPath</key>
\t\t<dict>
\t\t\t<key>/System/Library/CoreServices/Finder.app</key>
\t\t\t<string>com.apple.finder</string>
\t\t</dict>
\t\t<key>applicationPath</key>
\t\t<string>/System/Library/CoreServices/Finder.app</string>
\t\t<key>applicationPaths</key>
\t\t<array>
\t\t\t<string>/System/Library/CoreServices/Finder.app</string>
\t\t</array>
\t\t<key>inputTypeIdentifier</key>
\t\t<string>com.apple.Automator.fileSystemObject</string>
\t\t<key>outputTypeIdentifier</key>
\t\t<string>com.apple.Automator.nothing</string>
\t\t<key>presentationMode</key>
\t\t<integer>15</integer>
\t\t<key>processesInput</key>
\t\t<false/>
\t\t<key>serviceApplicationBundleID</key>
\t\t<string>com.apple.finder</string>
\t\t<key>serviceApplicationPath</key>
\t\t<string>/System/Library/CoreServices/Finder.app</string>
\t\t<key>serviceInputTypeIdentifier</key>
\t\t<string>com.apple.Automator.fileSystemObject</string>
\t\t<key>serviceOutputTypeIdentifier</key>
\t\t<string>com.apple.Automator.nothing</string>
\t\t<key>serviceProcessesInput</key>
\t\t<false/>
\t\t<key>systemImageName</key>
\t\t<string>NSActionTemplate</string>
\t\t<key>useAutomaticInputType</key>
\t\t<false/>
\t\t<key>workflowTypeIdentifier</key>
\t\t<string>com.apple.Automator.servicesMenu</string>
\t</dict>
</dict>
</plist>
'''

with open(os.path.join(wf_dir, "Info.plist"), "w", encoding="utf-8") as f:
    f.write(info_plist)
with open(os.path.join(wf_dir, "document.wflow"), "w", encoding="utf-8") as f:
    f.write(doc_wflow)
PYEOF

    plutil -lint "${wf_dir}/Info.plist" >/dev/null
    plutil -lint "${wf_dir}/document.wflow" >/dev/null
}

mkdir -p "$SERVICES_DIR"

# ─── 1. 圧縮ワークフロー用シェルスクリプト (APLZ v4 .aplz) ──────────────────────
read -r -d '' COMPRESS_SCRIPT <<'EOF' || true
export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
export LANG="ja_JP.UTF-8"
export LC_CTYPE="UTF-8"

exec >> /tmp/aplz_service.log 2>&1
echo "=== $(date) [compress] ==="

APLZ_FALLBACK="__APLZ_FALLBACK__"

notify() {
    osascript \
        -e 'on run argv' \
        -e 'display notification (item 3 of argv) with title (item 1 of argv) subtitle (item 2 of argv)' \
        -e 'end run' \
        -- "$1" "$2" "$3" >/dev/null 2>&1 || true
}

resolve_aplz() {
    local c
    for c in "$HOME/.local/bin/aplz" "/usr/local/bin/aplz" "/opt/homebrew/bin/aplz" "$APLZ_FALLBACK"; do
        if [ -x "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

unique_path() {
    local dir="$1"
    local name="$2"
    local is_dir="$3"
    local cand="${dir}/${name}"
    if [ ! -e "$cand" ] && [ ! -L "$cand" ]; then
        echo "$cand"
        return 0
    fi
    local stem="$name"
    local ext=""
    if [ "$is_dir" -eq 0 ]; then
        local no_lead="${name#.}"
        if [[ "$no_lead" == *.* ]]; then
            stem="${name%.*}"
            ext=".${name##*.}"
        fi
    fi
    local n=2
    while :; do
        cand="${dir}/${stem} ${n}${ext}"
        if [ ! -e "$cand" ] && [ ! -L "$cand" ]; then
            echo "$cand"
            return 0
        fi
        n=$((n + 1))
    done
}

copy_with_retry() {
    local src="$1"
    local dst="$2"
    local attempt
    for attempt in 1 2 3 4 5; do
        cp -p "$src" "$dst" 2>/dev/null && return 0
        ditto -- "$src" "$dst" 2>/dev/null && return 0
        cat "$src" > "$dst" 2>/dev/null && return 0
        rm -f "$dst"
        sleep 1
    done
    return 1
}

stage_tree() {
    local src="$1"
    local dst="$2"
    local item rel item_target link_target
    mkdir -p "$dst" || return 1
    while IFS= read -r -d '' item; do
        [ "$item" = "$src" ] && continue
        rel="${item#$src/}"
        item_target="${dst}/${rel}"
        if [ -L "$item" ]; then
            mkdir -p "$(dirname "$item_target")" || return 1
            link_target="$(readlink "$item")" || return 1
            ln -s "$link_target" "$item_target" || return 1
        elif [ -d "$item" ]; then
            mkdir -p "$item_target" || return 1
        elif [ -f "$item" ]; then
            mkdir -p "$(dirname "$item_target")" || return 1
            copy_with_retry "$item" "$item_target" || return 1
            chmod "$(stat -f %Lp "$item" 2>/dev/null)" "$item_target" 2>/dev/null || true
        fi
    done < <(find "$src" -print0)
}

APLZ_BIN="$(resolve_aplz)" || {
    echo "[ERR] aplz binary not found"
    notify "APLZ" "圧縮失敗" "aplz コマンドが見つかりません"
    exit 1
}

_active_work=""
trap '[ -n "$_active_work" ] && rm -rf "$_active_work" 2>/dev/null' EXIT INT TERM

total=$#
index=0
for raw_f in "$@"; do
    index=$((index + 1))
    f="$raw_f"
    [ "$f" != "/" ] && f="${f%/}"
    base="$(basename "$f")"
    dir="$(dirname "$f")"

    if [ ! -e "$f" ] && [ ! -L "$f" ]; then
        echo "[ERR] Not found: $f"
        notify "APLZ" "圧縮失敗" "${base} が見つかりません"
        continue
    fi

    echo "compressing [${index}/${total}]: $f"
    notify "APLZ" "圧縮中 (${index}/${total})" "$base"

    work_root="$(mktemp -d "${dir}/.aplz_work_XXXXXX" 2>/dev/null || mktemp -d /tmp/aplz_work.XXXXXX)"
    _active_work="$work_root"
    tmp_out="${work_root}/${base}.aplz"
    log_out="${work_root}/aplz.out"
    rc=0

    # 高速ゼロコピー直圧縮を試行
    if "$APLZ_BIN" c -3 -o "$tmp_out" "$f" > "$log_out" 2>&1; then
        rc=0
    else
        cat "$log_out" 2>/dev/null || true
        echo "direct compress failed, retrying via local staging (cloud drive fallback)..."
        rm -f "$tmp_out"
        staged="${work_root}/${base}"
        if [ -d "$f" ] && [ ! -L "$f" ]; then
            if stage_tree "$f" "$staged" && "$APLZ_BIN" c -3 -o "$tmp_out" "$staged" > "$log_out" 2>&1; then
                rc=0
            else
                rc=$?
            fi
        else
            if copy_with_retry "$f" "$staged" && "$APLZ_BIN" c -3 -o "$tmp_out" "$staged" > "$log_out" 2>&1; then
                rc=0
            else
                rc=$?
            fi
        fi
    fi

    cat "$log_out" 2>/dev/null || true

    if [ $rc -eq 0 ] && [ -f "$tmp_out" ]; then
        target="$(unique_path "$dir" "${base}.aplz" 0)"
        if mv "$tmp_out" "$target" 2>/dev/null || (cp "$tmp_out" "$target" && rm -f "$tmp_out"); then
            out_name="$(basename "$target")"
            summary="$(awk -F'|' '/^[[:space:]]*Size[[:space:]]*:/ {s=$1; sub(/^[[:space:]]*Size[[:space:]]*:[[:space:]]*/, "", s); sub(/[[:space:]]*$/, "", s); print s; exit}' "$log_out" 2>/dev/null || true)"
            echo "done: $target ($summary)"
            if [ -n "$summary" ]; then
                notify "APLZ" "圧縮完了 (${summary})" "$out_name"
            else
                notify "APLZ" "圧縮完了" "$out_name"
            fi
        else
            rc=1
            echo "[ERR] failed to move archive to: $target"
            notify "APLZ" "圧縮失敗" "${base} の保存に失敗しました"
        fi
    else
        echo "exit code: $rc"
        notify "APLZ" "圧縮失敗" "${base}（ログ: /tmp/aplz_service.log）"
    fi

    rm -rf "$work_root"
    _active_work=""
done
EOF

# ─── 2. 解凍ワークフロー用シェルスクリプト (APLZ v4 .aplz 専用) ─────────────────
read -r -d '' EXTRACT_SCRIPT <<'EOF' || true
export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"
export LANG="ja_JP.UTF-8"
export LC_CTYPE="UTF-8"

exec >> /tmp/aplz_service.log 2>&1
echo "=== $(date) [decompress] ==="

APLZ_FALLBACK="__APLZ_FALLBACK__"

notify() {
    osascript \
        -e 'on run argv' \
        -e 'display notification (item 3 of argv) with title (item 1 of argv) subtitle (item 2 of argv)' \
        -e 'end run' \
        -- "$1" "$2" "$3" >/dev/null 2>&1 || true
}

resolve_aplz() {
    local c
    for c in "$HOME/.local/bin/aplz" "/usr/local/bin/aplz" "/opt/homebrew/bin/aplz" "$APLZ_FALLBACK"; do
        if [ -x "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

unique_path() {
    local dir="$1"
    local name="$2"
    local is_dir="$3"
    local cand="${dir}/${name}"
    if [ ! -e "$cand" ] && [ ! -L "$cand" ]; then
        echo "$cand"
        return 0
    fi
    local stem="$name"
    local ext=""
    if [ "$is_dir" -eq 0 ]; then
        local no_lead="${name#.}"
        if [[ "$no_lead" == *.* ]]; then
            stem="${name%.*}"
            ext=".${name##*.}"
        fi
    fi
    local n=2
    while :; do
        cand="${dir}/${stem} ${n}${ext}"
        if [ ! -e "$cand" ] && [ ! -L "$cand" ]; then
            echo "$cand"
            return 0
        fi
        n=$((n + 1))
    done
}

copy_with_retry() {
    local src="$1"
    local dst="$2"
    local attempt
    for attempt in 1 2 3 4 5; do
        cp -p "$src" "$dst" 2>/dev/null && return 0
        ditto -- "$src" "$dst" 2>/dev/null && return 0
        cat "$src" > "$dst" 2>/dev/null && return 0
        rm -f "$dst"
        sleep 1
    done
    return 1
}

read_magic8() {
    LC_ALL=C dd if="$1" bs=8 count=1 2>/dev/null | od -An -tx1 | tr -d ' \t\n'
}

EXTRACTED_DEST=""
move_extracted_tree() {
    local stage_dir="$1"
    local target_dir="$2"
    local fallback_name="$3"
    EXTRACTED_DEST=""

    local entries=()
    shopt -s dotglob nullglob
    entries=("$stage_dir"/*)
    shopt -u dotglob nullglob

    local count=${#entries[@]}
    if [ "$count" -eq 0 ]; then
        local dest="$(unique_path "$target_dir" "$fallback_name" 1)"
        mkdir -p "$dest" || return 1
        EXTRACTED_DEST="$dest"
        return 0
    elif [ "$count" -eq 1 ]; then
        local item="${entries[0]}"
        local item_name="$(basename "$item")"
        local is_dir=0
        [ -d "$item" ] && [ ! -L "$item" ] && is_dir=1
        local dest="$(unique_path "$target_dir" "$item_name" "$is_dir")"
        if mv "$item" "$dest" 2>/dev/null || (cp -a "$item" "$dest" && rm -rf "$item"); then
            EXTRACTED_DEST="$dest"
            return 0
        fi
        return 1
    else
        local dest="$(unique_path "$target_dir" "$fallback_name" 1)"
        mkdir -p "$dest" || return 1
        if mv "${entries[@]}" "$dest/" 2>/dev/null || (cp -a "${entries[@]}" "$dest/" && rm -rf "${entries[@]}"); then
            EXTRACTED_DEST="$dest"
            return 0
        fi
        return 1
    fi
}

APLZ_BIN="$(resolve_aplz)" || {
    echo "[ERR] aplz binary not found"
    notify "APLZ" "解凍失敗" "aplz コマンドが見つかりません"
    exit 1
}

_active_work=""
trap '[ -n "$_active_work" ] && rm -rf "$_active_work" 2>/dev/null' EXIT INT TERM

total=$#
index=0
for raw_f in "$@"; do
    index=$((index + 1))
    f="$raw_f"
    [ "$f" != "/" ] && f="${f%/}"
    base="$(basename "$f")"
    dir="$(dirname "$f")"

    if [ ! -f "$f" ]; then
        echo "[SKIP] Not a regular file: $f"
        notify "APLZ" "解凍スキップ" "${base} はファイルではありません"
        continue
    fi

    echo "extracting [${index}/${total}]: $f"
    notify "APLZ" "解凍中 (${index}/${total})" "$base"

    work_root="$(mktemp -d "${dir}/.aplz_extract_XXXXXX" 2>/dev/null || mktemp -d /tmp/aplz_extract.XXXXXX)"
    _active_work="$work_root"
    stage_dir="${work_root}/out"
    mkdir -p "$stage_dir"

    src_file="$f"
    magic8="$(read_magic8 "$src_file")"
    if [ -z "$magic8" ]; then
        if copy_with_retry "$f" "${work_root}/${base}"; then
            src_file="${work_root}/${base}"
            magic8="$(read_magic8 "$src_file")"
        fi
    fi

    if [ "$magic8" != "41504c5a04000000" ]; then
        if [[ "$magic8" == 41504c5a* ]]; then
            echo "[ERR] Unsupported legacy APLZ version (magic8=${magic8}): $f"
            notify "APLZ" "非対応の旧バージョン" "${base} は旧 APLZ 形式です（APLZ v4 非対応）"
        else
            echo "[ERR] Not an APLZ v4 archive (magic8=${magic8:-none}): $f"
            notify "APLZ" "非対応の形式" "${base} は .aplz アーカイブではありません"
        fi
        rm -rf "$work_root"
        _active_work=""
        continue
    fi

    rc=0
    if "$APLZ_BIN" x "$src_file" -C "$stage_dir"; then
        rc=0
    elif [ "$src_file" = "$f" ]; then
        echo "direct extract failed, retrying via local staging..."
        rm -rf "$stage_dir" && mkdir -p "$stage_dir"
        if copy_with_retry "$f" "${work_root}/${base}" && "$APLZ_BIN" x "${work_root}/${base}" -C "$stage_dir"; then
            rc=0
        else
            rc=$?
        fi
    else
        rc=1
    fi

    if [ $rc -eq 0 ]; then
        fallback="${base%.aplz}"
        [ "$fallback" = "$base" ] && fallback="${base}_extracted"
        if move_extracted_tree "$stage_dir" "$dir" "$fallback"; then
            echo "done: $EXTRACTED_DEST"
            notify "APLZ" "解凍完了" "$(basename "$EXTRACTED_DEST")"
        else
            echo "[ERR] failed to move extracted items to $dir"
            notify "APLZ" "解凍失敗" "${base} の配置に失敗しました"
        fi
    else
        echo "exit code: $rc"
        notify "APLZ" "解凍失敗" "${base}（ログ: /tmp/aplz_service.log）"
    fi

    rm -rf "$work_root"
    _active_work=""
done
EOF

echo "==> Creating: ${SERVICES_DIR}/APLZ で圧縮.workflow"
create_workflow "APLZ で圧縮" "$COMPRESS_SCRIPT"

echo "==> Creating: ${SERVICES_DIR}/APLZ で解凍.workflow"
create_workflow "APLZ で解凍" "$EXTRACT_SCRIPT"

# macOS サービスキャッシュを更新
/System/Library/CoreServices/pbs -flush 2>/dev/null || true

echo ""
echo "============================================================"
echo " Finder サービス / クイックアクションをインストールしました"
echo ""
echo " 登録されたサービス:"
echo "   - APLZ で圧縮  → 選択したファイル／フォルダを .aplz 圧縮"
echo "   - APLZ で解凍  → 選択した .aplz ファイルを解凍"
echo ""
echo " 動作仕様:"
echo "   - 同名ファイル／フォルダがある場合は自動リネーム (例: foo 2.aplz / foo 2)"
echo "   - ログ出力先: /tmp/aplz_service.log"
echo ""
echo " アンインストール:"
echo "   ./setup_finder.sh --uninstall"
echo "============================================================"
