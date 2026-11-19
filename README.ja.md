# APLZ v4 (`.aplz`) — Apple Silicon UMA 異種協調アーカイバ

[![CI](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml/badge.svg)](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml)

[English README](README.md)

**APLZ（`aplz`）** は、Apple Silicon（M1〜M4+）のユニファイドメモリアーキテクチャ（UMA）とハードウェア構造から逆算して設計した、次世代ロスレス圧縮・ネイティブアーカイブツールです。

CPU の P-core・E-core と Apple GPU を 3 スロット完全非同期パイプラインで協調動作させ、さらに **FastCDC 内容定義チャンク重複排除**、**型別 Solid Grouping**、**256 KiB ブロック単位の O(1) 部分ランダムアクセス展開** を単一バイナリフォーマット `.aplz`（v4）に統合しています。

---

## 主な特長

- **Apple Silicon「32-way 対称性」**: Apple GPU の SIMD-group 幅（32）と ARM64 NEON の 128-bit レジスタ群（32 本、`uint32x4_t × 8`）に 1:1 対応する **32-way Coalesced rANS** を採用。GPU と CPU で **1 ビット残らず完全一致** する出力を生成します。
- **Zero-Cold-Start Hybrid CPU/GPU 制御**:
  - 小〜中ファイル（`< 96 MiB`）: Metal 起動コストを避け、ARM64 NEON＋4 レーン ILP ブランチレス 32-way rANS で起動遅延 0 ms のまま完走。
  - 大ファイル（`>= 96 MiB`）: 先頭バッチを CPU で即時処理しつつ裏側で Metal を非同期初期化し、完了次第 **3 スロット並列パイプライン（CPU LZ+前処理 || Apple GPU rANS || 並列パッキング＋fwrite）** へ自動移行。
- **OpenZL 型ブロック別可逆前処理**: 256 KiB ブロックごとに高速プローブを行い、`SHUFFLE2`（FP16/BF16 AI 重み・プレーン別 rANS＋高エントロピー仮数部 RAW バイパス）、`SHUFFLE4`（FP32/Int32 数値・科学技術データ）、`ARM64_BCJ`（Apple Silicon Mach-O 実行バイナリ）、`RAW`/`RLE` を自動選択。
- **一時 `.tar` 廃止の `.aplz` v4 ネイティブコンテナ**:
  - **正規化 FastCDC**（Gear Hash, 16〜256 KiB, 160-bit 強指紋）により、ファイル間・世代バックアップ間・先頭バイト挿入シフト後でも重複チャンクを自動排除。
  - **型別 Solid Grouping** により、64 KiB 未満の小ファイルをデータ種別（`TEXT`/`CODE`/`NUM`/`MEDIA`/`BIN`）かつ拡張子順で 4 MiB 上限の Solid Pack へ集約。
  - **O(1) 一覧表示＆部分展開**: 末尾の圧縮セントラルディレクトリを読むだけで数 ms で一覧表示し、巨大アーカイブから 1 ファイルだけ取り出す場合も該当する 256 KiB ブロックのみを選択復号します。
- **3 重整合性検証と安全性**: ARM64 ハードウェア CRC32C と $\text{GF}(2)$ 結合により Header・Pack・256 KiB Block・Chunk・File を全層検証。Zip-Slip およびシンボリックリンク経由のディレクトリ逸脱を厳格に拒否します。

---

## インストール（macOS / Apple Silicon）

```bash
# ~/.local/bin/aplz へインストール（sudo 不要）
./install.sh

# システム全体（/usr/local/bin/aplz）へインストールする場合
./install.sh --system

# macOS Finder「サービス / クイックアクション」登録（APLZ で圧縮 / APLZ で解凍）
./setup_finder.sh

# アンインストール
./install.sh --uninstall
./setup_finder.sh --uninstall
```

### ソースからのビルド & 自動テスト

```bash
./build.sh test
```

---

## CLI の使い方

```bash
# ファイルやディレクトリを .aplz へ圧縮 (-1 高速 / -3 標準 / -6 高圧縮)
aplz c -3 -o backup.aplz ./my_project
aplz c -1 --no-dedup -o fast.aplz ./large_file.bin

# アーカイブ内容と重複排除サマリを O(1) で一覧表示
aplz l backup.aplz

# ディスクへ書き出さずに全層 CRC32C 整合性を検証
aplz t backup.aplz

# アーカイブ全体を展開
aplz x backup.aplz -C ./restored

# 特定ファイル / サブフォルダのみを O(1) ブロック部分展開
aplz x backup.aplz -C ./restored my_project/src/main.cpp

# 単一ファイルアーカイブを直接解凍
aplz d archive.aplz restored.bin
```

---

## 実測ベンチマーク（Apple M4, End-to-End ファイル対ファイル）

測定環境: Apple M4（10-core CPU / 10-core GPU, UMA, macOS, `aplz` v4.0.0）。すべて `cmp` / `dircmp` によるバイト完全一致を確認済みです。

### 1. 実プロジェクトディレクトリ（979 ファイル / 89.4 MiB：3 世代ソース＋FP16 重み 2 世代＋FP32 テレメトリ＋Mach-O）

| ツール | 圧縮後サイズ | 圧縮率 | 圧縮 | 全展開 | 一覧 (`l`/`-t`) | 1 ファイル部分展開 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`aplz -1`** | **12.39 MiB** | **13.9%** | **49.1 ms** | **76.0 ms** | **2.9 ms** | **3.23 ms** |
| **`aplz -3`** | **12.32 MiB** | **13.8%** | **49.8 ms** | **76.3 ms** | **3.0 ms** | **3.19 ms** |
| `tar+zstd -1 -T0` | 33.08 MiB | 37.0% | 346.5 ms | 364.3 ms | 47.0 ms | 46.17 ms |
| `tar+zstd -3 -T0` | 32.39 MiB | 36.2% | 373.1 ms | 368.1 ms | 58.0 ms | 57.63 ms |
| `tar+lz4 -1` | 47.97 MiB | 53.7% | 380.5 ms | 366.3 ms | 42.1 ms | 41.28 ms |

### 2. 大容量単一ファイル — 1.00 GiB 混在データ（`FP32 + FP16 + Mach-O + Mixed`）

| ツール | 圧縮率 | 出力サイズ | 圧縮速度 | 解凍 (`d`) | 展開 (`x`) | 検証 (`t`) / 一覧 (`l`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`aplz -1`** | **38.25%** | **391.6 MiB** | **3,010 MB/s (0.34 秒)** | **3,450 MB/s (0.30 秒)** | **3,192 MB/s (0.32 秒)** | 検証 `0.14 秒` (7,066 MB/s) / 一覧 `4.6 ms` |
| **`aplz -3`** | **37.55%** | **384.5 MiB** | **2,797 MB/s (0.37 秒)** | **3,496 MB/s (0.29 秒)** | **3,355 MB/s (0.31 秒)** | 検証 `0.15 秒` (7,020 MB/s) / 一覧 `4.4 ms` |
| `aplz -1 --no-dedup` | 58.11% | 595.1 MiB | 2,417 MB/s (0.42 秒) | 3,290 MB/s (0.31 秒) | 2,852 MB/s (0.36 秒) | 検証 `0.22 秒` (4,692 MB/s) / 一覧 `4.1 ms` |
| `zstd -3 -T0` | 66.20% | 677.9 MiB | 2,153 MB/s (0.48 秒) | 1,204 MB/s (0.85 秒) | — | — |

### 3. 単一ファイル — 256 MiB（`aplz` vs `zstd -3 -T0` vs `lz4 -1`）

| データセット | `aplz -1` (圧縮率 / 圧縮 / 解凍) | `aplz -3` (圧縮率 / 圧縮 / 解凍) | `zstd -3 -T0` (圧縮率 / 圧縮 / 解凍) | `lz4 -1` (圧縮率 / 圧縮 / 解凍) |
| :--- | :--- | :--- | :--- | :--- |
| **Source+Docs** | **3.0%** / 3,794 / 3,603 MB/s | **2.8%** / 3,650 / **3,826 MB/s** | 0.3% / 9,170 / 4,216 MB/s | 33.7% / 1,054 / 1,589 MB/s |
| **Mach-O ARM64** | **1.9%** / **4,869** / **4,187 MB/s** | **1.8%** / **4,991** / **3,987 MB/s** | 39.8% / 1,792 / 1,184 MB/s | 57.4% / 719 / 1,439 MB/s |
| **FP16 Weights** | **84.1%** / 1,840 / **3,542 MB/s** | **84.1%** / 2,105 / **3,596 MB/s** | 91.6% / 3,189 / 1,211 MB/s | 100.0% / 2,281 / 1,824 MB/s |
| **FP32 Telemetry** | **41.0%** / **2,215** / **3,998 MB/s** | **38.3%** / **1,693** / **3,959 MB/s** | 75.5% / 1,193 / 1,023 MB/s | 99.9% / 1,530 / 1,759 MB/s |
| **Mixed Real** | **48.5%** / 2,188 / **2,984 MB/s** | **48.4%** / 2,118 / **2,968 MB/s** | 57.9% / 2,933 / 1,867 MB/s | 72.7% / 1,082 / 1,747 MB/s |

> **ヒント**: チャンク間重複がない単一の数値テンソル・AI 重みファイルでは、`--no-dedup` を指定することで FastCDC 走査を省略し 256 MiB 単位でダイレクト圧縮できます（`FP16 Weights`: 圧縮 **3,211 MB/s** / 解凍 **3,905 MB/s**、`FP32 Telemetry`: 圧縮 **2,710 MB/s** / 解凍 **3,748 MB/s**）。

---

## ディレクトリ構成

- [SPEC.md](SPEC.md) — 詳細設計・数理・バイナリ仕様書
- [src/rans/](src/rans/) — 32-way Coalesced rANS コア（Metal GPU / ARM64 NEON / Scalar リファレンス）
- [src/engine/](src/engine/) — 3 スロット UMA 異種協調ブロックエンジン（`ASZ2`）
- [src/archive/](src/archive/) — `.aplz` v4 ネイティブコンテナ・FastCDC・Solid Grouping・`aplz` CLI
- [tests/](tests/) — 自動検証スイートおよびベンチマーク
- [build.sh](build.sh) / [install.sh](install.sh) / [setup_finder.sh](setup_finder.sh) — ビルド・インストール・macOS Finder サービス登録スクリプト
