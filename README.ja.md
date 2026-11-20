# APLZ v4 (`.aplz`) — Apple Silicon UMA 異種協調アーカイバ

[![CI](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml/badge.svg)](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml)

[English README](README.md)

**APLZ（`aplz`）** は、Apple Silicon（M1〜M4+）のユニファイドメモリアーキテクチャ（UMA）とハードウェア構造から逆算して設計した、次世代ロスレス圧縮・ネイティブアーカイブツールです。

CPU の P-core・E-core と Apple GPU を 3 スロット完全非同期パイプラインで協調動作させ、さらに **FastCDC 内容定義チャンク重複排除**、**型別 Solid Grouping**、**256 KiB ブロック単位の O(1) 部分ランダムアクセス展開** を単一バイナリフォーマット `.aplz`（v4）に統合しています。

---

## 主な特長

- **Apple Silicon「32-way 対称性」**: Apple GPU の SIMD-group 幅（32）と ARM64 NEON の 128-bit レジスタ群（32 本、`uint32x4_t × 8`）に 1:1 対応する **32-way Coalesced rANS** を採用。GPU と CPU で **1 ビット残らず完全一致** する出力を生成します。
- **Zero-Cold-Start Hybrid CPU/GPU 制御**:
  - 小〜中ファイル（`< 384 MiB`）: Metal 起動コストを避け、ARM64 NEON（`vqtbl1q_u8` 32-way rANS 復号・64-bit 1 乗算 4-way ILP 符号化・L1D 融合 LUT）で起動遅延 0 ms のまま完走。
  - 大ファイル（`>= 384 MiB`）: 先頭バッチを CPU で即時処理しつつ裏側で Metal を非同期初期化し、完了次第 **3 スロット並列パイプライン（CPU LZ+前処理 || Apple GPU rANS || 並列パッキング＋fwrite）** へ自動移行。
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
| **`aplz -1`** | **12.39 MiB** | **13.9%** | **43.9 ms** | **61.7 ms** | **3.0 ms** | **3.13 ms** |
| **`aplz -3`** | **12.33 MiB** | **13.8%** | **46.5 ms** | **61.5 ms** | **2.9 ms** | **3.08 ms** |
| `tar+zstd -1 -T0` | 33.07 MiB | 37.0% | 338.9 ms | 355.5 ms | 47.0 ms | 46.03 ms |
| `tar+zstd -3 -T0` | 32.39 MiB | 36.2% | 357.4 ms | 353.3 ms | 58.2 ms | 57.36 ms |
| `tar+lz4 -1` | 47.97 MiB | 53.7% | 371.5 ms | 354.0 ms | 41.8 ms | 41.02 ms |

### 2. 大容量単一ファイル — 1.00 GiB 混在データ（`FP32 + FP16 + Mach-O + Mixed`）

| ツール | 圧縮率 | 出力サイズ | 圧縮速度 | 解凍 (`d`) | 展開 (`x`) | 検証 (`t`) / 一覧 (`l`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`aplz -1`** | **38.24%** | **391.6 MiB** | **3,297 MB/s (0.31 秒)** | **3,566 MB/s (0.29 秒)** | **3,109 MB/s (0.33 秒)** | 検証 `0.12 秒` (8,403 MB/s) / 一覧 `3.9 ms` |
| **`aplz -3`** | **37.56%** | **384.7 MiB** | **3,207 MB/s (0.32 秒)** | **3,438 MB/s (0.30 秒)** | **3,657 MB/s (0.28 秒)** | 検証 `0.12 秒` (8,768 MB/s) / 一覧 `3.7 ms` |
| `aplz -1 --no-dedup` | 58.12% | 595.2 MiB | 2,642 MB/s (0.39 秒) | 3,120 MB/s (0.33 秒) | 2,662 MB/s (0.38 秒) | 検証 `0.14 秒` (7,421 MB/s) / 一覧 `4.7 ms` |
| `zstd -3 -T0` | 66.20% | 677.9 MiB | 2,003 MB/s (0.51 秒) | 1,263 MB/s (0.81 秒) | — | — |

### 3. 単一ファイル — 256 MiB（`aplz` vs `zstd -3 -T0` vs `lz4 -1`）

| データセット | `aplz -1` (圧縮率 / 圧縮 / 解凍) | `aplz -3` (圧縮率 / 圧縮 / 解凍) | `zstd -3 -T0` (圧縮率 / 圧縮 / 解凍) | `lz4 -1` (圧縮率 / 圧縮 / 解凍) |
| :--- | :--- | :--- | :--- | :--- |
| **Source+Docs** | 2.9% / 5,026 / 4,566 MB/s | 2.7% / 4,756 / 4,406 MB/s | 0.4% / 9,656 / 5,600 MB/s | 33.8% / 1,103 / 1,739 MB/s |
| **Mach-O ARM64** | **1.9%** / **7,309** / **4,847 MB/s** | **1.8%** / **7,041** / **5,068 MB/s** | 39.8% / 1,966 / 1,213 MB/s | 57.4% / 737 / 1,575 MB/s |
| **FP16 Weights** | **84.1%** / 2,655 / **4,258 MB/s** | **84.1%** / 2,666 / **4,197 MB/s** | 91.6% / 3,678 / 1,220 MB/s | 100.0% / 2,622 / 2,291 MB/s |
| **FP32 Telemetry** | **41.0%** / **2,487** / **4,677 MB/s** | **38.4%** / **2,000** / **4,718 MB/s** | 75.5% / 1,314 / 1,030 MB/s | 99.9% / 1,613 / 1,945 MB/s |
| **Mixed Real** | **48.5%** / 2,857 / **2,849 MB/s** | **48.3%** / 2,710 / **3,513 MB/s** | 57.9% / 3,149 / 1,964 MB/s | 72.8% / 1,318 / 1,867 MB/s |

> **ヒント**: チャンク間重複がない単一の数値テンソル・AI 重みファイルでは、`--no-dedup` を指定することで FastCDC 走査を省略し 256 MiB 単位でダイレクト圧縮できます（`FP16 Weights`: 圧縮 **4,338 MB/s** / 解凍 **4,260 MB/s**、`FP32 Telemetry`: 圧縮 **3,390 MB/s** / 解凍 **4,749 MB/s**）。

---

## ディレクトリ構成

- [SPEC.md](SPEC.md) — 詳細設計・数理・バイナリ仕様書
- [src/rans/](src/rans/) — 32-way Coalesced rANS コア（Metal GPU / ARM64 NEON / Scalar リファレンス）
- [src/engine/](src/engine/) — 3 スロット UMA 異種協調ブロックエンジン（`ASZ2`）
- [src/archive/](src/archive/) — `.aplz` v4 ネイティブコンテナ・FastCDC・Solid Grouping・`aplz` CLI
- [tests/](tests/) — 自動検証スイートおよびベンチマーク
- [build.sh](build.sh) / [install.sh](install.sh) / [setup_finder.sh](setup_finder.sh) — ビルド・インストール・macOS Finder サービス登録スクリプト
