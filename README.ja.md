# APLZ — Apple Parallel LZ

[English](README.md) | **日本語** | [中文](README.zh.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

Apple Silicon GPU で動作する高速ロスレス圧縮/解凍エンジン。Metal Compute Shader (MSL 3.0) と tANS (Asymmetric Numeral Systems) を組み合わせた GPU パイプラインにより、圧縮・解凍の完全な Round-trip を実現する。

## 主な特徴

- Zero-copy I/O: Apple Silicon の Unified Memory Architecture (UMA) を使い、`mmap` + `newBufferWithBytesNoCopy` により CPU↔GPU 間の入力データコピーを排除する。
- 並列 LZ77: 128 スレッドが 64 KB チャンクを協調して処理する。4-gram 乗算ハッシュ (`h4`)、2-way ハッシュ (`atomic_fetch_min`)、Run-Skipping、4 バイト並列比較で高速にマッチを探索し、parallel prefix-sum でコンパクションする。
- GPU Per-chunk 128 Interleaved tANS: チャンクごとの最適な tANS テーブルを GPU (`build_tans_tables`) で動的に構築する。各スレッドが独立した ANS 状態を保持し、128 本のビットストリームを並列にエンコード/デコードする。
- 完全並列 LZ77 デコード: タイル単位の parallel prefix-scan でバリデーションとリテラル書き込みを進め、重複マッチ (`dist < len`) もシリアルフォールバックなしでモジュロ演算により 128 スレッドで並列コピーする。
- 非同期ダブルバッファリング: GCD (`dispatch_semaphore` / `dispatch_group`) によるバッチパイプラインで、GPU 実行と CPU テーブル構築・非同期ファイル I/O をオーバーラップさせる。
- O(1) ストリーミング・メモリ: 入力サイズに依存しない固定のメモリ使用量で動作する。メガバッチ (32 MB) 単位で処理し、5 GB+ のファイルでも安定して動作する。
- Perfect Round-trip: 圧縮と解凍でバイトレベルの完全一致を保証する。GPU 側で ANS/LZ ストリームのバリデーションも備える。

## アーキテクチャ

### 圧縮パイプライン (`-c`)

```
mmap(input)  <- zero-copy MTLBuffer (Apple Silicon UMA)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  メガバッチ・ストリーミング (512 chunks = 32 MB / batch)   ║
║  (Pass 1 はメガバッチ間でダブルバッファ事前プリフェッチ)    ║
║                                                           ║
║  1. GPU Pass 1: compress_chunk                            ║
║     ┌───────────────────────────────────────────────────┐ ║
║     │ 1 threadgroup (128 threads) = 1 chunk (64 KB)     │ ║
║     │   A-1  Hash table init                            │ ║
║     │   A-2  4-gram hash + Run-Skipping (atomic_min x2) │ ║
║     │   A-3  LZ77 4-byte parallel match -> sparse[]     │ ║
║     │   B    Parallel prefix-sum compaction -> compact[]│ ║
║     └───────────────────────────────────────────────────┘ ║
║                          |                                ║
║                          v                                ║
║  2. GPU テーブル構築: build_tans_tables                    ║
║     Per chunk: Histogram -> Normalize -> Spread -> EncTbl ║
║                          | SymInfo[512] + enc_table[1024] ║
║                          v                                ║
║  3. GPU Pass 2: tans_encode (ダブルバッファ, 32 ch/batch)  ║
║     チャンクあたり 128 本のインターリーブ ANS ストリーム    ║
║     addCompletedHandler -> serial write queue (tmp 書出)  ║
╚═══════════════════════════════════════════════════════════╝
    | シークテーブル書き戻し + アトミック rename
    v
  .aplz file
```

### 解凍パイプライン (`-d`)

```
mmap(.aplz) -> ヘッダ解析 + chunk_offsets バリデーション
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  メガバッチ・ストリーミング (ダブルバッファ 32 MB 出力)     ║
║  for each mega-batch:                                     ║
║    ┌───────────────────────────────────────────────────┐  ║
║    │ ダブルバッファ・バッチパイプライン (32 chunks/batch)│  ║
║    │                                                   │  ║
║    │ CPU (batch N+1, dispatch_apply で並列構築):        │  ║
║    │   per-chunk compact_freq + 128 streams 読み込み    │  ║
║    │   DecodeEntry[L] をチャンクごとに構築              │  ║
║    │                                                   │  ║
║    │ GPU (batch N, 128 threads/chunk):                 │  ║
║    │   tans_decode -> lz77_decode -> mb_out[slot]      │  ║
║    └───────────────────────────────────────────────────┘  ║
║    -> 非同期 fwrite(mb_out[slot]) -> バッファ再利用        ║
╚═══════════════════════════════════════════════════════════╝
    v
  復元データ
```

## ファイルフォーマット (.aplz)

| Offset | Size | Content |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic `"APLZ"`, `version = 3`, `original_size`, `chunk_size = 65536`, `num_chunks`) |
| 24 | 4 B | `n_streams` = 128 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8×N B | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (N チャンク、v3 形式) |

各チャンクのデータ構造 (v3 形式) は次のとおり。

- `token_cnt` (4 B): dense token 数
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` ペア (各 4 B、非ゼロシンボルのみ)
- `stream_sizes[128]` (各 2 B = 256 B): 各 ANS ストリームのバイトサイズ
- stream data: 連結された ANS ビットストリーム (128 インターリーブストリーム、各マッチトークンは 16-bit 後方距離を含む)

> 空ファイル (`original_size = 0`, `num_chunks = 0`) は 32 バイトのヘッダのみ (`FileHeader` + `n_streams` + `ans_log_l`) で保存される。

## インストール

macOS + Xcode Command Line Tools が必要。

```bash
# ~/.local/bin にインストール (-O3 -flto 最適化ビルド + 事前コンパイル metallib)
./install.sh

# /usr/local/bin にインストール (sudo 必要)
./install.sh /usr/local

# アンインストール
./install.sh uninstall
```

### Finder 統合 (オプション)

```bash
# 右クリック ->「APLZ で圧縮」「APLZ で解凍」を追加
./setup_finder.sh

# 削除
./setup_finder.sh uninstall
```

Google Drive やクラウドストレージ上のファイルにも対応する。サービス実行時は自動的に `/tmp` 経由で処理することで、macOS サンドボックス制限を回避する。

## 使い方

### aplz コマンド (推奨)

```bash
# ファイル圧縮
aplz compress myfile.txt              → myfile.txt.aplz

# ディレクトリ圧縮 (自動 tar)
aplz compress myproject/              → myproject.tar.aplz

# 解凍
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# ファイル情報
aplz info myfile.txt.aplz
```

### gpu_zip (低レベル API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal [--debug]
./gpu_zip -d <input.aplz> <output> compression.metal [--debug]
```

> `compression.metal` と同じ場所に事前コンパイル済みの `compression.metallib` がある場合は自動でロードし、存在しない場合は実行時にソースコンパイルへフォールバックする。

### 開発用ビルド

```bash
./build.sh        # -O2 ビルド (gpu_zip + compression.metallib を生成)
./build.sh clean   # 成果物削除
```

## パフォーマンス (Apple M4)

`-O3 -flto` リリースビルドで 512 MB の入力を 3 回実行したベスト値（バイト完全一致の Round-trip を検証済み）。

### 圧縮

| データ | 入力 | 出力 | 圧縮率 | スループット |
|---|---|---|---|---|
| テキスト (繰り返しパターン) | 512 MB | 10.3 MB | 2.0% | 297 MB/s |
| ランダムバイナリ | 512 MB | 535.9 MB | 104.7% | 134 MB/s |

### 解凍

| データ | 入力 | 速度 | 備考 |
|---|---|---|---|
| テキスト (繰り返しパターン) | 512 MB | 2.6 GB/s | 16 mega-batches、フルパイプラインオーバーラップ |
| ランダムバイナリ | 512 MB | 232 MB/s | 16 mega-batches、ほぼリテラル |

> ランダムデータは非圧縮のため、tANS や distance フィールドのオーバーヘッドにより出力が入力をわずかに超える。
> メガバッチストリーミングにより、入力サイズに関係なくメモリ使用量は一定。

## ソースファイル構成

| ファイル | 役割 |
|---|---|
| `APLZ.h` | 共有ヘッダ (CPU/GPU 共通の構造体・定数) |
| `compression.metal` | GPU カーネル (MSL 3.0): `compress_chunk`, `build_tans_tables`, `tans_encode`, `tans_decode`, `lz77_decode` |
| `main.mm` | ホストドライバ (Objective-C++): O(1) ストリーミング・メガバッチ + 非同期ダブルバッファパイプライン |
| `aplz` | ユーザー向け CLI ラッパー (Bash): ファイル/ディレクトリ対応、tar 統合 |
| `install.sh` | インストーラ: 最適化ビルド (`-O3 -flto` + `.metallib`) + システムワイドデプロイ |
| `setup_finder.sh` | macOS Finder Quick Action 生成 (右クリック圧縮/解凍) |
| `build.sh` | 開発用ビルドスクリプト |
| `bench_compress.sh` | gzip/zip/zstd/lz4/xz との比較ベンチマークスクリプト |

## 技術詳細

- LZ77: 4-gram 乗算ハッシュ (`h4`)、2-way ハッシュテーブル (`atomic_fetch_min`)、Run-Skipping、4 バイト並列マッチ比較、parallel prefix-sum コンパクション (`sparse` → `compact`)。最大一致長 255、最小一致長 3、後方距離最大 65534
- Per-chunk tANS (v3): 各 64 KB チャンクの最適な tANS テーブルを GPU (`build_tans_tables`) で動的に構築する。128 インターリーブ ANS ストリーム、512 シンボル (256 literals + 256 match lengths)、L=1024、Duda's fast spread、Threadgroup cooperative loading によりテーブルを高速に参照する
- ANS Encoding: 正規化付きフォワードエンコード、16-bit 距離フィールド、終端検出用センチネルビット
- ANS Decoding: 最終状態からの逆順デコード、`clz()` によるセンチネル検出、ワード単位バッチビット抽出、O(1) デコードテーブル参照、ストリーム・状態境界バリデーション
- Compact Frequency Serialization: Per-chunk 頻度テーブルを非ゼロエントリのみ保存 (`n_nonzero + [sym_id, freq]` ペア) してファイルサイズのオーバーヘッドを最小化する
- 完全並列 LZ77 デコード: タイル単位の parallel prefix-scan によるバリデーション・リテラル書き込みと、モジュロ演算 (`out[off - dist + (j % dist)]`) による重複・非重複マッチの統一並列コピー（シリアルフォールバックなし）
- 非同期ダブルバッファリング: GCD バッチパイプライン (32 chunks/batch, 2 slots)。`dispatch_semaphore` によるスロット排他、`dispatch_apply` による解凍テーブル並列構築、`addCompletedHandler` とシリアル `dispatch_queue` による順序保証付き非同期書き込み、圧縮時のアトミック `rename`
- O(1) ストリーミング・メモリ: メガバッチ構成 (512 chunks = 32 MB/batch)。バッファを再利用するため、入力サイズに依存しない固定のメモリ使用量となる

## 既知の制約

- BS_CAP = 1024 B: 1 ストリームの最大出力は 1024 バイト。極端に膨張する非圧縮データではビットが切り捨てられる可能性がある。
- チャンクサイズ固定 64 KB: `APLZ.h` / `compression.metal` で変更できるが、再ビルドが必要。

## License

MIT License. See [LICENSE](LICENSE) for details.
