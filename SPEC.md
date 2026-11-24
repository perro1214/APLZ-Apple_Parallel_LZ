# APLZ v4 (`.aplz`) 詳細設計仕様書 — Apple Silicon UMA 異種協調圧縮エンジン

> Status: APLZ v4 実装・実測検証・実用化完了（旧 APLZ v3 との後方互換なし）

---

## 1. 設計思想と全体アーキテクチャ

APLZ v4 は Apple Silicon（M1〜M4+）のハードウェア構造から逆算して設計された次世代ロスレス圧縮・解凍エンジンおよびネイティブアーカイバ（`.aplz`）である。

### 1.1 中核原則
1. **UMA（Unified Memory Architecture）真・異種協調**:
   - GPU 単騎に逐次 LZ パースを押し付けず、**CPU 全コア（256 KiB 分離ストリーム LZ パース・NEON 前処理・FastCDC 重複排除・HW CRC32C）** と **Apple GPU（32-way SIMD-group rANS バッチ符号化・復号）** を `MTLResourceStorageModeShared`（ゼロコピー UMA 共有バッファ）上の 3 スロット非同期パイプラインで連結し、全シリコンを同時稼働させる。
   - **Zero-Cold-Start Hybrid 実行**: 単発の小〜中規模データ（`< 384 MiB`）では Metal GPU 起動遅延（約 30–60 ms）を避け、GPU と 100% ビット完全一致する **CPU 全コア ブランチレス 32-way rANS（64-bit 1 乗算符号化 / NEON `vqtbl1q_u8` 復号）** で遅延 0 ms から即座に稼働する。大規模データ（`>= 384 MiB`）ではバックグラウンドで非同期 Metal 初期化を進め、完了次第ゼロ停止で GPU rANS へ自動シームレス切替する。
2. **Apple Silicon「32-way 対称性」**:
   - Apple GPU のネイティブ SIMD-group 幅 = **32**。
   - Apple CPU（ARM64）の 128-bit NEON レジスタ数 = **32 本**（32 レーンの 32-bit rANS 状態が `uint32x4_t × 8` = 8 レジスタに完全常駐）。
   - エントロピー符号を **32-way SIMD-group Coalesced rANS（16-bit 再正規化）** に統一することで、**同一バイナリを CPU と GPU のどちらでも 100% ビット一致でエンコード・デコード**できる。
3. **OpenZL 型軽量変換 DAG + Oodle Kraken 型分離ストリーム**:
   - ブロック（256 KiB）単位で 16 KiB 高速プローブ（`n*log2(n)` LUT）を行い、可逆前処理（`NONE` / `SHUFFLE2` / `SHUFFLE4` / `ARM64_BCJ`）とコーデック（`RAW` / `RLE` / `RANS_ONLY` / `LZ_RANS`）を自動選択する。
   - `RANS_ONLY` では `SHUFFLE2`/`SHUFFLE4` のプレーン単位でエントロピーを独立評価し、高エントロピー・プレーン（`> 7.80 bpb`、例: FP16 仮数部）を即座に RAW バイパスする。
   - `LZ_RANS` では LZ 出力を `Literals` / `LL` / `ML` / `OF`（各 rANS 圧縮）＋ `ExtraBits`（64-bit 生ビットストリーム）の 4+1 ストリームへ完全分離する。
4. **切り捨てゼロ・膨張上限保証・3 重破損検知**:
   - 圧縮結果が入力サイズ以上になるタイル／ブロックは必ず `RAW` にフォールバックし、単一バイトブロックは 0 バイトペイロードの `RLE` で表現する。
   - すべてのブロックに ARM64 ハードウェア Castagnoli CRC32C を付与し、さらに $\text{GF}(2)$ 事前計算行列による `hx_crc32c_combine`（~10 ns/結合）で並列ブロック CRC からパック／ファイル全体の厳密 CRC32C を再スキャンなしで合成する。

### 1.2 段階的開発ロードマップ（全 Phase 完了）

| Phase | スコープ | 状態・検証結果 |
|---|---|---|
| **Phase 1** (`src/rans/`) | **32-way Coalesced rANS コア**（Metal GPU + ARM64 NEON + Scalar + 4-way ILP ブランチレス CPU Enc、256 KiB ブロック / 64 KiB タイル、テーブル共有、RAW/RLE フォールバック、破損検知） | **完了** — GPU Enc ↔ CPU Scalar / NEON / Fast Enc ↔ GPU Dec 全クロス 100% ビット完全一致・破損入力 100% 検知 |
| **Phase 2** (`src/engine/`) | **UMA 異種パイプラインエンジン (`ASZ2`)**（3 スロット非同期 `CPU \|\| GPU \|\| Pack+fwrite`、Zero-Cold-Start Hybrid、NEON Byte-Shuffle / ARM64 BCJ、エポック ハッシュ分離ストリーム LZ、2 スロット非同期 GPU デコード、256 KiB ランダムアクセス `decompress_range`） | **完了** — 252 ラウンドトリップ + 1-bit 破損検知 + 18 GPU==CPU パリティ全 PASS（警告 0） |
| **Phase 3** (`src/archive/`) | **`.aplz` v4 ネイティブコンテナ & `aplz` CLI**（一時 `.tar` 廃止、5 クラス型別 Solid Grouping、並列正規化 FastCDC 重複排除、O(1) 一覧表示、256 KiB ブロック単位部分展開、3 重 CRC32C 検査、Zip-Slip / symlink 逸脱完全防御、`install.sh` / `setup_finder.sh`） | **完了** — 自動テスト 9/9 PASS（警告 0）、979 ファイル実ディレクトリ・256 MiB・1.00 GiB ストリーミング検証完了 |

---

## 2. Phase 1 詳細仕様: 32-way Coalesced rANS (`RANS32`)

### 2.1 定数と階層構造

| パラメータ | 値 | 設計根拠 |
|---|---|---|
| `RANS32_LANES` | `32` | Apple GPU SIMD-group 幅 (32) かつ ARM64 NEON `uint32x4_t × 8` に完全一致 |
| `RANS32_TILE_SIZE` | `65,536 B` (64 KiB) | 1 SIMD-group (32 threads) = 2,048 ステップ。並列度とオーバーヘッドの最適点 |
| `RANS32_TILES_PER_BLOCK` | `4` | 1 threadgroup (128 threads = 4 SIMD-groups) が 16 KB LUT を共有 |
| `RANS32_BLOCK_SIZE` | `262,144 B` (256 KiB) | 頻度テーブルヘッダ（240〜384 B）のオーバーヘッドを 0.1% 以下に圧縮 |
| `RANS32_PROB_BITS` | `12` ($M = 4096$) | 256 シンボル全員に最小頻度 1 を割り当てても余裕があり、高エントロピー損失 < 0.002 bit/sym |
| `RANS32_L` | `1u << 16` (`65536`) | 状態レンジ $[2^{16}, 2^{32}-1]$。16-bit ワード再正規化で 1 シンボル最大 1 回 (`if` のみ、`while` 不要) |
| `RANS32_MIN_RANS_LEN` | `256 B` | これ未満の端数タイルは常に `RAW` を選択しヘッダ膨張を防ぐ |

### 2.2 rANS 状態遷移の数理仕様

有効な rANS ブロックでは非ゼロシンボル数が必ず $\ge 2$ であるため（1 種類のみのブロックは `MODE_RLE` へ分岐）、全非ゼロシンボルの正規化頻度 $f_s$ は厳密に $1 \le f_s \le 4095$ かつ $\sum_{s=0}^{255} f_s = 4096$ を満たす。累積頻度 $\text{cdf}[s] = \sum_{k < s} f_k$ とする。

#### エンコード遷移（逆順: `step = S-1` → `0`）
初期状態: 各レーン $l \in [0, 31]$ に対し $x_l = 2^{16}$。
1. **再正規化判定**（最大 1 回）:
   $$x_{\max}(f_s) = f_s \ll 20 \quad (\text{最大 } 4095 \cdot 2^{20} < 2^{32})$$
   もし $x_l \ge (f_s \ll 20)$ ならば、下位 16 ビット $(x_l \mathbin{\&} \text{0xFFFF})$ を共有ストリームへ排出し、$x_l \leftarrow x_l \gg 16$。
2. **状態更新**:
   $$q = \lfloor x_l / f_s \rfloor, \quad r = x_l - q \cdot f_s$$
   $$x_l \leftarrow (q \ll 12) + \text{cdf}[s] + r$$
   - 除算回避には、GPU 側で Granlund-Montgomery 32-bit 逆数乗算（$f_s \ge 2$ に対し $k = \lfloor \log_2(f_s - 1) \rfloor$, $m = \lfloor 2^{32}(2^{k+1} - f_s)/f_s \rfloor + 1$, $t = \text{mulhi}(x_l, m)$, $q = (t + ((x_l - t) \gg 1)) \gg k$）を用い、CPU 高速エンコーダ（`rans32_encode_tile_fast`）では全 $f_s \in [1, 4095]$ に対し分岐ゼロで厳密一致する 64-bit 1 乗算式 $q = \lfloor (x_l \cdot \lceil 2^{44} / f_s \rceil) / 2^{44} \rfloor$ および $x_l \leftarrow (x_l + \text{cdf}[s]) + q \cdot (4096 - f_s)$ を用いる（全 $f_s \in [1, 4095]$・全有効 $x_l < f_s \cdot 2^{20}$ でオーバーフローなし・厳密一致を検証済み）。

#### デコード遷移（順方向: `step = 0` → `S-1`）
初期状態: ヘッダから `x_l = final_state[l]` をロード。
1. **16 KB 共有 LUT 1 回参照**（`uint32_t lut[4096]`）:
   - エントリ構造（32-bit パック）:
     - bits `0..7`: `symbol` (8-bit)
     - bits `8..19`: `freq` (12-bit, $1..4095$)
     - bits `20..31`: `bias = slot - cdf[symbol]` (12-bit, $0..4094$)
   - 1 回の 32-bit ロードで 3 要素を取得:
     ```c
     uint32_t entry = lut[x & 4095u];
     uint8_t  sym   = (uint8_t)(entry & 0xFFu);
     uint32_t freq  = (entry >> 8u) & 0xFFFu;
     uint32_t bias  = entry >> 20u;
     x = freq * (x >> 12u) + bias;
     ```
2. **再正規化補充**（最大 1 回）:
   - もし $x < 2^{16}$ ならば、共有ストリームから `uint16_t w` を 1 ワード補充し、$x \leftarrow (x \ll 16) \mid w$。
3. **終端厳密検証**:
   - 全ステップ完了後、**全 32 レーンで $x_l = 2^{16}$ かつ読み取りポインタ `rp == 0`** であることを検証する。不一致は即座に破損エラーとする。

### 2.3 32-way Coalesced ストリームレイアウト（コアレス規約）

入力タイル長 $N$ バイトに対し、ステップ数 $S = \lceil N / 32 \rceil$。レーン $l \in [0, 31]$ は位置 $\text{pos} = \text{step} \cdot 32 + l < N$ を担当する。

- **エンコード時（`step = S-1` → `0`）**:
  - ステップ内で `need_renorm[l]` が真のレーン集合に対し、レーン昇順の順位 $\text{rank}(l) = \text{popcount}(\text{mask} \mathbin{\&} ((1 \ll l) - 1))$、総数 $\text{count} = \text{popcount}(\text{mask})$ を求める（Metal では `simd_ballot` + `simd_prefix_exclusive_sum`）。
  - 共有配列 `uint16_t words[]` の書き込み位置 `wp` に対し、`words[wp + rank(l)] = uint16_t(x_l & 0xFFFF)` を書き込み、`wp += count` と進める。
  - もし `wp * 2 + 128 >= N`（オーバーヘッド込みで元サイズ以上）となった時点で、そのタイルは即座に `TILE_MODE_RAW` へフォールバックする。
- **デコード時（`step = 0` → `S-1`）**:
  - 共有配列 `words[0 .. total_words-1]` の末尾 `rp = total_words` から開始する。
  - 各ステップで遷移直後の `need_refill[l] = (active && x_l < 65536)` は、エンコード時の同ステップの `need_renorm[l]` と 32 レーンすべてで数学的に 100% 一致する。
  - `count = popcount(mask)` に対し、`rp < count` なら破損エラー。そうでなければ `rp -= count` とし、`need_refill[l]` のレーンが `words[rp + rank(l)]` を読み出す。
  - これにより、入力読み込み・出力書き込み・ビットストリーム読み書きの **3 つすべてが完全連続（コアレス）アクセス**となる。

### 2.4 Phase 1 バイナリコンテナレイアウト（プロトタイプ形式 `R32P`）

```text
[StreamHeader: 24 B]
  uint8_t  magic[4]         = "R32P"
  uint32_t version          = 1
  uint64_t original_size    // 元バイト長
  uint32_t block_size       = 262144 (256 KiB)
  uint32_t num_blocks       = ceil(original_size / block_size)

[BlockOffsetTable: 8 * num_blocks B]
  uint64_t block_offsets[num_blocks]   // 各スーパーブロックのファイル先頭からのオフセット

[SuperBlock × num_blocks]
  uint8_t  block_mode       // 0 = RAW, 1 = RLE, 2 = RANS32
  uint8_t  num_tiles        // 1..4
  uint16_t reserved         // 0
  uint32_t raw_size         // このスーパーブロックの元サイズ (<= 262144)

  if block_mode == RAW (0):
    uint8_t raw_bytes[raw_size]

  if block_mode == RLE (1):
    uint8_t rle_byte        // 1 バイトのみ (raw_size バイト分を復元)

  if block_mode == RANS32 (2):
    [CompactFreqTable]
      uint8_t  encoding_kind   // 0 = Sparse (n_nz < 160), 1 = Dense12 (384 B)
      uint8_t  n_nonzero_m1    // (非ゼロシンボル数 - 1), 1..255
      if encoding_kind == 0:
        { uint8_t sym; uint16_t freq; } × (n_nonzero_m1 + 1)   // packed 3 B/entry
      else:
        uint8_t dense12[384]   // 2 シンボル(12bit×2 = 24bit = 3B) × 128

    [TileDesc × num_tiles] (各 8 B)
      uint8_t  tile_mode       // 0 = RAW, 1 = RLE, 2 = RANS32
      uint8_t  rle_byte        // tile_mode == RLE のみ有効
      uint16_t reserved        // 0
      uint32_t payload_words   // RAW: バイト数(=tile_len), RLE: 0, RANS32: uint16_t ワード数

    [TilePayload × num_tiles]
      if tile_mode == RAW:    uint8_t  raw[tile_len]
      if tile_mode == RLE:    (0 バイト)
      if tile_mode == RANS32: uint32_t final_states[32] (128 B) + uint16_t words[payload_words]
```


---

## 3. Phase 2 詳細仕様: UMA 異種協調圧縮エンジン (`ASZ2`)

Phase 2（`src/engine/`）は、Phase 1 の 32-way Coalesced rANS をバックエンドに据え、ブロック単位の適応前処理・分離ストリーム LZ・非同期 UMA パイプラインを統合したコアエンジンである。

### 3.1 3 スロット非同期 UMA パイプラインと Zero-Cold-Start Hybrid

圧縮時は **8 MiB（256 KiB × 32 ブロック）を 1 スロット** とし、3 スロット（`N_SLOTS = 3`）を完全に重畳実行する。

```text
Slot N+2 (CPU 全コア):  [1-pass L2 融合: HW CRC32C + 16 KiB LUT Probe + NEON 前処理 + エポック LZ]
Slot N+1 (GPU / CPU):  [32-way Coalesced rANS バッチ符号化 (GPU 非同期 OR CPU 4-way ILP ブランチレス)]
Slot N   (Write Queue):[厳密順序保証・適応並列ゼロ中間コピーパッキング + ストリーミング書出 + GF(2) CRC 結合]
```

- **Zero-Cold-Start Hybrid ポリシー**:
  - 入力 `< 384 MiB`（`HYBRID_GPU_MIN_BYTES` 未満）かつ Metal 未初期化時: Metal を起動せず、**CPU 全コア ブランチレス 32-way rANS**（`rans32_encode_tile_fast`、64-bit 1 乗算 4-way ILP）で 0 ms 起動遅延のまま処理する。`n_tiles > n_streams && n_streams < 64` の場合はタイル単位並列に自動切替する。
  - 入力 `>= 384 MiB` 時: 先頭スロットを CPU rANS で即座に処理開始しつつバックグラウンドスレッド（`warmup_gpu_encode_async()`、`QOS_CLASS_USER_INITIATED`）で非同期 Metal 初期化を進め、完了したスロットから自動的に Metal GPU バッチ（`hx_rans_batch_build` + `hx_rans_batch_encode`）へ無停止切替する。
  - GPU エンコード／デコードとも、バッチごとの Metal コマンド投入を `@autoreleasepool` で囲み、GB 級ストリーミングでも Objective-C 一時オブジェクトが蓄積しない構造としている。
- **ゼロコピー ARM64 NEON 並列デコード & 2 スロット非同期 GPU デコード (`DEC_SLOTS = 2`, `OUT_WR_SLOTS = 3`)**:
  - 単発 CLI 復号では、32 レーンを NEON レジスタ群 `uint32x4_t × 8` に常駐させたまま `vqtbl1q_u8` による全レーン並列リフィルを行うゼロコピー 32-way NEON 復号（`rans32_decode_tile_neon_bytes`）＋スレッドローカル 16 KB L1D 常駐 LUT 融合構築（NEON 4 要素並列 `rans32_read_freq_table_to_lut`）＋3 スロット非同期ファイル書出（`OUT_WR_SLOTS = 3`）により、Metal 起動遅延・L2 キャッシュスラッシング・中間コピーをゼロ化する。
  - 常駐 GPU デコード有効時は、Slot `N+1` の GPU 32-way rANS 復号（`hx_rans_batch_decode`）を非同期投入しながら、CPU 全コアで Slot `N` の LZ 展開 + NEON 逆前処理 + HW CRC32C 検証を実行し、CPU/GPU を完全並列化する。

### 3.2 適応前処理（OpenZL 型軽量 DAG）とプローブ

各 256 KiB ブロックの先頭最大 16 KiB を `n * log2(n)` 事前計算 LUT で高速評価し、前処理フィルタ `xform_id` とコーデック `codec_id` を選択する。

| `xform_id` | 名称 | NEON 実装 / 対象データ |
|---|---|---|
| `0` (`HX_XFORM_NONE`) | 無変換 | テキスト・ソースコード・一般バイナリ |
| `1` (`HX_XFORM_SHUFFLE2`) | 2-byte プレーン分離 | `vld2q_u8` / `vst2q_u8` — FP16 / BF16 重み・16-bit 音声 |
| `2` (`HX_XFORM_SHUFFLE4`) | 4-byte プレーン分離 | `vld4q_u8` / `vst4q_u8` — FP32 テレメトリ・Int32 配列 |
| `3` (`HX_XFORM_ARM64_BCJ`) | ARM64 `BL` 相対→絶対変換 | NEON 4 命令並列（`vld1q_u32` / `vbslq_u32`）— ARM64 Mach-O / ELF 機械語（`0x94000000` 系 `BL` 命令） |

| `codec_id` | 名称 | 内容 |
|---|---|---|
| `0` (`HX_CODEC_RAW`) | 生格納 | エントロピーゲート通過（`> 7.80 bpb` かつ前処理効果なし）または圧縮後膨張時のフォールバック |
| `1` (`HX_CODEC_RLE`) | 単一バイト RLE | ブロック全体が同一バイト `rle_byte` のときペイロード 0 B |
| `2` (`HX_CODEC_RANS_ONLY`) | プレーン別直 rANS | `SHUFFLE2` は 2 プレーン、`SHUFFLE4` は 4 プレーン、その他は 1 プレーンに分割。各プレーンのエントロピーが `> 7.80 bpb` なら rANS をスキップし直接 RAW 格納（FP16 仮数部など） |
| `3` (`HX_CODEC_LZ_RANS`) | 分離ストリーム LZ + rANS | 256 KiB ウィンドウ LZ（エポック・ハッシュテーブル）+ 4 分離 rANS ストリーム + 64-bit `ExtraBits` |

### 3.3 分離ストリーム LZ 仕様 (`LZ_RANS`)

- **最小マッチ長**: `HX_MIN_MATCH = 4`
- **圧縮レベル**:
  - **Level 1 (`-1`)**: 高速 1-probe ハッシュ（`14-bit` テーブル＋6-bit タグ = 実効 20-bit フィルタ）、`anchor` 先行ハッシュ＋ループ外 `rep0` 高速パス、高速スキップ。
  - **Level 3 (`-3`, default)**: 64-bit 1-load / 1-store 2-way バケット（`15-bit × 2`＋6-bit タグ = 実効 21-bit フィルタ）、末尾バイト先行フィルタ付き 1-step lazy match（閾値 32）、`rep0/rep1/rep2` 最適化、後方延長（backward extend）。
  - **Level 6 (`-6`)**: 4-way タグ付きエポックバケット（`15-bit × 4`＋6-bit タグ = 実効 21-bit フィルタ）+ 2-step lazy match による高圧縮モード。
- **タグ付きエポック・ハッシュテーブル**: `thread_local HxLzThreadState` を遅延確保し、各エントリに `[epoch:8 | hash_tag:6 | pos:18]` をパック。ブロックごとにエポックを進めることで 256 KiB ブロック毎の `memset` を完全に排除しつつ、ハッシュ衝突による不要なランダムメモリアクセスの約 98.4% を `XOR+CMP` 1 発で事前棄却する。
- **高速 LZ デコーダ (`hx_lz_decode_block`)**: 結合デコードテーブル（`LL_DEC` / `ML_DEC` / `OF_DEC`）、無条件 64-bit ブランチレス `ExtraBits` ロード、`ll <= 16` / `ml <= 16 && dist >= 16` の 16 バイト固定 `wildcopy` 高速パス、および `dist == 1/2/4` の NEON `vdupq` RLE 展開を実装。
- **シンボル空間**:
  - `Literals` (8-bit 生リテラル列)
  - `LL` (Literal Length コード: `0..36` の 37 値。`0..15` は追加ビットなし、`16..36` は `1..17` extra bits で `0..262,187` を完全表現)
  - `ML` (Match Length - 4 コード: `LL` と同一の 37 コード表)
  - `OF` (Offset コード: `0..20` の 21 値。`0 = rep0`, `1 = rep1`, `2 = rep2`, `3..20` は距離 $[2^k, 2^{k+1}-1]$ かつ $k = 0..17$ extra bits)
  - `ExtraBits`: シーケンス順に `LL extra` → `ML extra` → `OF extra` を 64-bit リトルエンディアン・ビットストリームへブランチレス 1-store パック。

### 3.4 Phase 2 バイナリコンテナレイアウト (`ASZ2`)

```text
[HxArchiveHeader: 32 B]
  uint8_t  magic[4]         = "ASZ2"
  uint32_t version          = 1
  uint64_t original_size    // 元データ総バイト数
  uint32_t block_size       = 262144 (256 KiB)
  uint32_t num_blocks       = ceil(original_size / block_size)
  uint32_t level            // 1 / 3 / 6
  uint32_t reserved         = 0

[BlockOffsetTable: 8 * num_blocks B]
  uint64_t block_offsets[num_blocks]   // 各ブロックのアーカイブ先頭からのバイトオフセット

[Block × num_blocks]
  [HxBlockHeader: 16 B]
    uint8_t  xform_id       // HX_XFORM_* (0..3)
    uint8_t  codec_id       // HX_CODEC_* (0..3)
    uint8_t  rle_byte       // CODEC_RLE 時のみ有効
    uint8_t  reserved       = 0
    uint32_t raw_size       // 元ブロックサイズ (<= 262144)
    uint32_t crc32c         // 元ブロックデータの Castagnoli CRC32C
    uint32_t comp_payload   // 直後に続くペイロードのバイト数

  [BlockPayload: comp_payload B]
    if codec_id == HX_CODEC_RAW (0):
      uint8_t raw_bytes[raw_size]

    if codec_id == HX_CODEC_RLE (1):
      (0 バイト)

    if codec_id == HX_CODEC_RANS_ONLY (2):
      [SubStream × num_planes]        // SHUFFLE2=2, SHUFFLE4=4, その他=1

    if codec_id == HX_CODEC_LZ_RANS (3):
      [HxLzBlockMeta: 16 B]
        uint32_t num_seqs
        uint32_t lit_size
        uint32_t tail_lit_size        // 最終マッチ後のリテラル数 (lit_size に含まれる)
        uint32_t extra_bytes          // ExtraBits バイト長
      [SubStream × 4]                 // 0: Literals, 1: LL, 2: ML, 3: OF
      uint8_t extra_bits[extra_bytes]

[SubStream フォーマット]
  uint8_t  mode             // 0 = HX_SUB_RAW, 1 = HX_SUB_RLE, 2 = HX_SUB_RANS
  uint8_t  rle_byte         // mode == RLE 時のみ有効
  uint8_t  num_tiles        // 0..4 (0 は空ストリーム)
  uint8_t  reserved         = 0
  if mode == HX_SUB_RAW && num_tiles > 0:
    uint8_t raw[stream_len]
  if mode == HX_SUB_RLE:
    (0 バイト)
  if mode == HX_SUB_RANS:
    [CompactFreqTable]      // encoding_kind: 0=Sparse (n_nz < 128, 2 + n_nz*3 B), 1=Dense12 (2 + 384 B)
    [Rans32TileDiskDesc × num_tiles] (各 8 B)
    [TilePayload × num_tiles]
```

### 3.5 O(1) ブロック範囲展開 (`decompress_range`) と $\text{GF}(2)$ CRC32C 結合

- `BlockOffsetTable` により、任意のバイト範囲 `[req_off, req_off + req_len)` を含む最小ブロック範囲 `[b_start, b_end]` のみを O(1) でシーク・復号・CRC32C 検証できる（`HeteroEngine::decompress_range`）。
- 全体 CRC32C が必要な場合は、事前計算した 41 個の $32 \times 32$ $\text{GF}(2)$ シフト行列（5 KB、最大 $2^{40} = 1\text{ TiB}$ 対応）により `hx_crc32c_combine(crc1, crc2, len2)` を約 10 ns/回で実行し、全ブロック CRC32C から単一ストリーム全体とビット厳密一致する CRC32C を再スキャンなしで合成する。

---

## 4. Phase 3 詳細仕様: `.aplz` v4 ネイティブコンテナ & アーカイバ

Phase 3（`src/archive/`）は、一時 `.tar` ファイルを完全に廃止し、ディレクトリツリー・シンボリックリンク・POSIX 権限・更新時刻を単一 `.aplz` コンテナへ直接格納するネイティブアーカイバおよび `aplz` CLI である。

### 4.1 格納戦略: 型別 Solid Grouping + 並列正規化 FastCDC

| ファイル規模 | 格納方式 | パック上限 | 狙い |
|---|---|---|---|
| **小ファイル (`< 64 KiB`)** | **5 クラス型別 Solid Grouping** | `4 MiB` (`ASZ_SOLID_PACK_MAX`) | 類似コンテンツを連続結合してクロスファイル LZ 圧縮率を劇的に向上させつつ、部分展開時の復号上限を 4 MiB に抑える |
| **大ファイル (`>= 64 KiB`)** | **並列正規化 FastCDC 重複排除 + 連続スパン直圧縮** | `128 MiB` (`ASZ_LARGE_PACK_MAX`) / `256 MiB` (`ASZ_STREAM_PACK_MAX`, `--no-dedup`) | 同一・類似チャンクを 1 回のみ格納。重複のない連続ユニークチャンク列はコピーせず `mmap` 領域から直接 `ASZ2` パックへストリーミング圧縮（`--no-dedup` 時は FastCDC を省略し 256 MiB 単位で直圧縮） |

#### 5 クラス型分類（`asz_classify_file`）
先頭マジックバイト → 拡張子 → 先頭 256 B の制御文字比率の順で高速分類する。
- `0` (`TEXT`): ソースコード・JSON・Markdown・設定ファイル・テキスト
- `1` (`CODE`): ARM64/x86 Mach-O・ELF・`.dylib` / `.so` / `.o` / `.metallib`
- `2` (`NUM`): FP16/BF16/FP32 重み・`.safetensors` / `.gguf` / `.npy` / `.wav` / `.parquet`
- `3` (`MEDIA`): 既圧縮メディア・アーカイブ（PNG / JPEG / MP4 / ZIP / GZ / ZST / PDF 等）
- `4` (`BIN`): その他一般バイナリ

#### 正規化 FastCDC（`aplz_fastcdc.h`）
- **パラメータ**: `min = 16 KiB`、`normal = 64 KiB`、`max = 256 KiB`（Phase 2 ブロックサイズに一致）
- **Gear Hash マスク**:
  - `[16 KiB, 64 KiB)`: `ASZ_CDC_MASK_S = 0x0007590703530000ULL`（厳密 16 bit）
  - `[64 KiB, 256 KiB)`: `ASZ_CDC_MASK_L = 0x0003590303530000ULL`（厳密 14 bit）
  - 32-bit 一括ロードと並列加算ツリー（`fp4 = (fp << 4) + (((g0 << 1) + g1) << 2) + ((g2 << 1) + g3)`）により、1 バイト粒度カット判定と 100% ビット同一のカット位置をループ依存長 1 命令で高速算出。
- **マルチコア並列化 & APFS コールド読込最適化**:
  - `>= 16 MiB` の巨大ファイルは 8 MiB スーパースライス毎に並列 `F_RDADVISE` 先読み＋並列 FastCDC を実行（複数巨大ファイル 16〜128 MiB のスライスも 1 回の `dispatch_apply` へ統合）。
  - `64 KiB .. 16 MiB` の中ファイル群は 32 MiB 単位の 8 ワーカー並列 `read()` + 並列 FastCDC バッチで処理し、`< 64 KiB` 小ファイル群は並列 `fstatat` 走査＋ディレクトリ単位 `openat(dirfd)` 1 パス読込を行う。
  - 読込・FastCDC と圧縮・書出は 3 スロット非同期パックキュー（`pack_q`）＋再利用バッファプール（`PackBufPool`）＋`F_ALLOCATEALL` / `F_NOCACHE`（512 MiB 以上）で完全重畳する。
- **160-bit 強コンテンツ指紋**: 1 パス 32 B ロードで `wyhash128`（`hash_lo`, `hash_hi`）と 3-way 並列 ARM64 ハードウェア `__crc32cd`（`crc32c`）を同時計算し、`(hash_lo, hash_hi, raw_size, crc32c)` で重複判定する。

### 4.2 `.aplz` v4 バイナリコンテナフォーマット

```text
[AszFileHeader: 40 B] (オフセット 0 固定)
  uint8_t  magic[4]             = { 'A', 'P', 'L', 'Z' }  // 旧 v3 (version=3) とは version=4 で明確分離
  uint32_t version              = 4
  uint32_t flags                // bit0: ASZ_FLAG_DEDUP, bit1: ASZ_FLAG_SOLID
  uint32_t level                // 1 / 3 / 6
  uint64_t directory_offset     // セントラルディレクトリ開始位置
  uint64_t directory_comp_size  // 圧縮ディレクトリサイズ (ASZ2 ペイロード)
  uint32_t directory_raw_size   // 展開後ディレクトリサイズ
  uint32_t header_crc32c        // 先頭 36 バイトの CRC32C

[Data Packs: 0 .. num_packs-1]
  // 各パックは独立した Phase 2 "ASZ2" 圧縮ストリーム
  // pack_crc32c = CRC32C(ASZ2 Header + BlockOffsetTable) XOR CRC32C(ASZ2 Payload)

[Compressed Central Directory: directory_offset 位置]
  // 単一の Phase 2 "ASZ2" 圧縮ペイロード。展開後レイアウト:
  [AszDirHeader: 32 B]
    uint32_t num_packs
    uint32_t num_chunks
    uint32_t num_entries
    uint32_t num_extents
    uint64_t string_pool_size
    uint64_t total_raw_bytes

  [AszPackDesc × num_packs] (各 32 B)
    uint64_t file_offset
    uint64_t comp_size
    uint32_t raw_size           // <= 128 MiB (FastCDC) / <= 256 MiB (--no-dedup)
    uint32_t pack_crc32c
    uint8_t  solid_class        // ASZ_CLASS_*
    uint8_t  is_solid           // 1=Solid Pack, 0=Large FastCDC Pack
    uint16_t reserved
    uint32_t reserved2

  [AszChunkDesc × num_chunks] (各 32 B)
    uint32_t pack_id
    uint32_t offset_in_pack
    uint32_t raw_size
    uint32_t crc32c
    uint64_t hash_lo
    uint64_t hash_hi

  [AszEntryDesc × num_entries] (各 48 B)
    uint64_t raw_size           // DIR / SYMLINK は 0
    uint64_t mtime_ns           // UNIX epoch ナノ秒
    uint32_t path_off           // StringPool 内オフセット
    uint32_t path_len
    uint32_t target_off         // SYMLINK 先オフセット
    uint32_t target_len
    uint32_t first_extent
    uint32_t num_extents
    uint32_t file_crc32c        // ファイル全体の厳密 CRC32C (GF(2) 合成可)
    uint16_t mode               // POSIX 下位 12 bit (07777)
    uint8_t  entry_type         // 0=FILE, 1=DIR, 2=SYMLINK
    uint8_t  solid_class        // ASZ_CLASS_*

  [AszExtentDesc × num_extents] (各 16 B)
    uint32_t chunk_id
    uint32_t offset_in_chunk
    uint32_t length
    uint32_t reserved

  [StringPool: string_pool_size B]
    // UTF-8 相対パスおよびシンボリックリンクターゲット列 (非 NUL 終端、オフセット+長さ参照)
```

### 4.3 高速読み取り・部分展開・整合性・セキュリティ仕様

1. **O(1) 一覧表示 (`aplz l`)**: `AszFileHeader` (40 B) と末尾の `Central Directory` のみを読んで表示するため、GB 級アーカイブでも数ミリ秒で完了する。
2. **256 KiB ブロック単位 O(1) 部分展開 (`aplz x archive.aplz -C out <path>`)**:
   - 指定ファイルが参照する Extent の `(pack_id, offset_in_pack, length)` のみを抽出し、`HeteroEngine::decompress_range` で該当 256 KiB ブロックだけを選択復号する（例: 128 MiB パック内の 128 KiB ファイルでも 1 ブロック = 256 KiB のみ復号、0.3 ms 未満）。
3. **パック順序最適化・並列ファイル書出・ゼロバッファ直展開**:
   - 全展開 (`extract`) および検証 (`verify`) では、対象ファイルを `(first_pack_id, offset_in_pack)` 順にソートし、6 スロット Pack LRU キャッシュ＋`F_RDADVISE` 先読みを併用する。同一 Pack 内の複数ファイル（Solid Pack 等）は APFS ディレクトリ vnode ロック競合を回避する 4 ワーカー並列ワークスティーリングで書出を行い、`umask` 事前キャッシュによる不要 `fchmod` 省略と同一パック内連続エクステントのゼロコピー `fwrite` 結合も実施する。
   - 非共有の連続大パックでは、`extract` / `extract_single_file` 双方で APFS `F_PREALLOCATE` + 3 スロット非同期ストリーミング直書出を行い、`verify` でも出力バッファ確保ゼロのストリーミング検証を実行する。
4. **3 重破損検知 (`aplz t` / 展開時自動検証)**:
   - Layer 1: `AszFileHeader.header_crc32c` + `Central Directory` の `ASZ2` ブロック CRC32C
   - Layer 2: 各パックの `AszPackDesc.pack_crc32c`（圧縮ストリーム全体の破損検知）+ `HxBlockHeader.crc32c` + rANS 32 レーン終端状態厳密検証
   - Layer 3: 各ファイルの `AszEntryDesc.file_crc32c`（全展開・部分展開双方で検証）
5. **Zip-Slip / シンボリックリンク逸脱防御**:
   - `asz_is_safe_rel_path`: 空文字・先頭/末尾 `/`・連続 `//`・`'\0'`・`'\'`・`.`・`..` を全面拒否。
   - `asz_mkdir_rel_safe`: 出力ベースディレクトリ配下のディレクトリ生成で `lstat()` を用い、アーカイブ内のシンボリックリンクを経由したディレクトリ外への書き込みを構造的に遮断する。

---

## 5. Apple M4 実測ベンチマークサマリ

測定環境: Apple M4 (10-core CPU / 10-core GPU, UMA, macOS, Clang++ `-O3 -flto -march=armv8.5-a+crc`, Metal 3.0)。全ケースで元データとのバイト完全一致を検証済み。

### 5.1 1.00 GiB 異種混合単一ファイル（FP32 + FP16 + Mach-O + Mixed、エンドツーエンド CLI）

| ツール | 圧縮後サイズ (圧縮率) | 圧縮速度 (時間) | 解凍 `d` (時間) | 展開 `x` (時間) | 整合性検査 (`t`) | 一覧 (`l`) |
|---|---:|---:|---:|---:|---:|---:|
| **`aplz -1` (fast)** | **391.6 MiB (38.25%)** | **3,157 MB/s (0.32 s)** | **3,654 MB/s (0.28 s)** | **3,679 MB/s (0.28 s)** | **0.12 s** (8,384 MB/s) | **4.0 ms** |
| **`aplz -3` (default)** | **384.7 MiB (37.57%)** | **3,698 MB/s (0.28 s)** | **3,729 MB/s (0.27 s)** | **3,480 MB/s (0.29 s)** | **0.11 s** (9,128 MB/s) | **3.7 ms** |
| `aplz -1 --no-dedup` | 594.9 MiB (58.10%) | 2,972 MB/s (0.34 s) | 3,212 MB/s (0.32 s) | 2,894 MB/s (0.35 s) | 0.13 s (7,926 MB/s) | 3.7 ms |
| `zstd -3 -T0` | 678.0 MiB (66.21%) | 1,989 MB/s (0.51 s) | 1,259 MB/s (0.81 s) | — | — | — |

### 5.2 979 ファイル実ディレクトリ（`ml_project`, 89.4 MiB、エンドツーエンド CLI）

| ツール | アーカイブサイズ (比率) | 作成 (Pack) | 全展開 (Full) | 一覧 (List) | 1 ファイル部分展開 |
|---|---:|---:|---:|---:|---:|
| **`aplz -1`** | **12.40 MiB (13.9%)** | **30.2 ms** | **62.9 ms** | **3.2 ms** | **3.22 ms** |
| **`aplz -3`** | **12.34 MiB (13.8%)** | **32.3 ms** | **62.1 ms** | **3.5 ms** | **3.34 ms** |
| `tar + zstd -1 -T0` | 33.08 MiB (37.0%) | 337.9 ms | 343.8 ms | 46.7 ms | 46.32 ms |
| `tar + zstd -3 -T0` | 32.39 MiB (36.2%) | 354.4 ms | 352.3 ms | 57.9 ms | 57.52 ms |
| `tar + lz4 -1` | 47.97 MiB (53.7%) | 365.5 ms | 363.7 ms | 42.7 ms | 41.46 ms |

### 5.3 Phase 2 コア（64 MiB インメモリ、定常スループット）

| データセット (64 MiB) | `aplz` L1 圧縮率 / 圧縮 / 解凍 | `aplz` L3 圧縮率 / 圧縮 / 解凍 | `aplz` L6 圧縮率 / 圧縮 / 解凍 |
|---|---:|---:|---:|
| `Source+Docs` | 26.3% / 3,281 / 4,154 MB/s | 23.9% / 2,229 / 4,841 MB/s | 21.8% / 1,127 / 5,108 MB/s |
| `Mach-O ARM64` | 43.8% / 2,729 / 5,139 MB/s | 41.5% / 1,703 / 4,093 MB/s | 39.9% / 852 / 3,861 MB/s |
| `FP16 Weights` | 84.1% / **6,699 / 6,617 MB/s** | 84.1% / **7,462 / 7,529 MB/s** | 84.1% / **8,612 / 7,837 MB/s** |
| `FP32 Telemetry` | 41.0% / 5,330 / 8,755 MB/s | 38.3% / 3,233 / **12,596 MB/s** | 36.8% / 1,385 / **11,962 MB/s** |
| `Mixed Real` | 63.4% / 4,589 / 7,810 MB/s | 62.2% / 3,223 / 7,634 MB/s | 61.3% / 1,897 / 4,633 MB/s |
| `Random` | 100.0% / **21,801 / 32,832 MB/s** | 100.0% / **24,008 / 31,359 MB/s** | 100.0% / **24,602 / 31,340 MB/s** |
