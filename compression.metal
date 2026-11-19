// compression.metal — APLZ GPU Kernels: LZ77 + Per-Chunk tANS Encode/Decode
//
// カーネル構成:
//   compress_chunk    — LZ77 マッチング (4-gram ハッシュ) + parallel prefix-sum コンパクション
//   build_tans_tables — GPU per-chunk ヒストグラム・正規化・spread/エンコードテーブル構築
//   tans_encode       — 128 インターリーブ ANS ストリーム並列エンコード (per-chunk table)
//   tans_decode       — 128 インターリーブ ANS ストリーム並列デコード (per-chunk table)
//   lz77_decode       — LZ77 展開 (完全並列化: タイル prefix-scan + モジュロコピー)

#include <metal_stdlib>
using namespace metal;

// ─── LZ77 定数 ─────────────────────────────────────────────────────────────────
constant uint CHUNK_SIZE = 65536u;
constant uint HASH_BITS  = 11u;
constant uint HASH_SIZE  = 1u << HASH_BITS;
constant uint MAX_MATCH  = 255u;
constant uint MIN_MATCH  = 3u;

// ─── tANS 定数 ─────────────────────────────────────────────────────────────────
constant uint ANS_LOG_L  = 10u;
constant uint ANS_L      = 1u << ANS_LOG_L;
constant uint N_SYMBOLS  = 512u;
constant uint N_STREAMS  = 128u;
constant uint BS_CAP     = 1024u;

// ─── 構造体 ──────────────────────────────────────────────────────────────────────
struct LzToken {
    uint8_t  is_match;
    uint8_t  _pad0;
    uint16_t val;
    uint16_t dist;
    uint16_t _pad1;
};

struct SymInfo {
    uint16_t freq;
    uint16_t cum_freq;
};

struct DecodeEntry {
    uint16_t symbol;
    uint16_t num_bits;
    uint16_t new_base;
    uint16_t _pad;
};

// ─── 4-gram 乗算ハッシュ（衝突率低減）────────────────────────────────────────────
inline uint h4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return ((uint(a) << 24) | (uint(b) << 16) | (uint(c) << 8) | uint(d)) * 2654435761u
           >> (32u - HASH_BITS);
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 1: compress_chunk — LZ77 (変更なし)
// ════════════════════════════════════════════════════════════════════════════════
kernel void compress_chunk(
    device const uint8_t*  in_data     [[ buffer(0) ]],
    device       LzToken*  out_sparse  [[ buffer(1) ]],
    device       LzToken*  out_compact [[ buffer(2) ]],
    device       uint32_t* compact_cnt [[ buffer(3) ]],
    constant     uint32_t& total_bytes [[ buffer(4) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    threadgroup atomic_uint ht_a[HASH_SIZE];
    threadgroup atomic_uint ht_b[HASH_SIZE];

    const uint base = gid * CHUNK_SIZE;
    if (base >= total_bytes) return;
    const uint clen = min(CHUNK_SIZE, total_bytes - base);

    uint chunk_off = gid * CHUNK_SIZE;
    device LzToken* sparse  = out_sparse  + chunk_off;
    device LzToken* compact = out_compact + chunk_off;

    for (uint i = tid; i < HASH_SIZE; i += tg_size) {
        atomic_store_explicit(&ht_a[i], 0xFFFFFFFFu, memory_order_relaxed);
        atomic_store_explicit(&ht_b[i], 0xFFFFFFFFu, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i + 3u < clen; i += tg_size) {
        // Run-Skipping: 同一文字が4バイト以上連続するランの途中はハッシュ更新をスキップ。
        // ランの先頭のみ更新することで、同一アドレスへの atomic 集中を回避。
        if (i > 0u &&
            in_data[base+i]   == in_data[base+i-1u] &&
            in_data[base+i+1u] == in_data[base+i-1u] &&
            in_data[base+i+2u] == in_data[base+i-1u])
            continue;
        uint slot  = h4(in_data[base+i], in_data[base+i+1], in_data[base+i+2], in_data[base+i+3]);
        uint old_a = atomic_fetch_min_explicit(&ht_a[slot], i, memory_order_relaxed);
        if (old_a != 0xFFFFFFFFu && old_a > i)
            atomic_fetch_min_explicit(&ht_b[slot], old_a, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < clen; i += tg_size) {
        LzToken t;
        t.is_match = 0; t._pad0 = 0; t._pad1 = 0;
        t.val = in_data[base + i]; t.dist = 0;

        if (i + 3u < clen) {
            uint slot = h4(in_data[base+i], in_data[base+i+1], in_data[base+i+2], in_data[base+i+3]);
            uint ca = atomic_load_explicit(&ht_a[slot], memory_order_relaxed);
            uint cb = atomic_load_explicit(&ht_b[slot], memory_order_relaxed);

            uint best = 0xFFFFFFFFu;
            for (uint k = 0u; k < 2u; ++k) {
                uint c = (k == 0u) ? ca : cb;
                if (c != 0xFFFFFFFFu && c < i && (i - c) <= 65534u)
                    best = (best == 0xFFFFFFFFu) ? c : max(best, c);
            }
            if (best != 0xFFFFFFFFu) {
                uint maxl = min(MAX_MATCH, clen - i), mlen = 0u;
                // Fast 4-byte comparison loop
                while (mlen + 4u <= maxl) {
                    uint4 a = uint4(
                        in_data[base+best+mlen], in_data[base+best+mlen+1u],
                        in_data[base+best+mlen+2u], in_data[base+best+mlen+3u]
                    );
                    uint4 b = uint4(
                        in_data[base+i+mlen], in_data[base+i+mlen+1u],
                        in_data[base+i+mlen+2u], in_data[base+i+mlen+3u]
                    );
                    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) break;
                    mlen += 4u;
                }
                // Handle remaining bytes
                while (mlen < maxl &&
                       in_data[base+best+mlen] == in_data[base+i+mlen]) ++mlen;
                if (mlen >= MIN_MATCH) {
                    t.is_match = 1;
                    t.val      = uint16_t(mlen);
                    t.dist     = uint16_t(i - best);
                }
            }
        }
        sparse[i] = t;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── コンパクション並列化 ──────────────────────────────────────────────
    // Step 1 (tid==0): sparse を走査し、各スレッドの担当開始位置と書き込みオフセットを計算。
    // Step 2 (全スレッド): sparse → compact に直接書き込み (Step 3/4 不要)。
    threadgroup uint tg_starts[128];    // 各 tid の sparse 走査開始インデックス
    threadgroup uint tg_offsets[129];   // 各 tid の compact 書き込みオフセット (exclusive prefix-sum)
    threadgroup uint tg_counts[128];

    // Step 1a: tid==0 が sparse を走査して担当開始位置を決定
    if (tid == 0u) {
        uint total_tok = 0u;
        uint i = 0u;
        while (i < clen) {
            total_tok++;
            i += sparse[i].is_match ? uint(sparse[i].val) : 1u;
        }
        uint toks_per = (total_tok + tg_size - 1u) / tg_size;

        uint tok_count = 0u;
        uint cur_tid = 0u;
        tg_starts[0u] = 0u;
        i = 0u;
        while (i < clen) {
            tok_count++;
            uint step = sparse[i].is_match ? uint(sparse[i].val) : 1u;
            i += step;
            if (tok_count % toks_per == 0u && cur_tid + 1u < tg_size) {
                cur_tid++;
                tg_starts[cur_tid] = i;
                tg_offsets[cur_tid] = tok_count;
            }
        }
        for (uint t = cur_tid + 1u; t < tg_size; t++) {
            tg_starts[t] = clen;
        }
        tg_counts[0] = total_tok;
        compact_cnt[gid] = total_tok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1b: 各スレッドの担当トークン数を数え、parallel prefix-sum で書き込みオフセットを作る
    {
        uint i = tg_starts[tid];
        uint my_end = (tid + 1u < tg_size) ? tg_starts[tid + 1u] : clen;
        uint local_cnt = 0u;
        while (i < my_end) {
            local_cnt++;
            i += sparse[i].is_match ? uint(sparse[i].val) : 1u;
        }
        tg_counts[tid] = local_cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1u; off < tg_size; off <<= 1u) {
        uint add = (tid >= off) ? tg_counts[tid - off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_counts[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tg_offsets[tid] = (tid == 0u) ? 0u : tg_counts[tid - 1u];
    if (tid == tg_size - 1u) {
        tg_offsets[tg_size] = tg_counts[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: 各スレッドが sparse → compact に直接書き込み。
    // 各スレッドの書き込み先 compact[tg_offsets[tid] + k] は互いに重ならないため安全。
    {
        uint i = tg_starts[tid];
        uint my_end = (tid + 1u < tg_size) ? tg_starts[tid + 1u] : clen;
        uint dst = tg_offsets[tid];
        while (i < my_end) {
            compact[dst++] = sparse[i];
            i += sparse[i].is_match ? uint(sparse[i].val) : 1u;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 2: tans_encode — Per-chunk tANS テーブルで並列エンコード
//
// buffer(2): sym_arr   — SymInfo[N_SYMBOLS] の配列 (チャンクごと)
// buffer(3): enc_arr   — uint16_t[ANS_L] の配列 (チャンクごと)
// 各スレッドグループが chunk_id で自分のテーブルを threadgroup メモリにロードする。
// ════════════════════════════════════════════════════════════════════════════════
kernel void tans_encode(
    device const LzToken*   tokens     [[ buffer(0) ]],
    device const uint32_t*  token_cnt  [[ buffer(1) ]],
    device const SymInfo*   sym_arr    [[ buffer(2) ]],
    device const uint16_t*  enc_arr    [[ buffer(3) ]],
    device       uint8_t*   bs_out     [[ buffer(4) ]],
    device       uint32_t*  bs_sizes   [[ buffer(5) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    // ── Per-chunk テーブルを threadgroup メモリに協調ロード ────────────────
    threadgroup SymInfo   tg_sym[N_SYMBOLS];  // 512 * 4 = 2048 bytes
    threadgroup uint16_t  tg_enc[ANS_L];      // 1024 * 2 = 2048 bytes

    device const SymInfo*  my_sym = sym_arr + gid * N_SYMBOLS;
    device const uint16_t* my_enc = enc_arr + gid * ANS_L;

    for (uint i = tid; i < N_SYMBOLS; i += tg_size)
        tg_sym[i] = my_sym[i];
    for (uint i = tid; i < ANS_L; i += tg_size)
        tg_enc[i] = my_enc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── エンコード ────────────────────────────────────────────────────────
    const uint n_tok = token_cnt[gid];
    device const LzToken* ct = tokens + gid * CHUNK_SIZE;
    uint state   = ANS_L;
    uint bit_buf = 0u;
    uint bit_cnt = 0u;

    device uint8_t* my_out = bs_out + ((gid * tg_size + tid) * BS_CAP);
    uint bp = 0u;

    for (uint i = tid; i < n_tok; i += tg_size) {
        LzToken tok = ct[i];
        uint sym = tok.is_match ? (256u + uint(tok.val)) : uint(tok.val);
        uint f   = uint(tg_sym[sym].freq);
        uint cum = uint(tg_sym[sym].cum_freq);
        if (f == 0u) continue;

        uint nb = 0u, tmp = state;
        while (tmp >= 2u * f) { tmp >>= 1u; nb++; }

        if (nb > 0u) {
            bit_buf |= ((state & ((1u << nb) - 1u)) << bit_cnt);
            bit_cnt += nb;
            while (bit_cnt >= 8u && bp < BS_CAP - 4u) {
                my_out[bp++] = uint8_t(bit_buf & 0xFFu);
                bit_buf >>= 8u; bit_cnt -= 8u;
            }
        }

        state = uint(tg_enc[cum + (state >> nb) - f]);

        if (tok.is_match) {
            bit_buf |= (uint(tok.dist) << bit_cnt);
            bit_cnt += 16u;
            while (bit_cnt >= 8u && bp < BS_CAP - 4u) {
                my_out[bp++] = uint8_t(bit_buf & 0xFFu);
                bit_buf >>= 8u; bit_cnt -= 8u;
            }
        }
    }

    // Sentinel + flush
    bit_buf |= (1u << bit_cnt);
    bit_cnt += 1u;

    while (bit_cnt > 0u && bp < BS_CAP - 2u) {
        my_out[bp++] = uint8_t(bit_buf & 0xFFu);
        bit_buf >>= 8u;
        bit_cnt = bit_cnt >= 8u ? bit_cnt - 8u : 0u;
    }

    if (bp + 2u <= BS_CAP) {
        my_out[bp++] = uint8_t(state & 0xFFu);
        my_out[bp++] = uint8_t((state >> 8u) & 0xFFu);
    }

    bs_sizes[gid * tg_size + tid] = bp;
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 3: tans_decode — Per-chunk tANS テーブルで並列デコード
//
// buffer(2): dec_arr — DecodeEntry[ANS_L] の配列 (チャンクごと)
// 各スレッドグループが chunk_id で自分のデコードテーブルを threadgroup にロード。
// ════════════════════════════════════════════════════════════════════════════════
kernel void tans_decode(
    device const uint8_t*      bs_in       [[ buffer(0) ]],
    device const uint32_t*     bs_sizes    [[ buffer(1) ]],
    device const DecodeEntry*  dec_arr     [[ buffer(2) ]],
    device       LzToken*      out_tokens  [[ buffer(3) ]],
    device const uint32_t*     token_cnt   [[ buffer(4) ]],
    device       atomic_uint*  ans_err     [[ buffer(5) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    // ── Per-chunk デコードテーブルを threadgroup メモリにロード ────────────
    threadgroup DecodeEntry tg_dec[ANS_L];  // 1024 * 8 = 8192 bytes

    device const DecodeEntry* my_dec = dec_arr + gid * ANS_L;
    for (uint i = tid; i < ANS_L; i += tg_size)
        tg_dec[i] = my_dec[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u)
        atomic_store_explicit(&ans_err[gid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── デコード ──────────────────────────────────────────────────────────
    const uint n_tok = token_cnt[gid];
    uint my_count = (tid < n_tok) ? ((n_tok - 1u - tid) / tg_size + 1u) : 0u;
    if (my_count == 0u) return;

    uint stream_idx = gid * tg_size + tid;
    uint bp = bs_sizes[stream_idx];
    device const uint8_t* my_in = bs_in + stream_idx * BS_CAP;
    if (bp < 2u || bp > BS_CAP) {
        atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
        return;
    }

    uint state = uint(my_in[bp - 2u]) | (uint(my_in[bp - 1u]) << 8u);
    uint data_bytes = bp - 2u;
    if (state < ANS_L || state >= 2u * ANS_L) {
        atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
        return;
    }

    int bit_pos = -1;
    if (data_bytes > 0u) {
        uint last_byte = uint(my_in[data_bytes - 1u]);
        uint sentinel_pos = (last_byte > 0u) ? (31u - clz(last_byte)) : 0u;
        bit_pos = int((data_bytes - 1u) * 8u + sentinel_pos) - 1;
    }

    device LzToken* chunk_out = out_tokens + gid * CHUNK_SIZE;

    for (int t = int(my_count) - 1; t >= 0; t--) {
        if (state < ANS_L || state >= 2u * ANS_L) {
            atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
            return;
        }
        uint idx = state - ANS_L;
        DecodeEntry e = tg_dec[idx];

        LzToken tok;
        tok._pad0 = 0; tok._pad1 = 0;

        if (e.symbol >= 256u) {
            tok.is_match = 1;
            tok.val = uint16_t(e.symbol - 256u);
            // 距離フィールド 16 ビットをバッチ取得。
            // エンコード: bit_buf |= (dist << bit_cnt) → dist LSB が小ビット位置に書かれる。
            // デコード逆走: bit_pos は最後に書かれた dist MSB 位置を指す。
            // [bit_pos-15 .. bit_pos] の 16 ビットを取り出すと raw16 = dist そのまま (反転不要)。
            if (bit_pos < 15) {
                atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
                return;
            }
            uint lo16 = uint(bit_pos) - 15u;    // dist の LSB 側ビット位置
            uint byte_lo = lo16 >> 3u;
            uint bit_lo  = lo16 & 7u;
            // 最大 3 バイトのウィンドウで 16 ビットを取り出す
            uint w = uint(my_in[byte_lo]);
            if (byte_lo + 1u < data_bytes) w |= uint(my_in[byte_lo + 1u]) << 8u;
            if (byte_lo + 2u < data_bytes) w |= uint(my_in[byte_lo + 2u]) << 16u;
            uint raw16 = (w >> bit_lo) & 0xFFFFu;
            tok.dist = uint16_t(raw16);
            bit_pos -= 16;
        } else {
            tok.is_match = 0;
            tok.val = uint16_t(e.symbol);
            tok.dist = 0;
        }

        uint nb = uint(e.num_bits);
        uint bits = 0u;
        if (nb > 0u) {
            // ANS num_bits は最大 ANS_LOG_L = 10 ビット。
            // エンコード: bit_buf |= (state_lower_bits << bit_cnt) → LSB が小ビット位置
            // [bit_pos-nb+1 .. bit_pos] を取り出すと raw_nb = state_lower_bits そのまま (反転不要)
            if (bit_pos < int(nb) - 1) {
                atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
                return;
            }
            uint lo_nb = uint(bit_pos) - nb + 1u;
            uint byte_nb = lo_nb >> 3u;
            uint bit_nb  = lo_nb & 7u;
            uint w2 = uint(my_in[byte_nb]);
            if (byte_nb + 1u < data_bytes) w2 |= uint(my_in[byte_nb + 1u]) << 8u;
            if (byte_nb + 2u < data_bytes) w2 |= uint(my_in[byte_nb + 2u]) << 16u;
            bits = (w2 >> bit_nb) & ((1u << nb) - 1u);
            bit_pos -= int(nb);
        }
        state = uint(e.new_base) + bits;

        chunk_out[tid + tg_size * uint(t)] = tok;
    }

    if (state != ANS_L || bit_pos != -1)
        atomic_store_explicit(&ans_err[gid], 1u, memory_order_relaxed);
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 4: lz77_decode — LZ77 展開 (完全並列化)
//
// 並列化手法:
//   Phase A (tid==0): バリデーション + リテラル書き込み (出力オフセット計算)
//   Phase B (全スレッド): マッチ並列コピー (モジュロ演算で重複マッチ対応)
//
// 旧コードのボトルネックを排除:
//   - tg_serial_mode の直列フォールバックを完全削除
//   - 重複マッチ (dist < len) の tid==0 シリアルコピーをモジュロ並列コピーに置換
// ════════════════════════════════════════════════════════════════════════════════
kernel void lz77_decode(
    device const LzToken*   tokens      [[ buffer(0) ]],
    device const uint32_t*  token_cnt   [[ buffer(1) ]],
    device       uint8_t*   out_data    [[ buffer(2) ]],
    device       atomic_uint* lz_err    [[ buffer(3) ]],
    constant     uint32_t&  total_bytes [[ buffer(4) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    if (tid == 0u)
        atomic_store_explicit(&lz_err[gid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = gid * CHUNK_SIZE;
    if (base >= total_bytes) return;
    const uint clen = min(CHUNK_SIZE, total_bytes - base);
    const uint n_tok = token_cnt[gid];

    device const LzToken* ct = tokens + gid * CHUNK_SIZE;
    device uint8_t* out = out_data + base;

    // ── Phase A: タイル単位 parallel prefix-scan でバリデーション + リテラル書き込み ──
    threadgroup uint tg_invalid;
    threadgroup uint tg_tile_scan[128];
    threadgroup uint tg_pos_base;
    if (tid == 0u)
        { tg_invalid = 0u; tg_pos_base = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile = 0u; tile < n_tok; tile += tg_size) {
        uint idx = tile + tid;
        uint step = 0u;
        bool is_match = false;
        uint val = 0u, dist = 0u;
        if (idx < n_tok) {
            LzToken tok = ct[idx];
            is_match = tok.is_match != 0;
            val = uint(tok.val);
            dist = uint(tok.dist);
            step = is_match ? val : 1u;
        }
        tg_tile_scan[tid] = step;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint off = 1u; off < tg_size; off <<= 1u) {
            uint add = (tid >= off) ? tg_tile_scan[tid - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tg_tile_scan[tid] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint pos_before = tg_pos_base + tg_tile_scan[tid] - step;
        if (idx < n_tok) {
            if (is_match) {
                uint len = val;
                if (len == 0u || dist == 0u || dist > pos_before || pos_before + len > clen) {
                    tg_invalid = 1u;
                }
            } else {
                if (pos_before >= clen) tg_invalid = 1u;
                else out[pos_before] = uint8_t(val);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (tid == tg_size - 1u) tg_pos_base += tg_tile_scan[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tg_invalid != 0u) break;
    }
    if (tid == 0u && tg_pos_base != clen)
        tg_invalid = 1u;
    if (tid == 0u && tg_invalid != 0u)
        atomic_store_explicit(&lz_err[gid], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (tg_invalid != 0u)
        return;

    // ── Phase B (全スレッド): マッチ並列コピー ──────────────────────────
    // 全トークンを順次走査し、マッチごとに全スレッドで並列コピー。
    // 重複マッチ (dist < len) はモジュロ演算: out[off+j] = out[off-dist + (j%dist)]
    // これにより自己参照チェーンを O(1) インデックス計算に還元。
    {
        uint pos = 0u;
        uint max_mw = 0u;
        for (uint i = 0u; i < n_tok; i++) {
            if (!ct[i].is_match) {
                pos += 1u;
                continue;
            }

            uint off  = pos;
            uint len  = uint(ct[i].val);
            uint dist = uint(ct[i].dist);

            // 前のマッチの書き込み範囲とソースが重なる場合は同期
            if (off - dist < max_mw) {
                threadgroup_barrier(mem_flags::mem_device);
            }

            // モジュロ演算で重複・非重複マッチを統一的に並列コピー
            for (uint j = tid; j < len; j += tg_size)
                out[off + j] = out[off - dist + (j % dist)];

            max_mw = max(max_mw, off + len);
            pos += len;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 5: build_tans_tables — GPU側でtANSテーブル構築（CPUオフロード用）
//
// 各チャンクのLzTokenからヒストグラム計算→正規化→spread table→エンコードテーブルまで
// GPU内で完結させ、CPU-GPU同期を削減。
//
// buffer(0): tokens      — LzToken[CHUNK_SIZE] × num_chunks
// buffer(1): token_cnt   — uint32_t[num_chunks]
// buffer(2): sym_arr     — SymInfo[N_SYMBOLS] × num_chunks (出力)
// buffer(3): enc_arr     — uint16_t[ANS_L] × num_chunks (出力)
// buffer(4): match_cnt   — uint32_t[num_chunks] (出力: チャンクごとの実マッチ数)
// ════════════════════════════════════════════════════════════════════════════════
kernel void build_tans_tables(
    device const LzToken*   tokens     [[ buffer(0) ]],
    device const uint32_t*  token_cnt  [[ buffer(1) ]],
    device       SymInfo*   sym_arr    [[ buffer(2) ]],
    device       uint16_t*  enc_arr    [[ buffer(3) ]],
    device       uint32_t*  match_cnt  [[ buffer(4) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    // ── Phase 1: ヒストグラム計算（threadgroupメモリで集計）────────────────
    threadgroup atomic_uint tg_hist[N_SYMBOLS];  // 512 * 4 = 2048 bytes

    // 初期化
    for (uint i = tid; i < N_SYMBOLS; i += tg_size)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ヒストグラム集計（各スレッドが担当範囲を処理）
    const uint n_tok = token_cnt[gid];
    device const LzToken* ct = tokens + gid * CHUNK_SIZE;

    for (uint i = tid; i < n_tok; i += tg_size) {
        LzToken tok = ct[i];
        uint sym = tok.is_match ? (256u + uint(tok.val)) : uint(tok.val);
        atomic_fetch_add_explicit(&tg_hist[sym], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: 正規化（total計算→割り当て調整→累積和）────────────────────
    // CPU側と同じロジックをtid==0でシリアル実行
    threadgroup uint32_t tg_hist_copy[N_SYMBOLS];
    threadgroup uint32_t tg_nf[N_SYMBOLS];
    threadgroup uint16_t tg_cum[N_SYMBOLS];

    // ヒストグラムをコピー
    for (uint i = tid; i < N_SYMBOLS; i += tg_size)
        tg_hist_copy[i] = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        // Total計算 & 実マッチ数記録 (シンボル 256..511 がマッチトークン)
        uint64_t total = 0u;
        uint32_t matches = 0u;
        for (uint i = 0u; i < N_SYMBOLS; i++) {
            total += tg_hist_copy[i];
            if (i >= 256u) matches += tg_hist_copy[i];
        }
        match_cnt[gid] = matches;

        // 正規化
        if (total == 0u) {
            for (uint i = 0u; i < N_SYMBOLS; i++) {
                tg_nf[i] = 0u;
                tg_cum[i] = 0u;
            }
            tg_nf[0u] = ANS_L;
        } else {
            uint32_t assigned = 0u;
            for (uint i = 0u; i < N_SYMBOLS; i++) {
                if (tg_hist_copy[i] == 0u) {
                    tg_nf[i] = 0u;
                } else {
                    tg_nf[i] = max(1u, (uint32_t)(tg_hist_copy[i] * ANS_L / total));
                }
                assigned += tg_nf[i];
            }

            // 最大頻度のシンボルを探す
            uint best = 0u;
            for (uint i = 1u; i < N_SYMBOLS; i++)
                if (tg_hist_copy[i] > tg_hist_copy[best]) best = i;

            // 差分調整
            int32_t diff = (int32_t)ANS_L - (int32_t)assigned;
            if ((int32_t)tg_nf[best] + diff >= 1) {
                tg_nf[best] += (uint32_t)diff;
                assigned = ANS_L;
            } else {
                while (assigned != ANS_L) {
                    for (uint i = 0u; i < N_SYMBOLS && assigned != ANS_L; i++) {
                        if (assigned < ANS_L && tg_nf[i] > 0u) { tg_nf[i]++; assigned++; }
                        if (assigned > ANS_L && tg_nf[i] > 1u) { tg_nf[i]--; assigned--; }
                    }
                }
            }

            // 累積和計算
            uint16_t cum = 0u;
            for (uint i = 0u; i < N_SYMBOLS; i++) {
                tg_cum[i] = cum;
                cum += (uint16_t)tg_nf[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: SymInfoをデバイスメモリに書き出し──────────────────────────────
    device SymInfo* my_sym = sym_arr + gid * N_SYMBOLS;
    for (uint i = tid; i < N_SYMBOLS; i += tg_size) {
        my_sym[i].freq = (uint16_t)tg_nf[i];
        my_sym[i].cum_freq = tg_cum[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Duda's spread table構築───────────────────────────────────────
    threadgroup uint16_t tg_spread[ANS_L];  // 1024 * 2 = 2048 bytes

    const uint step = (ANS_L >> 1u) + (ANS_L >> 3u) + 3u;
    const uint mask = ANS_L - 1u;

    // CPU側と同じstepを使った順次配置（tid==0でシリアル実行）
    if (tid == 0u) {
        uint pos = 0u;
        for (uint s = 0u; s < N_SYMBOLS; s++) {
            uint f = tg_nf[s];
            for (uint j = 0u; j < f; j++) {
                tg_spread[pos] = (uint16_t)s;
                pos = (pos + step) & mask;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: エンコードテーブル構築────────────────────────────────────────
    device uint16_t* my_enc = enc_arr + gid * ANS_L;
    threadgroup uint32_t tg_rank[N_SYMBOLS];

    if (tid == 0u) {
        for (uint i = 0u; i < N_SYMBOLS; i++)
            tg_rank[i] = 0u;
        for (uint p = 0u; p < ANS_L; p++) {
            uint s = tg_spread[p];
            uint j = tg_rank[s]++;
            uint cum = tg_cum[s];
            my_enc[cum + j] = (uint16_t)(ANS_L + p);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
