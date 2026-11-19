// rans32.metal — APLZ-X Phase 1: Apple Silicon GPU 32-way SIMD-group Coalesced rANS
//
// カーネル構成:
//   rans32_build_tables — 256 KiB スーパーブロック単位で並列ヒストグラム・正規化・16 KB LUT 構築
//   rans32_encode       — 1 TG (128 threads = 4 SIMD-groups) = 4 × 64 KiB タイル並列エンコード
//   rans32_decode       — 1 TG (128 threads = 4 SIMD-groups) = 16 KB LUT 共有で 4 タイル並列デコード

#include <metal_stdlib>
using namespace metal;

// ─── 定数 (rans32_format.h と完全一致) ────────────────────────────────────────
constant uint RANS32_LANES           = 32u;
constant uint RANS32_TILE_SIZE       = 65536u;
constant uint RANS32_TILES_PER_BLOCK = 4u;
constant uint RANS32_BLOCK_SIZE      = 262144u;
constant uint RANS32_N_SYMBOLS       = 256u;
constant uint RANS32_PROB_BITS       = 12u;
constant uint RANS32_PROB_SCALE      = 4096u;
constant uint RANS32_PROB_MASK       = 4095u;
constant uint RANS32_L               = 65536u;
constant uint RANS32_MIN_RANS_LEN    = 256u;
constant uint RANS32_TILE_MAX_WORDS  = 32768u;

constant uint RANS32_MODE_RAW        = 0u;
constant uint RANS32_MODE_RLE        = 1u;
constant uint RANS32_MODE_RANS       = 2u;

// ─── 共有構造体 ────────────────────────────────────────────────────────────────
struct Rans32EncSym {
    uint16_t freq;
    uint16_t cdf;
    uint32_t rcp_mul;
    uint32_t rcp_shift;
    uint32_t x_max;
};

struct Rans32BlockMeta {
    uint32_t mode;
    uint32_t rle_byte;
    uint32_t n_nonzero;
    uint32_t raw_size;
};

struct Rans32TileMeta {
    uint32_t mode;
    uint32_t rle_byte;
    uint32_t raw_size;
    uint32_t word_count;
};

inline uint32_t pack_lut(uint32_t sym, uint32_t freq, uint32_t bias) {
    return (sym & 0xFFu) | ((freq & 0xFFFu) << 8u) | ((bias & 0xFFFu) << 20u);
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 1: rans32_build_tables
//   1 threadgroup (128 threads) = 1 superblock (256 KiB)
// ════════════════════════════════════════════════════════════════════════════════
kernel void rans32_build_tables(
    device const uint8_t*    in_data        [[ buffer(0) ]],
    device       Rans32EncSym* out_enc_sym  [[ buffer(1) ]],
    device       uint32_t*   out_dec_lut    [[ buffer(2) ]],
    device       Rans32BlockMeta* out_meta  [[ buffer(3) ]],
    constant     uint32_t&   total_bytes    [[ buffer(4) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const uint block_base = gid * RANS32_BLOCK_SIZE;
    if (block_base >= total_bytes) return;
    const uint raw_size = min(RANS32_BLOCK_SIZE, total_bytes - block_base);
    device const uint8_t* blk = in_data + block_base;

    // ── Phase 1: 並列ヒストグラム集計 (4-byte ロード + 連続ラン結合で atomic 競合回避) ─
    threadgroup atomic_uint tg_hist[RANS32_N_SYMBOLS];
    for (uint i = tid; i < RANS32_N_SYMBOLS; i += tg_size)
        atomic_store_explicit(&tg_hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_u32 = raw_size >> 2u;
    device const uint32_t* blk32 = (device const uint32_t*)blk;

    uint run_sym = 0xFFFFFFFFu;
    uint run_cnt = 0u;
    for (uint i = tid; i < n_u32; i += tg_size) {
        uint32_t w = blk32[i];
        uint b0 =  w         & 0xFFu;
        uint b1 = (w >>  8u) & 0xFFu;
        uint b2 = (w >> 16u) & 0xFFu;
        uint b3 = (w >> 24u) & 0xFFu;
        if (b0 == b1 && b0 == b2 && b0 == b3) {
            if (b0 == run_sym) {
                run_cnt += 4u;
            } else {
                if (run_cnt > 0u)
                    atomic_fetch_add_explicit(&tg_hist[run_sym], run_cnt, memory_order_relaxed);
                run_sym = b0;
                run_cnt = 4u;
            }
        } else {
            if (run_cnt > 0u) {
                atomic_fetch_add_explicit(&tg_hist[run_sym], run_cnt, memory_order_relaxed);
                run_cnt = 0u;
                run_sym = 0xFFFFFFFFu;
            }
            atomic_fetch_add_explicit(&tg_hist[b0], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tg_hist[b1], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tg_hist[b2], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tg_hist[b3], 1u, memory_order_relaxed);
        }
    }
    if (run_cnt > 0u)
        atomic_fetch_add_explicit(&tg_hist[run_sym], run_cnt, memory_order_relaxed);

    // 末尾端数 (0..3 bytes)
    const uint tail_start = n_u32 << 2u;
    for (uint i = tail_start + tid; i < raw_size; i += tg_size)
        atomic_fetch_add_explicit(&tg_hist[blk[i]], 1u, memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: 頻度正規化 (sum == 4096, 各非ゼロ 1..4095) ────────────────────
    threadgroup uint32_t tg_cnt[RANS32_N_SYMBOLS];
    threadgroup uint16_t tg_freq[RANS32_N_SYMBOLS];
    threadgroup uint16_t tg_cdf[RANS32_N_SYMBOLS];
    threadgroup uint16_t tg_end[RANS32_N_SYMBOLS];
    threadgroup uint32_t tg_mode[1];

    for (uint i = tid; i < RANS32_N_SYMBOLS; i += tg_size)
        tg_cnt[i] = atomic_load_explicit(&tg_hist[i], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint n_nz = 0u;
        uint first_nz = 0u;
        uint best = 0u;
        uint best_cnt = 0u;
        for (uint s = 0u; s < RANS32_N_SYMBOLS; ++s) {
            uint c = tg_cnt[s];
            if (c > 0u) {
                if (n_nz == 0u) first_nz = s;
                n_nz++;
                if (c > best_cnt) { best_cnt = c; best = s; }
            }
        }

        Rans32BlockMeta meta;
        meta.raw_size  = raw_size;
        meta.n_nonzero = n_nz;
        meta.rle_byte  = first_nz;

        if (raw_size < RANS32_MIN_RANS_LEN && n_nz != 1u) {
            meta.mode = RANS32_MODE_RAW;
        } else if (n_nz == 1u) {
            meta.mode = RANS32_MODE_RLE;
        } else if (n_nz == 0u) {
            meta.mode = RANS32_MODE_RAW;
        } else {
            meta.mode = RANS32_MODE_RANS;
            uint32_t assigned = 0u;
            for (uint s = 0u; s < RANS32_N_SYMBOLS; ++s) {
                uint c = tg_cnt[s];
                if (c == 0u) {
                    tg_freq[s] = 0u;
                } else {
                    uint32_t f = (uint32_t)(((uint64_t)c * (uint64_t)RANS32_PROB_SCALE) / (uint64_t)raw_size);
                    if (f == 0u) f = 1u;
                    if (f >= RANS32_PROB_SCALE) f = RANS32_PROB_SCALE - 1u;
                    tg_freq[s] = (uint16_t)f;
                    assigned += f;
                }
            }

            // 合計を厳密に 4096 へ調整 (全非ゼロで 1..4095 維持)
            int32_t diff = (int32_t)RANS32_PROB_SCALE - (int32_t)assigned;
            int32_t best_new = (int32_t)tg_freq[best] + diff;
            if (best_new >= 1 && best_new <= (int32_t)(RANS32_PROB_SCALE - 1u)) {
                tg_freq[best] = (uint16_t)best_new;
                assigned = RANS32_PROB_SCALE;
            } else {
                while (assigned < RANS32_PROB_SCALE) {
                    for (uint s = 0u; s < RANS32_N_SYMBOLS && assigned < RANS32_PROB_SCALE; ++s) {
                        if (tg_freq[s] > 0u && tg_freq[s] < RANS32_PROB_SCALE - 1u) {
                            tg_freq[s]++;
                            assigned++;
                        }
                    }
                }
                while (assigned > RANS32_PROB_SCALE) {
                    // 頻度の大きいシンボルから削って歪みを抑える
                    uint max_s = 0u;
                    uint max_f = 0u;
                    for (uint s = 0u; s < RANS32_N_SYMBOLS; ++s) {
                        if (tg_freq[s] > max_f) { max_f = tg_freq[s]; max_s = s; }
                    }
                    if (max_f <= 1u) break;
                    uint32_t dec = min(assigned - RANS32_PROB_SCALE, max_f - 1u);
                    tg_freq[max_s] -= (uint16_t)dec;
                    assigned -= dec;
                }
            }

            uint16_t cum = 0u;
            for (uint s = 0u; s < RANS32_N_SYMBOLS; ++s) {
                tg_cdf[s] = cum;
                cum += tg_freq[s];
                tg_end[s] = cum;
            }
        }

        tg_mode[0] = meta.mode;
        out_meta[gid] = meta;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_mode[0] != RANS32_MODE_RANS) return;

    // ── Phase 3: 並列にエンコードテーブル & 16 KB デコード LUT を構築 ──────────
    device Rans32EncSym* my_enc = out_enc_sym + gid * RANS32_N_SYMBOLS;
    for (uint s = tid; s < RANS32_N_SYMBOLS; s += tg_size) {
        uint16_t f = tg_freq[s];
        Rans32EncSym es;
        es.freq  = f;
        es.cdf   = tg_cdf[s];
        es.x_max = uint32_t(f) << 20u;
        if (f <= 1u) {
            es.rcp_mul   = 0u;
            es.rcp_shift = 0u;
        } else {
            uint32_t k = 31u - clz(uint32_t(f - 1u));
            uint64_t num = (1ull << 32u) * ((1ull << (k + 1u)) - uint64_t(f));
            es.rcp_mul   = uint32_t(num / uint64_t(f)) + 1u;
            es.rcp_shift = k;
        }
        my_enc[s] = es;
    }

    // 4096 スロットを 128 スレッド完全均等並列で二分探索構築 (8 反復で lo == hi 確定)
    device uint32_t* my_lut = out_dec_lut + gid * RANS32_PROB_SCALE;
    for (uint slot = tid; slot < RANS32_PROB_SCALE; slot += tg_size) {
        uint lo = 0u;
        uint hi = RANS32_N_SYMBOLS - 1u; // inclusive [0..255], 幅 256 = 2^8
        for (uint iter = 0u; iter < 8u; ++iter) {
            uint mid = (lo + hi) >> 1u;
            if (uint(tg_end[mid]) > slot) hi = mid;
            else                          lo = mid + 1u;
        }
        uint sym  = lo;
        uint freq = uint(tg_freq[sym]);
        uint bias = slot - uint(tg_cdf[sym]);
        my_lut[slot] = pack_lut(sym, freq, bias);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 2: rans32_encode
//   1 threadgroup (128 threads = 4 SIMD-groups) = 1 superblock (4 × 64 KiB tiles)
//   各 SIMD-group (32 lanes) が 1 本のコアレス uint16_t ストリームへ並列出力。
// ════════════════════════════════════════════════════════════════════════════════
kernel void rans32_encode(
    device const uint8_t*          in_data          [[ buffer(0) ]],
    device const Rans32EncSym*     in_enc_sym       [[ buffer(1) ]],
    device const Rans32BlockMeta*  in_block_meta    [[ buffer(2) ]],
    device       uint16_t*         out_words        [[ buffer(3) ]],
    device       uint32_t*         out_final_states [[ buffer(4) ]],
    device       Rans32TileMeta*   out_tile_meta    [[ buffer(5) ]],
    constant     uint32_t&         total_bytes      [[ buffer(6) ]],

    uint tid         [[ thread_index_in_threadgroup ]],
    uint tg_size     [[ threads_per_threadgroup ]],
    uint lane        [[ thread_index_in_simdgroup ]],
    uint sg_id       [[ simdgroup_index_in_threadgroup ]],
    uint gid         [[ threadgroup_position_in_grid ]]
) {
    const uint block_base = gid * RANS32_BLOCK_SIZE;
    if (block_base >= total_bytes) return;

    const Rans32BlockMeta bmeta = in_block_meta[gid];
    const uint tile_gid  = gid * RANS32_TILES_PER_BLOCK + sg_id;
    const uint tile_base = block_base + sg_id * RANS32_TILE_SIZE;

    // ブロック全体で共有するエンコードテーブルを threadgroup メモリへロード (4 KB)
    threadgroup Rans32EncSym tg_sym[RANS32_N_SYMBOLS];
    if (bmeta.mode == RANS32_MODE_RANS) {
        device const Rans32EncSym* my_sym = in_enc_sym + gid * RANS32_N_SYMBOLS;
        for (uint i = tid; i < RANS32_N_SYMBOLS; i += tg_size)
            tg_sym[i] = my_sym[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tile_base >= total_bytes) {
        if (lane == 0u) {
            Rans32TileMeta tm = { RANS32_MODE_RAW, 0u, 0u, 0u };
            out_tile_meta[tile_gid] = tm;
        }
        return;
    }

    const uint tile_len = min(RANS32_TILE_SIZE, total_bytes - tile_base);

    if (bmeta.mode == RANS32_MODE_RLE) {
        if (lane == 0u) {
            Rans32TileMeta tm = { RANS32_MODE_RLE, bmeta.rle_byte, tile_len, 0u };
            out_tile_meta[tile_gid] = tm;
        }
        return;
    }

    if (bmeta.mode == RANS32_MODE_RAW || tile_len < RANS32_MIN_RANS_LEN) {
        if (lane == 0u) {
            Rans32TileMeta tm = { RANS32_MODE_RAW, 0u, tile_len, tile_len };
            out_tile_meta[tile_gid] = tm;
        }
        return;
    }

    // ── 32-way SIMD-group Coalesced rANS エンコード (逆順: step = S-1 .. 0) ────
    device const uint8_t* tile_in  = in_data + tile_base;
    device       uint16_t* my_words = out_words + uint64_t(tile_gid) * uint64_t(RANS32_TILE_MAX_WORDS);

    // 128 B (final_states) + wp * 2 B < tile_len かつ ワークスペース上限未満のみ rANS 採用
    const uint32_t max_words = min(RANS32_TILE_MAX_WORDS, (tile_len - 129u) >> 1u);
    const uint32_t steps     = (tile_len + RANS32_LANES - 1u) >> 5u;
    const uint32_t lane_mask = (1u << lane) - 1u;

    uint32_t state = RANS32_L;
    uint32_t wp    = 0u;
    bool overflow  = false;

    for (int step = int(steps) - 1; step >= 0; --step) {
        const uint32_t pos = (uint32_t(step) << 5u) + lane;
        const bool active  = (pos < tile_len);

        Rans32EncSym es;
        bool need_renorm = false;
        if (active) {
            uint8_t sym = tile_in[pos];
            es = tg_sym[sym];
            need_renorm = (state >= es.x_max);
        }

        // SIMD-group 全体で排出ワードを連続コアレス書き込み
        uint32_t mask32 = uint32_t(simd_vote::vote_t(simd_ballot(need_renorm)));
        uint32_t count  = popcount(mask32);

        if (wp + count > max_words) {
            overflow = true;
            break;
        }

        if (need_renorm) {
            uint32_t rank = popcount(mask32 & lane_mask);
            my_words[wp + rank] = uint16_t(state & 0xFFFFu);
            state >>= 16u;
        }
        wp += count;

        if (active) {
            uint32_t f = uint32_t(es.freq);
            uint32_t t = mulhi(state, es.rcp_mul);
            uint32_t q_div = (t + ((state - t) >> 1u)) >> es.rcp_shift;
            uint32_t q = (f == 1u) ? state : q_div;
            uint32_t r = state - q * f;
            state = (q << RANS32_PROB_BITS) + uint32_t(es.cdf) + r;
        }
    }

    if (overflow) {
        if (lane == 0u) {
            Rans32TileMeta tm = { RANS32_MODE_RAW, 0u, tile_len, tile_len };
            out_tile_meta[tile_gid] = tm;
        }
    } else {
        out_final_states[tile_gid * RANS32_LANES + lane] = state;
        if (lane == 0u) {
            Rans32TileMeta tm = { RANS32_MODE_RANS, 0u, tile_len, wp };
            out_tile_meta[tile_gid] = tm;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 3: rans32_decode
//   1 threadgroup (128 threads = 4 SIMD-groups) = 1 superblock (4 × 64 KiB tiles)
//   16 KB 逆 CDF LUT を 4 SIMD-groups (128 threads) で共有して超並列復号。
// ════════════════════════════════════════════════════════════════════════════════
kernel void rans32_decode(
    device const uint16_t*        in_words         [[ buffer(0) ]],
    device const uint32_t*        in_final_states  [[ buffer(1) ]],
    device const uint32_t*        in_dec_lut       [[ buffer(2) ]],
    device const Rans32TileMeta*  in_tile_meta     [[ buffer(3) ]],
    device const uint8_t*         in_raw_fallback  [[ buffer(4) ]],
    device       uint8_t*         out_data         [[ buffer(5) ]],
    device       atomic_uint*     out_err          [[ buffer(6) ]],
    constant     uint32_t&        total_bytes      [[ buffer(7) ]],

    uint tid         [[ thread_index_in_threadgroup ]],
    uint tg_size     [[ threads_per_threadgroup ]],
    uint lane        [[ thread_index_in_simdgroup ]],
    uint sg_id       [[ simdgroup_index_in_threadgroup ]],
    uint gid         [[ threadgroup_position_in_grid ]]
) {
    const uint block_base = gid * RANS32_BLOCK_SIZE;
    if (block_base >= total_bytes) return;

    const uint tile_gid  = gid * RANS32_TILES_PER_BLOCK + sg_id;
    const uint tile_base = block_base + sg_id * RANS32_TILE_SIZE;

    if (lane == 0u)
        atomic_store_explicit(&out_err[tile_gid], 0u, memory_order_relaxed);

    // 16 KB LUT を 128 スレッドで協調ロード (1 スレッド 32 ワード)
    threadgroup uint32_t tg_lut[RANS32_PROB_SCALE];
    device const uint32_t* my_lut = in_dec_lut + gid * RANS32_PROB_SCALE;
    for (uint i = tid; i < RANS32_PROB_SCALE; i += tg_size)
        tg_lut[i] = my_lut[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tile_base >= total_bytes) return;

    const Rans32TileMeta tm = in_tile_meta[tile_gid];
    const uint tile_len = min(RANS32_TILE_SIZE, total_bytes - tile_base);
    if (tm.raw_size != tile_len) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 4u, memory_order_relaxed);
        return;
    }

    device uint8_t* tile_out = out_data + tile_base;

    if (tm.mode == RANS32_MODE_RLE) {
        const uint8_t fill = uint8_t(tm.rle_byte);
        for (uint i = lane; i < tile_len; i += RANS32_LANES)
            tile_out[i] = fill;
        return;
    }

    if (tm.mode == RANS32_MODE_RAW) {
        device const uint8_t* raw_src = in_raw_fallback + tile_base;
        for (uint i = lane; i < tile_len; i += RANS32_LANES)
            tile_out[i] = raw_src[i];
        return;
    }

    if (tm.mode != RANS32_MODE_RANS || tm.word_count > RANS32_TILE_MAX_WORDS) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 5u, memory_order_relaxed);
        return;
    }

    // ── 32-way SIMD-group Coalesced rANS デコード (順方向: step = 0 .. S-1) ────
    device const uint16_t* my_words = in_words + uint64_t(tile_gid) * uint64_t(RANS32_TILE_MAX_WORDS);
    uint32_t state = in_final_states[tile_gid * RANS32_LANES + lane];
    if (simd_any(state < RANS32_L)) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 1u, memory_order_relaxed);
        return;
    }

    uint32_t rp = tm.word_count;
    const uint32_t steps     = (tile_len + RANS32_LANES - 1u) >> 5u;
    const uint32_t lane_mask = (1u << lane) - 1u;

    for (uint32_t step = 0u; step < steps; ++step) {
        const uint32_t pos = (step << 5u) + lane;
        const bool active  = (pos < tile_len);
        bool need_refill   = false;

        if (active) {
            uint32_t entry = tg_lut[state & RANS32_PROB_MASK];
            tile_out[pos]  = uint8_t(entry & 0xFFu);
            uint32_t freq  = (entry >> 8u) & RANS32_PROB_MASK;
            uint32_t bias  = entry >> 20u;
            state = freq * (state >> RANS32_PROB_BITS) + bias;
            need_refill = (state < RANS32_L);
        }

        uint32_t mask32 = uint32_t(simd_vote::vote_t(simd_ballot(need_refill)));
        uint32_t count  = popcount(mask32);

        if (rp < count) {
            if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 2u, memory_order_relaxed);
            return;
        }
        rp -= count;

        if (need_refill) {
            uint32_t rank = popcount(mask32 & lane_mask);
            state = (state << 16u) | uint32_t(my_words[rp + rank]);
        }
    }

    // 全 32 レーンの終端状態 == RANS32_L かつ rp == 0 を厳密検証
    bool bad_end = (state != RANS32_L) || (rp != 0u);
    if (simd_any(bad_end)) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 3u, memory_order_relaxed);
    }
}
