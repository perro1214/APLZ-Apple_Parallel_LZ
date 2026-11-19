// hetero_kernels.metal — APLZ-X Phase 2: Apple Silicon GPU 前処理 & 可変長バッチ rANS カーネル
//
// カーネル構成:
//   hx_transform_forward      — 並列前処理 (SHUFFLE2 / SHUFFLE4 / ARM64_BCJ)
//   hx_transform_inverse      — 並列逆処理 (完全可逆 1 パス)
//   hx_rans_batch_build       — 可変長ストリーム群の並列ヒストグラム・正規化・16 KB LUT 構築
//   hx_rans_batch_encode      — 1 TG (128 threads = 4 SIMD-groups) = 1 ストリーム (1..4 タイル)
//   hx_rans_batch_decode      — 1 TG (128 threads = 4 SIMD-groups) = 16 KB LUT 共有並列復号

#include <metal_stdlib>
using namespace metal;

// ─── 定数 (hetero_format.h と完全一致) ────────────────────────────────────────
constant uint HX_TILE_SIZE       = 65536u;
constant uint HX_LANES           = 32u;
constant uint HX_N_SYMBOLS       = 256u;
constant uint HX_PROB_BITS       = 12u;
constant uint HX_PROB_SCALE      = 4096u;
constant uint HX_PROB_MASK       = 4095u;
constant uint HX_RANS_L          = 65536u;
constant uint HX_MIN_RANS_LEN    = 256u;
constant uint HX_TILE_MAX_WORDS  = 32768u;

constant uint HX_XFORM_NONE      = 0u;
constant uint HX_XFORM_SHUFFLE2  = 1u;
constant uint HX_XFORM_SHUFFLE4  = 2u;
constant uint HX_XFORM_ARM64_BCJ = 3u;

constant uint HX_SUB_RAW         = 0u;
constant uint HX_SUB_RLE         = 1u;
constant uint HX_SUB_RANS        = 2u;

struct HxTransformJob {
    uint32_t byte_offset;
    uint32_t byte_len;
    uint32_t xform_id;
    uint32_t _pad;
};

struct HxRansStreamJob {
    uint32_t data_offset;
    uint32_t raw_size;
    uint32_t first_tile_idx;
    uint32_t num_tiles;
    uint32_t mode;
    uint32_t rle_byte;
    uint32_t n_nonzero;
    uint32_t _pad;
};

struct HxEncSym {
    uint16_t freq;
    uint16_t cdf;
    uint32_t rcp_mul;
    uint32_t rcp_shift;
    uint32_t x_max;
};

struct HxTileMeta {
    uint32_t mode;
    uint32_t rle_byte;
    uint32_t raw_size;
    uint32_t word_count;
};

inline uint32_t hx_pack_lut(uint32_t sym, uint32_t freq, uint32_t bias) {
    return (sym & 0xFFu) | ((freq & 0xFFFu) << 8u) | ((bias & 0xFFFu) << 20u);
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 1: hx_transform_forward — OpenZL 型 GPU 並列前処理フィルタ
//   1 threadgroup (256 threads) = 1 ブロック (最大 256 KiB)
// ════════════════════════════════════════════════════════════════════════════════
kernel void hx_transform_forward(
    device const uint8_t*        in_buf  [[ buffer(0) ]],
    device       uint8_t*        out_buf [[ buffer(1) ]],
    device const HxTransformJob* jobs    [[ buffer(2) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const HxTransformJob job = jobs[gid];
    const uint n = job.byte_len;
    if (n == 0u) return;

    device const uint8_t* src = in_buf  + job.byte_offset;
    device       uint8_t* dst = out_buf + job.byte_offset;

    if (job.xform_id == HX_XFORM_NONE) {
        const uint n4 = n >> 2u;
        device const uint32_t* s4 = (device const uint32_t*)src;
        device       uint32_t* d4 = (device       uint32_t*)dst;
        for (uint i = tid; i < n4; i += tg_size) d4[i] = s4[i];
        for (uint i = (n4 << 2u) + tid; i < n; i += tg_size) dst[i] = src[i];
        return;
    }

    if (job.xform_id == HX_XFORM_SHUFFLE2) {
        // 2 バイト周期を 2 プレーンへ分離: [b0, b2, ... | b1, b3, ... | tail]
        // FP16 (LE) では plane1 = 符号+指数部 (超高圧縮), plane0 = 仮数部下位
        const uint count = n >> 1u;
        for (uint i = tid; i < count; i += tg_size) {
            uint8_t b0 = src[i * 2u];
            uint8_t b1 = src[i * 2u + 1u];
            dst[i]         = b0;
            dst[count + i] = b1;
        }
        if (tid == 0u && (n & 1u)) dst[n - 1u] = src[n - 1u];
        return;
    }

    if (job.xform_id == HX_XFORM_SHUFFLE4) {
        // 4 バイト周期を 4 プレーンへ分離: [p0 | p1 | p2 | p3 | tail]
        const uint count = n >> 2u;
        device const uint32_t* s4 = (device const uint32_t*)src;
        for (uint i = tid; i < count; i += tg_size) {
            uint32_t w = s4[i];
            dst[0u * count + i] = uint8_t( w         & 0xFFu);
            dst[1u * count + i] = uint8_t((w >>  8u) & 0xFFu);
            dst[2u * count + i] = uint8_t((w >> 16u) & 0xFFu);
            dst[3u * count + i] = uint8_t((w >> 24u) & 0xFFu);
        }
        for (uint i = (count << 2u) + tid; i < n; i += tg_size) dst[i] = src[i];
        return;
    }

    if (job.xform_id == HX_XFORM_ARM64_BCJ) {
        // ARM64 BL 命令 (上位 6 bit = 0b100101) の PC 相対 imm26 → 絶対ターゲット変換 (100% 全単射)
        const uint count = n >> 2u;
        device const uint32_t* s4 = (device const uint32_t*)src;
        device       uint32_t* d4 = (device       uint32_t*)dst;
        for (uint i = tid; i < count; i += tg_size) {
            uint32_t inst = s4[i];
            if ((inst & 0xFC000000u) == 0x94000000u) {
                uint32_t imm = (inst + i) & 0x03FFFFFFu;
                inst = 0x94000000u | imm;
            }
            d4[i] = inst;
        }
        for (uint i = (count << 2u) + tid; i < n; i += tg_size) dst[i] = src[i];
        return;
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 2: hx_transform_inverse — GPU 並列逆前処理フィルタ
// ════════════════════════════════════════════════════════════════════════════════
kernel void hx_transform_inverse(
    device const uint8_t*        in_buf  [[ buffer(0) ]],
    device       uint8_t*        out_buf [[ buffer(1) ]],
    device const HxTransformJob* jobs    [[ buffer(2) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const HxTransformJob job = jobs[gid];
    const uint n = job.byte_len;
    if (n == 0u) return;

    device const uint8_t* src = in_buf  + job.byte_offset;
    device       uint8_t* dst = out_buf + job.byte_offset;

    if (job.xform_id == HX_XFORM_NONE) {
        const uint n4 = n >> 2u;
        device const uint32_t* s4 = (device const uint32_t*)src;
        device       uint32_t* d4 = (device       uint32_t*)dst;
        for (uint i = tid; i < n4; i += tg_size) d4[i] = s4[i];
        for (uint i = (n4 << 2u) + tid; i < n; i += tg_size) dst[i] = src[i];
        return;
    }

    if (job.xform_id == HX_XFORM_SHUFFLE2) {
        const uint count = n >> 1u;
        for (uint i = tid; i < count; i += tg_size) {
            dst[i * 2u]      = src[i];
            dst[i * 2u + 1u] = src[count + i];
        }
        if (tid == 0u && (n & 1u)) dst[n - 1u] = src[n - 1u];
        return;
    }

    if (job.xform_id == HX_XFORM_SHUFFLE4) {
        const uint count = n >> 2u;
        device uint32_t* d4 = (device uint32_t*)dst;
        for (uint i = tid; i < count; i += tg_size) {
            uint32_t w = uint32_t(src[0u * count + i])
                       | (uint32_t(src[1u * count + i]) <<  8u)
                       | (uint32_t(src[2u * count + i]) << 16u)
                       | (uint32_t(src[3u * count + i]) << 24u);
            d4[i] = w;
        }
        for (uint i = (count << 2u) + tid; i < n; i += tg_size) dst[i] = src[i];
        return;
    }

    if (job.xform_id == HX_XFORM_ARM64_BCJ) {
        const uint count = n >> 2u;
        device const uint32_t* s4 = (device const uint32_t*)src;
        device       uint32_t* d4 = (device       uint32_t*)dst;
        for (uint i = tid; i < count; i += tg_size) {
            uint32_t inst = s4[i];
            if ((inst & 0xFC000000u) == 0x94000000u) {
                uint32_t imm = (inst - i) & 0x03FFFFFFu;
                inst = 0x94000000u | imm;
            }
            d4[i] = inst;
        }
        for (uint i = (count << 2u) + tid; i < n; i += tg_size) dst[i] = src[i];
        return;
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 3: hx_rans_batch_build — 可変長ストリーム群の一括ヒストグラム・正規化・エンコード表構築
//   1 threadgroup (128 threads) = 1 ストリーム (Literals / LL / ML / OF / RANS_ONLY)
//   ※圧縮時はデコード LUT を生成しないため計算量とメモリ帯域を大幅削減
// ════════════════════════════════════════════════════════════════════════════════
kernel void hx_rans_batch_build(
    device const uint8_t*        stream_data [[ buffer(0) ]],
    device       HxRansStreamJob* jobs       [[ buffer(1) ]],
    device       HxEncSym*       out_enc_sym [[ buffer(2) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint sg_id   [[ simdgroup_index_in_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    HxRansStreamJob job = jobs[gid];
    const uint raw_size = job.raw_size;
    if (raw_size == 0u) {
        if (tid == 0u) {
            job.mode = HX_SUB_RAW; job.rle_byte = 0u; job.n_nonzero = 0u;
            jobs[gid] = job;
        }
        return;
    }

    device const uint8_t* sdata = stream_data + job.data_offset;

    // 4 バンク (SIMD-group 毎) ヒストグラムで偏りの強い分布での atomic 競合を 1/4 に削減 (4 KB)
    threadgroup atomic_uint tg_hist[4][HX_N_SYMBOLS];
    for (uint i = tid; i < 4u * HX_N_SYMBOLS; i += tg_size)
        atomic_store_explicit(&tg_hist[i >> 8u][i & 255u], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 16 バイト整列オフセットなので 4 バイト一括ロード + ラン結合でメモリ帯域を 4 倍効率化
    const uint n4 = raw_size >> 2u;
    device const uint32_t* sdata4 = (device const uint32_t*)sdata;
    uint run_sym = 0xFFFFFFFFu;
    uint run_cnt = 0u;
    for (uint i = tid; i < n4; i += tg_size) {
        uint32_t w = sdata4[i];
        uint b0 =  w         & 0xFFu;
        uint b1 = (w >>  8u) & 0xFFu;
        uint b2 = (w >> 16u) & 0xFFu;
        uint b3 = (w >> 24u) & 0xFFu;
        if (b0 == b1 && b0 == b2 && b0 == b3) {
            if (b0 == run_sym) {
                run_cnt += 4u;
            } else {
                if (run_cnt > 0u)
                    atomic_fetch_add_explicit(&tg_hist[sg_id][run_sym], run_cnt, memory_order_relaxed);
                run_sym = b0;
                run_cnt = 4u;
            }
        } else {
            if (run_cnt > 0u) {
                atomic_fetch_add_explicit(&tg_hist[sg_id][run_sym], run_cnt, memory_order_relaxed);
                run_cnt = 0u;
                run_sym = 0xFFFFFFFFu;
            }
            atomic_fetch_add_explicit(&tg_hist[sg_id][b0], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tg_hist[sg_id][b1], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tg_hist[sg_id][b2], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tg_hist[sg_id][b3], 1u, memory_order_relaxed);
        }
    }
    if (run_cnt > 0u)
        atomic_fetch_add_explicit(&tg_hist[sg_id][run_sym], run_cnt, memory_order_relaxed);
    for (uint i = (n4 << 2u) + tid; i < raw_size; i += tg_size)
        atomic_fetch_add_explicit(&tg_hist[sg_id][sdata[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint32_t tg_cnt[HX_N_SYMBOLS];
    threadgroup uint16_t tg_freq[HX_N_SYMBOLS];
    threadgroup uint16_t tg_cdf[HX_N_SYMBOLS];
    threadgroup uint32_t tg_mode[1];

    for (uint i = tid; i < HX_N_SYMBOLS; i += tg_size) {
        tg_cnt[i] = atomic_load_explicit(&tg_hist[0][i], memory_order_relaxed)
                  + atomic_load_explicit(&tg_hist[1][i], memory_order_relaxed)
                  + atomic_load_explicit(&tg_hist[2][i], memory_order_relaxed)
                  + atomic_load_explicit(&tg_hist[3][i], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint n_nz = 0u, first_nz = 0u, best = 0u, best_cnt = 0u;
        for (uint s = 0u; s < HX_N_SYMBOLS; ++s) {
            uint c = tg_cnt[s];
            if (c > 0u) {
                if (n_nz == 0u) first_nz = s;
                n_nz++;
                if (c > best_cnt) { best_cnt = c; best = s; }
            }
        }
        job.n_nonzero = n_nz;
        job.rle_byte  = first_nz;

        if (n_nz == 1u) {
            job.mode = HX_SUB_RLE;
        } else if (raw_size < HX_MIN_RANS_LEN || n_nz == 0u) {
            job.mode = HX_SUB_RAW;
        } else {
            job.mode = HX_SUB_RANS;
            uint32_t assigned = 0u;
            for (uint s = 0u; s < HX_N_SYMBOLS; ++s) {
                uint c = tg_cnt[s];
                if (c == 0u) {
                    tg_freq[s] = 0u;
                } else {
                    uint32_t f = uint32_t((uint64_t(c) * uint64_t(HX_PROB_SCALE)) / uint64_t(raw_size));
                    if (f == 0u) f = 1u;
                    if (f >= HX_PROB_SCALE) f = HX_PROB_SCALE - 1u;
                    tg_freq[s] = uint16_t(f);
                    assigned += f;
                }
            }
            int32_t diff = int32_t(HX_PROB_SCALE) - int32_t(assigned);
            int32_t best_new = int32_t(tg_freq[best]) + diff;
            if (best_new >= 1 && best_new <= int32_t(HX_PROB_SCALE - 1u)) {
                tg_freq[best] = uint16_t(best_new);
                assigned = HX_PROB_SCALE;
            } else {
                while (assigned < HX_PROB_SCALE) {
                    for (uint s = 0u; s < HX_N_SYMBOLS && assigned < HX_PROB_SCALE; ++s) {
                        if (tg_freq[s] > 0u && tg_freq[s] < HX_PROB_SCALE - 1u) { tg_freq[s]++; assigned++; }
                    }
                }
                while (assigned > HX_PROB_SCALE) {
                    uint max_s = 0u, max_f = 0u;
                    for (uint s = 0u; s < HX_N_SYMBOLS; ++s)
                        if (tg_freq[s] > max_f) { max_f = tg_freq[s]; max_s = s; }
                    if (max_f <= 1u) break;
                    uint32_t dec = min(assigned - HX_PROB_SCALE, max_f - 1u);
                    tg_freq[max_s] -= uint16_t(dec);
                    assigned -= dec;
                }
            }
            uint16_t cum = 0u;
            for (uint s = 0u; s < HX_N_SYMBOLS; ++s) {
                tg_cdf[s] = cum;
                cum += tg_freq[s];
            }
        }
        tg_mode[0] = job.mode;
        jobs[gid] = job;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_mode[0] != HX_SUB_RANS) return;

    device HxEncSym* my_enc = out_enc_sym + gid * HX_N_SYMBOLS;
    for (uint s = tid; s < HX_N_SYMBOLS; s += tg_size) {
        uint16_t f = tg_freq[s];
        HxEncSym es;
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
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 4: hx_rans_batch_encode — 可変長ストリーム群の 32-way Coalesced rANS 符号化
//   1 threadgroup (128 threads = 4 SIMD-groups) = 1 ストリーム (1..4 タイル)
// ════════════════════════════════════════════════════════════════════════════════
kernel void hx_rans_batch_encode(
    device const uint8_t*        stream_data      [[ buffer(0) ]],
    device const HxRansStreamJob* jobs            [[ buffer(1) ]],
    device const HxEncSym*       in_enc_sym       [[ buffer(2) ]],
    device       uint16_t*       out_words        [[ buffer(3) ]],
    device       uint32_t*       out_final_states [[ buffer(4) ]],
    device       HxTileMeta*     out_tile_meta    [[ buffer(5) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint lane    [[ thread_index_in_simdgroup ]],
    uint sg_id   [[ simdgroup_index_in_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const HxRansStreamJob job = jobs[gid];

    threadgroup HxEncSym tg_sym[HX_N_SYMBOLS];
    if (job.mode == HX_SUB_RANS) {
        device const HxEncSym* my_sym = in_enc_sym + gid * HX_N_SYMBOLS;
        for (uint i = tid; i < HX_N_SYMBOLS; i += tg_size)
            tg_sym[i] = my_sym[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id >= job.num_tiles) return;

    const uint tile_gid  = job.first_tile_idx + sg_id;
    const uint local_off = sg_id * HX_TILE_SIZE;
    const uint tile_len  = min(HX_TILE_SIZE, job.raw_size - local_off);

    if (job.mode == HX_SUB_RLE) {
        if (lane == 0u) out_tile_meta[tile_gid] = { HX_SUB_RLE, job.rle_byte, tile_len, 0u };
        return;
    }
    if (job.mode == HX_SUB_RAW || tile_len < HX_MIN_RANS_LEN) {
        if (lane == 0u) out_tile_meta[tile_gid] = { HX_SUB_RAW, 0u, tile_len, tile_len };
        return;
    }

    device const uint8_t*  tile_in  = stream_data + job.data_offset + local_off;
    device       uint16_t* my_words = out_words + uint64_t(tile_gid) * uint64_t(HX_TILE_MAX_WORDS);

    const uint32_t max_words = min(HX_TILE_MAX_WORDS, (tile_len - 129u) >> 1u);
    const uint32_t steps     = (tile_len + HX_LANES - 1u) >> 5u;
    const uint32_t lane_mask = (1u << lane) - 1u;

    uint32_t state = HX_RANS_L;
    uint32_t wp    = 0u;
    bool overflow  = false;

    for (int step = int(steps) - 1; step >= 0; --step) {
        const uint32_t pos = (uint32_t(step) << 5u) + lane;
        const bool active  = (pos < tile_len);

        HxEncSym es;
        bool need_renorm = false;
        if (active) {
            es = tg_sym[tile_in[pos]];
            need_renorm = (state >= es.x_max);
        }

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
            state = (q << HX_PROB_BITS) + uint32_t(es.cdf) + r;
        }
    }

    if (overflow) {
        if (lane == 0u) out_tile_meta[tile_gid] = { HX_SUB_RAW, 0u, tile_len, tile_len };
    } else {
        out_final_states[tile_gid * HX_LANES + lane] = state;
        if (lane == 0u) out_tile_meta[tile_gid] = { HX_SUB_RANS, 0u, tile_len, wp };
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Kernel 5: hx_rans_batch_decode — 可変長ストリーム群の 32-way Coalesced rANS 復号
//   1 threadgroup (128 threads = 4 SIMD-groups) = 1 ストリーム (16 KB LUT 共有)
// ════════════════════════════════════════════════════════════════════════════════
kernel void hx_rans_batch_decode(
    device const uint16_t*        in_words         [[ buffer(0) ]],
    device const uint32_t*        in_final_states  [[ buffer(1) ]],
    device const uint32_t*        in_dec_lut       [[ buffer(2) ]],
    device const HxRansStreamJob* jobs             [[ buffer(3) ]],
    device const HxTileMeta*      in_tile_meta     [[ buffer(4) ]],
    device const uint8_t*         in_raw_data      [[ buffer(5) ]],
    device       uint8_t*         out_stream_data  [[ buffer(6) ]],
    device       atomic_uint*     out_err          [[ buffer(7) ]],

    uint tid     [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    uint lane    [[ thread_index_in_simdgroup ]],
    uint sg_id   [[ simdgroup_index_in_threadgroup ]],
    uint gid     [[ threadgroup_position_in_grid ]]
) {
    const HxRansStreamJob job = jobs[gid];

    threadgroup uint32_t tg_lut[HX_PROB_SCALE];
    if (job.mode == HX_SUB_RANS) {
        device const uint32_t* my_lut = in_dec_lut + gid * HX_PROB_SCALE;
        for (uint i = tid; i < HX_PROB_SCALE; i += tg_size)
            tg_lut[i] = my_lut[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id >= job.num_tiles) return;

    const uint tile_gid  = job.first_tile_idx + sg_id;
    const uint local_off = sg_id * HX_TILE_SIZE;
    const uint tile_len  = min(HX_TILE_SIZE, job.raw_size - local_off);

    if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 0u, memory_order_relaxed);

    const HxTileMeta tm = in_tile_meta[tile_gid];
    if (tm.raw_size != tile_len) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 4u, memory_order_relaxed);
        return;
    }

    device uint8_t* tile_out = out_stream_data + job.data_offset + local_off;

    if (tm.mode == HX_SUB_RLE) {
        const uint8_t fill = uint8_t(tm.rle_byte);
        for (uint i = lane; i < tile_len; i += HX_LANES) tile_out[i] = fill;
        return;
    }
    if (tm.mode == HX_SUB_RAW) {
        // ホスト側で out_stream_data に直接配置済みの場合はコピー不要 (in_raw_data == out_stream_data)
        if (in_raw_data != out_stream_data) {
            device const uint8_t* raw_src = in_raw_data + job.data_offset + local_off;
            for (uint i = lane; i < tile_len; i += HX_LANES) tile_out[i] = raw_src[i];
        }
        return;
    }
    if (tm.mode != HX_SUB_RANS || tm.word_count > HX_TILE_MAX_WORDS) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 5u, memory_order_relaxed);
        return;
    }

    device const uint16_t* my_words = in_words + uint64_t(tile_gid) * uint64_t(HX_TILE_MAX_WORDS);
    uint32_t state = in_final_states[tile_gid * HX_LANES + lane];
    if (simd_any(state < HX_RANS_L)) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 1u, memory_order_relaxed);
        return;
    }

    uint32_t rp = tm.word_count;
    const uint32_t steps     = (tile_len + HX_LANES - 1u) >> 5u;
    const uint32_t lane_mask = (1u << lane) - 1u;

    for (uint32_t step = 0u; step < steps; ++step) {
        const uint32_t pos = (step << 5u) + lane;
        const bool active  = (pos < tile_len);
        bool need_refill   = false;

        if (active) {
            uint32_t entry = tg_lut[state & HX_PROB_MASK];
            tile_out[pos]  = uint8_t(entry & 0xFFu);
            uint32_t freq  = (entry >> 8u) & HX_PROB_MASK;
            uint32_t bias  = entry >> 20u;
            state = freq * (state >> HX_PROB_BITS) + bias;
            need_refill = (state < HX_RANS_L);
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

    if (simd_any((state != HX_RANS_L) || (rp != 0u))) {
        if (lane == 0u) atomic_store_explicit(&out_err[tile_gid], 3u, memory_order_relaxed);
    }
}
