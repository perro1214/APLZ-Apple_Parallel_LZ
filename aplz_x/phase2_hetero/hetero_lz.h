// hetero_lz.h — APLZ-X Phase 2: エントロピーゲート・型プローブ・分離ストリーム LZ (Level 1/3/6)
//
// 最適化ポイント:
//   - ARM64 NEON ハードウェア前処理 / プレーン分離直書き (vld2q/vld4q/vst2q/vst4q)
//   - 16 KB n*log2(n) LUT による超高速ブロックプローブ (libm log2 ゼロ!)
//   - thread_local 遅延確保 + Zstd 型エポック方式ハッシュテーブル (memset ゼロ & L1/L2 常駐!)
//   - 8 バイト一括比較マッチ長 + ブランチレス 1-store 64-bit ExtraBits パッキング
#pragma once

#include "hetero_format.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// ─── ゼロ初期化を省く軽量 RAII バッファ (冷ページフォルト削減) ────────────────────
struct HxRawBuf {
    uint8_t* ptr = nullptr;
    size_t   cap = 0u;
    HxRawBuf() = default;
    ~HxRawBuf() { free(ptr); }
    HxRawBuf(const HxRawBuf&) = delete;
    HxRawBuf& operator=(const HxRawBuf&) = delete;
    HxRawBuf(HxRawBuf&& o) noexcept : ptr(o.ptr), cap(o.cap) { o.ptr = nullptr; o.cap = 0u; }
    HxRawBuf& operator=(HxRawBuf&& o) noexcept {
        if (this != &o) { free(ptr); ptr = o.ptr; cap = o.cap; o.ptr = nullptr; o.cap = 0u; }
        return *this;
    }
    void ensure(size_t need) {
        if (cap < need) {
            free(ptr);
            ptr = (uint8_t*)malloc(need);
            cap = ptr ? need : 0u;
        }
    }
    uint8_t*       data()       { return ptr; }
    const uint8_t* data() const { return ptr; }
};

// ─── ARM64 NEON ハードウェア前処理 / プレーン分離直書き / 逆前処理 ────────────────
static inline void hx_neon_shuffle2_planes(const uint8_t* __restrict src, uint32_t n,
                                           uint8_t* __restrict p0, uint8_t* __restrict p1) {
    uint32_t cnt = n >> 1u;
    uint32_t i = 0u;
#if defined(__aarch64__)
    for (; i + 16u <= cnt; i += 16u) {
        uint8x16x2_t v = vld2q_u8(src + (i << 1u));
        vst1q_u8(p0 + i, v.val[0]);
        vst1q_u8(p1 + i, v.val[1]);
    }
#endif
    for (; i < cnt; ++i) {
        p0[i] = src[i * 2u];
        p1[i] = src[i * 2u + 1u];
    }
    if (n & 1u) p1[cnt] = src[n - 1u];
}

static inline void hx_neon_shuffle4_planes(const uint8_t* __restrict src, uint32_t n,
                                           uint8_t* __restrict p0, uint8_t* __restrict p1,
                                           uint8_t* __restrict p2, uint8_t* __restrict p3) {
    uint32_t cnt = n >> 2u;
    uint32_t i = 0u;
#if defined(__aarch64__)
    for (; i + 16u <= cnt; i += 16u) {
        uint8x16x4_t v = vld4q_u8(src + (i << 2u));
        vst1q_u8(p0 + i, v.val[0]);
        vst1q_u8(p1 + i, v.val[1]);
        vst1q_u8(p2 + i, v.val[2]);
        vst1q_u8(p3 + i, v.val[3]);
    }
#endif
    for (; i < cnt; ++i) {
        uint32_t w;
        memcpy(&w, src + i * 4u, 4u);
        p0[i] = (uint8_t)( w         & 0xFFu);
        p1[i] = (uint8_t)((w >>  8u) & 0xFFu);
        p2[i] = (uint8_t)((w >> 16u) & 0xFFu);
        p3[i] = (uint8_t)((w >> 24u) & 0xFFu);
    }
    for (uint32_t j = 0u; j < (n & 3u); ++j) p3[cnt + j] = src[cnt * 4u + j];
}

static inline void hx_neon_unshuffle2_planes(const uint8_t* __restrict p0,
                                             const uint8_t* __restrict p1,
                                             uint8_t* __restrict dst, uint32_t n) {
    uint32_t cnt = n >> 1u;
    uint32_t i = 0u;
#if defined(__aarch64__)
    for (; i + 16u <= cnt; i += 16u) {
        uint8x16x2_t v;
        v.val[0] = vld1q_u8(p0 + i);
        v.val[1] = vld1q_u8(p1 + i);
        vst2q_u8(dst + (i << 1u), v);
    }
#endif
    for (; i < cnt; ++i) {
        dst[i * 2u]      = p0[i];
        dst[i * 2u + 1u] = p1[i];
    }
    if (n & 1u) dst[n - 1u] = p1[cnt];
}

static inline void hx_neon_unshuffle4_planes(const uint8_t* __restrict p0,
                                             const uint8_t* __restrict p1,
                                             const uint8_t* __restrict p2,
                                             const uint8_t* __restrict p3,
                                             uint8_t* __restrict dst, uint32_t n) {
    uint32_t cnt = n >> 2u;
    uint32_t i = 0u;
#if defined(__aarch64__)
    for (; i + 16u <= cnt; i += 16u) {
        uint8x16x4_t v;
        v.val[0] = vld1q_u8(p0 + i);
        v.val[1] = vld1q_u8(p1 + i);
        v.val[2] = vld1q_u8(p2 + i);
        v.val[3] = vld1q_u8(p3 + i);
        vst4q_u8(dst + (i << 2u), v);
    }
#endif
    for (; i < cnt; ++i) {
        uint32_t w = (uint32_t)p0[i]
                   | ((uint32_t)p1[i] <<  8u)
                   | ((uint32_t)p2[i] << 16u)
                   | ((uint32_t)p3[i] << 24u);
        memcpy(dst + i * 4u, &w, 4u);
    }
    for (uint32_t j = 0u; j < (n & 3u); ++j) dst[cnt * 4u + j] = p3[cnt + j];
}

static inline void hx_neon_transform_forward(const uint8_t* __restrict src, uint8_t* __restrict dst,
                                             uint32_t n, uint8_t xform_id) {
    if (xform_id == HX_XFORM_NONE) {
        memcpy(dst, src, n);
        return;
    }
    if (xform_id == HX_XFORM_SHUFFLE2) {
        uint32_t cnt = n >> 1u;
        hx_neon_shuffle2_planes(src, n, dst, dst + cnt);
        return;
    }
    if (xform_id == HX_XFORM_SHUFFLE4) {
        uint32_t cnt = n >> 2u;
        hx_neon_shuffle4_planes(src, n, dst, dst + cnt, dst + cnt * 2u, dst + cnt * 3u);
        return;
    }
    if (xform_id == HX_XFORM_ARM64_BCJ) {
        uint32_t cnt = n >> 2u;
        for (uint32_t i = 0u; i < cnt; ++i) {
            uint32_t inst;
            memcpy(&inst, src + i * 4u, 4u);
            if ((inst & 0xFC000000u) == 0x94000000u) {
                inst = 0x94000000u | ((inst + i) & 0x03FFFFFFu);
            }
            memcpy(dst + i * 4u, &inst, 4u);
        }
        for (uint32_t j = cnt * 4u; j < n; ++j) dst[j] = src[j];
    }
}

static inline void hx_neon_transform_inverse(const uint8_t* __restrict src, uint8_t* __restrict dst,
                                             uint32_t n, uint8_t xform_id) {
    if (xform_id == HX_XFORM_NONE) {
        memcpy(dst, src, n);
        return;
    }
    if (xform_id == HX_XFORM_SHUFFLE2) {
        uint32_t cnt = n >> 1u;
        hx_neon_unshuffle2_planes(src, src + cnt, dst, n);
        return;
    }
    if (xform_id == HX_XFORM_SHUFFLE4) {
        uint32_t cnt = n >> 2u;
        hx_neon_unshuffle4_planes(src, src + cnt, src + cnt * 2u, src + cnt * 3u, dst, n);
        return;
    }
    if (xform_id == HX_XFORM_ARM64_BCJ) {
        uint32_t cnt = n >> 2u;
        for (uint32_t i = 0u; i < cnt; ++i) {
            uint32_t inst;
            memcpy(&inst, src + i * 4u, 4u);
            if ((inst & 0xFC000000u) == 0x94000000u) {
                inst = 0x94000000u | ((inst - i) & 0x03FFFFFFu);
            }
            memcpy(dst + i * 4u, &inst, 4u);
        }
        for (uint32_t j = cnt * 4u; j < n; ++j) dst[j] = src[j];
    }
}

// ─── 高速 4-byte ハッシュ & ロードヘルパ ────────────────────────────────────────
static inline uint32_t hx_hash4(uint32_t v, uint32_t bits) {
    return (v * 2654435761u) >> (32u - bits);
}

static inline uint32_t hx_read32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4u);
    return v;
}

static inline uint64_t hx_read64(const uint8_t* p) {
    uint64_t v;
    memcpy(&v, p, 8u);
    return v;
}

// 8 バイト並列 XOR + ctzll による超高速マッチ長計測
static inline uint32_t hx_match_len(const uint8_t* a, const uint8_t* b, uint32_t max_len) {
    uint32_t len = 0u;
    while (len + 8u <= max_len) {
        uint64_t diff = hx_read64(a + len) ^ hx_read64(b + len);
        if (diff != 0ull) {
            return len + ((uint32_t)__builtin_ctzll(diff) >> 3u);
        }
        len += 8u;
    }
    while (len < max_len && a[len] == b[len]) len++;
    return len;
}

// 1 回の 8 バイトロードで 4〜7 バイト短マッチをループ無しで即座に計測
static inline uint32_t hx_match_len_from_diff8(uint64_t diff8, const uint8_t* a, const uint8_t* b, uint32_t max_len) {
    if (__builtin_expect(diff8 != 0ull, 1)) {
        return (uint32_t)__builtin_ctzll(diff8) >> 3u;
    }
    return 8u + hx_match_len(a + 8u, b + 8u, max_len - 8u);
}

// ─── 1. 高速ブロックプローブ (16 KB n*log2(n) LUT で libm log2 ゼロ!) ──────────
struct HxProbeResult {
    uint8_t xform_id;       // HX_XFORM_*
    uint8_t codec_hint;     // HX_CODEC_RAW / RLE / RANS_ONLY / LZ_RANS
    uint8_t rle_byte;
    uint8_t plane_raw_mask; // RANS_ONLY 時の高エントロピー・プレーン RAW バイパスビットマスク
};

static inline const float* hx_get_nlog2n_table() {
    static float tbl[4097];
    static bool init = []() {
        tbl[0] = 0.0f;
        for (uint32_t i = 1u; i <= 4096u; ++i) {
            tbl[i] = (float)i * log2f((float)i);
        }
        return true;
    }();
    (void)init;
    return tbl;
}

static inline float hx_calc_entropy_fast(const uint32_t* cnt, uint32_t n, const float* tbl) {
    if (n == 0u) return 0.0f;
    float sum = 0.0f;
    for (uint32_t i = 0u; i < 256u; ++i) {
        sum += tbl[cnt[i]];
    }
    return (tbl[n] - sum) / (float)n;
}

static inline HxProbeResult hx_probe_block(const uint8_t* data, uint32_t len) {
    HxProbeResult res = { HX_XFORM_NONE, HX_CODEC_LZ_RANS, 0u, 0u };
    if (len < 64u) {
        res.codec_hint = HX_CODEC_RAW;
        return res;
    }

    // 1) 超高速 RLE チェック (先頭・中央・末尾一致なら全スキャン)
    uint8_t b0 = data[0];
    if (data[len >> 1u] == b0 && data[len - 1u] == b0) {
        uint64_t pat = (uint64_t)b0 * 0x0101010101010101ull;
        bool all_same = true;
        uint32_t i = 0u;
        for (; i + 8u <= len; i += 8u) {
            if (hx_read64(data + i) != pat) { all_same = false; break; }
        }
        if (all_same) {
            for (; i < len; ++i) {
                if (data[i] != b0) { all_same = false; break; }
            }
        }
        if (all_same) {
            res.codec_hint = HX_CODEC_RLE;
            res.rle_byte   = b0;
            return res;
        }
    }

    // 2) 4 KiB サンプル (先頭 2 KiB + 中央 2 KiB) を 1 パス解析
    const uint32_t seg = std::min(2048u, len >> 1u);
    const uint32_t sample_len = (seg * 2u) & ~3u;
    const uint8_t* segs[2] = { data, data + ((len >> 1u) & ~3u) };

    uint32_t cnt_all[256] = {};
    uint32_t cnt_s2[2][256] = {};
    uint32_t cnt_s4[4][256] = {};
    uint32_t bl_hits = 0u;
    uint32_t total_inst = 0u;
    uint32_t match4_hits = 0u;

    uint32_t seen_val[1024];
    uint8_t  seen_occ[1024] = {};

    for (int s = 0; s < 2; ++s) {
        const uint8_t* sp = segs[s];
        uint32_t sl = seg & ~3u;
        for (uint32_t i = 0u; i < sl; i += 4u) {
            uint32_t w = hx_read32(sp + i);
            uint8_t c0 = (uint8_t)(w & 0xFFu);
            uint8_t c1 = (uint8_t)((w >> 8u) & 0xFFu);
            uint8_t c2 = (uint8_t)((w >> 16u) & 0xFFu);
            uint8_t c3 = (uint8_t)((w >> 24u) & 0xFFu);

            cnt_all[c0]++; cnt_all[c1]++; cnt_all[c2]++; cnt_all[c3]++;
            cnt_s2[0][c0]++; cnt_s2[1][c1]++; cnt_s2[0][c2]++; cnt_s2[1][c3]++;
            cnt_s4[0][c0]++; cnt_s4[1][c1]++; cnt_s4[2][c2]++; cnt_s4[3][c3]++;

            if ((w & 0xFC000000u) == 0x94000000u) {
                uint32_t hi4 = (w >> 22u) & 0xFu;
                if (hi4 == 0x0u || hi4 == 0xFu) bl_hits++;
            }
            total_inst++;

            uint32_t h = hx_hash4(w, 10u);
            if (seen_occ[h] && seen_val[h] == w) match4_hits++;
            seen_occ[h] = 1u;
            seen_val[h] = w;
        }
    }

    const float* tbl = hx_get_nlog2n_table();
    float h_all  = hx_calc_entropy_fast(cnt_all, sample_len, tbl);
    float h_s2_0 = hx_calc_entropy_fast(cnt_s2[0], sample_len >> 1u, tbl);
    float h_s2_1 = hx_calc_entropy_fast(cnt_s2[1], sample_len >> 1u, tbl);
    float h_s2   = 0.5f * (h_s2_0 + h_s2_1);

    float h_s4[4];
    for (int k = 0; k < 4; ++k) h_s4[k] = hx_calc_entropy_fast(cnt_s4[k], sample_len >> 2u, tbl);
    float h_s4_avg = 0.25f * (h_s4[0] + h_s4[1] + h_s4[2] + h_s4[3]);

    float gain_s2 = h_all - h_s2;
    float gain_s4 = h_all - h_s4_avg;

    bool is_arm64_code = (total_inst > 0u && bl_hits * 100u >= total_inst);

    // 3) 構造化数値配列 (SHUFFLE4 / SHUFFLE2) 判定
    if (!is_arm64_code && gain_s4 >= 0.45f && gain_s4 > gain_s2 + 0.15f) {
        res.xform_id = HX_XFORM_SHUFFLE4;
        int hi_lanes = 0;
        uint8_t rmask = 0u;
        for (int k = 0; k < 4; ++k) {
            if (h_s4[k] > 7.4f) hi_lanes++;
            if (h_s4[k] > 7.80f) rmask |= (uint8_t)(1u << k);
        }
        if (hi_lanes >= 3 && match4_hits < total_inst / 40u) {
            res.codec_hint     = HX_CODEC_RANS_ONLY;
            res.plane_raw_mask = rmask;
        } else {
            res.codec_hint = HX_CODEC_LZ_RANS;
        }
        return res;
    }
    if (!is_arm64_code && gain_s2 >= 0.45f) {
        res.xform_id = HX_XFORM_SHUFFLE2;
        float max_lane = std::max(h_s2_0, h_s2_1);
        if (max_lane > 7.0f && match4_hits < total_inst / 20u) {
            res.codec_hint = HX_CODEC_RANS_ONLY;
            uint8_t rmask = 0u;
            if (h_s2_0 > 7.80f) rmask |= 1u;
            if (h_s2_1 > 7.80f) rmask |= 2u;
            res.plane_raw_mask = rmask;
        } else {
            res.codec_hint = HX_CODEC_LZ_RANS;
        }
        return res;
    }

    // 4) エントロピーゲート: 既圧縮・ランダムデータ検知 → 即 RAW バイパス!
    if (h_all > 7.80f && match4_hits * 64u < total_inst) {
        res.xform_id   = HX_XFORM_NONE;
        res.codec_hint = HX_CODEC_RAW;
        return res;
    }

    // 5) ARM64 実行コード → ARM64_BCJ フィルタ選択
    if (is_arm64_code) {
        res.xform_id   = HX_XFORM_ARM64_BCJ;
        res.codec_hint = HX_CODEC_LZ_RANS;
        return res;
    }

    return res;
}

// ─── 2. 分離ストリーム LZ 出力バッファ (未初期化 malloc で冷ページゼロ!) ────────
struct HxLzBlockOutput {
    static const uint32_t MAX_SEQS = HX_BLOCK_SIZE / HX_MIN_MATCH + 16u;
    HxRawBuf literals;
    HxRawBuf ll_codes;
    HxRawBuf ml_codes;
    HxRawBuf of_codes;
    HxRawBuf extra_bits;
    uint32_t lit_size      = 0u;
    uint32_t seq_count     = 0u;
    uint32_t extra_size    = 0u;
    uint32_t tail_lit_size = 0u;

    void ensure_allocated() {
        literals.ensure(HX_BLOCK_SIZE + 32u);
        ll_codes.ensure(MAX_SEQS);
        ml_codes.ensure(MAX_SEQS);
        of_codes.ensure(MAX_SEQS);
        extra_bits.ensure(HX_BLOCK_SIZE + 32u);
        lit_size = 0u;
        seq_count = 0u;
        extra_size = 0u;
        tail_lit_size = 0u;
    }
};

// ─── 2b. スレッド固有 (thread_local) 遅延確保 + エポック方式ハッシュテーブル ────
// ・必要レベルのテーブルだけを初回遅延確保 (L1 は 64 KB のみ、Random/FP16/Decode は 0 B!)
// ・Zstd 型エポック加算により毎ブロック memset ゼロ & 他ブロックの偽マッチロードも完全ゼロ!
struct HxLzThreadState {
    static const uint32_t HBITS_FAST = 14u; // 16K * 4B = 64 KB (L1D 常駐!)
    static const uint32_t HBITS_DEF  = 15u; // 32K * 2 * 4B = 256 KB (L2 常駐!)
    static const uint32_t HBITS_HIGH = 15u; // 32K * 4 * 4B = 512 KB (L2 常駐!)

    uint32_t* ht_fast = nullptr;
    uint32_t* ht_def  = nullptr; // [1<<15][2]
    uint32_t* ht_high = nullptr; // [1<<15][4]
    uint32_t  ep_fast = HX_BLOCK_SIZE;
    uint32_t  ep_def  = HX_BLOCK_SIZE;
    uint32_t  ep_high = HX_BLOCK_SIZE;
    HxRawBuf  xtmp; // LZ+Xform 時のみ遅延確保 (256 KiB)

    ~HxLzThreadState() {
        free(ht_fast);
        free(ht_def);
        free(ht_high);
    }

    uint32_t* acquire_fast(uint32_t& out_ep) {
        if (!ht_fast) {
            ht_fast = (uint32_t*)calloc(1u << HBITS_FAST, sizeof(uint32_t));
            ep_fast = HX_BLOCK_SIZE;
        } else {
            ep_fast += HX_BLOCK_SIZE;
            if (ep_fast > 0xF0000000u) {
                memset(ht_fast, 0, (size_t)(1u << HBITS_FAST) * sizeof(uint32_t));
                ep_fast = HX_BLOCK_SIZE;
            }
        }
        out_ep = ep_fast;
        return ht_fast;
    }

    uint32_t* acquire_def(uint32_t& out_ep) {
        if (!ht_def) {
            ht_def = (uint32_t*)calloc((size_t)(1u << HBITS_DEF) * 2u, sizeof(uint32_t));
            ep_def = HX_BLOCK_SIZE;
        } else {
            ep_def += HX_BLOCK_SIZE;
            if (ep_def > 0xF0000000u) {
                memset(ht_def, 0, (size_t)(1u << HBITS_DEF) * 2u * sizeof(uint32_t));
                ep_def = HX_BLOCK_SIZE;
            }
        }
        out_ep = ep_def;
        return ht_def;
    }

    uint32_t* acquire_high(uint32_t& out_ep) {
        if (!ht_high) {
            ht_high = (uint32_t*)calloc((size_t)(1u << HBITS_HIGH) * 4u, sizeof(uint32_t));
            ep_high = HX_BLOCK_SIZE;
        } else {
            ep_high += HX_BLOCK_SIZE;
            if (ep_high > 0xF0000000u) {
                memset(ht_high, 0, (size_t)(1u << HBITS_HIGH) * 4u * sizeof(uint32_t));
                ep_high = HX_BLOCK_SIZE;
            }
        }
        out_ep = ep_high;
        return ht_high;
    }

    uint8_t* acquire_xtmp() {
        xtmp.ensure(HX_BLOCK_SIZE + 32u);
        return xtmp.data();
    }
};

static inline HxLzThreadState& hx_get_thread_lz_state() {
    thread_local HxLzThreadState st;
    return st;
}

// ─── 3. 分離ストリーム LZ エンコーダ (Level 1 / 3 / 6 — エポック + 1-store ExtraBits) ─
static inline bool hx_lz_encode_block(
    const uint8_t* __restrict src,
    uint32_t                  len,
    uint32_t                  level,
    HxLzThreadState&          ts,
    HxLzBlockOutput&          out
) {
    out.ensure_allocated();
    if (len < 16u) return false;

    uint8_t* __restrict lit_base = out.literals.data();
    uint8_t* __restrict lit_p    = lit_base;
    uint8_t* __restrict ll_base  = out.ll_codes.data();
    uint8_t* __restrict ll_p     = ll_base;
    uint8_t* __restrict ml_p     = out.ml_codes.data();
    uint8_t* __restrict of_p     = out.of_codes.data();
    uint8_t* __restrict ex_base  = out.extra_bits.data();
    uint8_t* __restrict ex_p     = ex_base;

    uint32_t rep0 = 1u, rep1 = 4u, rep2 = 8u;
    uint64_t bit_acc = 0ull;
    uint32_t bit_cnt = 0u; // 不変条件: 0 <= bit_cnt <= 7 (最大 7 + 51 = 58 < 64 bit!)

    auto emit_seq = [&](const uint8_t* lit_src, uint32_t ll, uint32_t ml, uint32_t dist) {
        if (ll > 0u) {
            if (__builtin_expect(ll <= 16u, 1)) {
                memcpy(lit_p, lit_src, 16u);
            } else {
                memcpy(lit_p, lit_src, ll);
            }
            lit_p += ll;
        }
        uint32_t ll_ex, ll_eb, ml_ex, ml_eb, of_ex = 0u, of_eb = 0u;
        uint8_t ll_c = hx_len_to_code(ll, &ll_ex, &ll_eb);
        uint8_t ml_c = hx_len_to_code(ml - HX_MIN_MATCH, &ml_ex, &ml_eb);
        uint8_t of_c;
        if (dist == rep0) {
            of_c = 0u;
        } else if (dist == rep1) {
            of_c = 1u;
            std::swap(rep0, rep1);
        } else if (dist == rep2) {
            of_c = 2u;
            uint32_t tmp = rep2; rep2 = rep1; rep1 = rep0; rep0 = tmp;
        } else {
            of_c = hx_dist_to_code(dist, &of_ex, &of_eb);
            rep2 = rep1; rep1 = rep0; rep0 = dist;
        }
        *ll_p++ = ll_c;
        *ml_p++ = ml_c;
        *of_p++ = of_c;

        // ブランチレス 1-store 64-bit パッキング (最大 51 bit + bit_cnt<=7 = 58 bit)
        uint32_t tot_eb = ll_eb + ml_eb + of_eb;
        if (tot_eb) {
            uint64_t packed = (uint64_t)ll_ex
                            | ((uint64_t)ml_ex << ll_eb)
                            | ((uint64_t)of_ex << (ll_eb + ml_eb));
            bit_acc |= (packed << bit_cnt);
            bit_cnt += tot_eb;
            memcpy(ex_p, &bit_acc, 8u);
            uint32_t adv = bit_cnt >> 3u;
            ex_p    += adv;
            bit_acc >>= (adv << 3u);
            bit_cnt &= 7u;
        }
    };

    const uint32_t mflimit = (len > 16u) ? (len - 16u) : 0u;
    uint32_t anchor = 0u;
    uint32_t pos    = 1u;
    uint32_t matched_bytes = 0u;

    if (level <= 1u) {
        // ── Level 1 (--fast): 1-way L1D 64 KB エポックハッシュ + 8B 即時マッチ + ステップ加速 ──
        uint32_t ep;
        uint32_t* __restrict ht = ts.acquire_fast(ep);
        ht[hx_hash4(hx_read32(src), HxLzThreadState::HBITS_FAST)] = ep;

        while (pos < mflimit) {
            uint32_t miss = 64u;
            uint32_t best_len = 0u, best_dist = 0u;

            // マッチ直後 (anchor..anchor+3) のみ高確率な rep0 を高速チェック
            uint32_t rep_limit = std::min(mflimit, anchor + 4u);
            while (pos < mflimit) {
                uint64_t cur8 = hx_read64(src + pos);
                uint32_t cur4 = (uint32_t)cur8;

                if (pos < rep_limit && pos >= rep0) {
                    uint64_t rdiff = hx_read64(src + pos - rep0) ^ cur8;
                    if ((uint32_t)rdiff == 0u) {
                        best_len  = hx_match_len_from_diff8(rdiff, src + pos - rep0, src + pos, len - pos);
                        best_dist = rep0;
                        ht[hx_hash4(cur4, HxLzThreadState::HBITS_FAST)] = ep + pos;
                        break;
                    }
                }

                uint32_t h    = hx_hash4(cur4, HxLzThreadState::HBITS_FAST);
                uint32_t rel  = ht[h] - ep;
                ht[h]         = ep + pos;
                if (rel < pos) {
                    uint64_t diff = hx_read64(src + rel) ^ cur8;
                    if ((uint32_t)diff == 0u) {
                        best_len  = hx_match_len_from_diff8(diff, src + rel, src + pos, len - pos);
                        best_dist = pos - rel;
                        break;
                    }
                }
                pos  += (miss >> 6u);
                miss += 2u;
            }
            if (best_len < HX_MIN_MATCH) break;

            while (pos > anchor && pos > best_dist && src[pos - 1u] == src[pos - best_dist - 1u]) {
                pos--;
                best_len++;
            }

            emit_seq(src + anchor, pos - anchor, best_len, best_dist);
            matched_bytes += best_len;
            pos += best_len;
            anchor = pos;
            if (pos < mflimit && pos >= 2u) {
                ht[hx_hash4(hx_read32(src + pos - 2u), HxLzThreadState::HBITS_FAST)] = ep + (pos - 2u);
            }
        }
    } else if (level <= 3u) {
        // ── Level 3 (--default): 2-way 32K エポックハッシュ + 近傍 Repeat + 1-byte Lazy + 後方拡張 ──
        uint32_t ep;
        uint32_t* __restrict ht = ts.acquire_def(ep);

        auto insert2 = [&](uint32_t p) {
            uint32_t idx = hx_hash4(hx_read32(src + p), HxLzThreadState::HBITS_DEF) << 1u;
            ht[idx + 1u] = ht[idx];
            ht[idx]      = ep + p;
        };
        auto find2 = [&](uint32_t p, bool check_rep, uint32_t& out_len, uint32_t& out_dist) {
            uint64_t cur8 = hx_read64(src + p);
            uint32_t cur4 = (uint32_t)cur8;
            out_len = 0u; out_dist = 0u;
            if (check_rep) {
                if (p >= rep0) {
                    uint64_t rd = hx_read64(src + p - rep0) ^ cur8;
                    if ((uint32_t)rd == 0u) {
                        out_len  = hx_match_len_from_diff8(rd, src + p - rep0, src + p, len - p);
                        out_dist = rep0;
                    }
                }
                if (out_len == 0u && p >= rep1) {
                    uint64_t rd = hx_read64(src + p - rep1) ^ cur8;
                    if ((uint32_t)rd == 0u) {
                        out_len  = hx_match_len_from_diff8(rd, src + p - rep1, src + p, len - p);
                        out_dist = rep1;
                    }
                }
            }
            uint32_t idx = hx_hash4(cur4, HxLzThreadState::HBITS_DEF) << 1u;
            uint32_t r0  = ht[idx] - ep;
            uint32_t r1  = ht[idx + 1u] - ep;
            ht[idx + 1u] = ht[idx];
            ht[idx]      = ep + p;

            if (out_len >= 32u) return;

            uint32_t thresh = (out_dist == rep0 || out_dist == rep1) ? (out_len + 1u) : out_len;
            if (r0 < p) {
                uint64_t diff = hx_read64(src + r0) ^ cur8;
                if ((uint32_t)diff == 0u) {
                    uint32_t ml = hx_match_len_from_diff8(diff, src + r0, src + p, len - p);
                    if (ml > thresh) { out_len = ml; out_dist = p - r0; thresh = ml; }
                }
            }
            if (out_len < 16u && r1 < p) {
                uint64_t diff = hx_read64(src + r1) ^ cur8;
                if ((uint32_t)diff == 0u) {
                    uint32_t ml = hx_match_len_from_diff8(diff, src + r1, src + p, len - p);
                    if (ml > thresh) { out_len = ml; out_dist = p - r1; }
                }
            }
        };

        insert2(0u);
        uint32_t miss = 0u;
        while (pos < mflimit) {
            uint32_t mlen, mdist;
            find2(pos, (pos < anchor + 8u), mlen, mdist);
            if (mlen < HX_MIN_MATCH) {
                pos += 1u + (miss >> 7u);
                miss++;
                continue;
            }
            if (miss >= 128u) {
                while (pos > anchor && pos > mdist && src[pos - 1u] == src[pos - mdist - 1u]) {
                    pos--; mlen++;
                }
            }
            miss = 0u;
            if (pos + 1u < mflimit && mlen < 32u) {
                uint32_t nlen, ndist;
                find2(pos + 1u, true, nlen, ndist);
                bool next_is_rep = (ndist == rep0 || ndist == rep1);
                bool cur_is_rep  = (mdist == rep0 || mdist == rep1);
                if (nlen > mlen + (cur_is_rep && !next_is_rep ? 1u : 0u)) {
                    pos++;
                    mlen = nlen;
                    mdist = ndist;
                }
            }

            emit_seq(src + anchor, pos - anchor, mlen, mdist);
            matched_bytes += mlen;
            uint32_t end_pos = pos + mlen;
            if (end_pos < mflimit) {
                insert2(pos + 1u);
                if (pos + 2u < end_pos) insert2(pos + 2u);
                if (end_pos >= 2u && end_pos - 2u > pos + 2u) insert2(end_pos - 2u);
            }
            pos = end_pos;
            anchor = pos;
        }
    } else {
        // ── Level 6 (--high): 4-way 32K エポックハッシュ + 2-step Lazy + 末尾バイト弾き + 挿入上限 ──
        uint32_t ep;
        uint32_t* __restrict ht = ts.acquire_high(ep);

        auto insert4 = [&](uint32_t p) {
            uint32_t idx = hx_hash4(hx_read32(src + p), HxLzThreadState::HBITS_HIGH) << 2u;
            ht[idx + 3u] = ht[idx + 2u];
            ht[idx + 2u] = ht[idx + 1u];
            ht[idx + 1u] = ht[idx];
            ht[idx]      = ep + p;
        };
        auto match_score = [&](uint32_t ml, uint32_t d) -> int32_t {
            if (ml < HX_MIN_MATCH) return -1000;
            int32_t cost_bits = (d == rep0) ? 2 : (d == rep1 || d == rep2) ? 4 : (int32_t)(31 - __builtin_clz(d)) + 5;
            return (int32_t)(ml * 7u) - cost_bits;
        };
        auto find4 = [&](uint32_t p, uint32_t& out_len, uint32_t& out_dist) {
            uint64_t cur8 = hx_read64(src + p);
            uint32_t cur4 = (uint32_t)cur8;
            out_len = 0u; out_dist = 0u;
            int32_t best_sc = -1000;
            uint32_t reps[3] = { rep0, rep1, rep2 };
            for (int r = 0; r < 3; ++r) {
                uint32_t d = reps[r];
                if (p >= d) {
                    uint64_t rd = hx_read64(src + p - d) ^ cur8;
                    if ((uint32_t)rd == 0u) {
                        uint32_t ml = hx_match_len_from_diff8(rd, src + p - d, src + p, len - p);
                        int32_t sc = match_score(ml, d);
                        if (sc > best_sc) { best_sc = sc; out_len = ml; out_dist = d; }
                    }
                }
            }
            uint32_t idx = hx_hash4(cur4, HxLzThreadState::HBITS_HIGH) << 2u;
            uint32_t c0 = ht[idx], c1 = ht[idx + 1u], c2 = ht[idx + 2u], c3 = ht[idx + 3u];
            ht[idx + 3u] = c2; ht[idx + 2u] = c1; ht[idx + 1u] = c0; ht[idx] = ep + p;

            if (out_len >= 64u) return;

            uint32_t rels[4] = { c0 - ep, c1 - ep, c2 - ep, c3 - ep };
            for (int k = 0; k < 4; ++k) {
                uint32_t rel = rels[k];
                if (rel < p) {
                    // 末尾バイト弾き (既に 8B 以上の長マッチがあるときのみ即棄却)
                    if (out_len >= 8u && p + out_len - 1u < len && src[rel + out_len - 1u] != src[p + out_len - 1u]) {
                        continue;
                    }
                    uint64_t diff = hx_read64(src + rel) ^ cur8;
                    if ((uint32_t)diff == 0u) {
                        uint32_t ml = hx_match_len_from_diff8(diff, src + rel, src + p, len - p);
                        uint32_t d  = p - rel;
                        int32_t sc = match_score(ml, d);
                        if (sc > best_sc) { best_sc = sc; out_len = ml; out_dist = d; }
                    }
                }
            }
        };

        insert4(0u);
        while (pos < mflimit) {
            uint32_t mlen, mdist;
            find4(pos, mlen, mdist);
            if (mlen < HX_MIN_MATCH) { pos++; continue; }

            for (int lazy = 0; lazy < 2 && pos + 1u < mflimit && mlen < 128u; ++lazy) {
                uint32_t nlen, ndist;
                find4(pos + 1u, nlen, ndist);
                if (match_score(nlen, ndist) > match_score(mlen, mdist) + 5) {
                    pos++;
                    mlen = nlen;
                    mdist = ndist;
                } else {
                    break;
                }
            }

            emit_seq(src + anchor, pos - anchor, mlen, mdist);
            matched_bytes += mlen;
            uint32_t end_pos = pos + mlen;
            uint32_t ins_lim = std::min(end_pos, mflimit);
            uint32_t ins_end = std::min(ins_lim, pos + 24u);
            for (uint32_t q = pos + 1u; q < ins_end; ++q) insert4(q);
            if (ins_lim >= 2u && ins_lim - 2u >= ins_end) insert4(ins_lim - 2u);
            if (ins_lim >= 1u && ins_lim - 1u >= ins_end) insert4(ins_lim - 1u);
            pos = end_pos;
            anchor = pos;
        }
    }

    uint32_t tail = len - anchor;
    if (tail > 0u) {
        memcpy(lit_p, src + anchor, tail);
        lit_p += tail;
    }
    out.tail_lit_size = tail;
    out.lit_size      = (uint32_t)(lit_p - lit_base);
    out.seq_count     = (uint32_t)(ll_p - ll_base);

    if (bit_cnt > 0u) {
        *ex_p++ = (uint8_t)(bit_acc & 0xFFu);
    }
    out.extra_size = (uint32_t)(ex_p - ex_base);

    if (out.seq_count == 0u || matched_bytes < (len / 32u)) {
        return false;
    }
    return true;
}

// ─── 4. 超高速 ARM64 分離ストリーム LZ デコーダ ─────────────────────────────────
// 1-load 64-bit extra bits + 高速マッチ展開 + 厳密境界検証
static inline bool hx_lz_decode_block(
    const HxLzBlockMeta& meta,
    const uint8_t* __restrict lit,
    const uint8_t* __restrict ll_c,
    const uint8_t* __restrict ml_c,
    const uint8_t* __restrict of_c,
    const uint8_t* __restrict extra_bits, // 末尾に 8 バイトの安全余白あり
    uint32_t                  extra_bytes,
    uint8_t*       __restrict dst,
    uint32_t                  raw_size
) {
    uint32_t rep0 = 1u, rep1 = 4u, rep2 = 8u;
    uint32_t out_pos = 0u;
    uint32_t lit_pos = 0u;
    uint64_t bit_pos = 0ull;
    const uint64_t max_bits = (uint64_t)extra_bytes * 8ull;

    const uint32_t n_seq = meta.num_seqs;
    for (uint32_t i = 0u; i < n_seq; ++i) {
        uint8_t lc = ll_c[i];
        uint8_t mc = ml_c[i];
        uint8_t oc = of_c[i];
        if (__builtin_expect(lc >= HX_LEN_CODES || mc >= HX_LEN_CODES || oc >= HX_OF_CODES, 0))
            return false;

        uint32_t ll_eb = HX_LEN_EB[lc];
        uint32_t ml_eb = HX_LEN_EB[mc];
        uint32_t of_eb = (oc >= 3u) ? (uint32_t)(oc - 3u) : 0u;
        uint32_t tot_eb = ll_eb + ml_eb + of_eb;

        uint32_t ll = HX_LEN_BASE[lc];
        uint32_t ml = HX_LEN_BASE[mc] + HX_MIN_MATCH;
        uint32_t dist;

        if (tot_eb > 0u) {
            if (__builtin_expect(bit_pos + tot_eb > max_bits, 0)) return false;
            uint64_t w;
            memcpy(&w, extra_bits + (bit_pos >> 3u), 8u);
            w >>= (bit_pos & 7u);
            bit_pos += tot_eb;

            ll += (uint32_t)(w & ((1ull << ll_eb) - 1ull)); w >>= ll_eb;
            ml += (uint32_t)(w & ((1ull << ml_eb) - 1ull)); w >>= ml_eb;

            if (oc == 0u) {
                dist = rep0;
            } else if (oc == 1u) {
                dist = rep1; std::swap(rep0, rep1);
            } else if (oc == 2u) {
                dist = rep2; uint32_t tmp = rep2; rep2 = rep1; rep1 = rep0; rep0 = tmp;
            } else {
                dist = (1u << of_eb) + (uint32_t)(w & ((1ull << of_eb) - 1ull));
                rep2 = rep1; rep1 = rep0; rep0 = dist;
            }
        } else {
            if (oc == 0u) {
                dist = rep0;
            } else if (oc == 1u) {
                dist = rep1; std::swap(rep0, rep1);
            } else if (oc == 2u) {
                dist = rep2; uint32_t tmp = rep2; rep2 = rep1; rep1 = rep0; rep0 = tmp;
            } else {
                dist = 1u;
                rep2 = rep1; rep1 = rep0; rep0 = 1u;
            }
        }

        // リテラルコピー
        if (__builtin_expect(lit_pos + ll > meta.lit_size || out_pos + ll > raw_size, 0))
            return false;
        if (ll > 0u) {
            memcpy(dst + out_pos, lit + lit_pos, ll);
            lit_pos += ll;
            out_pos += ll;
        }

        // マッチコピー
        if (__builtin_expect(dist == 0u || dist > out_pos || out_pos + ml > raw_size, 0))
            return false;

        uint8_t*       dp = dst + out_pos;
        const uint8_t* sp = dp - dist;
        if (dist >= ml) {
            memcpy(dp, sp, ml);
        } else if (dist == 1u) {
            memset(dp, sp[0], ml);
        } else {
            uint32_t copied = 0u;
            while (copied < ml) {
                uint32_t chunk = std::min(dist, ml - copied);
                memcpy(dp + copied, sp + copied, chunk);
                copied += chunk;
                if (dist < 64u && copied == dist) {
                    while (copied < ml) {
                        uint32_t dbl = std::min(copied, ml - copied);
                        memcpy(dp + copied, dp, dbl);
                        copied += dbl;
                    }
                    break;
                }
            }
        }
        out_pos += ml;
    }

    uint32_t tail = meta.tail_lit_size;
    if (lit_pos + tail != meta.lit_size || out_pos + tail != raw_size) return false;
    if (tail > 0u) memcpy(dst + out_pos, lit + lit_pos, tail);
    return true;
}
