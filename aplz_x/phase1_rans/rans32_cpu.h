// rans32_cpu.h — APLZ-X Phase 1: 32-way Coalesced rANS CPU リファレンス & ARM64 NEON 高速実装
#pragma once

#include "rans32_format.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <vector>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// ─── 1. CPU 側テーブル構築 (Metal rans32_build_tables とビット完全一致) ─────────
static inline void rans32_cpu_build_tables(
    const uint8_t*   blk,
    uint32_t         raw_size,
    Rans32EncSym*    out_enc_sym,  // [256]
    uint32_t*        out_dec_lut,  // [4096]
    Rans32BlockMeta* out_meta
) {
    uint32_t cnt[RANS32_N_SYMBOLS] = {};
    for (uint32_t i = 0; i < raw_size; ++i) {
        cnt[blk[i]]++;
    }

    uint32_t n_nz = 0, first_nz = 0, best = 0, best_cnt = 0;
    for (uint32_t s = 0; s < RANS32_N_SYMBOLS; ++s) {
        if (cnt[s] > 0) {
            if (n_nz == 0) first_nz = s;
            n_nz++;
            if (cnt[s] > best_cnt) { best_cnt = cnt[s]; best = s; }
        }
    }

    out_meta->raw_size  = raw_size;
    out_meta->n_nonzero = n_nz;
    out_meta->rle_byte  = first_nz;

    if (raw_size < RANS32_MIN_RANS_LEN && n_nz != 1u) {
        out_meta->mode = RANS32_MODE_RAW;
        return;
    }
    if (n_nz == 1u) {
        out_meta->mode = RANS32_MODE_RLE;
        return;
    }
    if (n_nz == 0u) {
        out_meta->mode = RANS32_MODE_RAW;
        return;
    }

    out_meta->mode = RANS32_MODE_RANS;

    uint16_t freq[RANS32_N_SYMBOLS] = {};
    uint32_t assigned = 0;
    for (uint32_t s = 0; s < RANS32_N_SYMBOLS; ++s) {
        if (cnt[s] == 0) {
            freq[s] = 0;
        } else {
            uint32_t f = (uint32_t)(((uint64_t)cnt[s] * RANS32_PROB_SCALE) / raw_size);
            if (f == 0u) f = 1u;
            if (f >= RANS32_PROB_SCALE) f = RANS32_PROB_SCALE - 1u;
            freq[s] = (uint16_t)f;
            assigned += f;
        }
    }

    int32_t diff = (int32_t)RANS32_PROB_SCALE - (int32_t)assigned;
    int32_t best_new = (int32_t)freq[best] + diff;
    if (best_new >= 1 && best_new <= (int32_t)(RANS32_PROB_SCALE - 1u)) {
        freq[best] = (uint16_t)best_new;
        assigned = RANS32_PROB_SCALE;
    } else {
        while (assigned < RANS32_PROB_SCALE) {
            for (uint32_t s = 0; s < RANS32_N_SYMBOLS && assigned < RANS32_PROB_SCALE; ++s) {
                if (freq[s] > 0 && freq[s] < RANS32_PROB_SCALE - 1u) {
                    freq[s]++;
                    assigned++;
                }
            }
        }
        while (assigned > RANS32_PROB_SCALE) {
            uint32_t max_s = 0, max_f = 0;
            for (uint32_t s = 0; s < RANS32_N_SYMBOLS; ++s) {
                if (freq[s] > max_f) { max_f = freq[s]; max_s = s; }
            }
            if (max_f <= 1u) break;
            uint32_t dec = std::min(assigned - RANS32_PROB_SCALE, max_f - 1u);
            freq[max_s] -= (uint16_t)dec;
            assigned -= dec;
        }
    }

    uint16_t cum = 0;
    for (uint32_t s = 0; s < RANS32_N_SYMBOLS; ++s) {
        uint16_t f = freq[s];
        Rans32EncSym es;
        es.freq  = f;
        es.cdf   = cum;
        es.x_max = (uint32_t)f << 20u;
        rans32_compute_rcp(f, &es.rcp_mul, &es.rcp_shift);
        out_enc_sym[s] = es;

        if (out_dec_lut && f > 0) {
            for (uint32_t bias = 0; bias < f; ++bias) {
                out_dec_lut[cum + bias] = rans32_pack_lut(s, f, bias);
            }
        }
        cum += f;
    }
}

// ─── 2. 頻度テーブルのコンパクトシリアライズ / デシリアライズ ────────────────────
static inline void rans32_append_freq_table(std::vector<uint8_t>& out, const Rans32EncSym* enc_sym) {
    uint32_t n_nz = 0;
    for (uint32_t s = 0; s < RANS32_N_SYMBOLS; ++s) {
        if (enc_sym[s].freq > 0) n_nz++;
    }
    uint8_t kind = (n_nz < RANS32_SPARSE_THRESH) ? RANS32_FREQ_SPARSE : RANS32_FREQ_DENSE12;
    out.push_back(kind);
    out.push_back((uint8_t)(n_nz - 1u));

    if (kind == RANS32_FREQ_SPARSE) {
        size_t base = out.size();
        out.resize(base + (size_t)n_nz * 3u);
        uint8_t* p = out.data() + base;
        for (uint32_t s = 0; s < RANS32_N_SYMBOLS; ++s) {
            uint16_t f = enc_sym[s].freq;
            if (f > 0) {
                p[0] = (uint8_t)s;
                p[1] = (uint8_t)(f & 0xFFu);
                p[2] = (uint8_t)((f >> 8u) & 0xFFu);
                p += 3;
            }
        }
    } else {
        size_t base = out.size();
        out.resize(base + 384u);
        uint8_t* p = out.data() + base;
        for (uint32_t s = 0; s < RANS32_N_SYMBOLS; s += 2u) {
            uint32_t f0 = enc_sym[s].freq;
            uint32_t f1 = enc_sym[s + 1u].freq;
            p[0] = (uint8_t)(f0 & 0xFFu);
            p[1] = (uint8_t)(((f0 >> 8u) & 0x0Fu) | ((f1 & 0x0Fu) << 4u));
            p[2] = (uint8_t)((f1 >> 4u) & 0xFFu);
            p += 3;
        }
    }
}

static inline bool rans32_read_freq_table_to_lut(
    const uint8_t*& p,
    const uint8_t*  end,
    uint32_t*       out_dec_lut  // [4096]
) {
    if ((size_t)(end - p) < 2u) return false;
    uint8_t kind = p[0];
    uint32_t n_nz_expected = (uint32_t)p[1] + 1u;
    p += 2;
    if (n_nz_expected < 2u || n_nz_expected > RANS32_N_SYMBOLS) return false;

    uint16_t freq[RANS32_N_SYMBOLS] = {};
    if (kind == RANS32_FREQ_SPARSE) {
        if ((size_t)(end - p) < (size_t)n_nz_expected * 3u) return false;
        for (uint32_t i = 0; i < n_nz_expected; ++i) {
            uint8_t sym = p[0];
            uint16_t f  = (uint16_t)p[1] | ((uint16_t)p[2] << 8u);
            p += 3;
            if (f == 0u || f >= RANS32_PROB_SCALE || freq[sym] != 0u) return false;
            freq[sym] = f;
        }
    } else if (kind == RANS32_FREQ_DENSE12) {
        if ((size_t)(end - p) < 384u) return false;
        uint32_t nz = 0;
        for (uint32_t s = 0; s < RANS32_N_SYMBOLS; s += 2u) {
            uint16_t f0 = (uint16_t)p[0] | (((uint16_t)p[1] & 0x0Fu) << 8u);
            uint16_t f1 = ((uint16_t)p[1] >> 4u) | ((uint16_t)p[2] << 4u);
            p += 3;
            if (f0 >= RANS32_PROB_SCALE || f1 >= RANS32_PROB_SCALE) return false;
            freq[s]      = f0;
            freq[s + 1u] = f1;
            if (f0 > 0) nz++;
            if (f1 > 0) nz++;
        }
        if (nz != n_nz_expected) return false;
    } else {
        return false;
    }

    uint32_t sum = 0;
    for (uint32_t s = 0; s < RANS32_N_SYMBOLS; ++s) {
        uint16_t f = freq[s];
        if (sum + f > RANS32_PROB_SCALE) return false;
        for (uint32_t bias = 0; bias < f; ++bias) {
            out_dec_lut[sum + bias] = rans32_pack_lut(s, f, bias);
        }
        sum += f;
    }
    return sum == RANS32_PROB_SCALE;
}

// ─── 3. Scalar リファレンス エンコーダ (1 タイル = 最大 64 KiB) ──────────────────
// GPU rans32_encode と 1 ビット残らず同一の words[] / final_states[] を生成する。
static inline void rans32_encode_tile_scalar(
    const uint8_t*      tile_in,
    uint32_t            tile_len,
    const Rans32EncSym* enc_sym,
    uint16_t*           out_words,        // [RANS32_TILE_MAX_WORDS]
    uint32_t*           out_final_states, // [32]
    Rans32TileMeta*     out_meta
) {
    if (tile_len < RANS32_MIN_RANS_LEN) {
        out_meta->mode = RANS32_MODE_RAW;
        out_meta->rle_byte = 0;
        out_meta->raw_size = tile_len;
        out_meta->word_count = tile_len;
        return;
    }

    const uint32_t max_words = std::min(RANS32_TILE_MAX_WORDS, (tile_len - 129u) >> 1u);
    const uint32_t steps     = (tile_len + RANS32_LANES - 1u) >> 5u;

    uint32_t st[RANS32_LANES];
    for (uint32_t l = 0; l < RANS32_LANES; ++l) st[l] = RANS32_L;

    uint32_t wp = 0;
    for (int step = (int)steps - 1; step >= 0; --step) {
        const uint32_t base = (uint32_t)step << 5u;
        const uint32_t n_active = std::min(RANS32_LANES, tile_len - base);

        uint32_t step_words_cnt = 0;
        uint16_t step_words[RANS32_LANES];
        for (uint32_t l = 0; l < n_active; ++l) {
            const Rans32EncSym& es = enc_sym[tile_in[base + l]];
            uint32_t x = st[l];
            if (x >= es.x_max) {
                step_words[step_words_cnt++] = (uint16_t)(x & 0xFFFFu);
                x >>= 16u;
            }
            uint32_t f = es.freq;
            uint32_t q;
            if (f == 1u) {
                q = x;
            } else {
                uint32_t t = (uint32_t)(((uint64_t)x * (uint64_t)es.rcp_mul) >> 32u);
                q = (t + ((x - t) >> 1u)) >> es.rcp_shift;
            }
            uint32_t r = x - q * f;
            st[l] = (q << RANS32_PROB_BITS) + (uint32_t)es.cdf + r;
        }

        if (wp + step_words_cnt > max_words) {
            out_meta->mode = RANS32_MODE_RAW;
            out_meta->rle_byte = 0;
            out_meta->raw_size = tile_len;
            out_meta->word_count = tile_len;
            return;
        }
        memcpy(out_words + wp, step_words, (size_t)step_words_cnt * sizeof(uint16_t));
        wp += step_words_cnt;
    }

    memcpy(out_final_states, st, RANS32_LANES * sizeof(uint32_t));
    out_meta->mode = RANS32_MODE_RANS;
    out_meta->rle_byte = 0;
    out_meta->raw_size = tile_len;
    out_meta->word_count = wp;
}

// ─── 3b. 超高速 4-way ヒストグラム + ブランチレス 32-way rANS CPU エンコーダ ──────
// GPU hx_rans_batch_build + hx_rans_batch_encode と 1 ビット残らず完全一致!
static inline void rans32_cpu_build_tables_fast(
    const uint8_t* __restrict blk,
    uint32_t                  raw_size,
    Rans32EncSym*  __restrict out_enc_sym,  // [256]
    Rans32BlockMeta*          out_meta
) {
    uint32_t c0[RANS32_N_SYMBOLS] = {};
    uint32_t c1[RANS32_N_SYMBOLS] = {};
    uint32_t c2[RANS32_N_SYMBOLS] = {};
    uint32_t c3[RANS32_N_SYMBOLS] = {};
    uint32_t i = 0u;
    for (; i + 4u <= raw_size; i += 4u) {
        uint32_t w;
        memcpy(&w, blk + i, 4u);
        c0[ w         & 0xFFu]++;
        c1[(w >>  8u) & 0xFFu]++;
        c2[(w >> 16u) & 0xFFu]++;
        c3[(w >> 24u) & 0xFFu]++;
    }
    for (; i < raw_size; ++i) c0[blk[i]]++;

    uint32_t cnt[RANS32_N_SYMBOLS];
    uint32_t n_nz = 0u, first_nz = 0u, best = 0u, best_cnt = 0u;
    for (uint32_t s = 0u; s < RANS32_N_SYMBOLS; ++s) {
        uint32_t c = c0[s] + c1[s] + c2[s] + c3[s];
        cnt[s] = c;
        if (c > 0u) {
            if (n_nz == 0u) first_nz = s;
            n_nz++;
            if (c > best_cnt) { best_cnt = c; best = s; }
        }
    }

    out_meta->raw_size  = raw_size;
    out_meta->n_nonzero = n_nz;
    out_meta->rle_byte  = first_nz;

    if (n_nz == 1u) {
        out_meta->mode = RANS32_MODE_RLE;
        return;
    }
    if (raw_size < RANS32_MIN_RANS_LEN || n_nz == 0u) {
        out_meta->mode = RANS32_MODE_RAW;
        return;
    }

    out_meta->mode = RANS32_MODE_RANS;

    uint16_t freq[RANS32_N_SYMBOLS] = {};
    uint32_t assigned = 0u;
    for (uint32_t s = 0u; s < RANS32_N_SYMBOLS; ++s) {
        uint32_t c = cnt[s];
        if (c > 0u) {
            uint32_t f = (uint32_t)(((uint64_t)c * RANS32_PROB_SCALE) / raw_size);
            if (f == 0u) f = 1u;
            if (f >= RANS32_PROB_SCALE) f = RANS32_PROB_SCALE - 1u;
            freq[s] = (uint16_t)f;
            assigned += f;
        }
    }

    int32_t diff = (int32_t)RANS32_PROB_SCALE - (int32_t)assigned;
    int32_t best_new = (int32_t)freq[best] + diff;
    if (best_new >= 1 && best_new <= (int32_t)(RANS32_PROB_SCALE - 1u)) {
        freq[best] = (uint16_t)best_new;
        assigned = RANS32_PROB_SCALE;
    } else {
        while (assigned < RANS32_PROB_SCALE) {
            for (uint32_t s = 0u; s < RANS32_N_SYMBOLS && assigned < RANS32_PROB_SCALE; ++s) {
                if (freq[s] > 0u && freq[s] < RANS32_PROB_SCALE - 1u) { freq[s]++; assigned++; }
            }
        }
        while (assigned > RANS32_PROB_SCALE) {
            uint32_t max_s = 0u, max_f = 0u;
            for (uint32_t s = 0u; s < RANS32_N_SYMBOLS; ++s) {
                if (freq[s] > max_f) { max_f = freq[s]; max_s = s; }
            }
            if (max_f <= 1u) break;
            uint32_t dec = std::min(assigned - RANS32_PROB_SCALE, max_f - 1u);
            freq[max_s] -= (uint16_t)dec;
            assigned -= dec;
        }
    }

    uint16_t cum = 0u;
    for (uint32_t s = 0u; s < RANS32_N_SYMBOLS; ++s) {
        uint16_t f = freq[s];
        Rans32EncSym es;
        es.freq  = f;
        es.cdf   = cum;
        es.x_max = (uint32_t)f << 20u;
        rans32_compute_rcp(f, &es.rcp_mul, &es.rcp_shift);
        out_enc_sym[s] = es;
        cum += f;
    }
}

static inline void rans32_encode_tile_fast(
    const uint8_t*      __restrict tile_in,
    uint32_t                       tile_len,
    const Rans32EncSym* __restrict enc_sym,
    uint16_t*           __restrict out_words,        // [RANS32_TILE_MAX_WORDS]
    uint32_t*           __restrict out_final_states, // [32]
    Rans32TileMeta*                out_meta
) {
    if (tile_len < RANS32_MIN_RANS_LEN) {
        out_meta->mode = RANS32_MODE_RAW;
        out_meta->rle_byte = 0u;
        out_meta->raw_size = tile_len;
        out_meta->word_count = tile_len;
        return;
    }

    const uint32_t max_words = std::min(RANS32_TILE_MAX_WORDS, (tile_len - 129u) >> 1u);
    const uint32_t steps     = (tile_len + RANS32_LANES - 1u) >> 5u;

    uint32_t st[RANS32_LANES];
    for (uint32_t l = 0u; l < RANS32_LANES; ++l) st[l] = RANS32_L;

    uint32_t wp = 0u;

    // 末尾の部分ステップ (rem = tile_len & 31) があれば先に処理
    int step = (int)steps - 1;
    const uint32_t rem = tile_len & 31u;
    if (rem > 0u) {
        const uint32_t base = (uint32_t)step << 5u;
        uint16_t step_words[RANS32_LANES];
        uint32_t cnt = 0u;
        for (uint32_t l = 0u; l < rem; ++l) {
            const Rans32EncSym es = enc_sym[tile_in[base + l]];
            uint32_t x = st[l];
            uint32_t ren = (x >= es.x_max);
            step_words[cnt] = (uint16_t)x;
            cnt += ren;
            x >>= (ren << 4u);
            uint32_t f = es.freq;
            uint32_t t = (uint32_t)(((uint64_t)x * (uint64_t)es.rcp_mul) >> 32u);
            uint32_t q_div = (t + ((x - t) >> 1u)) >> es.rcp_shift;
            uint32_t q = (f == 1u) ? x : q_div;
            uint32_t r = x - q * f;
            st[l] = (q << RANS32_PROB_BITS) + (uint32_t)es.cdf + r;
        }
        if (wp + cnt > max_words) {
            out_meta->mode = RANS32_MODE_RAW;
            out_meta->rle_byte = 0u;
            out_meta->raw_size = tile_len;
            out_meta->word_count = tile_len;
            return;
        }
        memcpy(out_words + wp, step_words, (size_t)cnt * sizeof(uint16_t));
        wp += cnt;
        step--;
    }

    // フル 32 レーン・メインループ: 4 レーン並列 ILP + プレフィクス加算ブランチレス直書き
    // (32 レーン直列の dst_w 依存チェーンを 1/4 に短縮し、Apple M4 の 4 ALU + 2 MUL を飽和!)
    for (; step >= 0; --step) {
        const uint8_t* p = tile_in + ((uint32_t)step << 5u);
        if (__builtin_expect(wp + RANS32_LANES <= max_words, 1)) {
            uint16_t* dst_w = out_words + wp;
            for (uint32_t l = 0u; l < RANS32_LANES; l += 4u) {
                const Rans32EncSym e0 = enc_sym[p[l + 0u]];
                const Rans32EncSym e1 = enc_sym[p[l + 1u]];
                const Rans32EncSym e2 = enc_sym[p[l + 2u]];
                const Rans32EncSym e3 = enc_sym[p[l + 3u]];

                uint32_t x0 = st[l + 0u];
                uint32_t x1 = st[l + 1u];
                uint32_t x2 = st[l + 2u];
                uint32_t x3 = st[l + 3u];

                uint32_t r0 = (x0 >= e0.x_max);
                uint32_t r1 = (x1 >= e1.x_max);
                uint32_t r2 = (x2 >= e2.x_max);
                uint32_t r3 = (x3 >= e3.x_max);

                uint32_t o1 = r0;
                uint32_t o2 = o1 + r1;
                uint32_t o3 = o2 + r2;
                uint32_t tot = o3 + r3;

                dst_w[0]  = (uint16_t)x0;
                dst_w[o1] = (uint16_t)x1;
                dst_w[o2] = (uint16_t)x2;
                dst_w[o3] = (uint16_t)x3;
                dst_w += tot;

                x0 >>= (r0 << 4u);
                x1 >>= (r1 << 4u);
                x2 >>= (r2 << 4u);
                x3 >>= (r3 << 4u);

                uint32_t f0 = e0.freq, f1 = e1.freq, f2 = e2.freq, f3 = e3.freq;
                uint32_t t0 = (uint32_t)(((uint64_t)x0 * (uint64_t)e0.rcp_mul) >> 32u);
                uint32_t t1 = (uint32_t)(((uint64_t)x1 * (uint64_t)e1.rcp_mul) >> 32u);
                uint32_t t2 = (uint32_t)(((uint64_t)x2 * (uint64_t)e2.rcp_mul) >> 32u);
                uint32_t t3 = (uint32_t)(((uint64_t)x3 * (uint64_t)e3.rcp_mul) >> 32u);

                uint32_t q0 = (f0 == 1u) ? x0 : ((t0 + ((x0 - t0) >> 1u)) >> e0.rcp_shift);
                uint32_t q1 = (f1 == 1u) ? x1 : ((t1 + ((x1 - t1) >> 1u)) >> e1.rcp_shift);
                uint32_t q2 = (f2 == 1u) ? x2 : ((t2 + ((x2 - t2) >> 1u)) >> e2.rcp_shift);
                uint32_t q3 = (f3 == 1u) ? x3 : ((t3 + ((x3 - t3) >> 1u)) >> e3.rcp_shift);

                st[l + 0u] = (q0 << RANS32_PROB_BITS) + (uint32_t)e0.cdf + (x0 - q0 * f0);
                st[l + 1u] = (q1 << RANS32_PROB_BITS) + (uint32_t)e1.cdf + (x1 - q1 * f1);
                st[l + 2u] = (q2 << RANS32_PROB_BITS) + (uint32_t)e2.cdf + (x2 - q2 * f2);
                st[l + 3u] = (q3 << RANS32_PROB_BITS) + (uint32_t)e3.cdf + (x3 - q3 * f3);
            }
            wp = (uint32_t)(dst_w - out_words);
        } else {
            uint16_t step_words[RANS32_LANES];
            uint32_t cnt = 0u;
            for (uint32_t l = 0u; l < RANS32_LANES; ++l) {
                const Rans32EncSym es = enc_sym[p[l]];
                uint32_t x = st[l];
                uint32_t ren = (x >= es.x_max);
                step_words[cnt] = (uint16_t)x;
                cnt += ren;
                x >>= (ren << 4u);
                uint32_t f = es.freq;
                uint32_t t = (uint32_t)(((uint64_t)x * (uint64_t)es.rcp_mul) >> 32u);
                uint32_t q_div = (t + ((x - t) >> 1u)) >> es.rcp_shift;
                uint32_t q = (f == 1u) ? x : q_div;
                uint32_t r = x - q * f;
                st[l] = (q << RANS32_PROB_BITS) + (uint32_t)es.cdf + r;
            }
            if (wp + cnt > max_words) {
                out_meta->mode = RANS32_MODE_RAW;
                out_meta->rle_byte = 0u;
                out_meta->raw_size = tile_len;
                out_meta->word_count = tile_len;
                return;
            }
            memcpy(out_words + wp, step_words, (size_t)cnt * sizeof(uint16_t));
            wp += cnt;
        }
    }

    memcpy(out_final_states, st, RANS32_LANES * sizeof(uint32_t));
    out_meta->mode = RANS32_MODE_RANS;
    out_meta->rle_byte = 0u;
    out_meta->raw_size = tile_len;
    out_meta->word_count = wp;
}

// ─── 4. Scalar リファレンス デコーダ (厳密オラクル) ────────────────────────────
static inline bool rans32_decode_tile_scalar(
    const uint16_t* words,
    uint32_t        word_count,
    const uint32_t* final_states,
    const uint32_t* dec_lut,
    uint8_t*        out_tile,
    uint32_t        tile_len
) {
    uint32_t st[RANS32_LANES];
    for (uint32_t l = 0; l < RANS32_LANES; ++l) {
        st[l] = final_states[l];
        if (st[l] < RANS32_L) return false;
    }

    uint32_t rp = word_count;
    const uint32_t steps = (tile_len + RANS32_LANES - 1u) >> 5u;

    for (uint32_t step = 0; step < steps; ++step) {
        const uint32_t base = step << 5u;
        const uint32_t n_active = std::min(RANS32_LANES, tile_len - base);

        uint32_t mask = 0;
        for (uint32_t l = 0; l < n_active; ++l) {
            uint32_t x = st[l];
            uint32_t entry = dec_lut[x & RANS32_PROB_MASK];
            out_tile[base + l] = (uint8_t)(entry & 0xFFu);
            uint32_t freq = (entry >> 8u) & RANS32_PROB_MASK;
            uint32_t bias = entry >> 20u;
            x = freq * (x >> RANS32_PROB_BITS) + bias;
            st[l] = x;
            if (x < RANS32_L) mask |= (1u << l);
        }

        uint32_t cnt = (uint32_t)__builtin_popcount(mask);
        if (rp < cnt) return false;
        rp -= cnt;

        const uint16_t* wp = words + rp;
        for (uint32_t l = 0; l < n_active; ++l) {
            if (mask & (1u << l)) {
                st[l] = (st[l] << 16u) | (uint32_t)(*wp++);
            }
        }
    }

    if (rp != 0u) return false;
    for (uint32_t l = 0; l < RANS32_LANES; ++l) {
        if (st[l] != RANS32_L) return false;
    }
    return true;
}

// ─── 5. ARM64 NEON 32-way レジスタ常駐・ブランチレス高速デコーダ ──────────────
// Apple Silicon の 32 本の 128-bit NEON レジスタを活かし、
// 32 レーンの rANS 状態を 8 本の uint32x4_t (v0..v7) に常駐させてスタックスピルを排除。
// 8 グループ並列の独立ポインタ + 完全ブランチレス補充で 32 B/step 展開する。
static inline bool rans32_decode_tile_neon(
    const uint16_t* __restrict words,
    uint32_t                   word_count,
    const uint32_t* __restrict final_states,
    const uint32_t* __restrict dec_lut,
    uint8_t*        __restrict out_tile,
    uint32_t                   tile_len
) {
#if !defined(__aarch64__)
    return rans32_decode_tile_scalar(words, word_count, final_states, dec_lut, out_tile, tile_len);
#else
    const uint32x4_t v_rans_l = vdupq_n_u32(RANS32_L);
    const uint32x4_t v_mask12 = vdupq_n_u32(RANS32_PROB_MASK);

    uint32x4_t st0 = vld1q_u32(final_states +  0);
    uint32x4_t st1 = vld1q_u32(final_states +  4);
    uint32x4_t st2 = vld1q_u32(final_states +  8);
    uint32x4_t st3 = vld1q_u32(final_states + 12);
    uint32x4_t st4 = vld1q_u32(final_states + 16);
    uint32x4_t st5 = vld1q_u32(final_states + 20);
    uint32x4_t st6 = vld1q_u32(final_states + 24);
    uint32x4_t st7 = vld1q_u32(final_states + 28);

    uint32x4_t bad_init = vorrq_u32(
        vorrq_u32(vorrq_u32(vcltq_u32(st0, v_rans_l), vcltq_u32(st1, v_rans_l)),
                  vorrq_u32(vcltq_u32(st2, v_rans_l), vcltq_u32(st3, v_rans_l))),
        vorrq_u32(vorrq_u32(vcltq_u32(st4, v_rans_l), vcltq_u32(st5, v_rans_l)),
                  vorrq_u32(vcltq_u32(st6, v_rans_l), vcltq_u32(st7, v_rans_l))));
    if (vmaxvq_u32(bad_init) != 0u) return false;

    uint32_t rp = word_count;
    const uint32_t full_steps = tile_len >> 5u;

    #define RANS32_NEON_STEP_GROUP(ST, OUT_PTR, NEED_VEC)                       \
        uint32x4_t _idx_##ST = vandq_u32(ST, v_mask12);                         \
        uint32_t _e0_##ST = dec_lut[vgetq_lane_u32(_idx_##ST, 0)];              \
        uint32_t _e1_##ST = dec_lut[vgetq_lane_u32(_idx_##ST, 1)];              \
        uint32_t _e2_##ST = dec_lut[vgetq_lane_u32(_idx_##ST, 2)];              \
        uint32_t _e3_##ST = dec_lut[vgetq_lane_u32(_idx_##ST, 3)];              \
        uint32_t _packed_##ST = (_e0_##ST & 0xFFu)                              \
                              | ((_e1_##ST & 0xFFu) <<  8u)                     \
                              | ((_e2_##ST & 0xFFu) << 16u)                     \
                              | ((_e3_##ST & 0xFFu) << 24u);                    \
        memcpy((OUT_PTR), &_packed_##ST, 4u);                                   \
        uint32x4_t _ent_##ST = {_e0_##ST, _e1_##ST, _e2_##ST, _e3_##ST};        \
        uint32x4_t _frq_##ST = vandq_u32(vshrq_n_u32(_ent_##ST, 8), v_mask12);  \
        uint32x4_t _bia_##ST = vshrq_n_u32(_ent_##ST, 20);                      \
        ST = vmlaq_u32(_bia_##ST, _frq_##ST, vshrq_n_u32(ST, 12));              \
        uint32x4_t NEED_VEC = vshrq_n_u32(vcltq_u32(ST, v_rans_l), 31)

    #define RANS32_BRANCHLESS_REFILL4(ST, P)                                    \
        do {                                                                    \
            uint32_t _s0 = vgetq_lane_u32(ST, 0);                               \
            uint32_t _n0 = (_s0 < RANS32_L);                                    \
            uint32_t _w0 = *(P);                                                \
            _s0 = _n0 ? ((_s0 << 16u) | _w0) : _s0;                             \
            (P) += _n0;                                                         \
            uint32_t _s1 = vgetq_lane_u32(ST, 1);                               \
            uint32_t _n1 = (_s1 < RANS32_L);                                    \
            uint32_t _w1 = *(P);                                                \
            _s1 = _n1 ? ((_s1 << 16u) | _w1) : _s1;                             \
            (P) += _n1;                                                         \
            uint32_t _s2 = vgetq_lane_u32(ST, 2);                               \
            uint32_t _n2 = (_s2 < RANS32_L);                                    \
            uint32_t _w2 = *(P);                                                \
            _s2 = _n2 ? ((_s2 << 16u) | _w2) : _s2;                             \
            (P) += _n2;                                                         \
            uint32_t _s3 = vgetq_lane_u32(ST, 3);                               \
            uint32_t _n3 = (_s3 < RANS32_L);                                    \
            uint32_t _w3 = *(P);                                                \
            _s3 = _n3 ? ((_s3 << 16u) | _w3) : _s3;                             \
            (P) += _n3;                                                         \
            ST = (uint32x4_t){_s0, _s1, _s2, _s3};                              \
        } while (0)

    #define RANS32_SAFE_REFILL4(ST, P)                                          \
        do {                                                                    \
            uint32_t _s0 = vgetq_lane_u32(ST, 0);                               \
            if (_s0 < RANS32_L) _s0 = (_s0 << 16u) | (uint32_t)(*(P)++);        \
            uint32_t _s1 = vgetq_lane_u32(ST, 1);                               \
            if (_s1 < RANS32_L) _s1 = (_s1 << 16u) | (uint32_t)(*(P)++);        \
            uint32_t _s2 = vgetq_lane_u32(ST, 2);                               \
            if (_s2 < RANS32_L) _s2 = (_s2 << 16u) | (uint32_t)(*(P)++);        \
            uint32_t _s3 = vgetq_lane_u32(ST, 3);                               \
            if (_s3 < RANS32_L) _s3 = (_s3 << 16u) | (uint32_t)(*(P)++);        \
            ST = (uint32x4_t){_s0, _s1, _s2, _s3};                              \
        } while (0)

    for (uint32_t step = 0; step < full_steps; ++step) {
        uint8_t* dst = out_tile + (step << 5u);

        RANS32_NEON_STEP_GROUP(st0, dst +  0, n0);
        RANS32_NEON_STEP_GROUP(st1, dst +  4, n1);
        RANS32_NEON_STEP_GROUP(st2, dst +  8, n2);
        RANS32_NEON_STEP_GROUP(st3, dst + 12, n3);
        RANS32_NEON_STEP_GROUP(st4, dst + 16, n4);
        RANS32_NEON_STEP_GROUP(st5, dst + 20, n5);
        RANS32_NEON_STEP_GROUP(st6, dst + 24, n6);
        RANS32_NEON_STEP_GROUP(st7, dst + 28, n7);

        const uint32_t c0 = vaddvq_u32(n0);
        const uint32_t c1 = vaddvq_u32(n1);
        const uint32_t c2 = vaddvq_u32(n2);
        const uint32_t c3 = vaddvq_u32(n3);
        const uint32_t c4 = vaddvq_u32(n4);
        const uint32_t c5 = vaddvq_u32(n5);
        const uint32_t c6 = vaddvq_u32(n6);
        const uint32_t c7 = vaddvq_u32(n7);

        const uint32_t o1 = c0,      o2 = o1 + c1, o3 = o2 + c2, o4 = o3 + c3;
        const uint32_t o5 = o4 + c4, o6 = o5 + c5, o7 = o6 + c6, cnt = o7 + c7;

        if (__builtin_expect(rp < cnt, 0)) return false;
        rp -= cnt;
        if (__builtin_expect(cnt == 0u, 0)) continue;

        const uint16_t* base_p = words + rp;
        const uint16_t* p0 = base_p;      const uint16_t* p1 = base_p + o1;
        const uint16_t* p2 = base_p + o2; const uint16_t* p3 = base_p + o3;
        const uint16_t* p4 = base_p + o4; const uint16_t* p5 = base_p + o5;
        const uint16_t* p6 = base_p + o6; const uint16_t* p7 = base_p + o7;

        if (__builtin_expect(rp + 32u <= word_count, 1)) {
            RANS32_BRANCHLESS_REFILL4(st0, p0);
            RANS32_BRANCHLESS_REFILL4(st1, p1);
            RANS32_BRANCHLESS_REFILL4(st2, p2);
            RANS32_BRANCHLESS_REFILL4(st3, p3);
            RANS32_BRANCHLESS_REFILL4(st4, p4);
            RANS32_BRANCHLESS_REFILL4(st5, p5);
            RANS32_BRANCHLESS_REFILL4(st6, p6);
            RANS32_BRANCHLESS_REFILL4(st7, p7);
        } else {
            RANS32_SAFE_REFILL4(st0, p0);
            RANS32_SAFE_REFILL4(st1, p1);
            RANS32_SAFE_REFILL4(st2, p2);
            RANS32_SAFE_REFILL4(st3, p3);
            RANS32_SAFE_REFILL4(st4, p4);
            RANS32_SAFE_REFILL4(st5, p5);
            RANS32_SAFE_REFILL4(st6, p6);
            RANS32_SAFE_REFILL4(st7, p7);
        }
    }

    #undef RANS32_NEON_STEP_GROUP
    #undef RANS32_BRANCHLESS_REFILL4
    #undef RANS32_SAFE_REFILL4

    uint32_t st[RANS32_LANES];
    vst1q_u32(st +  0, st0); vst1q_u32(st +  4, st1);
    vst1q_u32(st +  8, st2); vst1q_u32(st + 12, st3);
    vst1q_u32(st + 16, st4); vst1q_u32(st + 20, st5);
    vst1q_u32(st + 24, st6); vst1q_u32(st + 28, st7);

    const uint32_t rem = tile_len & 31u;
    if (rem > 0u) {
        const uint32_t base = full_steps << 5u;
        uint32_t mask = 0;
        for (uint32_t l = 0; l < rem; ++l) {
            uint32_t x = st[l];
            uint32_t entry = dec_lut[x & RANS32_PROB_MASK];
            out_tile[base + l] = (uint8_t)(entry & 0xFFu);
            uint32_t freq = (entry >> 8u) & RANS32_PROB_MASK;
            uint32_t bias = entry >> 20u;
            x = freq * (x >> RANS32_PROB_BITS) + bias;
            st[l] = x;
            if (x < RANS32_L) mask |= (1u << l);
        }
        uint32_t cnt = (uint32_t)__builtin_popcount(mask);
        if (rp < cnt) return false;
        rp -= cnt;
        const uint16_t* wp = words + rp;
        for (uint32_t l = 0; l < rem; ++l) {
            if (mask & (1u << l)) st[l] = (st[l] << 16u) | (uint32_t)(*wp++);
        }
    }

    if (rp != 0u) return false;
    for (uint32_t l = 0; l < RANS32_LANES; ++l) {
        if (st[l] != RANS32_L) return false;
    }
    return true;
#endif
}
