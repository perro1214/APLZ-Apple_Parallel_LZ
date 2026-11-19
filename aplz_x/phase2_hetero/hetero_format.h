// hetero_format.h — APLZ-X Phase 2: UMA 異種協調エンジン 共通定義
//
// ブロック構成:
//   - 256 KiB 独立ブロック単位 (LZ ウィンドウ 256 KiB)
//   - OpenZL 型軽量前処理 (NONE / SHUFFLE2 / SHUFFLE4 / ARM64_BCJ)
//   - 4 分離ストリーム LZ (Literals / LL / ML / OF + 64-bit ExtraBits)
//   - 32-way SIMD-group Coalesced rANS (可変長ストリーム一括 GPU バッチ処理)
//   - ARM64 ハードウェア CRC32C 検証
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

// ─── パイプライン定数 (hetero_kernels.metal と完全一致) ────────────────────────
static const uint32_t HX_BLOCK_SIZE      = 262144u;   // 256 KiB 独立ブロック
static const uint32_t HX_TILE_SIZE       = 65536u;    // 64 KiB rANS タイル
static const uint32_t HX_MAX_TILES_STR   = 4u;        // 1 ストリーム最大 4 タイル (256 KiB)
static const uint32_t HX_LANES           = 32u;       // Apple GPU SIMD-group = 32
static const uint32_t HX_TG_SIZE         = 128u;      // 4 SIMD-groups / threadgroup

static const uint32_t HX_N_SYMBOLS       = 256u;
static const uint32_t HX_PROB_BITS       = 12u;
static const uint32_t HX_PROB_SCALE      = 4096u;
static const uint32_t HX_PROB_MASK       = 4095u;
static const uint32_t HX_RANS_L          = 65536u;
static const uint32_t HX_MIN_RANS_LEN    = 256u;
static const uint32_t HX_TILE_MAX_WORDS  = 32768u;

static const uint32_t HX_MIN_MATCH       = 4u;

// ブロック単位前処理フィルタ ID (OpenZL 型軽量 DAG)
static const uint8_t HX_XFORM_NONE       = 0u;
static const uint8_t HX_XFORM_SHUFFLE2   = 1u; // 2-byte plane split (FP16 / BF16 / Int16)
static const uint8_t HX_XFORM_SHUFFLE4   = 2u; // 4-byte plane split (FP32 / Int32)
static const uint8_t HX_XFORM_ARM64_BCJ  = 3u; // ARM64 BL 相対→絶対オフセット変換

// ブロック単位コーデック ID
static const uint8_t HX_CODEC_RAW        = 0u; // そのまま格納 (エントロピーゲート / 非圧縮)
static const uint8_t HX_CODEC_RLE        = 1u; // 単一バイト 1B 格納
static const uint8_t HX_CODEC_RANS_ONLY  = 2u; // 前処理後ダイレクト 32-way rANS (FP16 等に最適)
static const uint8_t HX_CODEC_LZ_RANS    = 3u; // 分離ストリーム LZ + 32-way rANS

// サブストリーム圧縮モード
static const uint8_t HX_SUB_RAW          = 0u;
static const uint8_t HX_SUB_RLE          = 1u;
static const uint8_t HX_SUB_RANS         = 2u;

// 頻度表形式
static const uint8_t HX_FREQ_SPARSE      = 0u;
static const uint8_t HX_FREQ_DENSE12     = 1u;
static const uint32_t HX_SPARSE_THRESH   = 128u;

// ─── LL / ML 対数クラス表 (37 コード: 0..262187 を完全カバー, max extra = 17 bit) ─
static const uint32_t HX_LEN_CODES = 37u;
static const uint32_t HX_LEN_BASE[HX_LEN_CODES] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 18, 20, 24, 28, 36, 44, 60, 76, 108, 172, 300, 556,
    1068, 2092, 4140, 8236, 16428, 32812, 65580, 131116
};
static const uint8_t HX_LEN_EB[HX_LEN_CODES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17
};

// OF コード: 0=rep0, 1=rep1, 2=rep2, 3..20 = dist in [2^k, 2^(k+1)-1], eb = k (0..17)
static const uint32_t HX_OF_CODES = 21u;

static const uint8_t HX_LEN_CODE_SMALL[76] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 16, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20,
    20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22,
    22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23
};

static inline uint8_t hx_len_to_code(uint32_t val, uint32_t* out_extra, uint32_t* out_eb) {
    if (__builtin_expect(val < 76u, 1)) {
        uint8_t c  = HX_LEN_CODE_SMALL[val];
        *out_extra = val - HX_LEN_BASE[c];
        *out_eb    = HX_LEN_EB[c];
        return c;
    }
    uint32_t b = 31u - (uint32_t)__builtin_clz(val - 44u);
    uint8_t  c = (uint8_t)(19u + b);
    *out_extra = val - HX_LEN_BASE[c];
    *out_eb    = HX_LEN_EB[c];
    return c;
}

static inline uint8_t hx_dist_to_code(uint32_t dist, uint32_t* out_extra, uint32_t* out_eb) {
    uint32_t k = 31u - (uint32_t)__builtin_clz(dist);
    *out_extra = dist - (1u << k);
    *out_eb    = k;
    return (uint8_t)(3u + k);
}

// RANS_ONLY ブロックのプレーン分割定義 (SHUFFLE2=2プレーン, SHUFFLE4=4プレーン, その他=1プレーン)
static inline uint32_t hx_rans_only_plane_count(uint8_t xform_id, uint32_t raw_size) {
    if (xform_id == HX_XFORM_SHUFFLE2 && raw_size >= 2u) return 2u;
    if (xform_id == HX_XFORM_SHUFFLE4 && raw_size >= 4u) return 4u;
    return 1u;
}

static inline void hx_rans_only_plane_layout(uint8_t xform_id, uint32_t raw_size,
                                             uint32_t out_off[4], uint32_t out_len[4]) {
    if (xform_id == HX_XFORM_SHUFFLE2 && raw_size >= 2u) {
        uint32_t cnt = raw_size >> 1u;
        out_off[0] = 0u;  out_len[0] = cnt;
        out_off[1] = cnt; out_len[1] = raw_size - cnt;
    } else if (xform_id == HX_XFORM_SHUFFLE4 && raw_size >= 4u) {
        uint32_t cnt = raw_size >> 2u;
        out_off[0] = 0u;       out_len[0] = cnt;
        out_off[1] = cnt;      out_len[1] = cnt;
        out_off[2] = cnt * 2u; out_len[2] = cnt;
        out_off[3] = cnt * 3u; out_len[3] = raw_size - cnt * 3u;
    } else {
        out_off[0] = 0u; out_len[0] = raw_size;
    }
}

// ─── CPU / GPU 共有構造体 (hetero_kernels.metal とバイナリ完全一致) ────────────

// 前処理ジョブ記述子 (1 ブロック = 16 B)
struct HxTransformJob {
    uint32_t byte_offset;
    uint32_t byte_len;
    uint32_t xform_id;   // HX_XFORM_*
    uint32_t _pad;
};
static_assert(sizeof(HxTransformJob) == 16, "HxTransformJob size mismatch");

// 可変長ストリーム rANS ジョブ記述子 (1 ストリーム = 32 B)
// 1 ストリームは最大 256 KiB (最大 4 タイル)
struct HxRansStreamJob {
    uint32_t data_offset;     // 入力/出力生バイトバッファ内のオフセット
    uint32_t raw_size;        // ストリーム実バイト長 (0..262144)
    uint32_t first_tile_idx;  // タイル配列 (words / fstates / tmeta) の開始インデックス
    uint32_t num_tiles;       // ceil(raw_size / 65536), 0..4
    uint32_t mode;            // build_tables 出力 / decode 入力: HX_SUB_RAW / RLE / RANS
    uint32_t rle_byte;        // mode == RLE 時のバイト値
    uint32_t n_nonzero;
    uint32_t _pad;
};
static_assert(sizeof(HxRansStreamJob) == 32, "HxRansStreamJob size mismatch");

struct HxEncSym {
    uint16_t freq;
    uint16_t cdf;
    uint32_t rcp_mul;
    uint32_t rcp_shift;
    uint32_t x_max;
};
static_assert(sizeof(HxEncSym) == 16, "HxEncSym size mismatch");

struct HxTileMeta {
    uint32_t mode;       // HX_SUB_RAW / RLE / RANS
    uint32_t rle_byte;
    uint32_t raw_size;
    uint32_t word_count; // mode == RANS 時: uint16_t ワード数
};
static_assert(sizeof(HxTileMeta) == 16, "HxTileMeta size mismatch");

// ─── コンテナバイナリヘッダ ("ASZ2" Phase 2 フォーマット) ───────────────────────
#pragma pack(push, 1)
struct HxArchiveHeader {
    uint8_t  magic[4];       // "ASZ2"
    uint32_t version;        // 1
    uint64_t original_size;
    uint32_t block_size;     // 262144
    uint32_t num_blocks;
    uint32_t level;          // 1 / 3 / 6
    uint32_t reserved;
};
struct HxBlockHeader {
    uint8_t  xform_id;       // HX_XFORM_*
    uint8_t  codec_id;       // HX_CODEC_*
    uint8_t  rle_byte;       // CODEC_RLE 時
    uint8_t  reserved;
    uint32_t raw_size;       // <= 262144
    uint32_t crc32c;         // 元データの CRC32C
    uint32_t comp_payload;   // このヘッダの直後に続くペイロードバイト数
};
struct HxLzBlockMeta {
    uint32_t num_seqs;
    uint32_t lit_size;
    uint32_t tail_lit_size;  // 最終マッチ以降の残りリテラル数 (lit_size に含まれる)
    uint32_t extra_bytes;    // 64-bit ビットストリームのバイト数
};
#pragma pack(pop)

static_assert(sizeof(HxArchiveHeader) == 32, "HxArchiveHeader size mismatch");
static_assert(sizeof(HxBlockHeader)   == 16, "HxBlockHeader size mismatch");
static_assert(sizeof(HxLzBlockMeta)   == 16, "HxLzBlockMeta size mismatch");

// ─── ARM64 ハードウェア CRC32C (1 コア ~15+ GB/s) ──────────────────────────────
static inline uint32_t hx_crc32c(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    size_t i = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    for (; i + 8u <= len; i += 8u) {
        uint64_t v;
        memcpy(&v, data + i, 8u);
        crc = __crc32cd(crc, v);
    }
    if (i + 4u <= len) {
        uint32_t v;
        memcpy(&v, data + i, 4u);
        crc = __crc32cw(crc, v);
        i += 4u;
    }
    for (; i < len; ++i) crc = __crc32cb(crc, data[i]);
#else
    for (; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b)
            crc = (crc >> 1u) ^ (0x82F63B78u & (0u - (crc & 1u)));
    }
#endif
    return ~crc;
}

// ─── O(log len) GF(2) CRC32C (Castagnoli) 結合: CRC(A || B) = combine(CRC(A), CRC(B), |B|) ─
// 事前計算した 41 個の 32x32 GF(2) 基底行列 (5 KB) により、1 結合あたり ~10 ns で厳密一致!
static inline uint32_t hx_gf2_mat_vec(const uint32_t mat[32], uint32_t vec) {
    uint32_t sum = 0u;
    while (vec) {
        uint32_t b = (uint32_t)__builtin_ctz(vec);
        sum ^= mat[b];
        vec &= (vec - 1u);
    }
    return sum;
}

static inline void hx_gf2_mat_sq(uint32_t sq[32], const uint32_t mat[32]) {
    for (int n = 0; n < 32; ++n) sq[n] = hx_gf2_mat_vec(mat, mat[n]);
}

static inline uint32_t hx_crc32c_combine(uint32_t crc1, uint32_t crc2, uint64_t len2) {
    if (len2 == 0u) return crc1;
    // M[k] = ゼロ 2^k バイトシフトに対応する 32x32 GF(2) 行列 (k=0..40 → 最大 1 TiB)
    struct MatTable {
        uint32_t m[41][32];
        MatTable() {
            uint32_t odd[32], even[32];
            odd[0] = 0x82F63B78u; // Castagnoli bit-reflected polynomial
            uint32_t row = 1u;
            for (int n = 1; n < 32; ++n) { odd[n] = row; row <<= 1u; }
            hx_gf2_mat_sq(even, odd);  // 2 bits
            hx_gf2_mat_sq(odd, even);  // 4 bits
            hx_gf2_mat_sq(m[0], odd);  // 8 bits = 1 byte (2^0 bytes)
            for (int k = 1; k < 41; ++k) {
                hx_gf2_mat_sq(m[k], m[k - 1]);
            }
        }
    };
    static const MatTable tbl;
    uint32_t k = 0u;
    while (len2 && k < 41u) {
        if (len2 & 1ull) crc1 = hx_gf2_mat_vec(tbl.m[k], crc1);
        len2 >>= 1u;
        k++;
    }
    return crc1 ^ crc2;
}
