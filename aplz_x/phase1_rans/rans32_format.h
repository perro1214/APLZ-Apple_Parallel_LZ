// rans32_format.h — APLZ-X Phase 1: 32-way Coalesced rANS 共通定義 (CPU / Metal 共通仕様)
#pragma once

#include <stdint.h>
#include <stddef.h>

// ─── 定数 (rans32.metal 内の constant 定義と完全一致させること) ────────────────
static const uint32_t RANS32_LANES           = 32u;       // Apple GPU SIMD-group = 32 / ARM64 NEON 8x4 = 32
static const uint32_t RANS32_TILE_SIZE       = 65536u;    // 64 KiB / tile
static const uint32_t RANS32_TILES_PER_BLOCK = 4u;        // 4 tiles / superblock (= 128 threads/TG)
static const uint32_t RANS32_BLOCK_SIZE      = RANS32_TILE_SIZE * RANS32_TILES_PER_BLOCK; // 256 KiB
static const uint32_t RANS32_TG_SIZE         = RANS32_LANES * RANS32_TILES_PER_BLOCK;     // 128 threads

static const uint32_t RANS32_N_SYMBOLS       = 256u;
static const uint32_t RANS32_PROB_BITS       = 12u;
static const uint32_t RANS32_PROB_SCALE      = 1u << RANS32_PROB_BITS; // 4096
static const uint32_t RANS32_PROB_MASK       = RANS32_PROB_SCALE - 1u; // 4095
static const uint32_t RANS32_L               = 1u << 16u;              // 65536 (16-bit word renorm)
static const uint32_t RANS32_MIN_RANS_LEN    = 256u;                   // 256 B 未満の端数タイルは RAW

// 1 タイルあたり GPU ワークスペース上限 (uint16_t ワード数: 32768 words = 64 KiB)
static const uint32_t RANS32_TILE_MAX_WORDS  = RANS32_TILE_SIZE / 2u;

// モード種別
static const uint8_t  RANS32_MODE_RAW        = 0u;
static const uint8_t  RANS32_MODE_RLE        = 1u;
static const uint8_t  RANS32_MODE_RANS       = 2u;

// 頻度テーブル符号化種別
static const uint8_t  RANS32_FREQ_SPARSE     = 0u;
static const uint8_t  RANS32_FREQ_DENSE12    = 1u;
static const uint32_t RANS32_SPARSE_THRESH   = 128u; // n_nz * 3 < 384 となる境界 (128 * 3 = 384)

// ─── CPU / GPU 共有構造体 (Metal 構造体とバイナリ完全一致) ─────────────────────

// シンボルごとのエンコード情報 (16 B, 16-byte aligned)
// 1 スーパーブロックあたり 256 * 16 = 4,096 B
struct Rans32EncSym {
    uint16_t freq;       // 正規化頻度 (0 または 1..4095)
    uint16_t cdf;        // 累積頻度 (0..4095)
    uint32_t rcp_mul;    // Granlund-Montgomery 逆数乗数 (f >= 2)
    uint32_t rcp_shift;  // シフト量 k = floor(log2(f - 1))
    uint32_t x_max;      // 再正規化閾値 = (uint32_t)freq << 20
};
static_assert(sizeof(Rans32EncSym) == 16, "Rans32EncSym size mismatch");

// ブロック単位メタデータ (GPU build_tables 出力)
struct Rans32BlockMeta {
    uint32_t mode;       // RANS32_MODE_RAW / RLE / RANS
    uint32_t rle_byte;   // mode == RLE 時のバイト値
    uint32_t n_nonzero;  // 非ゼロシンボル数 (0..256)
    uint32_t raw_size;   // ブロック実バイト長 (<= 262144)
};
static_assert(sizeof(Rans32BlockMeta) == 16, "Rans32BlockMeta size mismatch");

// タイル単位エンコード結果メタデータ (GPU encode 出力 / GPU decode 入力)
// 1 タイル = 16 B + final_states[32] (別バッファまたは埋め込み)
struct Rans32TileMeta {
    uint32_t mode;       // RANS32_MODE_RAW / RLE / RANS
    uint32_t rle_byte;   // mode == RLE 時のバイト値
    uint32_t raw_size;   // タイル実バイト長 (<= 65536)
    uint32_t word_count; // mode == RANS 時の uint16_t ワード数
};
static_assert(sizeof(Rans32TileMeta) == 16, "Rans32TileMeta size mismatch");

// ─── シリアライズ用ヘッダ (ファイル/バッファ格納用) ────────────────────────────
#pragma pack(push, 1)
struct Rans32StreamHeader {
    uint8_t  magic[4];       // "R32P"
    uint32_t version;        // 1
    uint64_t original_size;
    uint32_t block_size;     // RANS32_BLOCK_SIZE (262144)
    uint32_t num_blocks;
};
struct Rans32BlockDiskHeader {
    uint8_t  block_mode;     // 0=RAW, 1=RLE, 2=RANS
    uint8_t  num_tiles;      // 1..4
    uint16_t reserved;       // 0
    uint32_t raw_size;       // <= 262144
};
struct Rans32TileDiskDesc {
    uint8_t  tile_mode;      // 0=RAW, 1=RLE, 2=RANS
    uint8_t  rle_byte;
    uint16_t reserved;       // 0
    uint32_t payload_words;  // RAW: byte_len, RLE: 0, RANS: word_count
};
#pragma pack(pop)

static_assert(sizeof(Rans32StreamHeader)    == 24, "Rans32StreamHeader size mismatch");
static_assert(sizeof(Rans32BlockDiskHeader) == 8,  "Rans32BlockDiskHeader size mismatch");
static_assert(sizeof(Rans32TileDiskDesc)    == 8,  "Rans32TileDiskDesc size mismatch");

// ─── LUT パックヘルパー ────────────────────────────────────────────────────────
// bits 0..7   : symbol (8-bit)
// bits 8..19  : freq   (12-bit, 1..4095)
// bits 20..31 : bias   (12-bit, 0..4094)
static inline uint32_t rans32_pack_lut(uint32_t sym, uint32_t freq, uint32_t bias) {
    return (sym & 0xFFu) | ((freq & 0xFFFu) << 8u) | ((bias & 0xFFFu) << 20u);
}

// ─── Granlund-Montgomery 32-bit 除算定数計算 (f in [1..4095]) ─────────────────
static inline void rans32_compute_rcp(uint16_t f, uint32_t* out_mul, uint32_t* out_shift) {
    if (f <= 1u) {
        *out_mul = 0u;
        *out_shift = 0u;
        return;
    }
    uint32_t k = 31u - (uint32_t)__builtin_clz((uint32_t)(f - 1u));
    uint64_t num = ((uint64_t)1u << 32u) * (((uint64_t)1u << (k + 1u)) - (uint64_t)f);
    *out_mul   = (uint32_t)(num / (uint64_t)f) + 1u;
    *out_shift = k;
}
