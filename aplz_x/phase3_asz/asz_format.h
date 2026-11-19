// asz_format.h — APLZ-X Phase 3: `.asz` ネイティブコンテナ形式・型分類・セキュリティ検証
//
// アーカイブレイアウト:
//   [AszFileHeader: 40 B] (magic="ASZ\x01", directory_offset, directory_comp_size, header_crc32c)
//   [Data Packs: Phase 2 ASZ2 圧縮パック列]
//     - 小ファイル (< 64 KiB): 型別 Solid Grouping (TEXT / CODE / NUMERIC / MEDIA / BINARY, 上限 4 MiB/Pack)
//     - 大ファイル (>= 64 KiB): FastCDC 重複排除済みユニークチャンク列 (上限 32 MiB/Pack)
//   [Compressed Central Directory: Phase 2 ASZ2 圧縮済みメタデータ]
//     - AszDirHeader
//     - AszPackDesc[num_packs]
//     - AszChunkDesc[num_chunks]
//     - AszEntryDesc[num_entries]
//     - AszExtentDesc[num_extents]
//     - String Pool (UTF-8 相対パス & シンボリックリンク先)
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <cctype>

#include "../phase2_hetero/hetero_format.h"

// ─── 定数・フラグ ──────────────────────────────────────────────────────────────
static const uint8_t  ASZ_MAGIC[4]          = { 'A', 'S', 'Z', 0x01 };
static const uint32_t ASZ_VERSION           = 1u;

static const uint32_t ASZ_FLAG_DEDUP        = 1u << 0u;
static const uint32_t ASZ_FLAG_SOLID        = 1u << 1u;

static const uint8_t  ASZ_ENTRY_FILE        = 0u;
static const uint8_t  ASZ_ENTRY_DIR         = 1u;
static const uint8_t  ASZ_ENTRY_SYMLINK     = 2u;

// Solid Grouping 型クラス
static const uint8_t  ASZ_CLASS_TEXT        = 0u; // ソースコード / JSON / Markdown / テキスト
static const uint8_t  ASZ_CLASS_CODE        = 1u; // ARM64 Mach-O / ELF / 共有ライブラリ
static const uint8_t  ASZ_CLASS_NUMERIC     = 2u; // FP16/FP32 重み・科学技術数値・音声
static const uint8_t  ASZ_CLASS_MEDIA_COMP  = 3u; // 既圧縮メディア・アーカイブ (PNG/JPG/MP4/ZIP/ZST)
static const uint8_t  ASZ_CLASS_BINARY      = 4u; // その他一般バイナリ
static const uint8_t  ASZ_NUM_CLASSES       = 5u;

// チャンク / パック設計定数
static const uint32_t ASZ_SMALL_FILE_THRESH = 65536u;             // < 64 KiB は型別 Solid Grouping
static const uint32_t ASZ_SOLID_PACK_MAX    = 4u * 1024u * 1024u; // Solid Pack 上限 4 MiB
static const uint32_t ASZ_LARGE_PACK_MAX    = 64u * 1024u * 1024u;// 大ファイル Pack 上限 64 MiB (内部 256 KiB 独立シーク)

// ─── バイナリ構造体 (1 バイトアライン固定) ──────────────────────────────────────
#pragma pack(push, 1)
struct AszFileHeader {
    uint8_t  magic[4];            // "ASZ\x01"
    uint32_t version;             // 1
    uint32_t flags;               // ASZ_FLAG_*
    uint32_t level;               // 1 / 3 / 6
    uint64_t directory_offset;    // ファイル先頭からのセントラルディレクトリ開始位置
    uint64_t directory_comp_size; // 圧縮ディレクトリサイズ (ASZ2 ペイロード)
    uint32_t directory_raw_size;  // 展開後ディレクトリサイズ
    uint32_t header_crc32c;       // 先頭 36 バイトの CRC32C
};

struct AszDirHeader {
    uint32_t num_packs;
    uint32_t num_chunks;
    uint32_t num_entries;
    uint32_t num_extents;
    uint64_t string_pool_size;
    uint64_t total_raw_bytes;     // アーカイブ内全ファイルの合計元サイズ
};

struct AszPackDesc {
    uint64_t file_offset;         // .asz ファイル内での ASZ2 パック開始位置
    uint64_t comp_size;           // ASZ2 圧縮バイト数
    uint32_t raw_size;            // パック内非圧縮バイト数 (<= 32 MiB)
    uint32_t pack_crc32c;         // 圧縮パックバイト列の CRC32C
    uint8_t  solid_class;         // ASZ_CLASS_*
    uint8_t  is_solid;            // 1 = 小ファイル Solid Pack, 0 = 大ファイル FastCDC Pack
    uint16_t reserved;
    uint32_t reserved2;
};

struct AszChunkDesc {
    uint32_t pack_id;             // 格納先 Pack インデックス
    uint32_t offset_in_pack;      // Pack 非圧縮データ内の開始オフセット
    uint32_t raw_size;            // チャンク実バイト長
    uint32_t crc32c;              // チャンク生バイトの CRC32C
    uint64_t hash_lo;             // 128-bit 強コンテンツフィンガープリント下位
    uint64_t hash_hi;             // 128-bit 強コンテンツフィンガープリント上位
};

struct AszEntryDesc {
    uint64_t raw_size;            // 元ファイルサイズ (DIR/SYMLINK は 0)
    uint64_t mtime_ns;            // 最終更新時刻 (UNIX ns)
    uint32_t path_off;            // String Pool 内パス開始オフセット
    uint32_t path_len;            // パス長 (バイト)
    uint32_t target_off;          // SYMLINK 先オフセット (非 SYMLINK は 0)
    uint32_t target_len;          // SYMLINK 先長さ
    uint32_t first_extent;        // Extent Table 開始インデックス
    uint32_t num_extents;         // このファイルを構成する Extent 数
    uint32_t file_crc32c;         // ファイル全体の CRC32C
    uint16_t mode;                // POSIX パーミッション下位 12 bit (07777)
    uint8_t  entry_type;          // ASZ_ENTRY_*
    uint8_t  solid_class;         // ASZ_CLASS_*
};

struct AszExtentDesc {
    uint32_t chunk_id;            // Chunk Table インデックス
    uint32_t offset_in_chunk;     // チャンク内オフセット (通常 0)
    uint32_t length;              // この Extent が参照する長さ
    uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(AszFileHeader) == 40, "AszFileHeader size mismatch");
static_assert(sizeof(AszDirHeader)  == 32, "AszDirHeader size mismatch");
static_assert(sizeof(AszPackDesc)   == 32, "AszPackDesc size mismatch");
static_assert(sizeof(AszChunkDesc)  == 32, "AszChunkDesc size mismatch");
static_assert(sizeof(AszEntryDesc)  == 48, "AszEntryDesc size mismatch");
static_assert(sizeof(AszExtentDesc) == 16, "AszExtentDesc size mismatch");

// ─── ストリーミング ARM64 ハードウェア CRC32C (ファイル全体検証用) ───────────────
static inline uint32_t asz_crc32c_update(uint32_t state, const uint8_t* data, size_t len) {
    uint32_t crc = state;
    size_t i = 0u;
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
    return crc;
}

static inline uint32_t asz_crc32c_init() { return 0xFFFFFFFFu; }
static inline uint32_t asz_crc32c_final(uint32_t state) { return ~state; }

// ─── セキュリティ: Zip-Slip / パストラバーサル防止 ──────────────────────────────
static inline bool asz_is_safe_rel_path(const std::string& path) {
    if (path.empty()) return false;
    if (path[0] == '/' || path.back() == '/') return false;
    for (char c : path) {
        if (c == '\0' || c == '\\') return false;
    }
    size_t i = 0u;
    while (i < path.size()) {
        size_t j = path.find('/', i);
        if (j == std::string::npos) j = path.size();
        size_t clen = j - i;
        if (clen == 0u) return false; // 連続スラッシュ禁止
        if ((clen == 1u && path[i] == '.') ||
            (clen == 2u && path[i] == '.' && path[i + 1u] == '.')) {
            return false;
        }
        i = j + 1u;
    }
    return true;
}

// ─── 拡張子抽出 (小文字化) ─────────────────────────────────────────────────────
static inline std::string asz_lower_ext(const std::string& path) {
    size_t slash = path.find_last_of('/');
    size_t dot   = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return "";
    std::string ext = path.substr(dot + 1u);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    return ext;
}

// ─── Solid Grouping 用 高速ファイル型分類 (拡張子 + 先頭マジックバイト) ─────────
static inline uint8_t asz_classify_file(const std::string& path, const uint8_t* head, size_t head_len) {
    // 1) 先頭マジックバイトによる高精度判定 (Mach-O / ELF / PNG / JPEG / ZIP / GZ / ZST / PDF)
    if (head && head_len >= 4u) {
        uint32_t m4;
        memcpy(&m4, head, 4u);
        // Mach-O 64/32/Fat (LE/BE) or ELF
        if (m4 == 0xFEEDFACFu || m4 == 0xCFFAEDFEu ||
            m4 == 0xFEEDFACEu || m4 == 0xCEFAEDFEu ||
            m4 == 0xCAFEBABEu || m4 == 0xBEBAFECAu ||
            m4 == 0x464C457Fu) {
            return ASZ_CLASS_CODE;
        }
        // PNG / ZIP / GZIP / ZSTD / XZ / JPEG / WebP
        if (m4 == 0x474E5089u || m4 == 0x04034B50u || m4 == 0xFD2FB528u ||
            (head[0] == 0x1Fu && head[1] == 0x8Bu) ||
            (head[0] == 0xFFu && head[1] == 0xD8u && head[2] == 0xFFu)) {
            return ASZ_CLASS_MEDIA_COMP;
        }
    }

    // 2) 拡張子による分類
    std::string ext = asz_lower_ext(path);
    if (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "webp" || ext == "heic" ||
        ext == "gif" || ext == "mp4" || ext == "mov" || ext == "mkv" || ext == "mp3" ||
        ext == "aac" || ext == "m4a" || ext == "flac" || ext == "ogg" || ext == "opus" ||
        ext == "zip" || ext == "gz" || ext == "tgz" || ext == "zst" || ext == "xz" ||
        ext == "bz2" || ext == "7z" || ext == "rar" || ext == "dmg" || ext == "pdf" ||
        ext == "asz" || ext == "aplz" || ext == "lz4" || ext == "woff2") {
        return ASZ_CLASS_MEDIA_COMP;
    }
    if (ext == "dylib" || ext == "so" || ext == "a" || ext == "o" || ext == "exe" ||
        ext == "dll" || ext == "wasm" || ext == "metallib" || ext == "air") {
        return ASZ_CLASS_CODE;
    }
    if (ext == "fp16" || ext == "bf16" || ext == "fp32" || ext == "safetensors" ||
        ext == "npy" || ext == "npz" || ext == "gguf" || ext == "pt" || ext == "pth" ||
        ext == "wav" || ext == "pcm" || ext == "raw" || ext == "parquet" || ext == "arrow") {
        return ASZ_CLASS_NUMERIC;
    }
    if (ext == "c" || ext == "cc" || ext == "cpp" || ext == "cxx" || ext == "h" || ext == "hpp" ||
        ext == "m" || ext == "mm" || ext == "metal" || ext == "swift" || ext == "rs" ||
        ext == "go" || ext == "py" || ext == "js" || ext == "ts" || ext == "tsx" || ext == "jsx" ||
        ext == "java" || ext == "kt" || ext == "rb" || ext == "php" || ext == "sh" || ext == "zsh" ||
        ext == "md" || ext == "txt" || ext == "json" || ext == "jsonl" || ext == "xml" ||
        ext == "yaml" || ext == "yml" || ext == "toml" || ext == "ini" || ext == "cfg" ||
        ext == "html" || ext == "htm" || ext == "css" || ext == "svg" || ext == "csv" ||
        ext == "tsv" || ext == "sql" || ext == "log" || ext == "cmake" || ext == "gitignore") {
        return ASZ_CLASS_TEXT;
    }

    // 3) 拡張子不明な小ファイルは先頭最大 256 バイトの NUL / 制御文字比率でテキスト判定
    if (head && head_len > 0u) {
        size_t check = std::min(head_len, (size_t)256u);
        size_t bad = 0u;
        for (size_t i = 0u; i < check; ++i) {
            uint8_t c = head[i];
            if (c == 0u || (c < 0x09u) || (c > 0x0Du && c < 0x20u)) bad++;
        }
        if (bad * 20u < check) return ASZ_CLASS_TEXT;
    }
    return ASZ_CLASS_BINARY;
}

static inline const char* asz_class_name(uint8_t cls) {
    switch (cls) {
        case ASZ_CLASS_TEXT:       return "TEXT";
        case ASZ_CLASS_CODE:       return "CODE";
        case ASZ_CLASS_NUMERIC:    return "NUM";
        case ASZ_CLASS_MEDIA_COMP: return "MEDIA";
        default:                   return "BIN";
    }
}
