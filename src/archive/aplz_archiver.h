// asz_archiver.h — APLZ-X Phase 3: `.asz` ネイティブアーカイバ本体
//
// 機能:
//   - ディレクトリ・複数ファイル一括アーカイブ (一時 .tar 完全廃止)
//   - 小ファイル (< 64 KiB) 型別 Solid Grouping (TEXT / CODE / NUMERIC / MEDIA / BINARY, 上限 4 MiB/Pack)
//   - 大ファイル (>= 64 KiB) 並列 FastCDC 重複排除 + ゼロコピー連続スパン直圧縮 (上限 32 MiB/Pack)
//   - O(1) フッタ読み取り一覧表示 (`list`)
//   - 256 KiB ブロック単位ランダムアクセス部分展開 (`extract` with filters → `HeteroEngine::decompress_range`)
//   - パック・ブロック・ファイル 3 重 CRC32C 整合性検査 (`verify`) + Zip-Slip 完全防御
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>

#include "aplz_format.h"
#include "aplz_fastcdc.h"
#include "../engine/hetero_engine.h"

struct AszCreateOptions {
    uint32_t level        = 3u;   // 1 (--fast) / 3 (--default) / 6 (--high)
    bool     enable_dedup = true; // FastCDC 重複排除 ON/OFF
    bool     enable_solid = true; // 小ファイル型別 Solid Grouping ON/OFF
};

struct AszCreateStats {
    uint64_t total_raw_bytes     = 0u;
    uint64_t unique_raw_bytes    = 0u;
    uint64_t dedup_saved_bytes   = 0u;
    uint64_t final_archive_bytes = 0u;
    uint32_t num_files           = 0u;
    uint32_t num_dirs            = 0u;
    uint32_t num_symlinks        = 0u;
    uint32_t num_packs           = 0u;
    uint32_t num_chunks_total    = 0u;
    uint32_t num_chunks_unique   = 0u;
    double   elapsed_s           = 0.0;
};

struct AszExtractStats {
    uint64_t extracted_bytes = 0u;
    uint32_t extracted_files = 0u;
    uint32_t extracted_dirs  = 0u;
    uint32_t extracted_links = 0u;
    uint32_t packs_decoded   = 0u;
    uint32_t blocks_decoded  = 0u; // 部分展開時に実際に復号した 256 KiB ブロック数
    bool     used_partial    = false;
    double   elapsed_s       = 0.0;
};

struct AszListEntry {
    std::string path;
    std::string target;
    uint64_t    raw_size;
    uint64_t    mtime_ns;
    uint32_t    file_crc32c;
    uint32_t    num_extents;
    uint16_t    mode;
    uint8_t     entry_type;
    uint8_t     solid_class;
};

struct AszArchiveSummary {
    uint32_t version             = 0u;
    uint32_t flags               = 0u;
    uint32_t level               = 0u;
    uint64_t archive_bytes       = 0u;
    uint64_t total_raw_bytes     = 0u;
    uint64_t unique_chunk_bytes  = 0u;
    uint32_t num_packs           = 0u;
    uint32_t num_chunks          = 0u;
    uint32_t num_entries         = 0u;
    uint32_t num_extents         = 0u;
    uint32_t dir_comp_bytes      = 0u;
};

struct AszParsedDir {
    AszFileHeader              fhdr;
    AszDirHeader               dhdr;
    std::vector<AszPackDesc>   packs;
    std::vector<AszChunkDesc>  chunks;
    std::vector<AszEntryDesc>  entries;
    std::vector<AszExtentDesc> extents;
    std::string                str_pool;

    std::string entry_path(const AszEntryDesc& e) const {
        return str_pool.substr(e.path_off, e.path_len);
    }
    std::string entry_target(const AszEntryDesc& e) const {
        return (e.target_len > 0u) ? str_pool.substr(e.target_off, e.target_len) : std::string();
    }
};

// ─── ファイルシステム補助 ──────────────────────────────────────────────────────
// 既存親ディレクトリ (/tmp -> /private/tmp 等のOSシンボリックリンク) を許容する mkdir -p
static inline bool asz_mkdir_p(const std::string& dir_path) {
    if (dir_path.empty()) return true;
    std::string cur;
    size_t i = 0u;
    if (dir_path[0] == '/') { cur = "/"; i = 1u; }
    while (i < dir_path.size()) {
        size_t j = dir_path.find('/', i);
        if (j == std::string::npos) j = dir_path.size();
        if (j > i) {
            if (!cur.empty() && cur.back() != '/') cur += '/';
            cur.append(dir_path, i, j - i);
            struct stat st;
            if (stat(cur.c_str(), &st) != 0) {
                if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
            } else if (!S_ISDIR(st.st_mode)) {
                return false;
            }
        }
        i = j + 1u;
    }
    return true;
}

// 展開先 base_dir 配下の相対ディレクトリをシンボリックリンク辿り禁止で安全に作成
static inline bool asz_mkdir_rel_safe(const std::string& base_dir, const std::string& rel_dir) {
    if (rel_dir.empty()) return true;
    std::string cur = base_dir;
    size_t i = 0u;
    while (i < rel_dir.size()) {
        size_t j = rel_dir.find('/', i);
        if (j == std::string::npos) j = rel_dir.size();
        if (j > i) {
            if (!cur.empty() && cur.back() != '/') cur += '/';
            cur.append(rel_dir, i, j - i);
            struct stat st;
            if (lstat(cur.c_str(), &st) != 0) {
                if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
            } else if (!S_ISDIR(st.st_mode)) {
                // アーカイブ内シンボリックリンク経由の外部ディレクトリ逸脱を拒否!
                return false;
            }
        }
        i = j + 1u;
    }
    return true;
}

static inline std::string asz_parent_dir(const std::string& p) {
    size_t slash = p.find_last_of('/');
    if (slash == std::string::npos) return "";
    return p.substr(0u, slash);
}

static inline std::string asz_strip_trailing_slashes(std::string s) {
    while (s.size() > 1u && s.back() == '/') s.pop_back();
    return s;
}

static inline std::string asz_basename(const std::string& p) {
    std::string s = asz_strip_trailing_slashes(p);
    size_t slash = s.find_last_of('/');
    return (slash == std::string::npos) ? s : s.substr(slash + 1u);
}

static inline uint64_t asz_stat_mtime_ns(const struct stat& st) {
#if defined(__APPLE__)
    return (uint64_t)st.st_mtimespec.tv_sec * 1000000000ULL + (uint64_t)st.st_mtimespec.tv_nsec;
#else
    return (uint64_t)st.st_mtim.tv_sec * 1000000000ULL + (uint64_t)st.st_mtim.tv_nsec;
#endif
}

static inline void asz_set_mtime_ns(const char* path, uint64_t mtime_ns) {
    if (mtime_ns == 0u) return;
    struct timeval tv[2];
    tv[0].tv_sec  = (long)(mtime_ns / 1000000000ULL);
    tv[0].tv_usec = (int)((mtime_ns % 1000000000ULL) / 1000ULL);
    tv[1] = tv[0];
    utimes(path, tv);
}

static inline void asz_set_mtime_ns_fd(int fd, uint64_t mtime_ns) {
    if (fd < 0 || mtime_ns == 0u) return;
    struct timespec ts[2];
    ts[0].tv_sec  = (time_t)(mtime_ns / 1000000000ULL);
    ts[0].tv_nsec = (long)(mtime_ns % 1000000000ULL);
    ts[1] = ts[0];
    futimens(fd, ts);
}

// シンボリックリンク辿り防止付き高速ファイル作成 (新規展開時は unlink システムコール 0 回!)
static inline int asz_open_create_safe(const char* path, mode_t mode) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode);
    if (fd < 0 && errno == EEXIST) {
        unlink(path);
        fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode);
    }
    return fd;
}

// ASZ2 パックのストリーム CRC32C 並列検証 (Header+SeekTable の CRC XOR ペイロード部の CRC)
static inline bool asz_verify_pack_stream_crc(const uint8_t* pb, size_t comp_size, uint32_t expected_crc) {
    if (comp_size < sizeof(HxArchiveHeader)) return false;
    HxArchiveHeader hdr;
    memcpy(&hdr, pb, sizeof(hdr));
    size_t htot = sizeof(hdr) + (size_t)hdr.num_blocks * sizeof(uint64_t);
    if (htot > comp_size) return false;
    uint32_t hcrc = hx_crc32c(pb, htot);
    const uint8_t* pay = pb + htot;
    size_t plen = comp_size - htot;
    uint32_t pcrc = 0u;
    if (plen >= 2u * 1024u * 1024u) {
        const size_t n_slices = 16u;
        const size_t step = (plen + n_slices - 1u) / n_slices;
        uint32_t scrcs[16] = {};
        size_t   slens[16] = {};
        uint32_t* sc_p = scrcs;
        size_t*   sl_p = slens;
        dispatch_apply(n_slices, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            size_t off = i * step;
            if (off >= plen) { sl_p[i] = 0u; sc_p[i] = 0u; return; }
            size_t len = std::min(step, plen - off);
            sl_p[i] = len;
            sc_p[i] = hx_crc32c(pay + off, len);
        });
        for (size_t i = 0u; i < n_slices; ++i) {
            if (slens[i] > 0u) pcrc = hx_crc32c_combine(pcrc, scrcs[i], slens[i]);
        }
    } else {
        pcrc = hx_crc32c(pay, plen);
    }
    return (hcrc ^ pcrc) == expected_crc;
}

// 16 KiB アライン済み領域の mmap ページ解放 (大容量ファイル処理時の RSS 膨張防止)
static inline void asz_madvise_dontneed(const void* base, size_t start_off, size_t end_off) {
    static const size_t PG = 16384u; // Apple Silicon 16 KiB ページ
    size_t a0 = (start_off + PG - 1u) & ~(PG - 1u);
    size_t a1 = end_off & ~(PG - 1u);
    if (a1 > a0) {
        madvise((void*)((const uint8_t*)base + a0), a1 - a0, MADV_DONTNEED);
    }
}

// ─── Phase 3 `.asz` アーカイバクラス ────────────────────────────────────────────
class AszArchiver {
public:
    HeteroEngine eng;
    std::string  last_error;

    bool init(const char* shader_dir, bool eager_gpu = false) {
        return eng.init(shader_dir, eager_gpu);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. アーカイブ作成 (`asz c`)
    // ═══════════════════════════════════════════════════════════════════════════
    bool create(
        const std::vector<std::string>& inputs,
        const std::string&              out_asz_path,
        const AszCreateOptions&         opts,
        AszCreateStats*                 out_stats = nullptr
    ) {
        last_error.clear();
        CFTimeInterval t0 = CACurrentMediaTime();
        if (inputs.empty()) { last_error = "No input files or directories specified"; return false; }

        struct ScanItem {
            std::string src_path;
            std::string rel_path;
            std::string target;
            uint64_t    raw_size    = 0u;
            uint64_t    mtime_ns    = 0u;
            uint16_t    mode        = 0644u;
            uint8_t     entry_type  = ASZ_ENTRY_FILE;
            uint8_t     solid_class = ASZ_CLASS_BINARY;
            uint32_t    entry_idx   = 0u;
        };

        std::vector<ScanItem> items;
        std::unordered_set<std::string> seen_paths;

        std::function<bool(const std::string&, const std::string&)> walk =
            [&](const std::string& src, const std::string& rel) -> bool {
            struct stat st;
            if (lstat(src.c_str(), &st) != 0) {
                last_error = "Cannot stat input: " + src;
                return false;
            }
            if (!asz_is_safe_rel_path(rel)) {
                last_error = "Unsafe relative path: " + rel;
                return false;
            }
            if (!seen_paths.insert(rel).second) {
                last_error = "Duplicate archive path: " + rel;
                return false;
            }

            ScanItem it;
            it.src_path = src;
            it.rel_path = rel;
            it.mtime_ns = asz_stat_mtime_ns(st);
            it.mode     = (uint16_t)(st.st_mode & 07777u);

            if (S_ISLNK(st.st_mode)) {
                char buf[4096];
                ssize_t n = readlink(src.c_str(), buf, sizeof(buf) - 1u);
                if (n < 0) { last_error = "readlink failed: " + src; return false; }
                buf[n] = '\0';
                it.entry_type = ASZ_ENTRY_SYMLINK;
                it.target     = std::string(buf, (size_t)n);
                items.push_back(std::move(it));
                return true;
            }
            if (S_ISDIR(st.st_mode)) {
                it.entry_type = ASZ_ENTRY_DIR;
                items.push_back(std::move(it));

                DIR* d = opendir(src.c_str());
                if (!d) { last_error = "Cannot open dir: " + src; return false; }
                std::vector<std::string> names;
                while (struct dirent* de = readdir(d)) {
                    if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0) continue;
                    names.emplace_back(de->d_name);
                }
                closedir(d);
                std::sort(names.begin(), names.end());
                for (const std::string& nm : names) {
                    if (!walk(src + "/" + nm, rel + "/" + nm)) return false;
                }
                return true;
            }
            if (S_ISREG(st.st_mode)) {
                it.entry_type = ASZ_ENTRY_FILE;
                it.raw_size   = (uint64_t)st.st_size;
                // 拡張子のみで高速一次判定 (未確定のみ後段で全コア並列マジック検査)
                it.solid_class = asz_classify_file(rel, nullptr, 0u);
                items.push_back(std::move(it));
                return true;
            }
            // FIFO / Socket / Device はスキップ
            return true;
        };

        for (const std::string& raw_in : inputs) {
            std::string clean = asz_strip_trailing_slashes(raw_in);
            std::string base  = asz_basename(clean);
            if (base.empty() || base == "." || base == "..") {
                // カレントディレクトリ "." が指定された場合は中身を展開追加
                DIR* d = opendir(clean.c_str());
                if (!d) { last_error = "Cannot open dir: " + clean; return false; }
                std::vector<std::string> names;
                while (struct dirent* de = readdir(d)) {
                    if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0) continue;
                    names.emplace_back(de->d_name);
                }
                closedir(d);
                std::sort(names.begin(), names.end());
                for (const std::string& nm : names) {
                    if (!walk(clean + "/" + nm, nm)) return false;
                }
            } else {
                if (!walk(clean, base)) return false;
            }
        }

        // 拡張子未確定 (ASZ_CLASS_BINARY) の通常ファイルのみ先頭 256B を全コア並列でマジック判定
        dispatch_queue_t par_q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        {
            std::vector<uint32_t> need_probe;
            for (size_t i = 0u; i < items.size(); ++i) {
                if (items[i].entry_type == ASZ_ENTRY_FILE && items[i].raw_size > 0u &&
                    items[i].solid_class == ASZ_CLASS_BINARY) {
                    need_probe.push_back((uint32_t)i);
                }
            }
            if (!need_probe.empty()) {
                ScanItem* items_p = items.data();
                const uint32_t* np_p = need_probe.data();
                std::atomic<int> first_err{-1};
                std::atomic<int>* first_err_p = &first_err;
                dispatch_apply(need_probe.size(), par_q, ^(size_t k) {
                    uint32_t idx = np_p[k];
                    ScanItem& it = items_p[idx];
                    uint8_t head[256];
                    int fd = open(it.src_path.c_str(), O_RDONLY);
                    if (fd < 0) {
                        int expected = -1;
                        first_err_p->compare_exchange_strong(expected, (int)idx);
                        return;
                    }
                    ssize_t got = read(fd, head, std::min((size_t)it.raw_size, sizeof(head)));
                    close(fd);
                    if (got > 0) {
                        it.solid_class = asz_classify_file(it.rel_path, head, (size_t)got);
                    }
                });
                int err_idx = first_err.load();
                if (err_idx >= 0) {
                    last_error = "Cannot open file: " + items[(size_t)err_idx].src_path;
                    return false;
                }
            }
        }

        AszCreateStats stats;
        uint64_t total_raw = 0u;
        for (size_t i = 0u; i < items.size(); ++i) {
            items[i].entry_idx = (uint32_t)i;
            if (items[i].entry_type == ASZ_ENTRY_FILE) {
                total_raw += items[i].raw_size;
                stats.num_files++;
            } else if (items[i].entry_type == ASZ_ENTRY_DIR) {
                stats.num_dirs++;
            } else {
                stats.num_symlinks++;
            }
        }
        stats.total_raw_bytes = total_raw;

        // 大容量アーカイブは先頭で非同期 Metal GPU 初期化を先行起動! (遅延ゼロで GPU 切替)
        if (total_raw >= HeteroEngine::HYBRID_GPU_MIN_BYTES) {
            eng.warmup_gpu_encode_async();
        }

        // 出力ファイル (.tmp 経由でアトミック置換, 無バッファ直接 write で余分な stdio コピー排除)
        std::string tmp_path = out_asz_path + ".tmp";
        FILE* fout = fopen(tmp_path.c_str(), "wb");
        if (!fout) { last_error = "Cannot create output file: " + tmp_path; return false; }
        setvbuf(fout, nullptr, _IONBF, 0);

        auto abort_out = [&]() -> bool {
            fclose(fout);
            unlink(tmp_path.c_str());
            return false;
        };

        AszFileHeader fhdr = {};
        if (fwrite(&fhdr, sizeof(fhdr), 1, fout) != 1) { last_error = "Header write failed"; return abort_out(); }
        uint64_t file_cursor = sizeof(fhdr);

        std::vector<AszPackDesc>   packs;
        std::vector<AszChunkDesc>  chunks;
        std::vector<AszEntryDesc>  entries(items.size());
        std::vector<AszExtentDesc> extents;
        std::string                str_pool;

        // Entry 基本メタデータ設定
        for (size_t i = 0u; i < items.size(); ++i) {
            const ScanItem& it = items[i];
            AszEntryDesc& e = entries[i];
            memset(&e, 0, sizeof(e));
            e.raw_size    = it.raw_size;
            e.mtime_ns    = it.mtime_ns;
            e.path_off    = (uint32_t)str_pool.size();
            e.path_len    = (uint32_t)it.rel_path.size();
            str_pool.append(it.rel_path);
            if (!it.target.empty()) {
                e.target_off = (uint32_t)str_pool.size();
                e.target_len = (uint32_t)it.target.size();
                str_pool.append(it.target);
            }
            e.mode        = it.mode;
            e.entry_type  = it.entry_type;
            e.solid_class = it.solid_class;
            e.file_crc32c = hx_crc32c((const uint8_t*)"", 0u);
        }

        // 128-bit + 32-bit CRC コンテンツ重複排除テーブル
        std::unordered_map<AszChunkKey, uint32_t, AszChunkKeyHash> chunk_map;
        chunk_map.reserve(std::min((size_t)262144u, (size_t)(total_raw / 49152u) + 1024u));

        // ステージング Pack バッファ & ゼロ中間バッファ直接ストリーミング圧縮フラッシュ
        std::vector<uint8_t> pack_buf;

        auto write_compressed_pack = [&](const uint8_t* raw_ptr, uint32_t raw_len,
                                         uint8_t sclass, bool is_solid,
                                         uint32_t* out_raw_crc = nullptr) -> bool {
            if (raw_len == 0u) {
                if (out_raw_crc) *out_raw_crc = 0u;
                return true;
            }
            uint64_t comp_bytes = 0u;
            uint32_t pcrc = 0u;
            if (!eng.compress_core(raw_ptr, raw_len, opts.level, nullptr, fout, nullptr,
                                   &comp_bytes, &pcrc, out_raw_crc)) {
                last_error = "Phase 2 HeteroEngine streaming compress failed";
                return false;
            }
            AszPackDesc pd = {};
            pd.file_offset = file_cursor;
            pd.comp_size   = comp_bytes;
            pd.raw_size    = raw_len;
            pd.pack_crc32c = pcrc;
            pd.solid_class = sclass;
            pd.is_solid    = is_solid ? 1u : 0u;
            packs.push_back(pd);
            file_cursor += comp_bytes;
            return true;
        };

        // ── Phase A: 小ファイル (< 64 KiB) 全コア並列読込 + 型別 Solid Grouping ───
        std::vector<uint32_t> small_by_class[ASZ_NUM_CLASSES];
        std::vector<uint32_t> large_by_class[ASZ_NUM_CLASSES];
        for (size_t i = 0u; i < items.size(); ++i) {
            if (items[i].entry_type != ASZ_ENTRY_FILE || items[i].raw_size == 0u) continue;
            uint8_t cls = opts.enable_solid ? items[i].solid_class : ASZ_CLASS_BINARY;
            if (items[i].raw_size < ASZ_SMALL_FILE_THRESH) {
                small_by_class[cls].push_back((uint32_t)i);
            } else {
                large_by_class[cls].push_back((uint32_t)i);
            }
        }

        std::vector<uint8_t> small_batch_buf;
        for (uint8_t cls = 0u; cls < ASZ_NUM_CLASSES; ++cls) {
            auto& idxs = small_by_class[cls];
            if (idxs.empty()) continue;

            // 拡張子→ファイル名順ソートで類似小ファイルを同一 256 KiB LZ 窓へ集約!
            std::sort(idxs.begin(), idxs.end(), [&](uint32_t a, uint32_t b) {
                std::string ea = asz_lower_ext(items[a].rel_path);
                std::string eb = asz_lower_ext(items[b].rel_path);
                if (ea != eb) return ea < eb;
                return items[a].rel_path < items[b].rel_path;
            });

            pack_buf.clear();
            const size_t SMALL_BATCH = 256u;
            for (size_t b0 = 0u; b0 < idxs.size(); b0 += SMALL_BATCH) {
                const size_t bc = std::min(SMALL_BATCH, idxs.size() - b0);
                std::vector<size_t>      boffs(bc);
                std::vector<AszChunkKey> bkeys(bc);
                size_t total_batch_bytes = 0u;
                for (size_t k = 0u; k < bc; ++k) {
                    boffs[k] = total_batch_bytes;
                    total_batch_bytes += (size_t)items[idxs[b0 + k]].raw_size;
                }
                small_batch_buf.resize(total_batch_bytes);

                const ScanItem* items_p = items.data();
                const uint32_t* idxs_p  = idxs.data() + b0;
                const size_t*   offs_p  = boffs.data();
                uint8_t*        buf_p   = small_batch_buf.data();
                AszChunkKey*    keys_p  = bkeys.data();
                const bool      do_dedup = opts.enable_dedup;
                std::atomic<int> read_err{-1};
                std::atomic<int>* read_err_p = &read_err;

                dispatch_apply(bc, par_q, ^(size_t k) {
                    uint32_t idx = idxs_p[k];
                    const ScanItem& it = items_p[idx];
                    uint32_t flen = (uint32_t)it.raw_size;
                    uint8_t* dst = buf_p + offs_p[k];
                    int fd = open(it.src_path.c_str(), O_RDONLY);
                    if (fd < 0) {
                        int exp = -1;
                        read_err_p->compare_exchange_strong(exp, (int)idx);
                        return;
                    }
                    size_t got_tot = 0u;
                    while (got_tot < flen) {
                        ssize_t r = read(fd, dst + got_tot, flen - got_tot);
                        if (r < 0) {
                            if (errno == EINTR) continue;
                            break;
                        }
                        if (r == 0) break;
                        got_tot += (size_t)r;
                    }
                    close(fd);
                    if (got_tot != flen) {
                        int exp = -1;
                        read_err_p->compare_exchange_strong(exp, (int)idx);
                        return;
                    }
                    keys_p[k] = do_dedup
                                    ? asz_fingerprint_chunk(dst, flen)
                                    : AszChunkKey{ 0ULL, 0ULL, flen, hx_crc32c(dst, flen) };
                });

                int eidx = read_err.load();
                if (eidx >= 0) {
                    last_error = "Cannot read file: " + items[(size_t)eidx].src_path;
                    return abort_out();
                }

                // 決定論的順序で chunk_map / pack_buf へ確定反映
                for (size_t k = 0u; k < bc; ++k) {
                    uint32_t idx = idxs[b0 + k];
                    uint32_t flen = (uint32_t)items[idx].raw_size;
                    const uint8_t* fptr = small_batch_buf.data() + boffs[k];
                    const AszChunkKey& key = bkeys[k];

                    entries[idx].file_crc32c  = key.crc32c;
                    entries[idx].first_extent = (uint32_t)extents.size();
                    entries[idx].num_extents  = 1u;
                    stats.num_chunks_total++;

                    bool hit = false;
                    if (opts.enable_dedup) {
                        auto mit = chunk_map.find(key);
                        if (mit != chunk_map.end()) {
                            extents.push_back({ mit->second, 0u, flen, 0u });
                            stats.dedup_saved_bytes += flen;
                            hit = true;
                        }
                    }
                    if (!hit) {
                        if (!pack_buf.empty() && pack_buf.size() + flen + 4u > ASZ_SOLID_PACK_MAX) {
                            if (!write_compressed_pack(pack_buf.data(), (uint32_t)pack_buf.size(), cls, true))
                                return abort_out();
                            pack_buf.clear();
                        }
                        if (cls == ASZ_CLASS_CODE || cls == ASZ_CLASS_NUMERIC) {
                            while ((pack_buf.size() & 3u) != 0u) pack_buf.push_back(0u);
                        }
                        uint32_t cid = (uint32_t)chunks.size();
                        uint32_t off = (uint32_t)pack_buf.size();
                        pack_buf.insert(pack_buf.end(), fptr, fptr + flen);

                        AszChunkDesc cd = { (uint32_t)packs.size(), off, flen, key.crc32c, key.hash_lo, key.hash_hi };
                        chunks.push_back(cd);
                        if (opts.enable_dedup) chunk_map.emplace(key, cid);
                        extents.push_back({ cid, 0u, flen, 0u });
                        stats.unique_raw_bytes += flen;
                        stats.num_chunks_unique++;
                    }
                }
            }
            if (!pack_buf.empty()) {
                if (!write_compressed_pack(pack_buf.data(), (uint32_t)pack_buf.size(), cls, true))
                    return abort_out();
                pack_buf.clear();
            }
        }

        // ── Phase B: 大ファイル (>= 64 KiB) 並列 FastCDC / 非重複直結ストリーミング ─
        std::vector<AszRawChunkSlice> raw_chunks;
        for (uint8_t cls = 0u; cls < ASZ_NUM_CLASSES; ++cls) {
            auto& idxs = large_by_class[cls];
            if (idxs.empty()) continue;

            pack_buf.clear();
            for (uint32_t idx : idxs) {
                const ScanItem& it = items[idx];
                int fd = open(it.src_path.c_str(), O_RDONLY);
                if (fd < 0) { last_error = "Cannot open file: " + it.src_path; return abort_out(); }
                void* m = mmap(nullptr, (size_t)it.raw_size, PROT_READ, MAP_PRIVATE, fd, 0);
                if (m == MAP_FAILED) { close(fd); last_error = "mmap failed: " + it.src_path; return abort_out(); }
                madvise(m, (size_t)it.raw_size, MADV_SEQUENTIAL);
                const uint8_t* fdata = (const uint8_t*)m;

                entries[idx].first_extent = (uint32_t)extents.size();

                // 高速パス: 重複排除 OFF 時は FastCDC・事前 CRC 走査を完全スキップし 256 MiB パック直結圧縮!
                // (compress_core の並列ブロック CRC を GF(2) 結合して file_crc32c を追加コスト 0 で取得!)
                if (!opts.enable_dedup) {
                    if (!pack_buf.empty()) {
                        if (!write_compressed_pack(pack_buf.data(), (uint32_t)pack_buf.size(), cls, false)) {
                            munmap(m, (size_t)it.raw_size); close(fd); return abort_out();
                        }
                        pack_buf.clear();
                    }
                    uint32_t file_crc = 0u;
                    size_t foff = 0u;
                    const size_t fsize = (size_t)it.raw_size;
                    while (foff < fsize) {
                        uint32_t slen = (uint32_t)std::min((size_t)ASZ_STREAM_PACK_MAX, fsize - foff);
                        uint32_t slice_crc = 0u;
                        uint32_t pid = (uint32_t)packs.size();
                        if (!write_compressed_pack(fdata + foff, slen, cls, false, &slice_crc)) {
                            munmap(m, fsize); close(fd); return abort_out();
                        }
                        uint32_t cid = (uint32_t)chunks.size();
                        AszChunkDesc cd = { pid, 0u, slen, slice_crc, 0ULL, 0ULL };
                        chunks.push_back(cd);
                        extents.push_back({ cid, 0u, slen, 0u });
                        file_crc = hx_crc32c_combine(file_crc, slice_crc, slen);
                        stats.num_chunks_total++;
                        stats.num_chunks_unique++;
                        stats.unique_raw_bytes += slen;

                        // 2 GiB 超の超巨大ファイルのみ完了ページを解放して RSS 膨張を防止
                        if (fsize >= 2048ull * 1024ull * 1024ull) {
                            asz_madvise_dontneed(fdata, foff, foff + slen);
                        }
                        foff += slen;
                    }
                    entries[idx].file_crc32c = file_crc;
                    entries[idx].num_extents = (uint32_t)extents.size() - entries[idx].first_extent;
                    munmap(m, fsize);
                    close(fd);
                    continue;
                }

                // 重複排除 ON: 128 MiB ウィンドウ単位で FastCDC + 圧縮を交互実行しキャッシュ局所性 & 低 RSS を両立!
                const size_t fsize = (size_t)it.raw_size;
                const size_t WIN_BYTES = (size_t)ASZ_LARGE_PACK_MAX; // 128 MiB (8 MiB スーパースライスの整数倍)
                uint32_t file_crc = 0u;

                const uint8_t* span_ptr = nullptr;
                uint32_t       span_len = 0u;

                auto flush_span_direct = [&]() -> bool {
                    if (span_len == 0u) return true;
                    bool ok = write_compressed_pack(span_ptr, span_len, cls, false);
                    span_ptr = nullptr;
                    span_len = 0u;
                    return ok;
                };
                auto materialize_span = [&]() {
                    if (span_len == 0u) return;
                    pack_buf.insert(pack_buf.end(), span_ptr, span_ptr + span_len);
                    span_ptr = nullptr;
                    span_len = 0u;
                };

                for (size_t win_off = 0u; win_off < fsize; win_off += WIN_BYTES) {
                    size_t win_len = std::min(WIN_BYTES, fsize - win_off);

                    // ウィンドウ内を全コア並列 FastCDC + 160-bit 指紋計算
                    asz_fastcdc_chunk_and_hash(fdata + win_off, win_len, true, raw_chunks);

                    for (const AszRawChunkSlice& rc_rel : raw_chunks) {
                        const size_t abs_off = win_off + rc_rel.offset;
                        file_crc = hx_crc32c_combine(file_crc, rc_rel.key.crc32c, rc_rel.length);
                        stats.num_chunks_total++;

                        auto mit = chunk_map.find(rc_rel.key);
                        if (mit != chunk_map.end()) {
                            extents.push_back({ mit->second, 0u, rc_rel.length, 0u });
                            stats.dedup_saved_bytes += rc_rel.length;
                            if (span_len >= 8u * 1024u * 1024u) {
                                if (!flush_span_direct()) { munmap(m, fsize); close(fd); return abort_out(); }
                            } else if (span_len > 0u) {
                                materialize_span();
                            }
                            continue;
                        }

                        const uint8_t* cptr = fdata + abs_off;
                        if (pack_buf.empty() &&
                            (span_len == 0u || (span_ptr + span_len == cptr && span_len + rc_rel.length <= ASZ_LARGE_PACK_MAX))) {
                            if (span_len == 0u) span_ptr = cptr;
                            uint32_t cid = (uint32_t)chunks.size();
                            AszChunkDesc cd = { (uint32_t)packs.size(), span_len, rc_rel.length,
                                                rc_rel.key.crc32c, rc_rel.key.hash_lo, rc_rel.key.hash_hi };
                            chunks.push_back(cd);
                            chunk_map.emplace(rc_rel.key, cid);
                            extents.push_back({ cid, 0u, rc_rel.length, 0u });
                            span_len += rc_rel.length;
                            stats.unique_raw_bytes += rc_rel.length;
                            stats.num_chunks_unique++;

                            if (span_len >= ASZ_LARGE_PACK_MAX) {
                                if (!flush_span_direct()) { munmap(m, fsize); close(fd); return abort_out(); }
                            }
                            continue;
                        }

                        if (span_len > 0u) {
                            if (!flush_span_direct()) { munmap(m, fsize); close(fd); return abort_out(); }
                            if (pack_buf.empty() && rc_rel.length <= ASZ_LARGE_PACK_MAX) {
                                span_ptr = cptr;
                                uint32_t cid = (uint32_t)chunks.size();
                                AszChunkDesc cd = { (uint32_t)packs.size(), 0u, rc_rel.length,
                                                    rc_rel.key.crc32c, rc_rel.key.hash_lo, rc_rel.key.hash_hi };
                                chunks.push_back(cd);
                                chunk_map.emplace(rc_rel.key, cid);
                                extents.push_back({ cid, 0u, rc_rel.length, 0u });
                                span_len = rc_rel.length;
                                stats.unique_raw_bytes += rc_rel.length;
                                stats.num_chunks_unique++;
                                continue;
                            }
                        }
                        if (!pack_buf.empty() && pack_buf.size() + rc_rel.length + 4u > ASZ_LARGE_PACK_MAX) {
                            if (!write_compressed_pack(pack_buf.data(), (uint32_t)pack_buf.size(), cls, false)) {
                                munmap(m, fsize); close(fd); return abort_out();
                            }
                            pack_buf.clear();
                        }
                        if (cls == ASZ_CLASS_CODE || cls == ASZ_CLASS_NUMERIC) {
                            uint32_t want_align = (uint32_t)(abs_off & 3u);
                            while ((pack_buf.size() & 3u) != want_align) pack_buf.push_back(0u);
                        }
                        uint32_t cid = (uint32_t)chunks.size();
                        uint32_t off = (uint32_t)pack_buf.size();
                        pack_buf.insert(pack_buf.end(), cptr, cptr + rc_rel.length);
                        AszChunkDesc cd = { (uint32_t)packs.size(), off, rc_rel.length,
                                            rc_rel.key.crc32c, rc_rel.key.hash_lo, rc_rel.key.hash_hi };
                        chunks.push_back(cd);
                        chunk_map.emplace(rc_rel.key, cid);
                        extents.push_back({ cid, 0u, rc_rel.length, 0u });
                        stats.unique_raw_bytes += rc_rel.length;
                        stats.num_chunks_unique++;
                    }

                    // ウィンドウ境界: 2 GiB 超の超巨大ファイルのみ完了済みページを解放
                    if (fsize >= 2048ull * 1024ull * 1024ull) {
                        if (span_len >= 4u * 1024u * 1024u) {
                            if (!flush_span_direct()) { munmap(m, fsize); close(fd); return abort_out(); }
                        } else if (span_len > 0u) {
                            materialize_span();
                        }
                        asz_madvise_dontneed(fdata, win_off, win_off + win_len);
                    }
                }

                entries[idx].file_crc32c = file_crc;

                // ファイル終端: munmap 前に span_ptr を確実にフラッシュ/退避
                if (span_len >= 4u * 1024u * 1024u || idxs.size() == 1u) {
                    if (!flush_span_direct()) { munmap(m, fsize); close(fd); return abort_out(); }
                } else if (span_len > 0u) {
                    materialize_span();
                }

                entries[idx].num_extents = (uint32_t)extents.size() - entries[idx].first_extent;
                munmap(m, fsize);
                close(fd);
            }

            if (!pack_buf.empty()) {
                if (!write_compressed_pack(pack_buf.data(), (uint32_t)pack_buf.size(), cls, false))
                    return abort_out();
                pack_buf.clear();
            }
        }

        // ── Phase C: セントラルディレクトリを Phase 2 圧縮して末尾へ書き出し ──────
        AszDirHeader dhdr = {};
        dhdr.num_packs        = (uint32_t)packs.size();
        dhdr.num_chunks       = (uint32_t)chunks.size();
        dhdr.num_entries      = (uint32_t)entries.size();
        dhdr.num_extents      = (uint32_t)extents.size();
        dhdr.string_pool_size = (uint64_t)str_pool.size();
        dhdr.total_raw_bytes  = total_raw;

        size_t dir_raw_size = sizeof(dhdr)
                            + packs.size()   * sizeof(AszPackDesc)
                            + chunks.size()  * sizeof(AszChunkDesc)
                            + entries.size() * sizeof(AszEntryDesc)
                            + extents.size() * sizeof(AszExtentDesc)
                            + str_pool.size();
        std::vector<uint8_t> dir_raw(dir_raw_size);
        uint8_t* dp = dir_raw.data();
        auto put_vec = [&](const void* src, size_t bytes) {
            if (bytes > 0u) { memcpy(dp, src, bytes); dp += bytes; }
        };
        put_vec(&dhdr, sizeof(dhdr));
        put_vec(packs.data(),   packs.size()   * sizeof(AszPackDesc));
        put_vec(chunks.data(),  chunks.size()  * sizeof(AszChunkDesc));
        put_vec(entries.data(), entries.size() * sizeof(AszEntryDesc));
        put_vec(extents.data(), extents.size() * sizeof(AszExtentDesc));
        put_vec(str_pool.data(), str_pool.size());

        std::vector<uint8_t> dir_comp;
        if (!eng.compress(dir_raw.data(), dir_raw.size(), 3u, dir_comp)) {
            last_error = "Failed to compress Central Directory";
            return abort_out();
        }
        uint64_t dir_off = file_cursor;
        if (fwrite(dir_comp.data(), 1, dir_comp.size(), fout) != dir_comp.size()) {
            last_error = "Failed to write Central Directory";
            return abort_out();
        }
        file_cursor += (uint64_t)dir_comp.size();

        // 先頭 AszFileHeader を書き戻し
        memcpy(fhdr.magic, ASZ_MAGIC, 4);
        fhdr.version             = ASZ_VERSION;
        fhdr.flags               = (opts.enable_dedup ? ASZ_FLAG_DEDUP : 0u)
                                 | (opts.enable_solid ? ASZ_FLAG_SOLID : 0u);
        fhdr.level               = opts.level;
        fhdr.directory_offset    = dir_off;
        fhdr.directory_comp_size = (uint64_t)dir_comp.size();
        fhdr.directory_raw_size  = (uint32_t)dir_raw.size();
        fhdr.header_crc32c       = hx_crc32c((const uint8_t*)&fhdr, offsetof(AszFileHeader, header_crc32c));

        if (fseek(fout, 0, SEEK_SET) != 0 || fwrite(&fhdr, sizeof(fhdr), 1, fout) != 1) {
            last_error = "Failed to finalize AszFileHeader";
            return abort_out();
        }
        if (fclose(fout) != 0) {
            unlink(tmp_path.c_str());
            last_error = "fclose failed";
            return false;
        }
        if (rename(tmp_path.c_str(), out_asz_path.c_str()) != 0) {
            unlink(tmp_path.c_str());
            last_error = "rename failed";
            return false;
        }

        stats.num_packs           = (uint32_t)packs.size();
        stats.final_archive_bytes = file_cursor;
        stats.elapsed_s           = CACurrentMediaTime() - t0;
        if (out_stats) *out_stats = stats;
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. O(1) セントラルディレクトリ読み込み & 厳密構造検証
    // ═══════════════════════════════════════════════════════════════════════════
    bool parse_directory_from_mem(const uint8_t* data, size_t size, AszParsedDir& out) {
        if (size < sizeof(AszFileHeader)) { last_error = "Archive too small"; return false; }
        memcpy(&out.fhdr, data, sizeof(AszFileHeader));
        if (memcmp(out.fhdr.magic, ASZ_MAGIC, 4) != 0) {
            last_error = "Invalid .aplz magic";
            return false;
        }
        if (out.fhdr.version != ASZ_VERSION) {
            last_error = "Unsupported .aplz version " + std::to_string(out.fhdr.version) + " (requires APLZ v4)";
            return false;
        }
        uint32_t exp_hcrc = hx_crc32c((const uint8_t*)&out.fhdr, offsetof(AszFileHeader, header_crc32c));
        if (out.fhdr.header_crc32c != exp_hcrc) {
            last_error = "AszFileHeader CRC32C mismatch (corrupted header)";
            return false;
        }
        if (out.fhdr.directory_offset > size ||
            out.fhdr.directory_comp_size > size - out.fhdr.directory_offset) {
            last_error = "Central Directory out of file bounds";
            return false;
        }

        std::vector<uint8_t> dir_raw;
        if (!eng.decompress(data + out.fhdr.directory_offset, (size_t)out.fhdr.directory_comp_size, dir_raw) ||
            dir_raw.size() != out.fhdr.directory_raw_size ||
            dir_raw.size() < sizeof(AszDirHeader)) {
            last_error = "Failed to decompress or verify Central Directory";
            return false;
        }

        const uint8_t* p   = dir_raw.data();
        const uint8_t* end = p + dir_raw.size();
        memcpy(&out.dhdr, p, sizeof(AszDirHeader));
        p += sizeof(AszDirHeader);

        size_t need = (size_t)out.dhdr.num_packs   * sizeof(AszPackDesc)
                    + (size_t)out.dhdr.num_chunks  * sizeof(AszChunkDesc)
                    + (size_t)out.dhdr.num_entries * sizeof(AszEntryDesc)
                    + (size_t)out.dhdr.num_extents * sizeof(AszExtentDesc)
                    + (size_t)out.dhdr.string_pool_size;
        if ((size_t)(end - p) != need) {
            last_error = "Central Directory layout size mismatch";
            return false;
        }

        out.packs.resize(out.dhdr.num_packs);
        if (!out.packs.empty()) { memcpy(out.packs.data(), p, out.packs.size() * sizeof(AszPackDesc)); p += out.packs.size() * sizeof(AszPackDesc); }
        out.chunks.resize(out.dhdr.num_chunks);
        if (!out.chunks.empty()) { memcpy(out.chunks.data(), p, out.chunks.size() * sizeof(AszChunkDesc)); p += out.chunks.size() * sizeof(AszChunkDesc); }
        out.entries.resize(out.dhdr.num_entries);
        if (!out.entries.empty()) { memcpy(out.entries.data(), p, out.entries.size() * sizeof(AszEntryDesc)); p += out.entries.size() * sizeof(AszEntryDesc); }
        out.extents.resize(out.dhdr.num_extents);
        if (!out.extents.empty()) { memcpy(out.extents.data(), p, out.extents.size() * sizeof(AszExtentDesc)); p += out.extents.size() * sizeof(AszExtentDesc); }
        out.str_pool.assign((const char*)p, (size_t)out.dhdr.string_pool_size);

        // 全テーブル境界 & セキュリティ事前検証
        for (const auto& pk : out.packs) {
            if (pk.file_offset > size || pk.comp_size > size - pk.file_offset) {
                last_error = "Pack offset out of bounds";
                return false;
            }
        }
        for (const auto& ch : out.chunks) {
            if (ch.pack_id >= out.packs.size() ||
                (uint64_t)ch.offset_in_pack + ch.raw_size > out.packs[ch.pack_id].raw_size) {
                last_error = "Chunk bounds invalid";
                return false;
            }
        }
        for (const auto& ex : out.extents) {
            if (ex.chunk_id >= out.chunks.size() ||
                (uint64_t)ex.offset_in_chunk + ex.length > out.chunks[ex.chunk_id].raw_size) {
                last_error = "Extent bounds invalid";
                return false;
            }
        }
        for (const auto& en : out.entries) {
            if ((uint64_t)en.path_off + en.path_len > out.str_pool.size() ||
                (uint64_t)en.target_off + en.target_len > out.str_pool.size() ||
                (uint64_t)en.first_extent + en.num_extents > out.extents.size()) {
                last_error = "Entry metadata out of bounds";
                return false;
            }
            if (!asz_is_safe_rel_path(out.entry_path(en))) {
                last_error = "Unsafe path rejected (Zip-Slip protection): " + out.entry_path(en);
                return false;
            }
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. O(1) 一覧取得 (`asz l`)
    // ═══════════════════════════════════════════════════════════════════════════
    bool list(const std::string& asz_path, std::vector<AszListEntry>& out_entries, AszArchiveSummary& out_sum) {
        last_error.clear();
        int fd = open(asz_path.c_str(), O_RDONLY);
        if (fd < 0) { last_error = "Cannot open archive: " + asz_path; return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); last_error = "fstat failed"; return false; }
        size_t sz = (size_t)st.st_size;
        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); last_error = "mmap failed"; return false; }

        AszParsedDir dir;
        bool ok = parse_directory_from_mem((const uint8_t*)m, sz, dir);
        munmap(m, sz);
        close(fd);
        if (!ok) return false;

        out_entries.clear();
        out_entries.reserve(dir.entries.size());
        for (const auto& e : dir.entries) {
            out_entries.push_back({
                dir.entry_path(e), dir.entry_target(e),
                e.raw_size, e.mtime_ns, e.file_crc32c, e.num_extents,
                e.mode, e.entry_type, e.solid_class
            });
        }
        uint64_t uniq_bytes = 0u;
        for (const auto& ch : dir.chunks) uniq_bytes += ch.raw_size;

        out_sum.version            = dir.fhdr.version;
        out_sum.flags              = dir.fhdr.flags;
        out_sum.level              = dir.fhdr.level;
        out_sum.archive_bytes      = sz;
        out_sum.total_raw_bytes    = dir.dhdr.total_raw_bytes;
        out_sum.unique_chunk_bytes = uniq_bytes;
        out_sum.num_packs          = dir.dhdr.num_packs;
        out_sum.num_chunks         = dir.dhdr.num_chunks;
        out_sum.num_entries        = dir.dhdr.num_entries;
        out_sum.num_extents        = dir.dhdr.num_extents;
        out_sum.dir_comp_bytes     = (uint32_t)dir.fhdr.directory_comp_size;
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. 展開 (`asz x`: 全展開 & O(1) 256 KiB ブロック単位ランダムアクセス部分展開)
    // ═══════════════════════════════════════════════════════════════════════════
    bool extract(
        const std::string&              asz_path,
        const std::string&              out_dir,
        const std::vector<std::string>& filters,
        AszExtractStats*                out_stats = nullptr
    ) {
        last_error.clear();
        CFTimeInterval t0 = CACurrentMediaTime();

        int fd = open(asz_path.c_str(), O_RDONLY);
        if (fd < 0) { last_error = "Cannot open archive: " + asz_path; return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); last_error = "fstat failed"; return false; }
        size_t sz = (size_t)st.st_size;
        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); last_error = "mmap failed"; return false; }
        const uint8_t* mapped = (const uint8_t*)m;

        AszParsedDir dir;
        if (!parse_directory_from_mem(mapped, sz, dir)) {
            munmap(m, sz); close(fd); return false;
        }

        // フィルタ対象エントリ選別
        std::vector<uint32_t> selected;
        selected.reserve(dir.entries.size());
        uint64_t sel_raw_bytes = 0u;
        for (uint32_t i = 0u; i < dir.entries.size(); ++i) {
            std::string ep = dir.entry_path(dir.entries[i]);
            bool match = filters.empty();
            for (const std::string& raw_f : filters) {
                std::string f = asz_strip_trailing_slashes(raw_f);
                if (ep == f || (ep.size() > f.size() && ep.compare(0, f.size(), f) == 0 && ep[f.size()] == '/')) {
                    match = true; break;
                }
            }
            if (match) {
                selected.push_back(i);
                if (dir.entries[i].entry_type == ASZ_ENTRY_FILE) sel_raw_bytes += dir.entries[i].raw_size;
            }
        }
        if (!filters.empty() && selected.empty()) {
            munmap(m, sz); close(fd);
            last_error = "No entries matched filter";
            return false;
        }

        if (!asz_mkdir_p(out_dir)) {
            munmap(m, sz); close(fd);
            last_error = "Cannot create output directory: " + out_dir;
            return false;
        }

        // フィルタ指定かつ選択バイト数が総パック容量の 25% 未満ならブロック単位部分復号を使用!
        uint64_t total_pack_raw = 0u;
        for (const auto& pk : dir.packs) total_pack_raw += pk.raw_size;
        bool use_partial = (!filters.empty() && total_pack_raw > 0u && sel_raw_bytes * 4u < total_pack_raw);

        AszExtractStats stats;
        stats.used_partial = use_partial;

        // 現在の umask を 1 回だけ取得 (99% の通常権限 0644/0755 で fchmod syscall を完全省略!)
        mode_t cur_umask = umask(0);
        umask(cur_umask);

        dispatch_queue_t par_q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // フル展開用 2 スロット Pack LRU キャッシュ (重複排除済みチャンクの多重参照で 100% ヒット!)
        struct PackCacheSlot {
            uint32_t             pack_id = UINT32_MAX;
            uint64_t             lru_ts  = 0u;
            std::vector<uint8_t> data;
        };
        PackCacheSlot cache[2];
        uint64_t lru_clock = 0u;

        auto get_decoded_pack = [&](uint32_t pid) -> const std::vector<uint8_t>* {
            for (int k = 0; k < 2; ++k) {
                if (cache[k].pack_id == pid) {
                    cache[k].lru_ts = ++lru_clock;
                    return &cache[k].data;
                }
            }
            int victim = (cache[0].lru_ts <= cache[1].lru_ts) ? 0 : 1;
            const AszPackDesc& pk = dir.packs[pid];
            const uint8_t* pbytes = mapped + pk.file_offset;
            if (!asz_verify_pack_stream_crc(pbytes, (size_t)pk.comp_size, pk.pack_crc32c)) {
                last_error = "Pack CRC32C mismatch (corrupted archive)";
                return nullptr;
            }
            if (!eng.decompress(pbytes, (size_t)pk.comp_size, cache[victim].data) ||
                cache[victim].data.size() != pk.raw_size) {
                last_error = "Pack decompression failed";
                return nullptr;
            }
            cache[victim].pack_id = pid;
            cache[victim].lru_ts  = ++lru_clock;
            stats.packs_decoded++;
            stats.blocks_decoded += (pk.raw_size + HX_BLOCK_SIZE - 1u) / HX_BLOCK_SIZE;
            return &cache[victim].data;
        };

        // 各 Pack への選択済み Extent 参照数を集計 (非共有大パックのゼロバッファ直展開判定に使用)
        std::vector<uint32_t> pack_ext_refs(dir.packs.size(), 0u);
        if (!use_partial) {
            for (uint32_t idx : selected) {
                const AszEntryDesc& en = dir.entries[idx];
                if (en.entry_type != ASZ_ENTRY_FILE) continue;
                for (uint32_t ei = 0u; ei < en.num_extents; ++ei) {
                    uint32_t pid = dir.chunks[dir.extents[en.first_extent + ei].chunk_id].pack_id;
                    pack_ext_refs[pid]++;
                }
            }
        }

        // Phase 1: ディレクトリ & 必要親ディレクトリを安全に一括先行作成 (並列ファイル書出前の安全性保証!)
        std::unordered_set<std::string> created_dirs;
        created_dirs.reserve(256u);
        std::vector<std::pair<std::string, uint16_t>> dir_chmods;
        std::vector<uint32_t> file_entries;
        std::vector<uint32_t> link_entries;
        file_entries.reserve(selected.size());

        bool ok = true;
        auto ensure_rel_dir_cached = [&](const std::string& rel_d) -> bool {
            if (rel_d.empty()) return true;
            if (created_dirs.find(rel_d) != created_dirs.end()) return true;
            if (!asz_mkdir_rel_safe(out_dir, rel_d)) return false;
            created_dirs.insert(rel_d);
            return true;
        };

        for (uint32_t idx : selected) {
            const AszEntryDesc& en = dir.entries[idx];
            std::string rel = dir.entry_path(en);
            if (en.entry_type == ASZ_ENTRY_DIR) {
                if (!ensure_rel_dir_cached(rel)) {
                    last_error = "mkdir failed: " + out_dir + "/" + rel;
                    ok = false; break;
                }
                dir_chmods.push_back({ out_dir + "/" + rel, (uint16_t)(en.mode & 0777u) });
                stats.extracted_dirs++;
            } else {
                std::string preldir = asz_parent_dir(rel);
                if (!ensure_rel_dir_cached(preldir)) {
                    last_error = "mkdir parent failed (possible symlink traversal): " + out_dir + "/" + rel;
                    ok = false; break;
                }
                if (en.entry_type == ASZ_ENTRY_FILE) file_entries.push_back(idx);
                else                                 link_entries.push_back(idx);
            }
        }

        // ファイルは Pack 順・オフセット順に整列 (各 Pack 復号を厳密 1 回に抑えつつ同 Pack 群を並列展開!)
        if (ok && !use_partial) {
            std::stable_sort(file_entries.begin(), file_entries.end(), [&](uint32_t a, uint32_t b) {
                const AszEntryDesc& ea = dir.entries[a];
                const AszEntryDesc& eb = dir.entries[b];
                uint32_t pa = (ea.num_extents > 0u) ? dir.chunks[dir.extents[ea.first_extent].chunk_id].pack_id : UINT32_MAX;
                uint32_t pb = (eb.num_extents > 0u) ? dir.chunks[dir.extents[eb.first_extent].chunk_id].pack_id : UINT32_MAX;
                if (pa != pb) return pa < pb;
                uint32_t offa = (ea.num_extents > 0u) ? dir.chunks[dir.extents[ea.first_extent].chunk_id].offset_in_pack : 0u;
                uint32_t offb = (eb.num_extents > 0u) ? dir.chunks[dir.extents[eb.first_extent].chunk_id].offset_in_pack : 0u;
                return offa < offb;
            });
        }

        // Phase 2: 通常ファイル展開
        std::vector<uint8_t> range_tmp;
        if (ok && use_partial) {
            // O(1) ブロック単位ランダムアクセス部分展開パス
            for (uint32_t idx : file_entries) {
                const AszEntryDesc& en = dir.entries[idx];
                std::string rel  = dir.entry_path(en);
                std::string full = out_dir + "/" + rel;

                int ofd = asz_open_create_safe(full.c_str(), 0600);
                if (ofd < 0) { last_error = "Cannot create file: " + full; ok = false; break; }
                if (en.raw_size >= 256u * 1024u) hx_preallocate_fd(ofd, en.raw_size);

                uint32_t crc_st = asz_crc32c_init();
                uint64_t written = 0u;
                for (uint32_t ei = 0u; ei < en.num_extents; ++ei) {
                    const AszExtentDesc& ex = dir.extents[en.first_extent + ei];
                    const AszChunkDesc&  ch = dir.chunks[ex.chunk_id];
                    const AszPackDesc&   pk = dir.packs[ch.pack_id];
                    range_tmp.resize(ex.length);
                    uint32_t bdec = 0u;
                    if (!eng.decompress_range(mapped + pk.file_offset, (size_t)pk.comp_size,
                                              (uint64_t)ch.offset_in_pack + ex.offset_in_chunk,
                                              ex.length, range_tmp.data(), &bdec)) {
                        last_error = "Partial range decompression failed";
                        ok = false; break;
                    }
                    stats.blocks_decoded += bdec;
                    if (ex.offset_in_chunk == 0u && ex.length == ch.raw_size &&
                        hx_crc32c(range_tmp.data(), ex.length) != ch.crc32c) {
                        last_error = "Chunk CRC32C mismatch";
                        ok = false; break;
                    }
                    crc_st = asz_crc32c_update(crc_st, range_tmp.data(), ex.length);
                    if (ex.length > 0u && !hx_write_all_fd(ofd, range_tmp.data(), ex.length)) {
                        last_error = "File write failed: " + full; ok = false; break;
                    }
                    written += ex.length;
                }
                if (ok && (written != en.raw_size || asz_crc32c_final(crc_st) != en.file_crc32c)) {
                    last_error = "File integrity mismatch (CRC32C/size): " + rel;
                    ok = false;
                }
                if (ok) {
                    fchmod(ofd, en.mode & 0777u);
                    asz_set_mtime_ns_fd(ofd, en.mtime_ns);
                }
                if (close(ofd) != 0) ok = false;
                if (!ok) { unlink(full.c_str()); break; }
                stats.extracted_bytes += written;
                stats.extracted_files++;
            }
        } else if (ok) {
            // フル展開パス: 同 Pack 複数ファイルは全コア並列書出、大単体ファイルはゼロバッファ直ストリーム展開!
            auto file_single_pack_id = [&](uint32_t idx) -> uint32_t {
                const AszEntryDesc& en = dir.entries[idx];
                if (en.num_extents == 0u) return UINT32_MAX - 1u; // 空ファイル群
                uint32_t pid0 = dir.chunks[dir.extents[en.first_extent].chunk_id].pack_id;
                for (uint32_t ei = 1u; ei < en.num_extents; ++ei) {
                    if (dir.chunks[dir.extents[en.first_extent + ei].chunk_id].pack_id != pid0)
                        return UINT32_MAX; // マルチパック跨ぎ
                }
                return pid0;
            };

            size_t fi = 0u;
            while (fi < file_entries.size() && ok) {
                uint32_t idx0 = file_entries[fi];
                uint32_t pid0 = file_single_pack_id(idx0);

                // (A) 同一 Pack (または空ファイル群) に属する連続ファイル群を収集
                size_t fj = fi + 1u;
                if (pid0 != UINT32_MAX) {
                    while (fj < file_entries.size() && file_single_pack_id(file_entries[fj]) == pid0) {
                        fj++;
                    }
                }
                size_t group_cnt = fj - fi;

                // Path 2B: 複数ファイル (Solid Pack 等) → Pack を 1 回復号し、APFS 最適な 4 ワーカー並列書出!
                // (10 スレッド同時 open は同一ディレクトリ vnode ロック競合を起こすため 4 ワーカーに制限)
                if (group_cnt >= 2u && pid0 != UINT32_MAX) {
                    const std::vector<uint8_t>* pdata = nullptr;
                    if (pid0 != UINT32_MAX - 1u) {
                        pdata = get_decoded_pack(pid0);
                        if (!pdata) { ok = false; break; }
                    }
                    const uint8_t* pack_bytes = pdata ? pdata->data() : nullptr;
                    const uint32_t* grp_ptr   = file_entries.data() + fi;
                    const AszParsedDir* dir_p = &dir;
                    const std::string*  od_p  = &out_dir;
                    std::atomic<int> first_err{-1};
                    std::atomic<uint64_t> batch_bytes{0u};
                    std::atomic<size_t> next_k{0u};
                    std::atomic<int>* first_err_p = &first_err;
                    std::atomic<uint64_t>* batch_bytes_p = &batch_bytes;
                    std::atomic<size_t>* next_k_p = &next_k;
                    const size_t n_workers = std::min((size_t)4u, group_cnt);

                    dispatch_apply(n_workers, par_q, ^(size_t /*w*/) {
                        while (true) {
                            if (first_err_p->load(std::memory_order_relaxed) >= 0) return;
                            size_t k = next_k_p->fetch_add(1u, std::memory_order_relaxed);
                            if (k >= group_cnt) break;

                            uint32_t eidx = grp_ptr[k];
                            const AszEntryDesc& en = dir_p->entries[eidx];
                            std::string rel  = dir_p->entry_path(en);
                            std::string full = *od_p + "/" + rel;
                            mode_t want_mode = (mode_t)(en.mode & 0777u);

                            int ofd = asz_open_create_safe(full.c_str(), want_mode);
                            if (ofd < 0) {
                                int exp = -1; first_err_p->compare_exchange_strong(exp, (int)eidx);
                                return;
                            }
                            if (en.raw_size >= 256u * 1024u) hx_preallocate_fd(ofd, en.raw_size);

                            uint32_t crc_st = asz_crc32c_init();
                            uint64_t written = 0u;
                            bool lok = true;
                            for (uint32_t ei = 0u; ei < en.num_extents; ++ei) {
                                const AszExtentDesc& ex = dir_p->extents[en.first_extent + ei];
                                const AszChunkDesc&  ch = dir_p->chunks[ex.chunk_id];
                                const uint8_t* ptr = pack_bytes + ch.offset_in_pack + ex.offset_in_chunk;
                                crc_st = asz_crc32c_update(crc_st, ptr, ex.length);
                                if (ex.length > 0u && !hx_write_all_fd(ofd, ptr, ex.length)) {
                                    lok = false; break;
                                }
                                written += ex.length;
                            }
                            if (lok && (written != en.raw_size || asz_crc32c_final(crc_st) != en.file_crc32c)) {
                                lok = false;
                            }
                            if (lok) {
                                if ((want_mode & cur_umask) != 0u) fchmod(ofd, want_mode);
                                asz_set_mtime_ns_fd(ofd, en.mtime_ns);
                            }
                            if (close(ofd) != 0) lok = false;
                            if (!lok) {
                                unlink(full.c_str());
                                int exp = -1; first_err_p->compare_exchange_strong(exp, (int)eidx);
                                return;
                            }
                            batch_bytes_p->fetch_add(written, std::memory_order_relaxed);
                        }
                    });

                    int eidx = first_err.load();
                    if (eidx >= 0) {
                        last_error = "Parallel file extraction failed: " + dir.entry_path(dir.entries[(size_t)eidx]);
                        ok = false; break;
                    }
                    stats.extracted_bytes += batch_bytes.load();
                    stats.extracted_files += (uint32_t)group_cnt;
                    fi = fj;
                    continue;
                }

                // Path 2A / 2C: 単一大ファイルまたはマルチパックファイル
                // 非共有の連続フルパックはゼロバッファ非同期ストリーミング + APFS F_PREALLOCATE で直展開!
                const AszEntryDesc& en = dir.entries[idx0];
                std::string rel  = dir.entry_path(en);
                std::string full = out_dir + "/" + rel;

                int ofd = asz_open_create_safe(full.c_str(), 0600);
                if (ofd < 0) { last_error = "Cannot create file: " + full; ok = false; break; }
                if (en.raw_size >= 256u * 1024u) hx_preallocate_fd(ofd, en.raw_size);
                FILE* fout = fdopen(ofd, "wb");
                if (!fout) { close(ofd); unlink(full.c_str()); last_error = "fdopen failed: " + full; ok = false; break; }
                setvbuf(fout, nullptr, _IONBF, 0);

                uint32_t file_crc = 0u;
                uint64_t written = 0u;
                uint32_t ei = 0u;
                while (ei < en.num_extents && ok) {
                    const AszExtentDesc& ex0 = dir.extents[en.first_extent + ei];
                    const AszChunkDesc&  ch0 = dir.chunks[ex0.chunk_id];
                    const uint32_t       pid = ch0.pack_id;
                    const AszPackDesc&   pk  = dir.packs[pid];

                    bool cached = (cache[0].pack_id == pid || cache[1].pack_id == pid);
                    if (!cached && ch0.offset_in_pack == 0u && ex0.offset_in_chunk == 0u &&
                        pk.raw_size >= 1024u * 1024u) {
                        uint32_t expect_off = 0u;
                        uint32_t ej = ei;
                        while (ej < en.num_extents && expect_off < pk.raw_size) {
                            const AszExtentDesc& ex = dir.extents[en.first_extent + ej];
                            const AszChunkDesc&  ch = dir.chunks[ex.chunk_id];
                            if (ch.pack_id != pid || ch.offset_in_pack + ex.offset_in_chunk != expect_off) break;
                            expect_off += ex.length;
                            ej++;
                        }
                        if (expect_off == pk.raw_size && pack_ext_refs[pid] == (ej - ei)) {
                            const uint8_t* pb = mapped + pk.file_offset;
                            if (!asz_verify_pack_stream_crc(pb, (size_t)pk.comp_size, pk.pack_crc32c)) {
                                last_error = "Pack CRC32C mismatch (corrupted archive)";
                                ok = false; break;
                            }
                            if (!eng.decompress_core(pb, (size_t)pk.comp_size, nullptr, fout, nullptr, &file_crc)) {
                                last_error = "Streaming pack decompression failed";
                                ok = false; break;
                            }
                            if (sz >= 2048ull * 1024ull * 1024ull) {
                                asz_madvise_dontneed(mapped, (size_t)pk.file_offset, (size_t)(pk.file_offset + pk.comp_size));
                            }
                            stats.packs_decoded++;
                            stats.blocks_decoded += (pk.raw_size + HX_BLOCK_SIZE - 1u) / HX_BLOCK_SIZE;
                            written += pk.raw_size;
                            ei = ej;
                            continue;
                        }
                    }

                    const std::vector<uint8_t>* pdata = get_decoded_pack(pid);
                    if (!pdata) { ok = false; break; }
                    // 同一 Pack 内の物理連続 Extent 群をラン結合して 1 回のゼロコピー書出へ統合!
                    const uint32_t run_start_off = ch0.offset_in_pack + ex0.offset_in_chunk;
                    uint64_t run_len = 0u;
                    while (ei < en.num_extents) {
                        const AszExtentDesc& ex = dir.extents[en.first_extent + ei];
                        const AszChunkDesc&  ch = dir.chunks[ex.chunk_id];
                        if (ch.pack_id != pid || (uint64_t)ch.offset_in_pack + ex.offset_in_chunk != (uint64_t)run_start_off + run_len)
                            break;
                        const uint8_t* ptr = pdata->data() + ch.offset_in_pack + ex.offset_in_chunk;
                        uint32_t ecrc = (ex.offset_in_chunk == 0u && ex.length == ch.raw_size)
                                            ? ch.crc32c : hx_crc32c(ptr, ex.length);
                        file_crc = hx_crc32c_combine(file_crc, ecrc, ex.length);
                        run_len += ex.length;
                        ei++;
                    }
                    if (run_len > 0u && fwrite(pdata->data() + run_start_off, 1, (size_t)run_len, fout) != (size_t)run_len) {
                        last_error = "File write failed: " + full; ok = false; break;
                    }
                    written += run_len;
                }

                if (ok) {
                    fchmod(fileno(fout), en.mode & 0777u);
                    asz_set_mtime_ns_fd(fileno(fout), en.mtime_ns);
                }
                if (fclose(fout) != 0) { last_error = "fclose failed: " + full; ok = false; }
                if (!ok) { unlink(full.c_str()); break; }

                if (written != en.raw_size || file_crc != en.file_crc32c) {
                    unlink(full.c_str());
                    last_error = "File integrity mismatch (CRC32C/size): " + rel;
                    ok = false; break;
                }
                stats.extracted_bytes += written;
                stats.extracted_files++;
                fi++;
            }
        }

        // Phase 3: シンボリックリンク復元 (全ファイル展開完了後に実行しリンク経由逸脱を構造的に根絶)
        if (ok) {
            for (uint32_t idx : link_entries) {
                const AszEntryDesc& en = dir.entries[idx];
                std::string full = out_dir + "/" + dir.entry_path(en);
                unlink(full.c_str());
                std::string tgt = dir.entry_target(en);
                if (symlink(tgt.c_str(), full.c_str()) != 0) {
                    last_error = "symlink failed: " + full; ok = false; break;
                }
                stats.extracted_links++;
            }
        }

        // Phase 4: ディレクトリ権限を深い順に復元
        if (ok && !dir_chmods.empty()) {
            std::sort(dir_chmods.begin(), dir_chmods.end(),
                      [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
            for (const auto& dm : dir_chmods) {
                chmod(dm.first.c_str(), dm.second);
            }
        }

        munmap(m, sz);
        close(fd);

        stats.elapsed_s = CACurrentMediaTime() - t0;
        if (out_stats) *out_stats = stats;
        return ok;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. 単一ファイル直接展開 (`asz d <archive.asz> <out_file>`)
    // ═══════════════════════════════════════════════════════════════════════════
    bool extract_single_file(const std::string& asz_path, const std::string& out_file, AszExtractStats* out_stats = nullptr) {
        last_error.clear();
        CFTimeInterval t0 = CACurrentMediaTime();

        int fd = open(asz_path.c_str(), O_RDONLY);
        if (fd < 0) { last_error = "Cannot open archive: " + asz_path; return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); last_error = "fstat failed"; return false; }
        size_t sz = (size_t)st.st_size;
        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); last_error = "mmap failed"; return false; }
        const uint8_t* mapped = (const uint8_t*)m;

        AszParsedDir dir;
        if (!parse_directory_from_mem(mapped, sz, dir)) {
            munmap(m, sz); close(fd); return false;
        }
        int file_idx = -1;
        for (size_t i = 0u; i < dir.entries.size(); ++i) {
            if (dir.entries[i].entry_type == ASZ_ENTRY_FILE) {
                if (file_idx >= 0) {
                    munmap(m, sz); close(fd);
                    last_error = "Archive contains multiple files; use 'asz x <archive.asz> -C <dir>'";
                    return false;
                }
                file_idx = (int)i;
            }
        }
        if (file_idx < 0) {
            munmap(m, sz); close(fd);
            last_error = "Archive contains no regular files";
            return false;
        }

        const AszEntryDesc& en = dir.entries[file_idx];
        int ofd = asz_open_create_safe(out_file.c_str(), 0644);
        if (ofd < 0) { munmap(m, sz); close(fd); last_error = "Cannot open output file: " + out_file; return false; }
        if (en.raw_size >= 256u * 1024u) hx_preallocate_fd(ofd, en.raw_size);
        FILE* fout = fdopen(ofd, "wb");
        if (!fout) { close(ofd); unlink(out_file.c_str()); munmap(m, sz); close(fd); last_error = "fdopen failed: " + out_file; return false; }
        setvbuf(fout, nullptr, _IONBF, 0);

        AszExtractStats stats;
        struct PackCacheSlot { uint32_t pack_id = UINT32_MAX; uint64_t lru_ts = 0u; std::vector<uint8_t> data; };
        PackCacheSlot cache[2];
        uint64_t lru_clock = 0u;

        auto get_pack = [&](uint32_t pid) -> const std::vector<uint8_t>* {
            for (int k = 0; k < 2; ++k) {
                if (cache[k].pack_id == pid) { cache[k].lru_ts = ++lru_clock; return &cache[k].data; }
            }
            int victim = (cache[0].lru_ts <= cache[1].lru_ts) ? 0 : 1;
            const AszPackDesc& pk = dir.packs[pid];
            const uint8_t* pb = mapped + pk.file_offset;
            if (!asz_verify_pack_stream_crc(pb, (size_t)pk.comp_size, pk.pack_crc32c)) {
                last_error = "Pack CRC32C mismatch"; return nullptr;
            }
            if (!eng.decompress(pb, (size_t)pk.comp_size, cache[victim].data) ||
                cache[victim].data.size() != pk.raw_size) {
                last_error = "Pack decompression failed"; return nullptr;
            }
            cache[victim].pack_id = pid;
            cache[victim].lru_ts  = ++lru_clock;
            stats.packs_decoded++;
            stats.blocks_decoded += (pk.raw_size + HX_BLOCK_SIZE - 1u) / HX_BLOCK_SIZE;
            return &cache[victim].data;
        };

        // 連続パック全体を参照する Extent 列は、メモリ確保ゼロの 2 スロット非同期書出 + GF(2) ブロック CRC 結合で直接展開!
        uint32_t file_crc = 0u;
        uint64_t written = 0u;
        bool ok = true;
        uint32_t ei = 0u;
        while (ei < en.num_extents) {
            const AszExtentDesc& ex0 = dir.extents[en.first_extent + ei];
            const AszChunkDesc&  ch0 = dir.chunks[ex0.chunk_id];
            const uint32_t       pid = ch0.pack_id;
            const AszPackDesc&   pk  = dir.packs[pid];

            // Pack 全体 [0 .. pk.raw_size) を連続参照し、かつキャッシュ未載なら直接ストリーミング展開
            bool cached = (cache[0].pack_id == pid || cache[1].pack_id == pid);
            if (!cached && ch0.offset_in_pack == 0u && ex0.offset_in_chunk == 0u && pk.raw_size >= 1024u * 1024u) {
                uint32_t expect_off = 0u;
                uint32_t ej = ei;
                while (ej < en.num_extents && expect_off < pk.raw_size) {
                    const AszExtentDesc& ex = dir.extents[en.first_extent + ej];
                    const AszChunkDesc&  ch = dir.chunks[ex.chunk_id];
                    if (ch.pack_id != pid || ch.offset_in_pack + ex.offset_in_chunk != expect_off) break;
                    expect_off += ex.length;
                    ej++;
                }
                // 後続 Extent で同じ Pack を再参照しない場合のみダイレクトストリーミング (重複参照ありならキャッシュ!)
                if (expect_off == pk.raw_size) {
                    bool reused_later = false;
                    for (uint32_t ek = ej; ek < en.num_extents; ++ek) {
                        if (dir.chunks[dir.extents[en.first_extent + ek].chunk_id].pack_id == pid) {
                            reused_later = true;
                            break;
                        }
                    }
                    if (!reused_later) {
                        const uint8_t* pb = mapped + pk.file_offset;
                        if (!asz_verify_pack_stream_crc(pb, (size_t)pk.comp_size, pk.pack_crc32c)) {
                            last_error = "Pack CRC32C mismatch";
                            ok = false; break;
                        }
                        if (!eng.decompress_core(pb, (size_t)pk.comp_size, nullptr, fout, nullptr, &file_crc)) {
                            last_error = "Streaming pack decompression failed";
                            ok = false; break;
                        }
                        if (sz >= 2048ull * 1024ull * 1024ull) {
                            asz_madvise_dontneed(mapped, (size_t)pk.file_offset, (size_t)(pk.file_offset + pk.comp_size));
                        }
                        stats.packs_decoded++;
                        stats.blocks_decoded += (pk.raw_size + HX_BLOCK_SIZE - 1u) / HX_BLOCK_SIZE;
                        written += pk.raw_size;
                        ei = ej;
                        continue;
                    }
                }
            }

            const std::vector<uint8_t>* pdata = get_pack(pid);
            if (!pdata) { ok = false; break; }
            // 同一 Pack 内の物理連続 Extent 群をラン結合して 1 回のゼロコピー大ブロック書出へ統合!
            const uint32_t run_start_off = ch0.offset_in_pack + ex0.offset_in_chunk;
            uint64_t run_len = 0u;
            while (ei < en.num_extents) {
                const AszExtentDesc& ex = dir.extents[en.first_extent + ei];
                const AszChunkDesc&  ch = dir.chunks[ex.chunk_id];
                if (ch.pack_id != pid || (uint64_t)ch.offset_in_pack + ex.offset_in_chunk != (uint64_t)run_start_off + run_len)
                    break;
                const uint8_t* ptr = pdata->data() + ch.offset_in_pack + ex.offset_in_chunk;
                uint32_t ecrc = (ex.offset_in_chunk == 0u && ex.length == ch.raw_size)
                                    ? ch.crc32c : hx_crc32c(ptr, ex.length);
                file_crc = hx_crc32c_combine(file_crc, ecrc, ex.length);
                run_len += ex.length;
                ei++;
            }
            if (run_len > 0u && fwrite(pdata->data() + run_start_off, 1, (size_t)run_len, fout) != (size_t)run_len) {
                last_error = "Write failed"; ok = false; break;
            }
            written += run_len;
        }
        if (fclose(fout) != 0) ok = false;
        munmap(m, sz);
        close(fd);

        if (!ok || written != en.raw_size || file_crc != en.file_crc32c) {
            unlink(out_file.c_str());
            if (last_error.empty()) last_error = "File CRC32C mismatch";
            return false;
        }
        stats.extracted_bytes = written;
        stats.extracted_files = 1u;
        stats.elapsed_s       = CACurrentMediaTime() - t0;
        if (out_stats) *out_stats = stats;
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. 完全整合性テスト (`aplz t`: ディスク書出なしで全 Pack / Chunk / File 検証)
    // ═══════════════════════════════════════════════════════════════════════════
    bool verify(const std::string& asz_path, AszExtractStats* out_stats = nullptr) {
        last_error.clear();
        CFTimeInterval t0 = CACurrentMediaTime();

        int fd = open(asz_path.c_str(), O_RDONLY);
        if (fd < 0) { last_error = "Cannot open archive: " + asz_path; return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); last_error = "fstat failed"; return false; }
        size_t sz = (size_t)st.st_size;
        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); last_error = "mmap failed"; return false; }
        const uint8_t* mapped = (const uint8_t*)m;

        AszParsedDir dir;
        if (!parse_directory_from_mem(mapped, sz, dir)) {
            munmap(m, sz); close(fd); return false;
        }

        dispatch_queue_t par_q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        AszExtractStats stats;
        struct PackCacheSlot { uint32_t pack_id = UINT32_MAX; uint64_t lru_ts = 0u; std::vector<uint8_t> data; };
        PackCacheSlot cache[2];
        uint64_t lru_clock = 0u;
        std::vector<uint8_t> pack_verified(dir.packs.size(), 0u);
        std::vector<uint32_t> pack_ext_refs(dir.packs.size(), 0u);
        for (const auto& en : dir.entries) {
            if (en.entry_type != ASZ_ENTRY_FILE) continue;
            for (uint32_t ei = 0u; ei < en.num_extents; ++ei) {
                pack_ext_refs[dir.chunks[dir.extents[en.first_extent + ei].chunk_id].pack_id]++;
            }
        }

        auto get_pack = [&](uint32_t pid) -> const std::vector<uint8_t>* {
            for (int k = 0; k < 2; ++k) {
                if (cache[k].pack_id == pid) { cache[k].lru_ts = ++lru_clock; return &cache[k].data; }
            }
            int victim = (cache[0].lru_ts <= cache[1].lru_ts) ? 0 : 1;
            const AszPackDesc& pk = dir.packs[pid];
            const uint8_t* pb = mapped + pk.file_offset;
            if (!asz_verify_pack_stream_crc(pb, (size_t)pk.comp_size, pk.pack_crc32c)) {
                last_error = "Pack CRC32C mismatch"; return nullptr;
            }
            if (!eng.decompress(pb, (size_t)pk.comp_size, cache[victim].data) ||
                cache[victim].data.size() != pk.raw_size) {
                last_error = "Pack decode failed"; return nullptr;
            }
            cache[victim].pack_id = pid;
            cache[victim].lru_ts  = ++lru_clock;
            pack_verified[pid]    = 1u;
            stats.packs_decoded++;
            stats.blocks_decoded += (pk.raw_size + HX_BLOCK_SIZE - 1u) / HX_BLOCK_SIZE;
            return &cache[victim].data;
        };

        // Pack 順にエントリを検証し、各 Pack の復号回数を最小化 (LRU スラッシング完全防止)
        std::vector<uint32_t> order(dir.entries.size());
        for (uint32_t i = 0u; i < dir.entries.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const AszEntryDesc& ea = dir.entries[a];
            const AszEntryDesc& eb = dir.entries[b];
            uint32_t pa = (ea.entry_type == ASZ_ENTRY_FILE && ea.num_extents > 0u)
                              ? dir.chunks[dir.extents[ea.first_extent].chunk_id].pack_id : UINT32_MAX;
            uint32_t pb = (eb.entry_type == ASZ_ENTRY_FILE && eb.num_extents > 0u)
                              ? dir.chunks[dir.extents[eb.first_extent].chunk_id].pack_id : UINT32_MAX;
            return pa < pb;
        });

        bool ok = true;
        for (uint32_t idx : order) {
            const AszEntryDesc& en = dir.entries[idx];
            if (en.entry_type == ASZ_ENTRY_DIR)     { stats.extracted_dirs++;  continue; }
            if (en.entry_type == ASZ_ENTRY_SYMLINK) { stats.extracted_links++; continue; }

            uint32_t file_crc = 0u;
            uint64_t seen = 0u;
            uint32_t ei = 0u;
            while (ei < en.num_extents && ok) {
                const AszExtentDesc& ex0 = dir.extents[en.first_extent + ei];
                const AszChunkDesc&  ch0 = dir.chunks[ex0.chunk_id];
                const uint32_t       pid = ch0.pack_id;
                const AszPackDesc&   pk  = dir.packs[pid];

                // 非共有の単一フルチャンク Pack はゼロメモリ確保のストリーミング検証!
                bool cached = (cache[0].pack_id == pid || cache[1].pack_id == pid);
                if (!cached && pack_ext_refs[pid] == 1u &&
                    ch0.offset_in_pack == 0u && ex0.offset_in_chunk == 0u &&
                    ex0.length == pk.raw_size && ch0.raw_size == pk.raw_size &&
                    pk.raw_size >= 1024u * 1024u) {
                    const uint8_t* pb = mapped + pk.file_offset;
                    if (!asz_verify_pack_stream_crc(pb, (size_t)pk.comp_size, pk.pack_crc32c)) {
                        last_error = "Pack CRC32C mismatch"; ok = false; break;
                    }
                    uint32_t pack_raw_crc = 0u;
                    if (!eng.decompress_core(pb, (size_t)pk.comp_size, nullptr, nullptr, nullptr, &pack_raw_crc)) {
                        last_error = "Pack decode failed"; ok = false; break;
                    }
                    if (pack_raw_crc != ch0.crc32c) {
                        last_error = "Chunk CRC32C mismatch"; ok = false; break;
                    }
                    file_crc = hx_crc32c_combine(file_crc, pack_raw_crc, pk.raw_size);
                    seen += pk.raw_size;
                    pack_verified[pid] = 1u;
                    stats.packs_decoded++;
                    stats.blocks_decoded += (pk.raw_size + HX_BLOCK_SIZE - 1u) / HX_BLOCK_SIZE;
                    ei++;
                    continue;
                }

                // 同一 Pack 内の連続 Extent 群をまとめて全コア並列 Chunk CRC 検証 + GF(2) 結合!
                const std::vector<uint8_t>* pdata = get_pack(pid);
                if (!pdata) { ok = false; break; }
                uint32_t ej = ei + 1u;
                while (ej < en.num_extents &&
                       dir.chunks[dir.extents[en.first_extent + ej].chunk_id].pack_id == pid) {
                    ej++;
                }
                uint32_t cnt = ej - ei;
                const uint8_t* pbytes = pdata->data();
                if (cnt >= 8u) {
                    std::vector<uint32_t> ecrcs(cnt);
                    uint32_t* ec_p = ecrcs.data();
                    const AszExtentDesc* ex_base = &dir.extents[en.first_extent + ei];
                    const AszChunkDesc*  ch_arr  = dir.chunks.data();
                    std::atomic<bool> chunk_ok{true};
                    std::atomic<bool>* chunk_ok_p = &chunk_ok;
                    dispatch_apply(cnt, par_q, ^(size_t k) {
                        const AszExtentDesc& ex = ex_base[k];
                        const AszChunkDesc&  ch = ch_arr[ex.chunk_id];
                        const uint8_t* cptr = pbytes + ch.offset_in_pack;
                        uint32_t ccrc = hx_crc32c(cptr, ch.raw_size);
                        if (ccrc != ch.crc32c) { chunk_ok_p->store(false, std::memory_order_relaxed); return; }
                        ec_p[k] = (ex.offset_in_chunk == 0u && ex.length == ch.raw_size)
                                      ? ccrc : hx_crc32c(cptr + ex.offset_in_chunk, ex.length);
                    });
                    if (!chunk_ok.load()) { last_error = "Chunk CRC32C mismatch"; ok = false; break; }
                    for (uint32_t k = 0u; k < cnt; ++k) {
                        uint32_t elen = ex_base[k].length;
                        file_crc = hx_crc32c_combine(file_crc, ecrcs[k], elen);
                        seen += elen;
                    }
                } else {
                    for (uint32_t k = ei; k < ej; ++k) {
                        const AszExtentDesc& ex = dir.extents[en.first_extent + k];
                        const AszChunkDesc&  ch = dir.chunks[ex.chunk_id];
                        const uint8_t* cptr = pbytes + ch.offset_in_pack;
                        uint32_t ccrc = hx_crc32c(cptr, ch.raw_size);
                        if (ccrc != ch.crc32c) { last_error = "Chunk CRC32C mismatch"; ok = false; break; }
                        uint32_t ecrc = (ex.offset_in_chunk == 0u && ex.length == ch.raw_size)
                                            ? ccrc : hx_crc32c(cptr + ex.offset_in_chunk, ex.length);
                        file_crc = hx_crc32c_combine(file_crc, ecrc, ex.length);
                        seen += ex.length;
                    }
                }
                ei = ej;
            }
            if (!ok) break;
            if (seen != en.raw_size || file_crc != en.file_crc32c) {
                last_error = "File CRC32C mismatch: " + dir.entry_path(en);
                ok = false; break;
            }
            stats.extracted_bytes += seen;
            stats.extracted_files++;
        }

        // どのファイルからも未参照のパックが残っていれば念のため検証
        for (uint32_t pid = 0u; pid < dir.packs.size() && ok; ++pid) {
            if (!pack_verified[pid] && !get_pack(pid)) ok = false;
        }

        munmap(m, sz);
        close(fd);
        stats.elapsed_s = CACurrentMediaTime() - t0;
        if (out_stats) *out_stats = stats;
        return ok;
    }
};
