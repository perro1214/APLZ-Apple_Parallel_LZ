// asz_main.mm — APLZ-X Phase 3: `.asz` ネイティブ圧縮・アーカイブ CLI (`asz`)
//
// 使い方:
//   asz c [-1|-3|-6] [--no-dedup] [--no-solid] [-q] [-o <out.asz>] <input1> [input2 ...]
//   asz x <archive.asz> [-C <out_dir>] [-q] [path_filter ...]
//   asz d <archive.asz> <out_file> [-q]
//   asz l <archive.asz>
//   asz t <archive.asz>
#import <Foundation/Foundation.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <vector>

#include "asz_archiver.h"

static const char* ASZ_CLI_VERSION = "0.3.0 (APLZ-X Phase 3, ASZ v1)";

static std::string find_shader_dir() {
    if (const char* env = getenv("ASZ_SHADER_DIR")) {
        if (env[0] != '\0') return std::string(env);
    }
    char exe_buf[4096];
    uint32_t sz = sizeof(exe_buf);
    if (_NSGetExecutablePath(exe_buf, &sz) == 0) {
        char real_buf[4096];
        const char* p = realpath(exe_buf, real_buf) ? real_buf : exe_buf;
        std::string dir = asz_parent_dir(p);
        struct stat st;
        // 1) バイナリと同階層
        if (stat((dir + "/hetero_kernels.metallib").c_str(), &st) == 0 ||
            stat((dir + "/hetero_kernels.metal").c_str(), &st) == 0) {
            return dir;
        }
        // 2) 標準インストール先 (<prefix>/bin/asz → <prefix>/share/asz/)
        std::string share = dir + "/../share/asz";
        if (stat((share + "/hetero_kernels.metallib").c_str(), &st) == 0) {
            return share;
        }
        // 3) 開発ツリー (<repo>/aplz_x/phase3_asz/asz → ../phase2_hetero/)
        std::string p2 = dir + "/../phase2_hetero";
        if (stat((p2 + "/hetero_kernels.metallib").c_str(), &st) == 0 ||
            stat((p2 + "/hetero_kernels.metal").c_str(), &st) == 0) {
            return p2;
        }
    }
    return ".";
}

static std::string fmt_bytes(uint64_t b) {
    char buf[64];
    if (b >= 1024ULL * 1024ULL * 1024ULL) snprintf(buf, sizeof(buf), "%.2f GiB", (double)b / (1024.0 * 1024.0 * 1024.0));
    else if (b >= 1024ULL * 1024ULL)      snprintf(buf, sizeof(buf), "%.2f MiB", (double)b / (1024.0 * 1024.0));
    else if (b >= 1024ULL)                snprintf(buf, sizeof(buf), "%.1f KiB", (double)b / 1024.0);
    else                                  snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)b);
    return buf;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "APLZ-X (.asz) — Apple Silicon UMA Heterogeneous Archiver (Phase 3)\n\n"
        "Usage:\n"
        "  %s c [-1|-3|-6] [--no-dedup] [--no-solid] [-q] [-o <out.asz>] <input> [input2 ...]\n"
        "      Create .asz archive from file(s) and/or directory(ies)\n\n"
        "  %s x <archive.asz> [-C <out_dir>] [-q] [filter_path ...]\n"
        "      Extract full archive or specific file(s)/folder(s) (O(1) block random access)\n\n"
        "  %s d <archive.asz> <out_file> [-q]\n"
        "      Decompress single-file .asz directly to <out_file>\n\n"
        "  %s l <archive.asz>\n"
        "      List archive entries and dedup/pack summary (O(1) footer read)\n\n"
        "  %s t <archive.asz>\n"
        "      Verify full integrity (Header + Pack + Block + File CRC32C)\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc >= 2) {
            std::string arg1 = argv[1];
            if (arg1 == "-h" || arg1 == "--help" || arg1 == "help") {
                print_usage(argv[0]);
                return 0;
            }
            if (arg1 == "-V" || arg1 == "--version" || arg1 == "version") {
                printf("asz %s\n", ASZ_CLI_VERSION);
                return 0;
            }
        }
        if (argc < 3) {
            print_usage(argv[0]);
            return 1;
        }

        std::string cmd = argv[1];
        std::string shader_dir = find_shader_dir();

        AszArchiver arc;
        if (!arc.init(shader_dir.c_str(), false)) {
            fprintf(stderr, "[ERR] Failed to initialize AszArchiver.\n");
            return 1;
        }

        // ── Create (`c` / `create`) ────────────────────────────────────────────
        if (cmd == "c" || cmd == "create") {
            AszCreateOptions opts;
            std::string out_path;
            std::vector<std::string> inputs;
            bool quiet = false;

            for (int i = 2; i < argc; ++i) {
                std::string a = argv[i];
                if      (a == "-1" || a == "--fast")    opts.level = 1u;
                else if (a == "-3" || a == "--default") opts.level = 3u;
                else if (a == "-6" || a == "--high")    opts.level = 6u;
                else if (a == "--no-dedup")             opts.enable_dedup = false;
                else if (a == "--no-solid")             opts.enable_solid = false;
                else if (a == "-q" || a == "--quiet")   quiet = true;
                else if ((a == "-o" || a == "--output") && i + 1 < argc) out_path = argv[++i];
                else if (a.size() > 0 && a[0] == '-') {
                    fprintf(stderr, "[ERR] Unknown option: %s\n", a.c_str());
                    return 1;
                } else {
                    inputs.push_back(a);
                }
            }
            if (inputs.empty()) { print_usage(argv[0]); return 1; }
            if (out_path.empty()) {
                out_path = asz_strip_trailing_slashes(inputs[0]) + ".asz";
            }

            AszCreateStats st;
            if (!arc.create(inputs, out_path, opts, &st)) {
                fprintf(stderr, "[ERR] create failed: %s\n", arc.last_error.c_str());
                return 1;
            }
            if (!quiet) {
                double ratio = (st.total_raw_bytes > 0)
                                   ? (100.0 * (double)st.final_archive_bytes / (double)st.total_raw_bytes)
                                   : 0.0;
                double mbs   = (st.elapsed_s > 0.0)
                                   ? ((double)st.total_raw_bytes / (1024.0 * 1024.0) / st.elapsed_s)
                                   : 0.0;
                double ddp   = (st.total_raw_bytes > 0)
                                   ? (100.0 * (double)st.dedup_saved_bytes / (double)st.total_raw_bytes)
                                   : 0.0;
                printf("Created %s (L%u)\n", out_path.c_str(), opts.level);
                printf("  Entries : %u files, %u dirs, %u symlinks\n", st.num_files, st.num_dirs, st.num_symlinks);
                printf("  Size    : %s -> %s (%.2f%%) | Dedup saved: %s (%.1f%%)\n",
                       fmt_bytes(st.total_raw_bytes).c_str(),
                       fmt_bytes(st.final_archive_bytes).c_str(),
                       ratio,
                       fmt_bytes(st.dedup_saved_bytes).c_str(), ddp);
                printf("  Packs   : %u packs (%u unique / %u total chunks) | %.1f ms (%.0f MB/s)\n",
                       st.num_packs, st.num_chunks_unique, st.num_chunks_total,
                       st.elapsed_s * 1e3, mbs);
            }
            return 0;
        }

        // ── Extract (`x` / `extract`) ──────────────────────────────────────────
        if (cmd == "x" || cmd == "extract") {
            std::string asz_path = argv[2];
            std::string out_dir  = ".";
            std::vector<std::string> filters;
            bool quiet = false;

            for (int i = 3; i < argc; ++i) {
                std::string a = argv[i];
                if ((a == "-C" || a == "--dir") && i + 1 < argc) out_dir = argv[++i];
                else if (a == "-q" || a == "--quiet")            quiet = true;
                else filters.push_back(a);
            }

            AszExtractStats st;
            if (!arc.extract(asz_path, out_dir, filters, &st)) {
                fprintf(stderr, "[ERR] extract failed: %s\n", arc.last_error.c_str());
                return 1;
            }
            if (!quiet) {
                double mbs = (st.elapsed_s > 0.0)
                                 ? ((double)st.extracted_bytes / (1024.0 * 1024.0) / st.elapsed_s)
                                 : 0.0;
                printf("Extracted %u files, %u dirs, %u symlinks (%s) to %s\n",
                       st.extracted_files, st.extracted_dirs, st.extracted_links,
                       fmt_bytes(st.extracted_bytes).c_str(), out_dir.c_str());
                printf("  Mode    : %s (%u blocks decoded) | %.1f ms (%.0f MB/s)\n",
                       st.used_partial ? "O(1) Partial Block Random-Access" : "Full Pack Decode",
                       st.blocks_decoded, st.elapsed_s * 1e3, mbs);
            }
            return 0;
        }

        // ── Single-file direct decompress (`d`) ────────────────────────────────
        if (cmd == "d" || cmd == "decompress") {
            if (argc < 4) { print_usage(argv[0]); return 1; }
            std::string asz_path = argv[2];
            std::string out_file = argv[3];
            bool quiet = (argc >= 5 && (strcmp(argv[4], "-q") == 0 || strcmp(argv[4], "--quiet") == 0));

            AszExtractStats st;
            if (!arc.extract_single_file(asz_path, out_file, &st)) {
                fprintf(stderr, "[ERR] decompress failed: %s\n", arc.last_error.c_str());
                return 1;
            }
            if (!quiet) {
                double mbs = (st.elapsed_s > 0.0)
                                 ? ((double)st.extracted_bytes / (1024.0 * 1024.0) / st.elapsed_s)
                                 : 0.0;
                printf("Decompressed %s -> %s (%s) in %.1f ms (%.0f MB/s)\n",
                       asz_path.c_str(), out_file.c_str(), fmt_bytes(st.extracted_bytes).c_str(),
                       st.elapsed_s * 1e3, mbs);
            }
            return 0;
        }

        // ── List (`l` / `list`) ────────────────────────────────────────────────
        if (cmd == "l" || cmd == "list") {
            std::string asz_path = argv[2];
            std::vector<AszListEntry> list;
            AszArchiveSummary sum;
            CFTimeInterval t0 = CACurrentMediaTime();
            if (!arc.list(asz_path, list, sum)) {
                fprintf(stderr, "[ERR] list failed: %s\n", arc.last_error.c_str());
                return 1;
            }
            double ms = (CACurrentMediaTime() - t0) * 1e3;

            printf("Archive: %s (ASZ v%u, L%u, flags=%s%s)\n",
                   asz_path.c_str(), sum.version, sum.level,
                   (sum.flags & ASZ_FLAG_DEDUP) ? "DEDUP " : "",
                   (sum.flags & ASZ_FLAG_SOLID) ? "SOLID" : "");
            double ratio = (sum.total_raw_bytes > 0)
                               ? (100.0 * (double)sum.archive_bytes / (double)sum.total_raw_bytes) : 0.0;
            double ddp   = (sum.total_raw_bytes > 0 && sum.total_raw_bytes >= sum.unique_chunk_bytes)
                               ? (100.0 * (double)(sum.total_raw_bytes - sum.unique_chunk_bytes) / (double)sum.total_raw_bytes)
                               : 0.0;
            printf("Summary: %u entries | %s -> %s (%.2f%%) | Dedup: %.1f%% | %u packs, %u unique chunks | Dir: %u B (read in %.2f ms)\n\n",
                   sum.num_entries,
                   fmt_bytes(sum.total_raw_bytes).c_str(),
                   fmt_bytes(sum.archive_bytes).c_str(), ratio, ddp,
                   sum.num_packs, sum.num_chunks, sum.dir_comp_bytes, ms);

            printf("%-5s %-5s %12s  %8s  %s\n", "Type", "Class", "Size", "CRC32C", "Path");
            printf("----- ----- ------------  --------  ------------------------------------\n");
            for (const auto& e : list) {
                const char* tp = (e.entry_type == ASZ_ENTRY_DIR) ? "DIR" :
                                 (e.entry_type == ASZ_ENTRY_SYMLINK) ? "LINK" : "FILE";
                if (e.entry_type == ASZ_ENTRY_SYMLINK) {
                    printf("%-5s %-5s %12llu  %08x  %s -> %s\n",
                           tp, "-", (unsigned long long)e.raw_size, e.file_crc32c,
                           e.path.c_str(), e.target.c_str());
                } else {
                    printf("%-5s %-5s %12llu  %08x  %s\n",
                           tp, (e.entry_type == ASZ_ENTRY_FILE ? asz_class_name(e.solid_class) : "-"),
                           (unsigned long long)e.raw_size, e.file_crc32c, e.path.c_str());
                }
            }
            return 0;
        }

        // ── Verify (`t` / `test`) ──────────────────────────────────────────────
        if (cmd == "t" || cmd == "test") {
            std::string asz_path = argv[2];
            AszExtractStats st;
            if (!arc.verify(asz_path, &st)) {
                fprintf(stderr, "[FAIL] Integrity check FAILED: %s\n", arc.last_error.c_str());
                return 1;
            }
            double mbs = (st.elapsed_s > 0.0)
                             ? ((double)st.extracted_bytes / (1024.0 * 1024.0) / st.elapsed_s) : 0.0;
            printf("[OK] Verified %s: %u files, %u dirs, %u symlinks (%s, %u packs) in %.1f ms (%.0f MB/s)\n",
                   asz_path.c_str(), st.extracted_files, st.extracted_dirs, st.extracted_links,
                   fmt_bytes(st.extracted_bytes).c_str(), st.packs_decoded, st.elapsed_s * 1e3, mbs);
            return 0;
        }

        print_usage(argv[0]);
        return 1;
    }
}
