// bench_phase2.mm — APLZ-X Phase 2: 正確性検証 & zstd(-T0) / lz4 / APLZ v3 ガチ比較ベンチマーク
//
// 使い方:
//   ./bench_phase2 <shader_dir> [compare_script_mode]
//   ./bench_phase2 <shader_dir> --encode <level> <in_file> <out_asz2>
//   ./bench_phase2 <shader_dir> --decode <in_asz2> <out_file>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CABase.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <vector>
#include <string>
#include <algorithm>

#include "../src/engine/hetero_engine.h"

// ─── 決定的 PRNG ──────────────────────────────────────────────────────────────
static uint64_t g_rng = 0x123456789ABCDEF0ull;
static inline uint64_t rng64() {
    uint64_t x = g_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_rng = x;
    return x;
}

static inline uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static std::vector<uint8_t> read_file_bytes(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return {}; }
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)sz);
    size_t got = fread(buf.data(), 1, (size_t)sz, f);
    fclose(f);
    buf.resize(got);
    return buf;
}

static bool write_file_bytes(const char* path, const uint8_t* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = (n == 0) || (fwrite(data, 1, n, f) == n);
    fclose(f);
    return ok;
}

// ─── 6 種リアルデータセット生成 ────────────────────────────────────────────────
static void gen_dataset(std::vector<uint8_t>& dst, size_t size, int kind, const char* repo_root) {
    dst.resize(size);
    if (size == 0) return;
    g_rng = 0x517CC1B727220A95ull + (uint64_t)kind * 99991ull;

    if (kind == 0) {
        // 0: Real Source + Docs + JSON (リポジトリの実ファイルを結合し、ページ単位で微変化)
        const char* files[] = {
            "README.md", "README.ja.md", "SPEC.md",
            "src/rans/rans32.metal", "src/rans/rans32_cpu.h",
            "src/engine/hetero_lz.h", "src/engine/hetero_engine.h", "src/engine/hetero_kernels.metal",
            "src/archive/aplz_archiver.h", "src/archive/aplz_main.mm"
        };
        std::vector<uint8_t> corpus;
        for (const char* rel : files) {
            std::string p = std::string(repo_root) + "/" + rel;
            auto b = read_file_bytes(p.c_str());
            corpus.insert(corpus.end(), b.begin(), b.end());
        }
        if (corpus.size() < 4096) {
            const char* fb = "The quick brown fox jumps over the lazy dog. {\"id\":12345,\"ok\":true}\n";
            while (corpus.size() < 65536) corpus.insert(corpus.end(), fb, fb + strlen(fb));
        }
        size_t pos = 0;
        uint32_t seq = 0;
        while (pos < size) {
            size_t chunk = std::min(corpus.size(), size - pos);
            memcpy(dst.data() + pos, corpus.data(), chunk);
            // チャンクごとの完全一致を避けるため先頭の数バイトだけシーケンス番号を混ぜる
            if (chunk >= 16) {
                char tag[16];
                int tn = snprintf(tag, sizeof(tag), "/*#%08u*/", seq++);
                memcpy(dst.data() + pos, tag, (size_t)std::min(tn, 12));
            }
            pos += chunk;
        }
    } else if (kind == 1) {
        // 1: Real Mach-O ARM64 Binary (macOS 実バイナリから読み込み)
        const char* bins[] = {
            "/usr/bin/clang", "/usr/bin/git", "/usr/bin/python3",
            "/bin/zsh", "/bin/bash", "/usr/bin/tar", "/usr/bin/curl",
            "/opt/anaconda3/bin/zstd", "/opt/anaconda3/bin/python3"
        };
        std::vector<uint8_t> bin_corpus;
        for (const char* bp : bins) {
            auto b = read_file_bytes(bp);
            if (!b.empty()) {
                // 各バイナリから最大 4 MiB まで収集
                size_t take = std::min(b.size(), (size_t)(4 * 1024 * 1024));
                bin_corpus.insert(bin_corpus.end(), b.begin(), b.begin() + take);
                if (bin_corpus.size() >= size) break;
            }
        }
        if (bin_corpus.size() < 65536) {
            // フォールバック: リアルな ARM64 命令ストリーム (BL 命令多数含む)
            bin_corpus.resize(262144);
            uint32_t* w = (uint32_t*)bin_corpus.data();
            for (size_t i = 0; i < 262144 / 4; ++i) {
                uint64_t r = rng64();
                if ((r & 15u) == 0u) {
                    // BL to one of 64 common target functions (PC-relative)
                    int32_t target = (int32_t)((r >> 4) & 63u) * 32;
                    uint32_t imm26 = (uint32_t)(target - (int32_t)i) & 0x03FFFFFFu;
                    w[i] = 0x94000000u | imm26;
                } else {
                    static const uint32_t common_ops[8] = {
                        0xA9BF7BFDu, 0x910003FDu, 0xF9400000u, 0xAA0003E1u,
                        0xD65F03C0u, 0x52800000u, 0xB4000040u, 0x91000400u
                    };
                    w[i] = common_ops[(r >> 4) & 7u] | (((uint32_t)(r >> 12) & 0x1Fu) << 5u);
                }
            }
        }
        size_t pos = 0;
        while (pos < size) {
            size_t chunk = std::min(bin_corpus.size(), size - pos);
            memcpy(dst.data() + pos, bin_corpus.data(), chunk);
            pos += chunk;
        }
    } else if (kind == 2) {
        // 2: FP16 AI Weights (LLM 重み相当: N(0, 0.02) ガウス分布を IEEE-754 fp16 へ)
        size_t n16 = size / 2u;
        uint16_t* p16 = (uint16_t*)dst.data();
        for (size_t i = 0; i < n16; ++i) {
            // Irwin-Hall 近似で超高速ガウス乱数 (平均 0, 標準偏差 ~0.025)
            int32_t s = 0;
            uint64_t r = rng64();
            s += (int32_t)(r & 0xFFFFu); r >>= 16;
            s += (int32_t)(r & 0xFFFFu); r >>= 16;
            s += (int32_t)(r & 0xFFFFu); r >>= 16;
            s += (int32_t)(r & 0xFFFFu);
            float val = ((float)s / 65536.0f - 2.0f) * 0.04f;
            p16[i] = float_to_fp16(val);
        }
        if (size & 1u) dst[size - 1u] = 0;
    } else if (kind == 3) {
        // 3: FP32 Telemetry / Scientific (滑らかな正弦波＋微小ノイズの時系列 FP32 配列)
        size_t n32 = size / 4u;
        float* pf = (float*)dst.data();
        for (size_t i = 0; i < n32; ++i) {
            float noise = ((float)(int32_t)(rng64() & 255u) - 128.0f) * 1e-5f;
            pf[i] = 100.0f + 12.5f * sinf((float)(i & 1023u) * 0.006135923f) + noise;
        }
        for (size_t i = n32 * 4u; i < size; ++i) dst[i] = 0;
    } else if (kind == 4) {
        // 4: Mixed Realistic (25% Source + 25% ARM64 + 25% FP16 + 25% Random)
        size_t q = size / 4u;
        std::vector<uint8_t> part;
        gen_dataset(part, q, 0, repo_root); memcpy(dst.data() + 0 * q, part.data(), q);
        gen_dataset(part, q, 1, repo_root); memcpy(dst.data() + 1 * q, part.data(), q);
        gen_dataset(part, q, 2, repo_root); memcpy(dst.data() + 2 * q, part.data(), q);
        gen_dataset(part, size - 3 * q, 5, repo_root); memcpy(dst.data() + 3 * q, part.data(), size - 3 * q);
    } else {
        // 5: Random (暗号化・圧縮済み相当)
        uint64_t* p64 = (uint64_t*)dst.data();
        for (size_t i = 0; i < size / 8u; ++i) p64[i] = rng64();
        for (size_t i = (size / 8u) * 8u; i < size; ++i) dst[i] = (uint8_t)rng64();
    }
}

// ─── 1. 厳密正確性スイート ─────────────────────────────────────────────────────
static void run_correctness(HeteroEngine& eng, const char* repo_root) {
    printf("=== [1] Phase 2 Correctness & Corruption Detection Suite ===\n");
    const size_t sizes[] = {
        0, 1, 15, 63, 64, 255, 256, 4096, 65535, 65536, 262143, 262144, 262145, 524300
    };
    const uint32_t levels[] = { 1u, 3u, 6u };

    std::vector<uint8_t> src, arch, dec;
    uint32_t passed = 0;
    for (size_t sz : sizes) {
        for (int kind = 0; kind < 6; ++kind) {
            gen_dataset(src, sz, kind, repo_root);
            for (uint32_t lv : levels) {
                if (!eng.compress(src.data(), src.size(), lv, arch)) {
                    fprintf(stderr, "[FAIL] compress failed sz=%zu kind=%d lv=%u\n", sz, kind, lv);
                    exit(1);
                }
                dec.clear();
                if (!eng.decompress(arch.data(), arch.size(), dec) ||
                    dec.size() != src.size() ||
                    (sz > 0 && memcmp(dec.data(), src.data(), sz) != 0)) {
                    fprintf(stderr, "[FAIL] round-trip mismatch sz=%zu kind=%d lv=%u\n", sz, kind, lv);
                    exit(1);
                }
                passed++;
            }
        }
    }

    // 破損検知テスト (1-bit フリップ)
    gen_dataset(src, 300000, 0, repo_root);
    eng.compress(src.data(), src.size(), 3u, arch);
    size_t flip_pos = arch.size() / 2u;
    arch[flip_pos] ^= 0x01u;
    if (eng.decompress(arch.data(), arch.size(), dec)) {
        fprintf(stderr, "[FAIL] 1-bit corruption was NOT detected!\n");
        exit(1);
    }

    // GPU 32-way rANS と CPU ブランチレス 32-way rANS のビット完全一致テスト (全6データセット×3レベル)
    std::vector<uint8_t> arch_gpu, arch_cpu, dec_cross;
    for (int kind = 0; kind < 6; ++kind) {
        gen_dataset(src, 300000, kind, repo_root);
        for (uint32_t lv : levels) {
            eng.force_cpu_rans = false;
            if (!eng.compress(src.data(), src.size(), lv, arch_gpu)) {
                fprintf(stderr, "[FAIL] GPU parity compress failed kind=%d lv=%u\n", kind, lv);
                exit(1);
            }
            eng.force_cpu_rans = true;
            if (!eng.compress(src.data(), src.size(), lv, arch_cpu)) {
                fprintf(stderr, "[FAIL] CPU parity compress failed kind=%d lv=%u\n", kind, lv);
                exit(1);
            }
            if (arch_gpu != arch_cpu) {
                fprintf(stderr, "[FAIL] GPU vs CPU archive NOT bit-exact! kind=%d lv=%u (gpu=%zu cpu=%zu)\n",
                        kind, lv, arch_gpu.size(), arch_cpu.size());
                exit(1);
            }
            // CPU 復号で GPU 生成アーカイブを検証 & GPU 復号で CPU 生成アーカイブを検証
            if (!eng.decompress(arch_gpu.data(), arch_gpu.size(), dec_cross) || dec_cross != src) {
                fprintf(stderr, "[FAIL] CPU decode of GPU archive mismatch kind=%d lv=%u\n", kind, lv);
                exit(1);
            }
            eng.force_cpu_rans = false;
            if (!eng.decompress(arch_cpu.data(), arch_cpu.size(), dec_cross) || dec_cross != src) {
                fprintf(stderr, "[FAIL] GPU decode of CPU archive mismatch kind=%d lv=%u\n", kind, lv);
                exit(1);
            }
        }
    }

    printf("  [PASS] %u boundary/dataset/level round-trip cases + 1-bit corruption + 18 GPU==CPU bit-exact parity: ALL OK!\n\n", passed);
}

// ─── 2. 64 MiB インメモリ詳細ベンチマーク (各ステージ内訳付き) ──────────────────
static void run_inmem_bench(HeteroEngine& eng, const char* repo_root) {
    const size_t SIZE_MB = 64u;
    const size_t SIZE    = SIZE_MB * 1024u * 1024u;
    const char* names[6] = {
        "Source+Docs", "Mach-O ARM64", "FP16 Weights", "FP32 Telemetry", "Mixed Real", "Random"
    };

    printf("=== [2] APLZ-X Phase 2 In-Memory Benchmark (%zu MiB, best of 3 runs) ===\n", SIZE_MB);
    printf("Device: %s\n\n", eng.dev.name.UTF8String);

    std::vector<uint8_t> src, arch, dec;

    // Warmup
    gen_dataset(src, 4 * 1024 * 1024, 0, repo_root);
    eng.compress(src.data(), src.size(), 3u, arch);
    eng.decompress(arch.data(), arch.size(), dec);

    printf("%-15s | Lv |  Ratio |  Comp MB/s | Decomp MB/s | Comp(ms): LZ/Sub/Wait  | Dec(ms): Prs/rANS/LZ  | Codecs(LZ/rA/RL/RW) Xfm(S2/S4/BCJ)\n",
           "Dataset");
    printf("----------------+----+--------+------------+-------------+------------------------+-----------------------+-----------------------------------\n");

    for (int kind = 0; kind < 6; ++kind) {
        gen_dataset(src, SIZE, kind, repo_root);
        for (uint32_t lv : { 1u, 3u, 6u }) {
            HxTimingStats best_cs, best_ds;
            best_cs.total_s = 1e9;
            best_ds.total_s = 1e9;
            for (int r = 0; r < 3; ++r) {
                HxTimingStats cs;
                eng.compress(src.data(), src.size(), lv, arch, &cs);
                if (cs.total_s < best_cs.total_s) best_cs = cs;
            }
            for (int r = 0; r < 3; ++r) {
                HxTimingStats ds;
                bool ok = eng.decompress(arch.data(), arch.size(), dec, &ds);
                if (!ok || dec.size() != src.size() || memcmp(dec.data(), src.data(), SIZE) != 0) {
                    fprintf(stderr, "[FAIL] Bench RT mismatch kind=%d lv=%u\n", kind, lv);
                    exit(1);
                }
                if (ds.total_s < best_ds.total_s) best_ds = ds;
            }

            double ratio = 100.0 * (double)arch.size() / (double)SIZE;
            double cmbs  = (double)SIZE_MB / best_cs.total_s;
            double dmbs  = (double)SIZE_MB / best_ds.total_s;

            printf("%-15s | L%u | %5.1f%% | %7.0f MB/s | %8.0f MB/s | %5.1f / %4.1f / %5.1f  | %4.1f / %5.1f / %5.1f | %3u/%3u/%2u/%3u   %3u/%3u/%3u\n",
                   (lv == 1u ? names[kind] : ""),
                   lv, ratio, cmbs, dmbs,
                   best_cs.lz_s * 1e3, best_cs.rans_s * 1e3, best_cs.pack_s * 1e3,
                   best_ds.pack_s * 1e3, best_ds.rans_s * 1e3, best_ds.lz_s * 1e3,
                   best_cs.cnt_lz, best_cs.cnt_rans_only, best_cs.cnt_rle, best_cs.cnt_raw,
                   best_cs.cnt_shuf2, best_cs.cnt_shuf4, best_cs.cnt_bcj);
        }
        printf("----------------+----+--------+------------+-------------+------------------------+-----------------------+-----------------------------------\n");
    }
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage:\n"
                            "  %s <shader_dir> [repo_root]\n"
                            "  %s <shader_dir> --gen-datasets <out_dir> <size_mb> <repo_root>\n"
                            "  %s <shader_dir> --encode <level> <in_file> <out_asz2>\n"
                            "  %s <shader_dir> --decode <in_asz2> <out_file>\n",
                    argv[0], argv[0], argv[0], argv[0]);
            return 1;
        }
        const char* shader_dir = argv[1];

        if (argc >= 6 && strcmp(argv[2], "--gen-datasets") == 0) {
            const char* out_dir = argv[3];
            size_t size_mb = (size_t)atoi(argv[4]);
            const char* repo_root = argv[5];
            const char* fnames[6] = {
                "source.bin", "macho_arm64.bin", "fp16_weights.bin",
                "fp32_telemetry.bin", "mixed_real.bin", "random.bin"
            };
            std::vector<uint8_t> buf;
            for (int k = 0; k < 6; ++k) {
                gen_dataset(buf, size_mb * 1024u * 1024u, k, repo_root);
                std::string p = std::string(out_dir) + "/" + fnames[k];
                if (!write_file_bytes(p.c_str(), buf.data(), buf.size())) {
                    fprintf(stderr, "[ERR] Failed writing %s\n", p.c_str());
                    return 1;
                }
            }
            return 0;
        }

        HeteroEngine eng;
        // --encode / --decode 単発起動は遅延 Metal 初期化 (RAW や小ファイルで起動遅延ゼロ!)
        bool is_cli_oneshot = (argc >= 5 && (strcmp(argv[2], "--encode") == 0 || strcmp(argv[2], "--decode") == 0));
        if (!eng.init(shader_dir, !is_cli_oneshot)) {
            fprintf(stderr, "[ERR] Failed to initialize HeteroEngine.\n");
            return 1;
        }

        if (argc >= 6 && strcmp(argv[2], "--encode") == 0) {
            uint32_t lv = (uint32_t)atoi(argv[3]);
            return eng.compress_file(argv[4], argv[5], lv) ? 0 : 1;
        }
        if (argc >= 5 && strcmp(argv[2], "--decode") == 0) {
            return eng.decompress_file(argv[3], argv[4]) ? 0 : 1;
        }

        const char* repo_root = (argc >= 3) ? argv[2] : "../..";
        run_correctness(eng, repo_root);
        run_inmem_bench(eng, repo_root);
        return 0;
    }
}
