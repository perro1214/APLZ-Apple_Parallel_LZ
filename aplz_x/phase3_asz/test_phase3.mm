// test_phase3.mm — APLZ-X Phase 3: `.asz` 包括的正当性・重複排除・部分展開・セキュリティ検証スイート
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <string>

#include "asz_archiver.h"

static uint64_t g_rng = 0x123456789ABCDEF0ULL;
static inline uint64_t next_rng() {
    uint64_t x = g_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_rng = x;
    return x;
}

static void write_bytes(const std::string& p, const uint8_t* data, size_t n) {
    std::string pd = asz_parent_dir(p);
    if (!pd.empty()) asz_mkdir_p(pd);
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) { fprintf(stderr, "[FAIL] cannot write %s\n", p.c_str()); exit(1); }
    if (n > 0u && fwrite(data, 1, n, f) != n) { fprintf(stderr, "[FAIL] short write %s\n", p.c_str()); exit(1); }
    fclose(f);
}

static std::vector<uint8_t> read_bytes(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "[FAIL] cannot read %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b((size_t)sz);
    if (sz > 0 && fread(b.data(), 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "[FAIL] short read %s\n", p.c_str()); exit(1);
    }
    fclose(f);
    return b;
}

static void rm_rf(const std::string& path) {
    std::string cmd = "rm -rf \"" + path + "\"";
    int rc = system(cmd.c_str());
    (void)rc;
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "Usage: %s <shader_dir>\n", argv[0]); return 1; }
        const char* shader_dir = argv[1];

        AszArchiver arc;
        if (!arc.init(shader_dir, false)) {
            fprintf(stderr, "[FAIL] AszArchiver init failed\n");
            return 1;
        }

        char tmpl[] = "/tmp/aplzx_p3_test_XXXXXX";
        char* td_c = mkdtemp(tmpl);
        if (!td_c) { fprintf(stderr, "[FAIL] mkdtemp failed\n"); return 1; }
        std::string td = td_c;

        printf("=== Phase 3 (.asz) Automated Verification Suite ===\n");

        // ── Test 1: 境界サイズ単一ファイル往復 (0B .. 524300B, L1/L3/L6) ────────
        {
            const size_t sizes[] = { 0, 1, 63, 65535, 65536, 262144, 524300 };
            for (size_t sz : sizes) {
                std::vector<uint8_t> buf(sz);
                for (size_t i = 0; i < sz; ++i) buf[i] = (uint8_t)((i * 31u + (i >> 6u)) & 0xFFu);
                std::string fpath = td + "/t1_in.bin";
                std::string apath = td + "/t1.asz";
                std::string dpath = td + "/t1_out.bin";
                write_bytes(fpath, buf.data(), buf.size());

                for (uint32_t lv : { 1u, 3u, 6u }) {
                    AszCreateOptions opt; opt.level = lv;
                    if (!arc.create({ fpath }, apath, opt)) {
                        fprintf(stderr, "[FAIL] T1 create sz=%zu lv=%u: %s\n", sz, lv, arc.last_error.c_str());
                        rm_rf(td); return 1;
                    }
                    if (!arc.verify(apath)) {
                        fprintf(stderr, "[FAIL] T1 verify sz=%zu lv=%u: %s\n", sz, lv, arc.last_error.c_str());
                        rm_rf(td); return 1;
                    }
                    if (!arc.extract_single_file(apath, dpath)) {
                        fprintf(stderr, "[FAIL] T1 extract sz=%zu lv=%u: %s\n", sz, lv, arc.last_error.c_str());
                        rm_rf(td); return 1;
                    }
                    auto got = read_bytes(dpath);
                    if (got != buf) {
                        fprintf(stderr, "[FAIL] T1 mismatch sz=%zu lv=%u\n", sz, lv);
                        rm_rf(td); return 1;
                    }
                }
            }
            printf("  [PASS] Test 1: Boundary single-file round-trip (0B..524KB x L1/L3/L6)\n");
        }

        // ── Test 2: 複合ディレクトリ (空Dir, 空File, Symlink, 実行権限, 異種ファイル混在) ─
        {
            std::string root = td + "/proj";
            asz_mkdir_p(root + "/empty_dir");
            asz_mkdir_p(root + "/src/deep");
            asz_mkdir_p(root + "/bin");
            asz_mkdir_p(root + "/data");

            write_bytes(root + "/empty.txt", nullptr, 0);
            std::string code = "#include <stdio.h>\nint main() { printf(\"Hello APLZ-X!\\n\"); return 0; }\n";
            write_bytes(root + "/src/deep/main.cpp", (const uint8_t*)code.data(), code.size());
            std::string sh = "#!/bin/sh\necho ok\n";
            write_bytes(root + "/bin/run.sh", (const uint8_t*)sh.data(), sh.size());
            chmod((root + "/bin/run.sh").c_str(), 0755);

            // Symlink
            if (symlink("src/deep/main.cpp", (root + "/main_link.cpp").c_str()) != 0) {
                fprintf(stderr, "[FAIL] T2 symlink create failed\n"); rm_rf(td); return 1;
            }

            // 100 KiB 数値ファイル + その完全複製 (重複排除確認)
            std::vector<uint8_t> f32(100000);
            float* pf = (float*)f32.data();
            for (size_t i = 0; i < 25000; ++i) pf[i] = 42.0f + 3.5f * (float)(i & 255u);
            write_bytes(root + "/data/v1.fp32", f32.data(), f32.size());
            write_bytes(root + "/data/v2_copy.fp32", f32.data(), f32.size());

            std::string apath = td + "/proj.asz";
            AszCreateOptions opt; opt.level = 3u;
            AszCreateStats cs;
            if (!arc.create({ root }, apath, opt, &cs)) {
                fprintf(stderr, "[FAIL] T2 create: %s\n", arc.last_error.c_str()); rm_rf(td); return 1;
            }
            if (cs.dedup_saved_bytes < 95000u) {
                fprintf(stderr, "[FAIL] T2 expected dedup >= 95KB, got %llu\n", (unsigned long long)cs.dedup_saved_bytes);
                rm_rf(td); return 1;
            }

            std::string out_dir = td + "/proj_out";
            AszExtractStats es;
            if (!arc.extract(apath, out_dir, {}, &es)) {
                fprintf(stderr, "[FAIL] T2 extract: %s\n", arc.last_error.c_str()); rm_rf(td); return 1;
            }

            // 空ディレクトリ・空ファイル・実行権限・シンボリックリンク・バイナリ一致を厳密検査
            struct stat st;
            if (lstat((out_dir + "/proj/empty_dir").c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                fprintf(stderr, "[FAIL] T2 empty_dir missing\n"); rm_rf(td); return 1;
            }
            if (read_bytes(out_dir + "/proj/empty.txt").size() != 0u) {
                fprintf(stderr, "[FAIL] T2 empty.txt mismatch\n"); rm_rf(td); return 1;
            }
            if (lstat((out_dir + "/proj/bin/run.sh").c_str(), &st) != 0 || ((st.st_mode & 0111u) == 0u)) {
                fprintf(stderr, "[FAIL] T2 executable bit lost on run.sh\n"); rm_rf(td); return 1;
            }
            char lbuf[256];
            ssize_t ln = readlink((out_dir + "/proj/main_link.cpp").c_str(), lbuf, sizeof(lbuf) - 1u);
            if (ln < 0 || std::string(lbuf, (size_t)ln) != "src/deep/main.cpp") {
                fprintf(stderr, "[FAIL] T2 symlink target mismatch\n"); rm_rf(td); return 1;
            }
            if (read_bytes(out_dir + "/proj/data/v1.fp32") != f32 ||
                read_bytes(out_dir + "/proj/data/v2_copy.fp32") != f32) {
                fprintf(stderr, "[FAIL] T2 deduped fp32 mismatch\n"); rm_rf(td); return 1;
            }
            printf("  [PASS] Test 2: Directory tree (empty dir/file, symlink, chmod 0755, dedup=%llu B)\n",
                   (unsigned long long)cs.dedup_saved_bytes);
        }

        // ── Test 3: FastCDC 17バイト先頭シフト重複排除耐性テスト ──────────────────
        {
            std::string d = td + "/shift_test";
            asz_mkdir_p(d);
            const size_t N = 2u * 1024u * 1024u;
            std::vector<uint8_t> base(N);
            for (size_t i = 0; i < N; ++i) base[i] = (uint8_t)(next_rng() & 0xFFu);
            // 先頭に 17 バイトを挿入したバージョン 2 (固定チャンクでは 0% 重複排除、FastCDC なら >90%!)
            std::vector<uint8_t> shifted;
            const char* ins = "INSERT_17_BYTES!!";
            shifted.insert(shifted.end(), ins, ins + 17);
            shifted.insert(shifted.end(), base.begin(), base.end());

            write_bytes(d + "/v1.bin", base.data(), base.size());
            write_bytes(d + "/v2_shifted.bin", shifted.data(), shifted.size());

            std::string apath = td + "/shift.asz";
            AszCreateOptions opt; opt.level = 1u; opt.enable_dedup = true;
            AszCreateStats cs;
            if (!arc.create({ d }, apath, opt, &cs)) {
                fprintf(stderr, "[FAIL] T3 create: %s\n", arc.last_error.c_str()); rm_rf(td); return 1;
            }
            double saved_pct = 100.0 * (double)cs.dedup_saved_bytes / (double)N;
            if (saved_pct < 85.0) {
                fprintf(stderr, "[FAIL] T3 FastCDC shift dedup too low: %.1f%%\n", saved_pct);
                rm_rf(td); return 1;
            }
            std::string out = td + "/shift_out";
            if (!arc.extract(apath, out, {}) ||
                read_bytes(out + "/shift_test/v1.bin") != base ||
                read_bytes(out + "/shift_test/v2_shifted.bin") != shifted) {
                fprintf(stderr, "[FAIL] T3 round-trip mismatch\n"); rm_rf(td); return 1;
            }
            printf("  [PASS] Test 3: FastCDC 17-byte-shift dedup saved %.1f%% of 2 MiB random file & bit-exact RT\n", saved_pct);
        }

        // ── Test 4: 型別 Solid Grouping (小ファイル 200 個のクロスファイル圧縮効果) ─
        {
            std::string d = td + "/solid_test";
            asz_mkdir_p(d);
            std::string tmpl_txt =
                "// APLZ-X Solid Grouping benchmark header\n"
                "#pragma once\n"
                "#include <stdint.h>\n"
                "namespace aplzx { struct MetricRecord { uint64_t ts; double latency_ms; uint32_t code; }; }\n";
            for (int k = 0; k < 200; ++k) {
                std::string body;
                for (int r = 0; r < 24; ++r) {
                    body += tmpl_txt;
                    body += "static const int REC_ID_" + std::to_string(k) + "_" + std::to_string(r) + " = " + std::to_string(k * 100 + r) + ";\n";
                }
                char fn[64];
                snprintf(fn, sizeof(fn), "/mod_%03d.hpp", k);
                write_bytes(d + fn, (const uint8_t*)body.data(), body.size());
            }
            std::string apath = td + "/solid.asz";
            AszCreateOptions opt; opt.level = 3u; opt.enable_solid = true;
            AszCreateStats cs;
            if (!arc.create({ d }, apath, opt, &cs)) {
                fprintf(stderr, "[FAIL] T4 create: %s\n", arc.last_error.c_str()); rm_rf(td); return 1;
            }
            double ratio = 100.0 * (double)cs.final_archive_bytes / (double)cs.total_raw_bytes;
            if (ratio > 5.0 || !arc.verify(apath)) {
                fprintf(stderr, "[FAIL] T4 solid ratio too high (%.2f%%) or verify failed\n", ratio);
                rm_rf(td); return 1;
            }
            printf("  [PASS] Test 4: Solid Grouping on 200 small .hpp files (%llu B -> %llu B, %.2f%%, %u pack)\n",
                   (unsigned long long)cs.total_raw_bytes, (unsigned long long)cs.final_archive_bytes, ratio, cs.num_packs);
        }

        // ── Test 5: O(1) 256 KiB ブロック単位ランダムアクセス部分展開テスト ────────
        {
            std::string d = td + "/partial_test";
            asz_mkdir_p(d);
            // 16 MiB (64 ブロック) 分のバイナリ + 4 KiB の目的テキストファイル 1 つ
            for (int k = 0; k < 4; ++k) {
                std::vector<uint8_t> big(4u * 1024u * 1024u);
                for (size_t i = 0; i < big.size(); ++i) big[i] = (uint8_t)((i * 17u + k * 53u) & 0xFFu);
                write_bytes(d + "/blob_" + std::to_string(k) + ".bin", big.data(), big.size());
            }
            std::string needle = "APLZ-X O(1) Partial Random-Access Extraction Verified!\n";
            while (needle.size() < 4096u) needle += "Line padding for 4 KiB target file inside 16 MiB archive.\n";
            write_bytes(d + "/needle.txt", (const uint8_t*)needle.data(), needle.size());

            std::string apath = td + "/partial.asz";
            AszCreateOptions opt; opt.level = 1u;
            if (!arc.create({ d }, apath, opt)) {
                fprintf(stderr, "[FAIL] T5 create: %s\n", arc.last_error.c_str()); rm_rf(td); return 1;
            }

            std::string out_p = td + "/partial_out";
            AszExtractStats es;
            if (!arc.extract(apath, out_p, { "partial_test/needle.txt" }, &es)) {
                fprintf(stderr, "[FAIL] T5 partial extract: %s\n", arc.last_error.c_str()); rm_rf(td); return 1;
            }
            auto got = read_bytes(out_p + "/partial_test/needle.txt");
            if (got.size() != needle.size() || memcmp(got.data(), needle.data(), needle.size()) != 0) {
                fprintf(stderr, "[FAIL] T5 needle content mismatch\n"); rm_rf(td); return 1;
            }
            if (!es.used_partial || es.blocks_decoded > 2u) {
                fprintf(stderr, "[FAIL] T5 expected partial mode with <=2 blocks decoded, got partial=%d blocks=%u\n",
                        (int)es.used_partial, es.blocks_decoded);
                rm_rf(td); return 1;
            }
            printf("  [PASS] Test 5: O(1) Partial extract of 4 KiB from 16 MiB archive decoded ONLY %u block (%.2f ms)\n",
                   es.blocks_decoded, es.elapsed_s * 1e3);
        }

        // ── Test 6: 3 重破損検知 (Header / Pack / Central Directory の 1-bit 反転) ─
        {
            std::string apath = td + "/proj.asz";
            auto orig = read_bytes(apath);
            AszFileHeader fh;
            memcpy(&fh, orig.data(), sizeof(fh));

            size_t flip_locs[3] = {
                8u,                                               // AszFileHeader 内
                sizeof(AszFileHeader) + 32u,                      // Pack ペイロード内
                (size_t)fh.directory_offset + 16u                 // Central Directory 内
            };
            for (int k = 0; k < 3; ++k) {
                auto bad = orig;
                bad[flip_locs[k]] ^= 0x01u;
                std::string bad_p = td + "/bad_" + std::to_string(k) + ".asz";
                write_bytes(bad_p, bad.data(), bad.size());
                if (arc.verify(bad_p)) {
                    fprintf(stderr, "[FAIL] T6 corruption #%d was NOT detected!\n", k);
                    rm_rf(td); return 1;
                }
            }
            printf("  [PASS] Test 6: 1-bit corruption detected in Header, Pack payload, and Central Directory\n");
        }

        // ── Test 7: Zip-Slip パストラバーサル防御ユニットテスト ────────────────────
        {
            const char* bad_paths[] = {
                "", "/etc/passwd", "../evil", "a/../../evil", "a/./b", "a//b", "a/b/", "a\\b"
            };
            for (const char* bp : bad_paths) {
                if (asz_is_safe_rel_path(bp)) {
                    fprintf(stderr, "[FAIL] T7 unsafe path accepted: '%s'\n", bp);
                    rm_rf(td); return 1;
                }
            }
            const char* good_paths[] = {
                "file.txt", "dir/sub/file.cpp", ".gitignore", "dir/.hidden/a..b"
            };
            for (const char* gp : good_paths) {
                if (!asz_is_safe_rel_path(gp)) {
                    fprintf(stderr, "[FAIL] T7 safe path rejected: '%s'\n", gp);
                    rm_rf(td); return 1;
                }
            }
            printf("  [PASS] Test 7: Zip-Slip path sanitization verified\n");
        }

        // ── Test 8: GF(2) hx_crc32c_combine の厳密数学一致テスト (100 分割ケース) ──
        {
            std::vector<uint8_t> buf(300000);
            for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t)(next_rng() & 0xFFu);
            uint32_t full_crc = hx_crc32c(buf.data(), buf.size());
            const size_t splits[] = { 0, 1, 2, 3, 7, 8, 15, 16, 63, 64, 255, 4096, 65535, 65536, 131072, 262144, 299999, 300000 };
            for (size_t sp : splits) {
                uint32_t c1 = hx_crc32c(buf.data(), sp);
                uint32_t c2 = hx_crc32c(buf.data() + sp, buf.size() - sp);
                if (hx_crc32c_combine(c1, c2, buf.size() - sp) != full_crc) {
                    fprintf(stderr, "[FAIL] T8 crc32c_combine mismatch at split=%zu\n", sp);
                    rm_rf(td); return 1;
                }
            }
            // 多段チャンク結合 (ランダム長チャンク群の逐次結合)
            uint32_t acc = 0u;
            size_t pos = 0u;
            while (pos < buf.size()) {
                size_t clen = std::min(buf.size() - pos, (size_t)(1u + (next_rng() % 7777u)));
                acc = hx_crc32c_combine(acc, hx_crc32c(buf.data() + pos, clen), clen);
                pos += clen;
            }
            if (acc != full_crc) {
                fprintf(stderr, "[FAIL] T8 multi-chunk crc32c_combine mismatch!\n");
                rm_rf(td); return 1;
            }
            printf("  [PASS] Test 8: GF(2) hx_crc32c_combine bit-exact across boundary & multi-chunk splits\n");
        }

        rm_rf(td);
        printf("=== ALL 8 PHASE 3 TESTS PASSED! ===\n");
        return 0;
    }
}
