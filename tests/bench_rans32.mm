// bench_rans32.mm — APLZ-X Phase 1: 32-way Coalesced rANS 正確性検証 & ベンチマーク
//
// 検証項目:
//   1. 境界サイズ (0 B .. 524300 B) での GPU Enc == CPU Scalar Enc ビット完全一致
//   2. GPU Enc → GPU Dec / CPU Scalar Dec / CPU NEON Dec の全クロス完全一致
//   3. R32P バイナリシリアライズ / デシリアライズ (Sparse + Dense12 両経路)
//   4. 破損データ (不正 state, 1-bit 反転, 切り詰め) の 100% 検知
//   5. 64 MiB × 6 種データセットでの GPU / CPU-NEON(1c/all) / CPU-Scalar / Hybrid 実測

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CABase.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <dispatch/dispatch.h>

#include "../src/rans/rans32_format.h"
#include "../src/rans/rans32_cpu.h"

// ─── シェーダロード ───────────────────────────────────────────────────────────
static id<MTLLibrary> load_rans32_library(id<MTLDevice> dev, const char* dir) {
    NSError* err = nil;
    NSString* metallib_path = [NSString stringWithFormat:@"%s/rans32.metallib", dir];
    if ([[NSFileManager defaultManager] fileExistsAtPath:metallib_path]) {
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:metallib_path] error:&err];
        if (lib) return lib;
        fprintf(stderr, "[WARN] metallib load failed: %s\n", err.localizedDescription.UTF8String);
    }
    NSString* src_path = [NSString stringWithFormat:@"%s/rans32.metal", dir];
    NSString* src = [NSString stringWithContentsOfFile:src_path encoding:NSUTF8StringEncoding error:&err];
    if (!src) {
        fprintf(stderr, "[ERR] Cannot read %s: %s\n", src_path.UTF8String, err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.languageVersion = MTLLanguageVersion3_0;
    opt.mathMode        = MTLMathModeFast;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) {
        fprintf(stderr, "[ERR] Metal compile error:\n%s\n", err.localizedDescription.UTF8String);
        exit(EXIT_FAILURE);
    }
    return lib;
}

// ─── GPU コンテキスト ─────────────────────────────────────────────────────────
struct GpuRans32 {
    id<MTLDevice>               dev;
    id<MTLCommandQueue>         cq;
    id<MTLComputePipelineState> pso_build;
    id<MTLComputePipelineState> pso_encode;
    id<MTLComputePipelineState> pso_decode;

    bool init(const char* dir) {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) return false;
        cq = [dev newCommandQueue];
        id<MTLLibrary> lib = load_rans32_library(dev, dir);
        NSError* err = nil;
        auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) { fprintf(stderr, "[ERR] kernel '%s' not found\n", name.UTF8String); exit(1); }
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { fprintf(stderr, "[ERR] PSO '%s': %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
            return pso;
        };
        pso_build  = make_pso(@"rans32_build_tables");
        pso_encode = make_pso(@"rans32_encode");
        pso_decode = make_pso(@"rans32_decode");
        return true;
    }
};

// ─── GPU ワークスペース (UMA Shared) ──────────────────────────────────────────
struct GpuWorkspace {
    uint32_t max_bytes  = 0;
    uint32_t max_blocks = 0;
    uint32_t max_tiles  = 0;

    id<MTLBuffer> buf_in;
    id<MTLBuffer> buf_out;
    id<MTLBuffer> buf_enc_sym;
    id<MTLBuffer> buf_dec_lut;
    id<MTLBuffer> buf_bmeta;
    id<MTLBuffer> buf_words;
    id<MTLBuffer> buf_fstates;
    id<MTLBuffer> buf_tmeta;
    id<MTLBuffer> buf_err;

    void allocate(id<MTLDevice> dev, uint32_t bytes) {
        max_bytes  = std::max(bytes, (uint32_t)RANS32_BLOCK_SIZE);
        max_blocks = (max_bytes + RANS32_BLOCK_SIZE - 1u) / RANS32_BLOCK_SIZE;
        max_tiles  = max_blocks * RANS32_TILES_PER_BLOCK;

        auto alloc = [&](size_t len) {
            id<MTLBuffer> b = [dev newBufferWithLength:std::max(len, (size_t)64)
                                               options:MTLResourceStorageModeShared];
            if (!b) { fprintf(stderr, "[ERR] MTLBuffer alloc failed (%zu B)\n", len); exit(1); }
            return b;
        };
        buf_in      = alloc(max_bytes);
        buf_out     = alloc(max_bytes);
        buf_enc_sym = alloc((size_t)max_blocks * RANS32_N_SYMBOLS * sizeof(Rans32EncSym));
        buf_dec_lut = alloc((size_t)max_blocks * RANS32_PROB_SCALE * sizeof(uint32_t));
        buf_bmeta   = alloc((size_t)max_blocks * sizeof(Rans32BlockMeta));
        buf_words   = alloc((size_t)max_tiles  * RANS32_TILE_MAX_WORDS * sizeof(uint16_t));
        buf_fstates = alloc((size_t)max_tiles  * RANS32_LANES * sizeof(uint32_t));
        buf_tmeta   = alloc((size_t)max_tiles  * sizeof(Rans32TileMeta));
        buf_err     = alloc((size_t)max_tiles  * sizeof(uint32_t));
    }
};

// GPU エンコード実行 (1 コマンドバッファで build_tables → encode 連続投入)
static double gpu_run_encode(GpuRans32& g, GpuWorkspace& ws, uint32_t total_bytes) {
    if (total_bytes == 0) return 0.0;
    uint32_t num_blocks = (total_bytes + RANS32_BLOCK_SIZE - 1u) / RANS32_BLOCK_SIZE;

    CFTimeInterval t0 = CACurrentMediaTime();
    id<MTLCommandBuffer> cb = [g.cq commandBuffer];
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g.pso_build];
        [enc setBuffer:ws.buf_in      offset:0 atIndex:0];
        [enc setBuffer:ws.buf_enc_sym offset:0 atIndex:1];
        [enc setBuffer:ws.buf_dec_lut offset:0 atIndex:2];
        [enc setBuffer:ws.buf_bmeta   offset:0 atIndex:3];
        [enc setBytes:&total_bytes    length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(RANS32_TG_SIZE, 1, 1)];
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g.pso_encode];
        [enc setBuffer:ws.buf_in      offset:0 atIndex:0];
        [enc setBuffer:ws.buf_enc_sym offset:0 atIndex:1];
        [enc setBuffer:ws.buf_bmeta   offset:0 atIndex:2];
        [enc setBuffer:ws.buf_words   offset:0 atIndex:3];
        [enc setBuffer:ws.buf_fstates offset:0 atIndex:4];
        [enc setBuffer:ws.buf_tmeta   offset:0 atIndex:5];
        [enc setBytes:&total_bytes    length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(RANS32_TG_SIZE, 1, 1)];
        [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "[ERR] GPU encode error: %s\n", cb.error.localizedDescription.UTF8String);
        exit(1);
    }
    return CACurrentMediaTime() - t0;
}

// GPU デコード実行 (部分ブロック範囲にも対応: block_start..block_start+num_blocks-1)
static double gpu_run_decode_range(
    GpuRans32& g, GpuWorkspace& ws,
    uint32_t total_bytes, uint32_t block_start, uint32_t block_count,
    bool* out_ok
) {
    if (total_bytes == 0 || block_count == 0) { if (out_ok) *out_ok = true; return 0.0; }
    uint32_t byte_off   = block_start * RANS32_BLOCK_SIZE;
    uint32_t sub_bytes  = std::min(block_count * RANS32_BLOCK_SIZE, total_bytes - byte_off);
    uint32_t tile_start = block_start * RANS32_TILES_PER_BLOCK;

    CFTimeInterval t0 = CACurrentMediaTime();
    id<MTLCommandBuffer> cb = [g.cq commandBuffer];
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g.pso_decode];
        [enc setBuffer:ws.buf_words   offset:(size_t)tile_start * RANS32_TILE_MAX_WORDS * sizeof(uint16_t) atIndex:0];
        [enc setBuffer:ws.buf_fstates offset:(size_t)tile_start * RANS32_LANES * sizeof(uint32_t)          atIndex:1];
        [enc setBuffer:ws.buf_dec_lut offset:(size_t)block_start * RANS32_PROB_SCALE * sizeof(uint32_t)    atIndex:2];
        [enc setBuffer:ws.buf_tmeta   offset:(size_t)tile_start * sizeof(Rans32TileMeta)                   atIndex:3];
        [enc setBuffer:ws.buf_in      offset:(size_t)byte_off                                              atIndex:4];
        [enc setBuffer:ws.buf_out     offset:(size_t)byte_off                                              atIndex:5];
        [enc setBuffer:ws.buf_err     offset:(size_t)tile_start * sizeof(uint32_t)                         atIndex:6];
        [enc setBytes:&sub_bytes      length:4 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(block_count, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(RANS32_TG_SIZE, 1, 1)];
        [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    double dt = CACurrentMediaTime() - t0;

    if (cb.error) {
        fprintf(stderr, "[ERR] GPU decode error: %s\n", cb.error.localizedDescription.UTF8String);
        exit(1);
    }

    const uint32_t* errs = (const uint32_t*)ws.buf_err.contents + tile_start;
    bool ok = true;
    for (uint32_t i = 0; i < block_count * RANS32_TILES_PER_BLOCK; ++i) {
        if (errs[i] != 0u) { ok = false; break; }
    }
    if (out_ok) *out_ok = ok;
    return dt;
}

// ─── R32P バイナリシリアライズ / デシリアライズ (圧縮サイズ計測・コンテナ検証) ────
static std::vector<uint8_t> serialize_r32p(const GpuWorkspace& ws, uint32_t total_bytes) {
    std::vector<uint8_t> out;
    uint32_t num_blocks = (total_bytes + RANS32_BLOCK_SIZE - 1u) / RANS32_BLOCK_SIZE;

    Rans32StreamHeader hdr;
    memcpy(hdr.magic, "R32P", 4);
    hdr.version       = 1u;
    hdr.original_size = total_bytes;
    hdr.block_size    = RANS32_BLOCK_SIZE;
    hdr.num_blocks    = num_blocks;

    out.resize(sizeof(hdr) + (size_t)num_blocks * sizeof(uint64_t));
    memcpy(out.data(), &hdr, sizeof(hdr));
    std::vector<uint64_t> offsets(num_blocks, 0);

    const uint8_t*          in_ptr   = (const uint8_t*)ws.buf_in.contents;
    const Rans32EncSym*     sym_ptr  = (const Rans32EncSym*)ws.buf_enc_sym.contents;
    const Rans32BlockMeta*  bm_ptr   = (const Rans32BlockMeta*)ws.buf_bmeta.contents;
    const Rans32TileMeta*   tm_ptr   = (const Rans32TileMeta*)ws.buf_tmeta.contents;
    const uint16_t*         w_ptr    = (const uint16_t*)ws.buf_words.contents;
    const uint32_t*         fs_ptr   = (const uint32_t*)ws.buf_fstates.contents;

    auto append_raw = [&](const void* src, size_t n) {
        size_t pos = out.size();
        out.resize(pos + n);
        memcpy(out.data() + pos, src, n);
    };

    for (uint32_t b = 0; b < num_blocks; ++b) {
        offsets[b] = (uint64_t)out.size();
        const Rans32BlockMeta& bm = bm_ptr[b];
        uint32_t blk_base  = b * RANS32_BLOCK_SIZE;
        uint32_t num_tiles = (bm.raw_size + RANS32_TILE_SIZE - 1u) / RANS32_TILE_SIZE;

        // 全タイル RAW フォールバックした場合はブロック自体を RAW に昇格してテーブルを省く
        bool any_rans = false;
        if (bm.mode == RANS32_MODE_RANS) {
            for (uint32_t t = 0; t < num_tiles; ++t) {
                if (tm_ptr[b * RANS32_TILES_PER_BLOCK + t].mode == RANS32_MODE_RANS) {
                    any_rans = true;
                    break;
                }
            }
        }
        uint8_t disk_mode = (bm.mode == RANS32_MODE_RANS && !any_rans) ? RANS32_MODE_RAW : (uint8_t)bm.mode;

        Rans32BlockDiskHeader bh;
        bh.block_mode = disk_mode;
        bh.num_tiles  = (uint8_t)num_tiles;
        bh.reserved   = 0;
        bh.raw_size   = bm.raw_size;
        append_raw(&bh, sizeof(bh));

        if (disk_mode == RANS32_MODE_RAW) {
            append_raw(in_ptr + blk_base, bm.raw_size);
        } else if (disk_mode == RANS32_MODE_RLE) {
            uint8_t rb = (uint8_t)bm.rle_byte;
            append_raw(&rb, 1);
        } else {
            rans32_append_freq_table(out, sym_ptr + (size_t)b * RANS32_N_SYMBOLS);
            for (uint32_t t = 0; t < num_tiles; ++t) {
                const Rans32TileMeta& tm = tm_ptr[b * RANS32_TILES_PER_BLOCK + t];
                Rans32TileDiskDesc td;
                td.tile_mode     = (uint8_t)tm.mode;
                td.rle_byte      = (uint8_t)tm.rle_byte;
                td.reserved      = 0;
                td.payload_words = (tm.mode == RANS32_MODE_RANS) ? tm.word_count :
                                   (tm.mode == RANS32_MODE_RAW)  ? tm.raw_size   : 0u;
                append_raw(&td, sizeof(td));
            }
            for (uint32_t t = 0; t < num_tiles; ++t) {
                uint32_t tgid = b * RANS32_TILES_PER_BLOCK + t;
                const Rans32TileMeta& tm = tm_ptr[tgid];
                uint32_t tbase = blk_base + t * RANS32_TILE_SIZE;
                if (tm.mode == RANS32_MODE_RAW) {
                    append_raw(in_ptr + tbase, tm.raw_size);
                } else if (tm.mode == RANS32_MODE_RANS) {
                    append_raw(fs_ptr + (size_t)tgid * RANS32_LANES, RANS32_LANES * sizeof(uint32_t));
                    append_raw(w_ptr + (size_t)tgid * RANS32_TILE_MAX_WORDS, (size_t)tm.word_count * sizeof(uint16_t));
                }
            }
        }
    }
    if (num_blocks > 0) {
        memcpy(out.data() + sizeof(hdr), offsets.data(), (size_t)num_blocks * sizeof(uint64_t));
    }
    return out;
}

static bool deserialize_and_decode_cpu_neon(const std::vector<uint8_t>& blob, std::vector<uint8_t>& restored) {
    const uint8_t* base = blob.data();
    const uint8_t* end  = base + blob.size();
    if (blob.size() < sizeof(Rans32StreamHeader)) return false;

    Rans32StreamHeader hdr;
    memcpy(&hdr, base, sizeof(hdr));
    if (memcmp(hdr.magic, "R32P", 4) != 0 || hdr.version != 1u || hdr.block_size != RANS32_BLOCK_SIZE)
        return false;

    restored.resize((size_t)hdr.original_size);
    if (hdr.num_blocks == 0) return hdr.original_size == 0;

    const size_t off_tbl_bytes = (size_t)hdr.num_blocks * sizeof(uint64_t);
    if (blob.size() < sizeof(hdr) + off_tbl_bytes) return false;
    std::vector<uint64_t> offsets(hdr.num_blocks);
    memcpy(offsets.data(), base + sizeof(hdr), off_tbl_bytes);

    std::vector<uint32_t> lut(RANS32_PROB_SCALE);
    for (uint32_t b = 0; b < hdr.num_blocks; ++b) {
        if (offsets[b] >= blob.size()) return false;
        const uint8_t* p = base + offsets[b];
        if ((size_t)(end - p) < sizeof(Rans32BlockDiskHeader)) return false;
        Rans32BlockDiskHeader bh;
        memcpy(&bh, p, sizeof(bh));
        p += sizeof(bh);

        uint32_t blk_base = b * RANS32_BLOCK_SIZE;
        uint32_t exp_raw  = (uint32_t)std::min((uint64_t)RANS32_BLOCK_SIZE, hdr.original_size - blk_base);
        if (bh.raw_size != exp_raw) return false;

        if (bh.block_mode == RANS32_MODE_RAW) {
            if ((size_t)(end - p) < bh.raw_size) return false;
            memcpy(restored.data() + blk_base, p, bh.raw_size);
        } else if (bh.block_mode == RANS32_MODE_RLE) {
            if ((size_t)(end - p) < 1u) return false;
            memset(restored.data() + blk_base, p[0], bh.raw_size);
        } else if (bh.block_mode == RANS32_MODE_RANS) {
            if (!rans32_read_freq_table_to_lut(p, end, lut.data())) return false;
            if ((size_t)(end - p) < (size_t)bh.num_tiles * sizeof(Rans32TileDiskDesc)) return false;
            std::vector<Rans32TileDiskDesc> tds(bh.num_tiles);
            memcpy(tds.data(), p, (size_t)bh.num_tiles * sizeof(Rans32TileDiskDesc));
            p += (size_t)bh.num_tiles * sizeof(Rans32TileDiskDesc);

            for (uint32_t t = 0; t < bh.num_tiles; ++t) {
                uint32_t tbase = blk_base + t * RANS32_TILE_SIZE;
                uint32_t tlen  = std::min(RANS32_TILE_SIZE, bh.raw_size - t * RANS32_TILE_SIZE);
                if (tds[t].tile_mode == RANS32_MODE_RAW) {
                    if (tds[t].payload_words != tlen || (size_t)(end - p) < tlen) return false;
                    memcpy(restored.data() + tbase, p, tlen);
                    p += tlen;
                } else if (tds[t].tile_mode == RANS32_MODE_RLE) {
                    memset(restored.data() + tbase, tds[t].rle_byte, tlen);
                } else if (tds[t].tile_mode == RANS32_MODE_RANS) {
                    size_t need = RANS32_LANES * sizeof(uint32_t) + (size_t)tds[t].payload_words * sizeof(uint16_t);
                    if ((size_t)(end - p) < need) return false;
                    uint32_t fstates[RANS32_LANES];
                    memcpy(fstates, p, sizeof(fstates));
                    p += sizeof(fstates);
                    std::vector<uint16_t> words(tds[t].payload_words);
                    memcpy(words.data(), p, (size_t)tds[t].payload_words * sizeof(uint16_t));
                    p += (size_t)tds[t].payload_words * sizeof(uint16_t);
                    if (!rans32_decode_tile_neon(words.data(), tds[t].payload_words, fstates,
                                                 lut.data(), restored.data() + tbase, tlen))
                        return false;
                } else {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
    return true;
}

// ─── テストデータ生成 (決定的 PRNG) ────────────────────────────────────────────
static uint64_t rng_state = 0x123456789ABCDEF0ULL;
static inline uint64_t xorshift64() {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return x;
}

static void fill_dataset(uint8_t* dst, size_t n, int kind) {
    rng_state = 0x9E3779B97F4A7C15ULL + (uint64_t)kind * 1337ULL;
    if (kind == 0) {
        // Text (English-like distribution: lowercase + space + punctuation, ~4.2 bpb)
        static const char vocab[] =
            "eeeeeeeeeeeeetttttttttaaaaaaaaaooooooooiiiiiiinnnnnnnsssssshhhhhrrrrrr"
            "ddddllllcccuuummwwffggyyppbbvvkkjxqz      \n\n.,;:-()0123456789ABCDEFGH";
        const size_t vlen = sizeof(vocab) - 1;
        for (size_t i = 0; i < n; ++i) dst[i] = (uint8_t)vocab[xorshift64() % vlen];
    } else if (kind == 1) {
        // Skewed (LZ token class stream simulation: ~1.8 bpb)
        for (size_t i = 0; i < n; ++i) {
            uint64_t r = xorshift64() & 1023u;
            if      (r < 750) dst[i] = (uint8_t)(r & 3u);
            else if (r < 950) dst[i] = (uint8_t)(4u + (r & 15u));
            else              dst[i] = (uint8_t)(20u + (r & 63u));
        }
    } else if (kind == 2) {
        // FP16 Exponent-byte-like (AI weights byte-shuffled high byte: concentrated around 0x38..0x3F & 0xB8..0xBF)
        for (size_t i = 0; i < n; ++i) {
            uint64_t r = xorshift64();
            uint8_t sign = (uint8_t)((r & 1u) << 7u);
            uint32_t u = (uint32_t)((r >> 1u) & 255u);
            uint8_t exp_val;
            if      (u < 110) exp_val = 0x3Cu;
            else if (u < 180) exp_val = 0x3Bu;
            else if (u < 220) exp_val = 0x3Du;
            else if (u < 245) exp_val = 0x3Au;
            else              exp_val = (uint8_t)(0x30u + (u & 15u));
            dst[i] = sign | exp_val;
        }
    } else if (kind == 3) {
        // Dense binary (all 256 symbols active, moderately biased: tests Dense12 freq table!)
        for (size_t i = 0; i < n; ++i) {
            uint64_t r = xorshift64();
            if ((r & 3u) != 0u) dst[i] = (uint8_t)((r >> 8u) & 63u);
            else                dst[i] = (uint8_t)((r >> 8u) & 255u);
        }
    } else if (kind == 4) {
        // Zeros (RLE fast path)
        memset(dst, 0x00, n);
    } else {
        // Random (incompressible -> RAW fallback)
        uint64_t* p64 = (uint64_t*)dst;
        size_t n64 = n / 8u;
        for (size_t i = 0; i < n64; ++i) p64[i] = xorshift64();
        for (size_t i = n64 * 8u; i < n; ++i) dst[i] = (uint8_t)xorshift64();
    }
}

static double shannon_entropy_bpb(const uint8_t* data, size_t n) {
    if (n == 0) return 0.0;
    uint64_t cnt[256] = {};
    for (size_t i = 0; i < n; ++i) cnt[data[i]]++;
    double h = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (cnt[i] > 0) {
            double p = (double)cnt[i] / (double)n;
            h -= p * log2(p);
        }
    }
    return h;
}

// ─── 正確性 & クロス検証スイート ──────────────────────────────────────────────
static void run_correctness_suite(GpuRans32& g) {
    printf("=== [1] Correctness, Bit-Exact CPU/GPU Parity & Corruption Suite ===\n");
    const uint32_t sizes[] = {
        0, 1, 15, 31, 32, 33, 255, 256, 257, 1023,
        65535, 65536, 65537, 262143, 262144, 262145, 524300
    };

    GpuWorkspace ws;
    ws.allocate(g.dev, 524300u);

    for (uint32_t sz : sizes) {
        for (int kind = 0; kind < 6; ++kind) {
            uint8_t* in_ptr = (uint8_t*)ws.buf_in.contents;
            fill_dataset(in_ptr, sz, kind);

            // 1) GPU エンコード
            gpu_run_encode(g, ws, sz);

            // 2) CPU Scalar エンコードと 1 ビット残らず完全一致検証
            uint32_t num_blocks = (sz + RANS32_BLOCK_SIZE - 1u) / RANS32_BLOCK_SIZE;
            const Rans32BlockMeta* gpu_bm = (const Rans32BlockMeta*)ws.buf_bmeta.contents;
            const Rans32EncSym*    gpu_es = (const Rans32EncSym*)ws.buf_enc_sym.contents;
            const uint32_t*        gpu_lt = (const uint32_t*)ws.buf_dec_lut.contents;
            const Rans32TileMeta*  gpu_tm = (const Rans32TileMeta*)ws.buf_tmeta.contents;
            const uint16_t*        gpu_wd = (const uint16_t*)ws.buf_words.contents;
            const uint32_t*        gpu_fs = (const uint32_t*)ws.buf_fstates.contents;

            std::vector<uint16_t> cpu_words(RANS32_TILE_MAX_WORDS);
            std::vector<uint32_t> cpu_lut(RANS32_PROB_SCALE);
            std::vector<uint8_t>  cpu_out(RANS32_TILE_SIZE);

            for (uint32_t b = 0; b < num_blocks; ++b) {
                uint32_t bbase = b * RANS32_BLOCK_SIZE;
                uint32_t blen  = std::min(RANS32_BLOCK_SIZE, sz - bbase);
                Rans32EncSym cpu_sym[RANS32_N_SYMBOLS];
                Rans32BlockMeta cpu_bm;
                rans32_cpu_build_tables(in_ptr + bbase, blen, cpu_sym, cpu_lut.data(), &cpu_bm);

                if (memcmp(&cpu_bm, &gpu_bm[b], sizeof(cpu_bm)) != 0) {
                    fprintf(stderr, "[FAIL] BlockMeta mismatch sz=%u kind=%d b=%u\n", sz, kind, b);
                    exit(1);
                }
                if (cpu_bm.mode == RANS32_MODE_RANS) {
                    if (memcmp(cpu_sym, gpu_es + (size_t)b * RANS32_N_SYMBOLS, sizeof(cpu_sym)) != 0) {
                        fprintf(stderr, "[FAIL] EncSym mismatch sz=%u kind=%d b=%u\n", sz, kind, b);
                        exit(1);
                    }
                    if (memcmp(cpu_lut.data(), gpu_lt + (size_t)b * RANS32_PROB_SCALE, RANS32_PROB_SCALE * 4u) != 0) {
                        fprintf(stderr, "[FAIL] DecLUT mismatch sz=%u kind=%d b=%u\n", sz, kind, b);
                        exit(1);
                    }
                }

                uint32_t n_tiles = (blen + RANS32_TILE_SIZE - 1u) / RANS32_TILE_SIZE;
                for (uint32_t t = 0; t < n_tiles; ++t) {
                    uint32_t tgid  = b * RANS32_TILES_PER_BLOCK + t;
                    uint32_t tbase = bbase + t * RANS32_TILE_SIZE;
                    uint32_t tlen  = std::min(RANS32_TILE_SIZE, sz - tbase);

                    if (cpu_bm.mode == RANS32_MODE_RANS) {
                        uint32_t cpu_fs[RANS32_LANES];
                        Rans32TileMeta cpu_tm;
                        rans32_encode_tile_scalar(in_ptr + tbase, tlen, cpu_sym,
                                                  cpu_words.data(), cpu_fs, &cpu_tm);
                        if (memcmp(&cpu_tm, &gpu_tm[tgid], sizeof(cpu_tm)) != 0) {
                            fprintf(stderr, "[FAIL] TileMeta mismatch sz=%u kind=%d tgid=%u (cpu mode=%u wc=%u vs gpu mode=%u wc=%u)\n",
                                    sz, kind, tgid, cpu_tm.mode, cpu_tm.word_count, gpu_tm[tgid].mode, gpu_tm[tgid].word_count);
                            exit(1);
                        }
                        if (cpu_tm.mode == RANS32_MODE_RANS) {
                            if (memcmp(cpu_fs, gpu_fs + (size_t)tgid * RANS32_LANES, sizeof(cpu_fs)) != 0) {
                                fprintf(stderr, "[FAIL] FinalStates bit-mismatch sz=%u kind=%d tgid=%u\n", sz, kind, tgid);
                                exit(1);
                            }
                            if (memcmp(cpu_words.data(), gpu_wd + (size_t)tgid * RANS32_TILE_MAX_WORDS,
                                       (size_t)cpu_tm.word_count * sizeof(uint16_t)) != 0) {
                                fprintf(stderr, "[FAIL] Words bit-mismatch sz=%u kind=%d tgid=%u\n", sz, kind, tgid);
                                exit(1);
                            }

                            // 3) GPU 出力を CPU Scalar & CPU NEON 両方でデコード検証
                            memset(cpu_out.data(), 0xA5, tlen);
                            if (!rans32_decode_tile_scalar(gpu_wd + (size_t)tgid * RANS32_TILE_MAX_WORDS,
                                                           gpu_tm[tgid].word_count,
                                                           gpu_fs + (size_t)tgid * RANS32_LANES,
                                                           gpu_lt + (size_t)b * RANS32_PROB_SCALE,
                                                           cpu_out.data(), tlen) ||
                                memcmp(cpu_out.data(), in_ptr + tbase, tlen) != 0) {
                                fprintf(stderr, "[FAIL] CPU Scalar decode failed sz=%u kind=%d tgid=%u\n", sz, kind, tgid);
                                exit(1);
                            }

                            memset(cpu_out.data(), 0x5A, tlen);
                            if (!rans32_decode_tile_neon(gpu_wd + (size_t)tgid * RANS32_TILE_MAX_WORDS,
                                                         gpu_tm[tgid].word_count,
                                                         gpu_fs + (size_t)tgid * RANS32_LANES,
                                                         gpu_lt + (size_t)b * RANS32_PROB_SCALE,
                                                         cpu_out.data(), tlen) ||
                                memcmp(cpu_out.data(), in_ptr + tbase, tlen) != 0) {
                                fprintf(stderr, "[FAIL] CPU NEON decode failed sz=%u kind=%d tgid=%u\n", sz, kind, tgid);
                                exit(1);
                            }
                        }
                    }
                }
            }

            // 4) GPU デコード検証
            if (sz > 0) {
                memset(ws.buf_out.contents, 0xCC, sz);
                bool gpu_ok = false;
                gpu_run_decode_range(g, ws, sz, 0, num_blocks, &gpu_ok);
                if (!gpu_ok || memcmp(ws.buf_out.contents, in_ptr, sz) != 0) {
                    fprintf(stderr, "[FAIL] GPU decode failed sz=%u kind=%d ok=%d\n", sz, kind, (int)gpu_ok);
                    exit(1);
                }
            }

            // 5) R32P シリアライズ → デシリアライズ (Sparse / Dense12 両方通過)
            std::vector<uint8_t> blob = serialize_r32p(ws, sz);
            std::vector<uint8_t> restored;
            if (!deserialize_and_decode_cpu_neon(blob, restored) ||
                restored.size() != sz ||
                (sz > 0 && memcmp(restored.data(), in_ptr, sz) != 0)) {
                fprintf(stderr, "[FAIL] R32P serialize/deserialize failed sz=%u kind=%d\n", sz, kind);
                exit(1);
            }
        }
    }
    printf("  [PASS] All %zu boundary sizes x 6 datasets: GPU Enc == CPU Enc (bit-exact),\n"
           "         GPU Dec / CPU Scalar Dec / CPU NEON Dec / R32P container all 100%% exact!\n",
           sizeof(sizes) / sizeof(sizes[0]));

    // 6) 破損検知テスト (Negative tests)
    const uint32_t csz = RANS32_BLOCK_SIZE;
    fill_dataset((uint8_t*)ws.buf_in.contents, csz, 0); // Text
    gpu_run_encode(g, ws, csz);

    uint32_t*       fs = (uint32_t*)ws.buf_fstates.contents;
    uint16_t*       wd = (uint16_t*)ws.buf_words.contents;
    Rans32TileMeta* tm = (Rans32TileMeta*)ws.buf_tmeta.contents;
    const uint32_t* lt = (const uint32_t*)ws.buf_dec_lut.contents;
    std::vector<uint8_t> tmp(RANS32_TILE_SIZE);

    // Case A: final_state < RANS32_L
    {
        uint32_t saved = fs[5];
        fs[5] = RANS32_L - 1u;
        bool gpu_ok = true;
        gpu_run_decode_range(g, ws, csz, 0, 1, &gpu_ok);
        bool sc_ok = rans32_decode_tile_scalar(wd, tm[0].word_count, fs, lt, tmp.data(), RANS32_TILE_SIZE);
        bool ne_ok = rans32_decode_tile_neon  (wd, tm[0].word_count, fs, lt, tmp.data(), RANS32_TILE_SIZE);
        fs[5] = saved;
        if (gpu_ok || sc_ok || ne_ok) { fprintf(stderr, "[FAIL] Case A undetected!\n"); exit(1); }
    }
    // Case B: words[] 1-bit flip
    {
        uint16_t saved = wd[100];
        wd[100] ^= 0x0001u;
        bool gpu_ok = true;
        gpu_run_decode_range(g, ws, csz, 0, 1, &gpu_ok);
        bool sc_ok = rans32_decode_tile_scalar(wd, tm[0].word_count, fs, lt, tmp.data(), RANS32_TILE_SIZE);
        bool ne_ok = rans32_decode_tile_neon  (wd, tm[0].word_count, fs, lt, tmp.data(), RANS32_TILE_SIZE);
        wd[100] = saved;
        if (gpu_ok || sc_ok || ne_ok) { fprintf(stderr, "[FAIL] Case B undetected!\n"); exit(1); }
    }
    // Case C: word_count truncation
    {
        uint32_t saved = tm[0].word_count;
        tm[0].word_count = saved / 2u;
        bool gpu_ok = true;
        gpu_run_decode_range(g, ws, csz, 0, 1, &gpu_ok);
        bool sc_ok = rans32_decode_tile_scalar(wd, tm[0].word_count, fs, lt, tmp.data(), RANS32_TILE_SIZE);
        bool ne_ok = rans32_decode_tile_neon  (wd, tm[0].word_count, fs, lt, tmp.data(), RANS32_TILE_SIZE);
        tm[0].word_count = saved;
        if (gpu_ok || sc_ok || ne_ok) { fprintf(stderr, "[FAIL] Case C undetected!\n"); exit(1); }
    }
    printf("  [PASS] Corruption detection (bad state / 1-bit flip / truncation): GPU, Scalar, NEON all 100%% caught!\n\n");
}

// ─── 64 MiB スループットベンチマーク ───────────────────────────────────────────
static void run_benchmarks(GpuRans32& g) {
    const uint32_t SIZE_MB    = 64u;
    const uint32_t TOTAL_SIZE = SIZE_MB * 1024u * 1024u; // 64 MiB = 256 superblocks = 1024 tiles
    const uint32_t NUM_BLOCKS = TOTAL_SIZE / RANS32_BLOCK_SIZE;
    const int      RUNS       = 5;

    printf("=== [2] Throughput Benchmark (%u MiB = %u superblocks = %u tiles, best of %d runs) ===\n",
           SIZE_MB, NUM_BLOCKS, NUM_BLOCKS * RANS32_TILES_PER_BLOCK, RUNS);
    printf("Device: %s\n\n", g.dev.name.UTF8String);

    GpuWorkspace ws;
    ws.allocate(g.dev, TOTAL_SIZE);

    // Warmup
    fill_dataset((uint8_t*)ws.buf_in.contents, TOTAL_SIZE, 0);
    gpu_run_encode(g, ws, TOTAL_SIZE);
    bool dummy_ok = false;
    gpu_run_decode_range(g, ws, TOTAL_SIZE, 0, NUM_BLOCKS, &dummy_ok);

    const char* names[6] = {
        "Text (English)",
        "Skewed (LZ-tok)",
        "FP16 Exp-byte",
        "Dense Binary",
        "Zeros (RLE)",
        "Random (RAW)"
    };

    printf("%-16s | %5s | %6s | %6s | %9s | %9s | %9s | %9s | %9s | %9s\n",
           "Dataset", "H(bpb)", "Out%", "Overhead",
           "GPU Enc", "GPU Dec", "NEON 1c", "Sclr 1c", "NEON All", "Hybrid");
    printf("-----------------+-------+--------+----------+-----------+-----------+-----------+-----------+-----------+----------\n");

    const Rans32TileMeta*  tm_ptr  = (const Rans32TileMeta*)ws.buf_tmeta.contents;
    const uint16_t*        w_ptr   = (const uint16_t*)ws.buf_words.contents;
    const uint32_t*        fs_ptr  = (const uint32_t*)ws.buf_fstates.contents;
    const uint32_t*        lut_ptr = (const uint32_t*)ws.buf_dec_lut.contents;
    const uint8_t*         in_ptr  = (const uint8_t*)ws.buf_in.contents;
    uint8_t*               out_ptr = (uint8_t*)ws.buf_out.contents;

    auto cpu_decode_blocks = [&](uint32_t b_start, uint32_t b_count, bool use_neon, bool parallel) -> double {
        CFTimeInterval t0 = CACurrentMediaTime();
        auto do_block = ^(size_t ib) {
            uint32_t b = b_start + (uint32_t)ib;
            const uint32_t* lut = lut_ptr + (size_t)b * RANS32_PROB_SCALE;
            for (uint32_t t = 0; t < RANS32_TILES_PER_BLOCK; ++t) {
                uint32_t tgid  = b * RANS32_TILES_PER_BLOCK + t;
                uint32_t tbase = (b * RANS32_TILES_PER_BLOCK + t) * RANS32_TILE_SIZE;
                const Rans32TileMeta& tm = tm_ptr[tgid];
                if (tm.mode == RANS32_MODE_RLE) {
                    memset(out_ptr + tbase, (int)tm.rle_byte, tm.raw_size);
                } else if (tm.mode == RANS32_MODE_RAW) {
                    memcpy(out_ptr + tbase, in_ptr + tbase, tm.raw_size);
                } else if (use_neon) {
                    bool ok = rans32_decode_tile_neon(
                        w_ptr + (size_t)tgid * RANS32_TILE_MAX_WORDS, tm.word_count,
                        fs_ptr + (size_t)tgid * RANS32_LANES, lut, out_ptr + tbase, tm.raw_size);
                    if (!ok) { fprintf(stderr, "[ERR] NEON decode fail!\n"); exit(1); }
                } else {
                    bool ok = rans32_decode_tile_scalar(
                        w_ptr + (size_t)tgid * RANS32_TILE_MAX_WORDS, tm.word_count,
                        fs_ptr + (size_t)tgid * RANS32_LANES, lut, out_ptr + tbase, tm.raw_size);
                    if (!ok) { fprintf(stderr, "[ERR] Scalar decode fail!\n"); exit(1); }
                }
            }
        };
        if (parallel) {
            dispatch_apply(b_count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), do_block);
        } else {
            for (uint32_t ib = 0; ib < b_count; ++ib) do_block(ib);
        }
        return CACurrentMediaTime() - t0;
    };

    for (int kind = 0; kind < 6; ++kind) {
        fill_dataset((uint8_t*)ws.buf_in.contents, TOTAL_SIZE, kind);
        double h_bpb = shannon_entropy_bpb(in_ptr, TOTAL_SIZE);

        double best_gpu_enc = 1e9, best_gpu_dec = 1e9;
        double best_neon_1c = 1e9, best_sclr_1c = 1e9, best_neon_all = 1e9, best_hybrid = 1e9;

        for (int r = 0; r < RUNS; ++r) {
            best_gpu_enc = std::min(best_gpu_enc, gpu_run_encode(g, ws, TOTAL_SIZE));
        }

        std::vector<uint8_t> blob = serialize_r32p(ws, TOTAL_SIZE);
        double ratio_pct = 100.0 * (double)blob.size() / (double)TOTAL_SIZE;
        double actual_bpb = 8.0 * (double)blob.size() / (double)TOTAL_SIZE;
        double overhead_bpb = (kind >= 4) ? 0.0 : (actual_bpb - h_bpb);

        for (int r = 0; r < RUNS; ++r) {
            bool ok = false;
            best_gpu_dec = std::min(best_gpu_dec, gpu_run_decode_range(g, ws, TOTAL_SIZE, 0, NUM_BLOCKS, &ok));
            if (!ok) { fprintf(stderr, "[ERR] GPU dec bench fail\n"); exit(1); }
        }
        if (memcmp(out_ptr, in_ptr, TOTAL_SIZE) != 0) { fprintf(stderr, "[ERR] GPU dec mismatch\n"); exit(1); }

        for (int r = 0; r < 2; ++r) {
            best_neon_1c = std::min(best_neon_1c, cpu_decode_blocks(0, NUM_BLOCKS, true,  false));
            best_sclr_1c = std::min(best_sclr_1c, cpu_decode_blocks(0, NUM_BLOCKS, false, false));
        }
        for (int r = 0; r < RUNS; ++r) {
            best_neon_all = std::min(best_neon_all, cpu_decode_blocks(0, NUM_BLOCKS, true, true));
        }
        if (memcmp(out_ptr, in_ptr, TOTAL_SIZE) != 0) { fprintf(stderr, "[ERR] NEON dec mismatch\n"); exit(1); }

        // Hybrid Decode: GPU (80% blocks) + CPU NEON All-cores (20% blocks) 同時実行!
        uint32_t gpu_blocks = (NUM_BLOCKS * 80u) / 100u;
        uint32_t cpu_blocks = NUM_BLOCKS - gpu_blocks;
        for (int r = 0; r < RUNS; ++r) {
            CFTimeInterval t0 = CACurrentMediaTime();
            // GPU 先行投入 (非同期)
            uint32_t sub_bytes = gpu_blocks * RANS32_BLOCK_SIZE;
            id<MTLCommandBuffer> cb = [g.cq commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:g.pso_decode];
                [enc setBuffer:ws.buf_words   offset:0 atIndex:0];
                [enc setBuffer:ws.buf_fstates offset:0 atIndex:1];
                [enc setBuffer:ws.buf_dec_lut offset:0 atIndex:2];
                [enc setBuffer:ws.buf_tmeta   offset:0 atIndex:3];
                [enc setBuffer:ws.buf_in      offset:0 atIndex:4];
                [enc setBuffer:ws.buf_out     offset:0 atIndex:5];
                [enc setBuffer:ws.buf_err     offset:0 atIndex:6];
                [enc setBytes:&sub_bytes      length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(gpu_blocks, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(RANS32_TG_SIZE, 1, 1)];
                [enc endEncoding];
            }
            [cb commit];
            // GPU 実行中に残りブロックを CPU NEON 全コアで同時デコード
            cpu_decode_blocks(gpu_blocks, cpu_blocks, true, true);
            [cb waitUntilCompleted];
            best_hybrid = std::min(best_hybrid, CACurrentMediaTime() - t0);
        }
        if (memcmp(out_ptr, in_ptr, TOTAL_SIZE) != 0) { fprintf(stderr, "[ERR] Hybrid dec mismatch\n"); exit(1); }

        const double gib = (double)TOTAL_SIZE / (1024.0 * 1024.0 * 1024.0);
        printf("%-16s | %5.2f | %5.1f%% | %+6.3fb | %6.2f GB/s | %6.2f GB/s | %6.2f GB/s | %6.2f GB/s | %6.2f GB/s | %6.2f GB/s\n",
               names[kind], h_bpb, ratio_pct, overhead_bpb,
               gib / best_gpu_enc, gib / best_gpu_dec,
               gib / best_neon_1c, gib / best_sclr_1c,
               gib / best_neon_all, gib / best_hybrid);
    }
    printf("\n(Overhead = actual_bpb - Shannon_H; includes R32P headers, freq tables & 128B/tile states)\n");
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        const char* dir = (argc >= 2) ? argv[1] : ".";
        GpuRans32 g;
        if (!g.init(dir)) {
            fprintf(stderr, "[ERR] Metal unavailable.\n");
            return EXIT_FAILURE;
        }
        run_correctness_suite(g);
        run_benchmarks(g);
        return EXIT_SUCCESS;
    }
}
