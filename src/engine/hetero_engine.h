// hetero_engine.h — APLZ-X Phase 2: 3 スロット完全非同期 UMA 異種協調エンジン (Zero-Cold-Start Hybrid)
//
// 圧縮 3 段オーバーラップ・パイプライン (8 MiB × 3 スロット):
//   CPU 全コア (Slot N+2): [1パス L2 融合: HW CRC32C + LUT Probe + NEON 前処理 + エポック LZ]
//                          → [並列ストリーム集約 (RANS_ONLY はプレーン単位 NEON 直書き & 高エントロピー即バイパス)]
//   rANS 層   (Slot N+1):
//     - 常駐 / 大ファイル (>= 96 MiB): Apple GPU バッチ 32-way rANS (非同期投入で CPU 次スロットと完全並列)
//       ※初回大ファイルは非同期バックグラウンド Metal 初期化中も CPU ブランチレス 32-way rANS が即稼働 (起動遅延 0ms!)
//     - 単発小〜中ファイル (< 96 MiB): Metal 未起動ゼロ遅延 CPU 全コア ブランチレス 32-way rANS (GPU とビット完全一致!)
//   Write Q   (Slot N)  : [厳密順序保証・10 コア並列ゼロ中間コピーパッキング → 非同期ストリーミング fwrite]
//
// 解凍:
//   - 単発小〜中ファイル: Metal 起動ゼロ + HX_SUB_RAW ゼロコピー + CPU NEON 32-way + NEON 逆変換
//   - 常駐 (gpu_dec_ready): Apple GPU バッチ rANS 復号 + CPU 全コア並列 LZ + NEON 逆変換
//   - ファイル出力は 2 スロット非同期ダブルバッファ fwrite で I/O 隠蔽
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CABase.h>

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <dispatch/dispatch.h>

#include "hetero_format.h"
#include "hetero_lz.h"
#include "../rans/rans32_cpu.h"

// APFS 連続エクステント事前確保 (大量メタデータ更新・断片化を回避)
static inline void hx_preallocate_fd(int fd, uint64_t expected_size) {
#if defined(__APPLE__) && defined(F_PREALLOCATE)
    if (fd < 0 || expected_size < 256u * 1024u) return;
    fstore_t fst = {};
    fst.fst_flags   = F_ALLOCATECONTIG | F_ALLOCATEALL;
    fst.fst_posmode = F_PEOFPOSMODE;
    fst.fst_offset  = 0;
    fst.fst_length  = (off_t)expected_size;
    if (fcntl(fd, F_PREALLOCATE, &fst) == -1) {
        fst.fst_flags = F_ALLOCATEALL;
        (void)fcntl(fd, F_PREALLOCATE, &fst);
    }
#else
    (void)fd; (void)expected_size;
#endif
}

static inline bool hx_write_all_fd(int fd, const uint8_t* data, size_t len) {
    size_t off = 0u;
    while (off < len) {
        ssize_t w = write(fd, data + off, len - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        off += (size_t)w;
    }
    return true;
}

static_assert(sizeof(HxEncSym)   == sizeof(Rans32EncSym),   "EncSym ABI mismatch");
static_assert(sizeof(HxTileMeta) == sizeof(Rans32TileMeta), "TileMeta ABI mismatch");

// ─── 頻度表シリアライズ (生ポインタ直書き版 — 並列パッキング対応) ──────────────
static inline uint32_t hx_freq_table_bytes(uint32_t n_nz) {
    return 2u + ((n_nz < HX_SPARSE_THRESH) ? (n_nz * 3u) : 384u);
}

static inline uint8_t* hx_write_freq_table(uint8_t* p, const HxEncSym* enc_sym, uint32_t n_nz) {
    uint8_t kind = (n_nz < HX_SPARSE_THRESH) ? HX_FREQ_SPARSE : HX_FREQ_DENSE12;
    p[0] = kind;
    p[1] = (uint8_t)(n_nz - 1u);
    p += 2u;
    if (kind == HX_FREQ_SPARSE) {
        for (uint32_t s = 0u; s < HX_N_SYMBOLS; ++s) {
            uint16_t f = enc_sym[s].freq;
            if (f > 0u) {
                p[0] = (uint8_t)s;
                p[1] = (uint8_t)(f & 0xFFu);
                p[2] = (uint8_t)((f >> 8u) & 0xFFu);
                p += 3u;
            }
        }
    } else {
        for (uint32_t s = 0u; s < HX_N_SYMBOLS; s += 2u) {
            uint32_t f0 = enc_sym[s].freq;
            uint32_t f1 = enc_sym[s + 1u].freq;
            p[0] = (uint8_t)(f0 & 0xFFu);
            p[1] = (uint8_t)(((f0 >> 8u) & 0x0Fu) | ((f1 & 0x0Fu) << 4u));
            p[2] = (uint8_t)((f1 >> 4u) & 0xFFu);
            p += 3u;
        }
    }
    return p;
}

struct HxTimingStats {
    double probe_s    = 0.0;
    double xform_s    = 0.0;
    double lz_s       = 0.0; // 圧縮: CPU 融合(CRC+Probe+Xform+LZ), 解凍: LZ+NEON+CRC
    double rans_s     = 0.0; // 圧縮: rANS 投入/CPU rANS, 解凍: rANS 復号
    double pack_s     = 0.0; // 圧縮: スロット待機/完了待機, 解凍: ヘッダ解析+LUT構築
    double total_s    = 0.0;
    uint32_t cnt_raw = 0, cnt_rle = 0, cnt_rans_only = 0, cnt_lz = 0;
    uint32_t cnt_shuf2 = 0, cnt_shuf4 = 0, cnt_bcj = 0;
};

// ─── 3 スロット完全非同期 UMA 異種協調エンジン ─────────────────────────────────
class HeteroEngine {
public:
    static const uint32_t N_SLOTS        = 3u;                          // CPU || GPU || Pack+fwrite 3 段重畳
    static const uint32_t MB_BLOCKS      = 32u;                         // 8 MiB / スロット
    static const uint32_t MB_BYTES       = MB_BLOCKS * HX_BLOCK_SIZE;   // 8 MiB
    static const uint32_t MAX_STREAMS_MB = MB_BLOCKS * 4u;              // 最大 128 ストリーム/スロット
    static const uint32_t MAX_TILES_MB   = MB_BLOCKS * 7u;              // 最大 224 タイル/スロット
    static const uint32_t STREAM_BUF_CAP = MB_BYTES + MB_BLOCKS * 65536u; // 10 MiB
    // 単発 CLI 起動時、Metal コールドスタート(~70ms)を十分回収できるサイズ以上でのみ非同期 Metal 初期化を起動
    static const size_t   HYBRID_GPU_MIN_BYTES = 384ull * 1024ull * 1024ull;

    struct BlockPlan {
        uint32_t raw_size;
        uint32_t crc32c;
        uint8_t  xform_id;
        uint8_t  codec_id;
        uint8_t  rle_byte;
        uint8_t  plane_raw_mask;
        uint32_t lz_seq_count;
        uint32_t lz_lit_size;
        uint32_t lz_tail_lit_size;
        uint32_t lz_extra_size;
        uint32_t lz_extra_off;
        // sidx[k]: >=0 = rANS stream idx, -1 = 空 (0B), -2 = RAW バイパスプレーン (raw_off/raw_len)
        int32_t  sidx[4];
        uint32_t raw_off[4];
        uint32_t raw_len[4];
    };

    // エンコード用独立スロット (3 スロットが CPU / GPU / Pack+fwrite で完全並列稼働)
    struct EncSlot {
        BlockPlan plans[MB_BLOCKS];
        HxRawBuf  extra_pool; // LZ extra_bits のみコンパクト保持
        HxRawBuf  pack_tmp;   // out_file 書出用 (未初期化 malloc)

        // GPU バックエンド (ensure_gpu_encode で遅延確保)
        id<MTLBuffer> buf_streams = nil;
        id<MTLBuffer> buf_sjobs   = nil;
        id<MTLBuffer> buf_enc_sym = nil;
        id<MTLBuffer> buf_words   = nil;
        id<MTLBuffer> buf_fstates = nil;
        id<MTLBuffer> buf_tmeta   = nil;
        dispatch_semaphore_t gpu_done_sem = nil;

        // CPU バックエンド (CPU 32-way rANS 使用時のみ遅延確保)
        HxRawBuf cpu_streams;
        HxRawBuf cpu_sjobs;
        HxRawBuf cpu_enc_sym;
        HxRawBuf cpu_words;
        HxRawBuf cpu_fstates;
        HxRawBuf cpu_tmeta;

        uint32_t b0        = 0u;
        uint32_t bc        = 0u;
        size_t   batch_off = 0u;
        uint32_t n_streams = 0u;
        bool     used_gpu  = false;
    };

    std::string                 shader_dir_str;
    std::mutex                  gpu_init_mtx;
    bool                        gpu_core_ready = false;
    std::atomic<bool>           gpu_enc_ready_atomic{false};
    std::atomic<bool>           async_init_started{false};
    dispatch_group_t            async_init_grp = nil;
    std::atomic<bool>           gpu_dec_ready_atomic{false};
    std::atomic<bool>           async_dec_init_started{false};
    dispatch_group_t            async_dec_init_grp = nil;
    bool                        force_cpu_rans = false; // 検証・ベンチ用の強制 CPU rANS フラグ

    ~HeteroEngine() {
        if (async_init_grp) {
            dispatch_group_wait(async_init_grp, DISPATCH_TIME_FOREVER);
        }
        if (async_dec_init_grp) {
            dispatch_group_wait(async_dec_init_grp, DISPATCH_TIME_FOREVER);
        }
    }

    id<MTLDevice>               dev            = nil;
    id<MTLCommandQueue>         cq             = nil;
    id<MTLLibrary>              lib            = nil;
    id<MTLComputePipelineState> pso_rans_build = nil;
    id<MTLComputePipelineState> pso_rans_enc   = nil;
    id<MTLComputePipelineState> pso_rans_dec   = nil;

    struct DecBlockPlan {
        HxBlockHeader  bh;
        HxLzBlockMeta  lm;
        const uint8_t* payload_ptr;
        const uint8_t* extra_ptr;
        uint32_t       extra_bytes;
        int32_t        sidx[4];
        const uint8_t* s_ptr[4];
        uint32_t       s_len[4];
    };

    // GPU デコード用 2 スロット (Slot N+1 の GPU rANS 復号と Slot N の CPU LZ+NEON+CRC を完全並列!)
    static const uint32_t DEC_SLOTS = 2u;
    struct DecSlot {
        DecBlockPlan  dplans[MB_BLOCKS];
        id<MTLBuffer> buf_streams = nil;
        id<MTLBuffer> buf_sjobs   = nil;
        id<MTLBuffer> buf_lut     = nil;
        id<MTLBuffer> buf_words   = nil;
        id<MTLBuffer> buf_fstates = nil;
        id<MTLBuffer> buf_tmeta   = nil;
        id<MTLBuffer> buf_err     = nil;
        dispatch_semaphore_t gpu_done_sem = nil;

        uint32_t bc               = 0u;
        uint32_t n_streams        = 0u;
        uint32_t n_tiles          = 0u;
        uint32_t total_rans_bytes = 0u;
        size_t   batch_off        = 0u;
        size_t   batch_bytes      = 0u;
        bool     used_gpu         = false;
    };

    EncSlot         enc_slots[N_SLOTS];
    DecSlot         dec_slots[DEC_SLOTS];
    HxLzBlockOutput cpu_lz_outs[MB_BLOCKS]; // CPU ステージ専用 1 セット (遅延 malloc, ゼロ初期化なし!)

    // デコード用 CPU ワークスペース (未初期化 malloc、必要時のみ遅延確保)
    HxRawBuf cpu_dec_stream_buf;
    HxRawBuf cpu_dec_lut_buf;

    bool init(const char* shader_dir, bool eager_gpu = false) {
        shader_dir_str = shader_dir ? shader_dir : ".";
        async_init_grp = dispatch_group_create();
        async_dec_init_grp = dispatch_group_create();
        for (uint32_t s = 0u; s < N_SLOTS; ++s) {
            enc_slots[s].gpu_done_sem = dispatch_semaphore_create(0);
        }
        for (uint32_t s = 0u; s < DEC_SLOTS; ++s) {
            dec_slots[s].gpu_done_sem = dispatch_semaphore_create(0);
        }
        if (eager_gpu) {
            return ensure_gpu_encode() && ensure_gpu_decode();
        }
        return true;
    }

    // 大ファイル圧縮向けに非同期 Metal GPU エンコード初期化を先行起動
    // (QOS_CLASS_USER_INITIATED で CPU 飽和中も餓死させず確実に初期化完了 → GPU 3 スロットへ自動切替!)
    void warmup_gpu_encode_async() {
        if (force_cpu_rans || gpu_enc_ready_atomic.load(std::memory_order_acquire)) return;
        bool expected = false;
        if (async_init_started.compare_exchange_strong(expected, true)) {
            HeteroEngine* self_ptr = this;
            dispatch_group_async(async_init_grp, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                self_ptr->ensure_gpu_encode();
            });
        }
    }

    // 大ファイル解凍向けに非同期 Metal GPU デコード初期化を先行起動
    void warmup_gpu_decode_async() {
        if (force_cpu_rans || gpu_dec_ready_atomic.load(std::memory_order_acquire)) return;
        bool expected = false;
        if (async_dec_init_started.compare_exchange_strong(expected, true)) {
            HeteroEngine* self_ptr = this;
            dispatch_group_async(async_dec_init_grp, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                self_ptr->ensure_gpu_decode();
            });
        }
    }

    bool ensure_gpu_core() {
        if (gpu_core_ready) return true;
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) return false;
        cq = [dev newCommandQueue];

        NSError* err = nil;
        NSString* mlib = [NSString stringWithFormat:@"%s/hetero_kernels.metallib", shader_dir_str.c_str()];
        if ([[NSFileManager defaultManager] fileExistsAtPath:mlib]) {
            lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:mlib] error:&err];
        }
        if (!lib) {
            NSString* msrc = [NSString stringWithFormat:@"%s/hetero_kernels.metal", shader_dir_str.c_str()];
            NSString* src  = [NSString stringWithContentsOfFile:msrc encoding:NSUTF8StringEncoding error:&err];
            if (!src) { fprintf(stderr, "[ERR] Cannot read %s\n", msrc.UTF8String); return false; }
            MTLCompileOptions* opt = [MTLCompileOptions new];
            opt.languageVersion = MTLLanguageVersion3_0;
            opt.mathMode        = MTLMathModeFast;
            lib = [dev newLibraryWithSource:src options:opt error:&err];
            if (!lib) { fprintf(stderr, "[ERR] Metal error:\n%s\n", err.localizedDescription.UTF8String); return false; }
        }
        gpu_core_ready = true;
        return true;
    }

    id<MTLComputePipelineState> make_pso(NSString* name) {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "[ERR] kernel '%s' missing\n", name.UTF8String); exit(1); }
        NSError* pe = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!pso) { fprintf(stderr, "[ERR] PSO '%s': %s\n", name.UTF8String, pe.localizedDescription.UTF8String); exit(1); }
        return pso;
    }

    id<MTLBuffer> alloc_shared(size_t bytes) {
        id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) { fprintf(stderr, "[ERR] MTLBuffer alloc failed (%zu B)\n", bytes); exit(1); }
        return b;
    }

    bool ensure_gpu_encode() {
        if (gpu_enc_ready_atomic.load(std::memory_order_acquire)) return true;
        std::lock_guard<std::mutex> lk(gpu_init_mtx);
        if (gpu_enc_ready_atomic.load(std::memory_order_acquire)) return true;
        if (!ensure_gpu_core()) return false;
        if (!pso_rans_build) pso_rans_build = make_pso(@"hx_rans_batch_build");
        if (!pso_rans_enc)   pso_rans_enc   = make_pso(@"hx_rans_batch_encode");
        for (uint32_t s = 0u; s < N_SLOTS; ++s) {
            if (!enc_slots[s].buf_streams) {
                enc_slots[s].buf_streams = alloc_shared(STREAM_BUF_CAP);
                enc_slots[s].buf_sjobs   = alloc_shared((size_t)MAX_STREAMS_MB * sizeof(HxRansStreamJob));
                enc_slots[s].buf_enc_sym = alloc_shared((size_t)MAX_STREAMS_MB * HX_N_SYMBOLS * sizeof(HxEncSym));
                enc_slots[s].buf_words   = alloc_shared((size_t)MAX_TILES_MB * HX_TILE_MAX_WORDS * sizeof(uint16_t));
                enc_slots[s].buf_fstates = alloc_shared((size_t)MAX_TILES_MB * HX_LANES * sizeof(uint32_t));
                enc_slots[s].buf_tmeta   = alloc_shared((size_t)MAX_TILES_MB * sizeof(HxTileMeta));
            }
        }
        gpu_enc_ready_atomic.store(true, std::memory_order_release);
        return true;
    }

    bool ensure_gpu_decode() {
        if (gpu_dec_ready_atomic.load(std::memory_order_acquire)) return true;
        std::lock_guard<std::mutex> lk(gpu_init_mtx);
        if (gpu_dec_ready_atomic.load(std::memory_order_acquire)) return true;
        if (!ensure_gpu_core()) return false;
        if (!pso_rans_dec) pso_rans_dec = make_pso(@"hx_rans_batch_decode");
        for (uint32_t s = 0u; s < DEC_SLOTS; ++s) {
            if (!dec_slots[s].buf_streams) {
                dec_slots[s].buf_streams = alloc_shared(STREAM_BUF_CAP);
                dec_slots[s].buf_sjobs   = alloc_shared((size_t)MAX_STREAMS_MB * sizeof(HxRansStreamJob));
                dec_slots[s].buf_lut     = alloc_shared((size_t)MAX_STREAMS_MB * HX_PROB_SCALE * sizeof(uint32_t));
                dec_slots[s].buf_words   = alloc_shared((size_t)MAX_TILES_MB * HX_TILE_MAX_WORDS * sizeof(uint16_t));
                dec_slots[s].buf_fstates = alloc_shared((size_t)MAX_TILES_MB * HX_LANES * sizeof(uint32_t));
                dec_slots[s].buf_tmeta   = alloc_shared((size_t)MAX_TILES_MB * sizeof(HxTileMeta));
                dec_slots[s].buf_err     = alloc_shared((size_t)MAX_TILES_MB * sizeof(uint32_t));
            }
        }
        gpu_dec_ready_atomic.store(true, std::memory_order_release);
        return true;
    }

    static void ensure_slot_cpu_buffers(EncSlot& slot) {
        slot.cpu_streams.ensure(STREAM_BUF_CAP);
        slot.cpu_sjobs.ensure((size_t)MAX_STREAMS_MB * sizeof(HxRansStreamJob));
        slot.cpu_enc_sym.ensure((size_t)MAX_STREAMS_MB * HX_N_SYMBOLS * sizeof(HxEncSym));
        slot.cpu_words.ensure((size_t)MAX_TILES_MB * HX_TILE_MAX_WORDS * sizeof(uint16_t));
        slot.cpu_fstates.ensure((size_t)MAX_TILES_MB * HX_LANES * sizeof(uint32_t));
        slot.cpu_tmeta.ensure((size_t)MAX_TILES_MB * sizeof(HxTileMeta));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 圧縮コア: 3 スロット完全非同期パイプライン (Zero-Cold-Start Hybrid CPU/GPU)
    // ═══════════════════════════════════════════════════════════════════════════
    bool compress_core(
        const uint8_t*        in_data,
        size_t                in_size,
        uint32_t              level,
        std::vector<uint8_t>* out_vec,    // メモリ出力時 (非 nullptr)
        FILE*                 out_file,   // ストリーミングファイル出力時 (非 nullptr)
        HxTimingStats*        stats = nullptr,
        uint64_t*             out_written_bytes = nullptr,
        uint32_t*             out_stream_crc    = nullptr,
        uint32_t*             out_raw_crc       = nullptr
    ) {
        HxTimingStats st;
        CFTimeInterval t_all0 = CACurrentMediaTime();

        const uint32_t num_blocks = (in_size == 0u) ? 0u : (uint32_t)((in_size + HX_BLOCK_SIZE - 1u) / HX_BLOCK_SIZE);

        HxArchiveHeader hdr;
        memcpy(hdr.magic, "ASZ2", 4);
        hdr.version       = 1u;
        hdr.original_size = (uint64_t)in_size;
        hdr.block_size    = HX_BLOCK_SIZE;
        hdr.num_blocks    = num_blocks;
        hdr.level         = level;
        hdr.reserved      = 0u;

        std::vector<uint64_t> block_offsets(num_blocks, 0ull);
        const size_t header_total = sizeof(hdr) + (size_t)num_blocks * sizeof(uint64_t);
        const off_t  file_base_off = out_file ? ftello(out_file) : (off_t)0;
        __block uint64_t write_cursor = (uint64_t)header_total;
        __block bool     pipe_failed  = false;
        __block uint32_t payload_crc  = 0u;
        __block uint32_t raw_acc_crc  = 0u;
        const bool       want_crc     = (out_stream_crc != nullptr);
        const bool       want_raw_crc = (out_raw_crc != nullptr);

        if (out_vec) {
            out_vec->clear();
            // 再確保ゼロ保証 (FP16 や Random でも途中 realloc + 64 MiB コピーを完全排除!)
            out_vec->reserve(in_size + header_total + (size_t)num_blocks * sizeof(HxBlockHeader) + 4096u);
            out_vec->resize(header_total);
            memcpy(out_vec->data(), &hdr, sizeof(hdr));
        } else if (out_file) {
            if (fwrite(&hdr, sizeof(hdr), 1, out_file) != 1) return false;
            if (num_blocks > 0u && fwrite(block_offsets.data(), sizeof(uint64_t), num_blocks, out_file) != num_blocks)
                return false;
        }

        if (num_blocks == 0u) {
            if (out_written_bytes) *out_written_bytes = (uint64_t)header_total;
            if (out_raw_crc)       *out_raw_crc       = 0u;
            if (out_stream_crc)    *out_stream_crc    = hx_crc32c((const uint8_t*)&hdr, sizeof(hdr)) ^ payload_crc;
            st.total_s = CACurrentMediaTime() - t_all0;
            if (stats) *stats = st;
            return true;
        }

        dispatch_queue_t     par_q      = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        dispatch_queue_t     write_q    = dispatch_queue_create("aplzx.enc.write", DISPATCH_QUEUE_SERIAL);
        dispatch_semaphore_t slot_sem   = dispatch_semaphore_create(N_SLOTS);
        dispatch_group_t     done_grp   = dispatch_group_create();

        // 大ファイル初回起動時: 先頭/中央ブロックの超高速プローブで圧縮対象なら非同期 Metal 初期化を 1 回だけ先行開始!
        // → Batch 0..k は待ちゼロで CPU 32-way rANS が処理し、初期化完了の瞬間から GPU へ自動移行!
        if (!force_cpu_rans && !gpu_enc_ready_atomic.load(std::memory_order_acquire) && in_size >= HYBRID_GPU_MIN_BYTES) {
            uint32_t b0_len = (uint32_t)std::min((size_t)HX_BLOCK_SIZE, in_size);
            HxProbeResult p0 = hx_probe_block(in_data, b0_len);
            size_t mid_off = (in_size >> 1u) & ~(size_t)(HX_BLOCK_SIZE - 1u);
            uint32_t bm_len = (uint32_t)std::min((size_t)HX_BLOCK_SIZE, in_size - mid_off);
            HxProbeResult pm = hx_probe_block(in_data + mid_off, bm_len);
            if (p0.codec_hint == HX_CODEC_LZ_RANS || p0.codec_hint == HX_CODEC_RANS_ONLY ||
                pm.codec_hint == HX_CODEC_LZ_RANS || pm.codec_hint == HX_CODEC_RANS_ONLY) {
                warmup_gpu_encode_async();
            }
        }

        uint64_t*        offsets_ptr = block_offsets.data();
        HxTimingStats*   st_ptr      = &st;
        HxLzBlockOutput* lz_outs_ptr = cpu_lz_outs;
        const uint32_t   n_mega      = (num_blocks + MB_BLOCKS - 1u) / MB_BLOCKS;

        for (uint32_t mb = 0u; mb < n_mega; ++mb) {
            if (pipe_failed) break;

            CFTimeInterval tw0 = CACurrentMediaTime();
            dispatch_semaphore_wait(slot_sem, DISPATCH_TIME_FOREVER);
            st.pack_s += CACurrentMediaTime() - tw0;

            const uint32_t s_id = mb % N_SLOTS;
            EncSlot& slot = enc_slots[s_id];
            const uint32_t b0 = mb * MB_BLOCKS;
            const uint32_t bc = std::min(MB_BLOCKS, num_blocks - b0);
            const size_t   batch_off   = (size_t)b0 * HX_BLOCK_SIZE;
            const uint32_t batch_bytes = (uint32_t)std::min((size_t)bc * HX_BLOCK_SIZE, in_size - batch_off);

            slot.b0        = b0;
            slot.bc        = bc;
            slot.batch_off = batch_off;
            slot.n_streams = 0u;
            slot.used_gpu  = false;

            // ── Stage 1+2+3 (L2 キャッシュ完全融合 CPU 全コア並列) ─────────────────
            CFTimeInterval tcpu0 = CACurrentMediaTime();
            BlockPlan* plans_ptr = slot.plans;

            dispatch_apply(bc, par_q, ^(size_t ib) {
                uint32_t loff = (uint32_t)ib * HX_BLOCK_SIZE;
                uint32_t blen = std::min(HX_BLOCK_SIZE, batch_bytes - loff);
                const uint8_t* src = in_data + batch_off + loff;

                BlockPlan bp;
                bp.raw_size         = blen;
                bp.crc32c           = hx_crc32c(src, blen);
                HxProbeResult pr    = hx_probe_block(src, blen);
                bp.xform_id         = pr.xform_id;
                bp.codec_id         = pr.codec_hint;
                bp.rle_byte         = pr.rle_byte;
                bp.plane_raw_mask   = pr.plane_raw_mask;
                bp.lz_seq_count     = 0u;
                bp.lz_lit_size      = 0u;
                bp.lz_tail_lit_size = 0u;
                bp.lz_extra_size    = 0u;
                bp.lz_extra_off     = 0u;
                bp.sidx[0] = bp.sidx[1] = bp.sidx[2] = bp.sidx[3] = -1;
                bp.raw_off[0] = bp.raw_off[1] = bp.raw_off[2] = bp.raw_off[3] = 0u;
                bp.raw_len[0] = bp.raw_len[1] = bp.raw_len[2] = bp.raw_len[3] = 0u;

                if (bp.codec_id == HX_CODEC_LZ_RANS) {
                    HxLzThreadState& ts = hx_get_thread_lz_state();
                    const uint8_t* lz_src = src;
                    if (bp.xform_id != HX_XFORM_NONE) {
                        uint8_t* xdst = ts.acquire_xtmp();
                        hx_neon_transform_forward(src, xdst, blen, bp.xform_id);
                        lz_src = xdst;
                    }
                    bool use_lz = hx_lz_encode_block(lz_src, blen, level, ts, lz_outs_ptr[ib]);
                    if (use_lz) {
                        bp.lz_seq_count     = lz_outs_ptr[ib].seq_count;
                        bp.lz_lit_size      = lz_outs_ptr[ib].lit_size;
                        bp.lz_tail_lit_size = lz_outs_ptr[ib].tail_lit_size;
                        bp.lz_extra_size    = lz_outs_ptr[ib].extra_size;
                    } else {
                        bp.codec_id       = HX_CODEC_RANS_ONLY;
                        bp.plane_raw_mask = 0u;
                    }
                }
                plans_ptr[ib] = bp;
            });
            st.lz_s += CACurrentMediaTime() - tcpu0;

            // ── Stage 4a: ストリーム計画 & プレーン分離 / 並列集約タスク構築 ─────────
            CFTimeInterval tr0 = CACurrentMediaTime();
            uint32_t n_streams    = 0u;
            uint32_t n_tiles      = 0u;
            uint32_t cur_off      = 0u;
            uint32_t extra_cursor = 0u;

            struct BlockFillTask {
                uint32_t ib;
                uint8_t  kind;     // 0=RANS_ONLY_PLANES, 1=LZ_4STREAMS
                uint8_t  xform_id;
                uint8_t  n_planes;
                uint32_t raw_size;
                uint32_t dst_off[4];
                uint32_t len[4];
                uint32_t extra_off;
                uint32_t extra_len;
            };
            BlockFillTask   fill_tasks[MB_BLOCKS];
            uint32_t        n_fill = 0u;
            HxRansStreamJob local_jobs[MAX_STREAMS_MB];

            auto reg_stream_at = [&](uint32_t off, uint32_t len) -> int32_t {
                if (len == 0u) return -1;
                uint32_t nt  = (len + HX_TILE_SIZE - 1u) / HX_TILE_SIZE;
                uint32_t idx = n_streams++;
                local_jobs[idx] = { off, len, n_tiles, nt, HX_SUB_RAW, 0u, 0u, 0u };
                n_tiles += nt;
                return (int32_t)idx;
            };

            for (uint32_t ib = 0u; ib < bc; ++ib) {
                BlockPlan& bp = plans_ptr[ib];
                if (bp.xform_id == HX_XFORM_SHUFFLE2)  st.cnt_shuf2++;
                if (bp.xform_id == HX_XFORM_SHUFFLE4)  st.cnt_shuf4++;
                if (bp.xform_id == HX_XFORM_ARM64_BCJ) st.cnt_bcj++;

                if (bp.codec_id == HX_CODEC_RANS_ONLY) {
                    uint32_t n_planes = hx_rans_only_plane_count(bp.xform_id, bp.raw_size);
                    uint8_t  all_mask = (uint8_t)((1u << n_planes) - 1u);
                    if ((bp.plane_raw_mask & all_mask) == all_mask) {
                        bp.codec_id = HX_CODEC_RAW;
                        bp.xform_id = HX_XFORM_NONE;
                        continue;
                    }
                    uint32_t p_off[4], p_len[4];
                    hx_rans_only_plane_layout(bp.xform_id, bp.raw_size, p_off, p_len);

                    BlockFillTask ft = {};
                    ft.ib       = ib;
                    ft.kind     = 0u;
                    ft.xform_id = bp.xform_id;
                    ft.n_planes = (uint8_t)n_planes;
                    ft.raw_size = bp.raw_size;
                    for (uint32_t p = 0u; p < n_planes; ++p) {
                        ft.dst_off[p] = cur_off;
                        ft.len[p]     = p_len[p];
                        if ((bp.plane_raw_mask & (1u << p)) != 0u) {
                            bp.sidx[p]    = -2; // 高エントロピー・プレーンは GPU/CPU rANS へ送らず直接 RAW!
                            bp.raw_off[p] = cur_off;
                            bp.raw_len[p] = p_len[p];
                        } else {
                            bp.sidx[p] = reg_stream_at(cur_off, p_len[p]);
                        }
                        cur_off += (p_len[p] + 15u) & ~15u;
                    }
                    fill_tasks[n_fill++] = ft;
                } else if (bp.codec_id == HX_CODEC_LZ_RANS) {
                    bp.lz_extra_off = extra_cursor;
                    extra_cursor   += bp.lz_extra_size;

                    uint32_t lens[4] = { bp.lz_lit_size, bp.lz_seq_count, bp.lz_seq_count, bp.lz_seq_count };
                    BlockFillTask ft = {};
                    ft.ib        = ib;
                    ft.kind      = 1u;
                    ft.extra_off = bp.lz_extra_off;
                    ft.extra_len = bp.lz_extra_size;
                    for (int s = 0; s < 4; ++s) {
                        ft.dst_off[s] = cur_off;
                        ft.len[s]     = lens[s];
                        bp.sidx[s]    = reg_stream_at(cur_off, lens[s]);
                        cur_off      += (lens[s] + 15u) & ~15u;
                    }
                    fill_tasks[n_fill++] = ft;
                }
            }

            slot.n_streams = n_streams;
            if (extra_cursor > 0u) slot.extra_pool.ensure(extra_cursor);
            uint8_t* extra_dst_base = (extra_cursor > 0u) ? slot.extra_pool.data() : nullptr;

            // 大ファイルで先頭が RAW だった場合も、最初の圧縮バッチで非同期 Metal 初期化を起動
            if (n_streams > 0u && !force_cpu_rans &&
                !gpu_enc_ready_atomic.load(std::memory_order_acquire) && in_size >= HYBRID_GPU_MIN_BYTES) {
                warmup_gpu_encode_async();
            }

            // ── Stage 4b: Apple GPU 非同期投入 OR CPU 全コア ブランチレス 32-way rANS ─
            if (n_streams > 0u) {
                const bool use_gpu = !force_cpu_rans && gpu_enc_ready_atomic.load(std::memory_order_acquire);
                uint8_t* sbuf = nullptr;
                if (use_gpu) {
                    sbuf = (uint8_t*)slot.buf_streams.contents;
                } else {
                    ensure_slot_cpu_buffers(slot);
                    sbuf = slot.cpu_streams.data();
                }

                // 並列ストリーム集約 (RANS_ONLY は NEON 1 パスで各プレーン直書き!)
                const BlockFillTask* ft_ptr = fill_tasks;
                dispatch_apply(n_fill, par_q, ^(size_t idx) {
                    const BlockFillTask& ft = ft_ptr[idx];
                    if (ft.kind == 0u) {
                        const uint8_t* src = in_data + batch_off + (size_t)ft.ib * HX_BLOCK_SIZE;
                        if (ft.xform_id == HX_XFORM_SHUFFLE2 && ft.n_planes == 2u) {
                            hx_neon_shuffle2_planes(src, ft.raw_size, sbuf + ft.dst_off[0], sbuf + ft.dst_off[1]);
                        } else if (ft.xform_id == HX_XFORM_SHUFFLE4 && ft.n_planes == 4u) {
                            hx_neon_shuffle4_planes(src, ft.raw_size,
                                                    sbuf + ft.dst_off[0], sbuf + ft.dst_off[1],
                                                    sbuf + ft.dst_off[2], sbuf + ft.dst_off[3]);
                        } else if (ft.xform_id != HX_XFORM_NONE) {
                            hx_neon_transform_forward(src, sbuf + ft.dst_off[0], ft.raw_size, ft.xform_id);
                        } else {
                            memcpy(sbuf + ft.dst_off[0], src, ft.raw_size);
                        }
                    } else {
                        const HxLzBlockOutput& lo = lz_outs_ptr[ft.ib];
                        if (ft.len[0] > 0u) memcpy(sbuf + ft.dst_off[0], lo.literals.data(), ft.len[0]);
                        if (ft.len[1] > 0u) memcpy(sbuf + ft.dst_off[1], lo.ll_codes.data(), ft.len[1]);
                        if (ft.len[2] > 0u) memcpy(sbuf + ft.dst_off[2], lo.ml_codes.data(), ft.len[2]);
                        if (ft.len[3] > 0u) memcpy(sbuf + ft.dst_off[3], lo.of_codes.data(), ft.len[3]);
                        if (ft.extra_len > 0u) memcpy(extra_dst_base + ft.extra_off, lo.extra_bits.data(), ft.extra_len);
                    }
                });

                if (use_gpu) {
                    memcpy(slot.buf_sjobs.contents, local_jobs, (size_t)n_streams * sizeof(HxRansStreamJob));
                    slot.used_gpu = true;
                    dispatch_semaphore_t gsem = slot.gpu_done_sem;

                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [cq commandBuffer];
                        {
                            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                            [enc setComputePipelineState:pso_rans_build];
                            [enc setBuffer:slot.buf_streams offset:0 atIndex:0];
                            [enc setBuffer:slot.buf_sjobs   offset:0 atIndex:1];
                            [enc setBuffer:slot.buf_enc_sym offset:0 atIndex:2];
                            [enc dispatchThreadgroups:MTLSizeMake(n_streams, 1, 1)
                                 threadsPerThreadgroup:MTLSizeMake(HX_TG_SIZE, 1, 1)];
                            [enc endEncoding];
                        }
                        {
                            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                            [enc setComputePipelineState:pso_rans_enc];
                            [enc setBuffer:slot.buf_streams offset:0 atIndex:0];
                            [enc setBuffer:slot.buf_sjobs   offset:0 atIndex:1];
                            [enc setBuffer:slot.buf_enc_sym offset:0 atIndex:2];
                            [enc setBuffer:slot.buf_words   offset:0 atIndex:3];
                            [enc setBuffer:slot.buf_fstates offset:0 atIndex:4];
                            [enc setBuffer:slot.buf_tmeta   offset:0 atIndex:5];
                            [enc dispatchThreadgroups:MTLSizeMake(n_streams, 1, 1)
                                 threadsPerThreadgroup:MTLSizeMake(HX_TG_SIZE, 1, 1)];
                            [enc endEncoding];
                        }
                        [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                            if (completed.error) {
                                fprintf(stderr, "[ERR] GPU batch rANS error: %s\n",
                                        completed.error.localizedDescription.UTF8String);
                                pipe_failed = true;
                            }
                            dispatch_semaphore_signal(gsem);
                        }];
                        [cb commit]; // 非同期投入! CPU は即座に次スロットへ!
                    }
                } else {
                    // CPU 全コア ブランチレス 32-way rANS (GPU とビット完全一致・Metal 起動遅延 0ms!)
                    HxRansStreamJob* cj   = (HxRansStreamJob*)slot.cpu_sjobs.data();
                    Rans32EncSym*    csym = (Rans32EncSym*)slot.cpu_enc_sym.data();
                    uint16_t*        cw   = (uint16_t*)slot.cpu_words.data();
                    uint32_t*        cfs  = (uint32_t*)slot.cpu_fstates.data();
                    Rans32TileMeta*  ctm  = (Rans32TileMeta*)slot.cpu_tmeta.data();
                    memcpy(cj, local_jobs, (size_t)n_streams * sizeof(HxRansStreamJob));

                    if (n_tiles > n_streams && n_streams < 64u) {
                        // ストリーム数が少なく複数タイル構成 (FP16 32ストリーム×2タイル等) はタイル粒度で並列化して E-core 偏りを解消!
                        struct TileTask { uint16_t si; uint16_t local_t; };
                        TileTask ttasks[MAX_TILES_MB];
                        for (uint32_t si = 0u; si < n_streams; ++si) {
                            for (uint32_t t = 0u; t < cj[si].num_tiles; ++t) {
                                ttasks[cj[si].first_tile_idx + t] = { (uint16_t)si, (uint16_t)t };
                            }
                        }
                        std::vector<Rans32CpuFastSym> fast_syms((size_t)n_streams * HX_N_SYMBOLS);
                        Rans32CpuFastSym* fs_pool = fast_syms.data();
                        dispatch_apply(n_streams, par_q, ^(size_t si) {
                            HxRansStreamJob& sj = cj[si];
                            Rans32BlockMeta bm;
                            Rans32EncSym* my_sym = csym + si * HX_N_SYMBOLS;
                            rans32_cpu_build_tables_fast(sbuf + sj.data_offset, sj.raw_size, my_sym, &bm);
                            sj.mode      = bm.mode;
                            sj.rle_byte  = bm.rle_byte;
                            sj.n_nonzero = bm.n_nonzero;
                            if (bm.mode == HX_SUB_RANS) {
                                rans32_build_fast_enc_sym(my_sym, fs_pool + si * HX_N_SYMBOLS);
                            }
                        });
                        const TileTask* tt_ptr = ttasks;
                        dispatch_apply(n_tiles, par_q, ^(size_t tgid) {
                            uint32_t si = tt_ptr[tgid].si;
                            uint32_t t  = tt_ptr[tgid].local_t;
                            const HxRansStreamJob& sj = cj[si];
                            uint32_t loff = t * HX_TILE_SIZE;
                            uint32_t tlen = std::min(HX_TILE_SIZE, sj.raw_size - loff);
                            if (sj.mode == HX_SUB_RLE) {
                                ctm[tgid] = { HX_SUB_RLE, sj.rle_byte, tlen, 0u };
                            } else if (sj.mode == HX_SUB_RAW || tlen < HX_MIN_RANS_LEN) {
                                ctm[tgid] = { HX_SUB_RAW, 0u, tlen, tlen };
                            } else {
                                rans32_encode_tile_fast_with_table(sbuf + sj.data_offset + loff, tlen,
                                                                   fs_pool + (size_t)si * HX_N_SYMBOLS,
                                                                   cw + (size_t)tgid * HX_TILE_MAX_WORDS,
                                                                   cfs + (size_t)tgid * HX_LANES,
                                                                   &ctm[tgid]);
                            }
                        });
                    } else {
                        dispatch_apply(n_streams, par_q, ^(size_t si) {
                            HxRansStreamJob& sj = cj[si];
                            const uint8_t* sdata = sbuf + sj.data_offset;
                            Rans32EncSym*  my_sym = csym + si * HX_N_SYMBOLS;
                            Rans32BlockMeta bm;
                            rans32_cpu_build_tables_fast(sdata, sj.raw_size, my_sym, &bm);
                            sj.mode      = bm.mode;
                            sj.rle_byte  = bm.rle_byte;
                            sj.n_nonzero = bm.n_nonzero;

                            Rans32CpuFastSym fsym[HX_N_SYMBOLS];
                            if (bm.mode == HX_SUB_RANS) {
                                rans32_build_fast_enc_sym(my_sym, fsym);
                            }
                            for (uint32_t t = 0u; t < sj.num_tiles; ++t) {
                                uint32_t tgid = sj.first_tile_idx + t;
                                uint32_t loff = t * HX_TILE_SIZE;
                                uint32_t tlen = std::min(HX_TILE_SIZE, sj.raw_size - loff);
                                if (bm.mode == HX_SUB_RLE) {
                                    ctm[tgid] = { HX_SUB_RLE, bm.rle_byte, tlen, 0u };
                                } else if (bm.mode == HX_SUB_RAW || tlen < HX_MIN_RANS_LEN) {
                                    ctm[tgid] = { HX_SUB_RAW, 0u, tlen, tlen };
                                } else {
                                    rans32_encode_tile_fast_with_table(sdata + loff, tlen, fsym,
                                                                       cw + (size_t)tgid * HX_TILE_MAX_WORDS,
                                                                       cfs + (size_t)tgid * HX_LANES,
                                                                       &ctm[tgid]);
                                }
                            }
                        });
                    }
                }
            }
            st.rans_s += CACurrentMediaTime() - tr0;

            // ── Stage 5: シリアル write_q + 10 コア並列ゼロ中間コピーパッキング ────
            EncSlot* slot_ptr = &slot;
            dispatch_group_enter(done_grp);
            dispatch_async(write_q, ^{
                if (slot_ptr->used_gpu) {
                    dispatch_semaphore_wait(slot_ptr->gpu_done_sem, DISPATCH_TIME_FOREVER);
                }
                if (!pipe_failed) {
                    const uint8_t*         sbuf    = slot_ptr->n_streams ? (slot_ptr->used_gpu ? (const uint8_t*)slot_ptr->buf_streams.contents         : slot_ptr->cpu_streams.data()) : nullptr;
                    const HxRansStreamJob* sjobs   = slot_ptr->n_streams ? (slot_ptr->used_gpu ? (const HxRansStreamJob*)slot_ptr->buf_sjobs.contents   : (const HxRansStreamJob*)slot_ptr->cpu_sjobs.data()) : nullptr;
                    const HxEncSym*        sym_ptr = slot_ptr->n_streams ? (slot_ptr->used_gpu ? (const HxEncSym*)slot_ptr->buf_enc_sym.contents        : (const HxEncSym*)slot_ptr->cpu_enc_sym.data()) : nullptr;
                    const HxTileMeta*      tm_ptr  = slot_ptr->n_streams ? (slot_ptr->used_gpu ? (const HxTileMeta*)slot_ptr->buf_tmeta.contents        : (const HxTileMeta*)slot_ptr->cpu_tmeta.data()) : nullptr;
                    const uint16_t*        w_ptr   = slot_ptr->n_streams ? (slot_ptr->used_gpu ? (const uint16_t*)slot_ptr->buf_words.contents          : (const uint16_t*)slot_ptr->cpu_words.data()) : nullptr;
                    const uint32_t*        fs_ptr  = slot_ptr->n_streams ? (slot_ptr->used_gpu ? (const uint32_t*)slot_ptr->buf_fstates.contents        : (const uint32_t*)slot_ptr->cpu_fstates.data()) : nullptr;
                    const uint8_t*         ex_pool = slot_ptr->extra_pool.data();

                    // 1) 全 bc ブロックの厳密パックサイズとオフセットを O(bc) で計算 (コピーゼロ!)
                    struct BlockPackInfo {
                        uint32_t local_off;
                        uint32_t payload_sz;
                        bool     use_raw;
                        uint8_t  sub_mode[4];
                    };
                    BlockPackInfo pinfo[MB_BLOCKS];
                    uint32_t batch_total = 0u;

                    auto calc_sub_size = [&](int32_t sidx, uint32_t raw_len, uint8_t& out_mode) -> uint32_t {
                        if (sidx == -1) { out_mode = HX_SUB_RAW; return 4u; }
                        if (sidx == -2) { out_mode = HX_SUB_RAW; return 4u + raw_len; }
                        const HxRansStreamJob& sj = sjobs[sidx];
                        bool any_rans = false;
                        if (sj.mode == HX_SUB_RANS) {
                            for (uint32_t t = 0u; t < sj.num_tiles; ++t) {
                                if (tm_ptr[sj.first_tile_idx + t].mode == HX_SUB_RANS) { any_rans = true; break; }
                            }
                        }
                        uint8_t dm = (sj.mode == HX_SUB_RANS && !any_rans) ? HX_SUB_RAW : (uint8_t)sj.mode;
                        out_mode = dm;
                        if (dm == HX_SUB_RAW) return 4u + sj.raw_size;
                        if (dm == HX_SUB_RLE) return 4u;
                        uint32_t sz = 4u + hx_freq_table_bytes(sj.n_nonzero)
                                    + sj.num_tiles * (uint32_t)sizeof(Rans32TileDiskDesc);
                        for (uint32_t t = 0u; t < sj.num_tiles; ++t) {
                            const HxTileMeta& tm = tm_ptr[sj.first_tile_idx + t];
                            if (tm.mode == HX_SUB_RAW)       sz += tm.raw_size;
                            else if (tm.mode == HX_SUB_RANS) sz += HX_LANES * (uint32_t)sizeof(uint32_t) + tm.word_count * (uint32_t)sizeof(uint16_t);
                        }
                        return sz;
                    };

                    for (uint32_t ib = 0u; ib < slot_ptr->bc; ++ib) {
                        const uint32_t gidx = slot_ptr->b0 + ib;
                        const BlockPlan& bp = slot_ptr->plans[ib];
                        BlockPackInfo& pi = pinfo[ib];
                        pi.local_off = batch_total;
                        offsets_ptr[gidx] = write_cursor + (uint64_t)batch_total;

                        uint32_t psz = 0u;
                        if (bp.codec_id == HX_CODEC_RAW) {
                            pi.use_raw = true;
                        } else if (bp.codec_id == HX_CODEC_RLE) {
                            pi.use_raw = false;
                            psz = 0u;
                        } else if (bp.codec_id == HX_CODEC_RANS_ONLY) {
                            uint32_t np = hx_rans_only_plane_count(bp.xform_id, bp.raw_size);
                            for (uint32_t p = 0u; p < np; ++p) psz += calc_sub_size(bp.sidx[p], bp.raw_len[p], pi.sub_mode[p]);
                            pi.use_raw = (psz >= bp.raw_size);
                        } else {
                            psz = (uint32_t)sizeof(HxLzBlockMeta);
                            for (int s = 0; s < 4; ++s) psz += calc_sub_size(bp.sidx[s], 0u, pi.sub_mode[s]);
                            psz += bp.lz_extra_size;
                            pi.use_raw = (psz >= bp.raw_size);
                        }

                        if (pi.use_raw) {
                            pi.payload_sz = bp.raw_size;
                            st_ptr->cnt_raw++;
                        } else {
                            pi.payload_sz = psz;
                            if      (bp.codec_id == HX_CODEC_RLE)       st_ptr->cnt_rle++;
                            else if (bp.codec_id == HX_CODEC_RANS_ONLY) st_ptr->cnt_rans_only++;
                            else                                        st_ptr->cnt_lz++;
                        }
                        batch_total += (uint32_t)sizeof(HxBlockHeader) + pi.payload_sz;
                    }

                    // 2) 最終出力先ポインタを 1 回だけ確保 (out_vec 時は中間コピー完全ゼロ!)
                    uint8_t* dst_base = nullptr;
                    if (out_vec) {
                        size_t old_sz = out_vec->size();
                        out_vec->resize(old_sz + batch_total);
                        dst_base = out_vec->data() + old_sz;
                    } else {
                        slot_ptr->pack_tmp.ensure(batch_total);
                        dst_base = slot_ptr->pack_tmp.data();
                    }

                    // 3) 適応型ゼロ中間コピーパッキング:
                    //    - >= 4 MiB (FP16 / Mixed / Random 等): 10 コア並列パックで 8〜15 GB/s!
                    //    - <  4 MiB (高圧縮 LZ 等): write_q 単スレッドで 0.1ms 完了させ、次スロット LZ の P-core を奪わない!
                    const BlockPackInfo* pi_ptr = pinfo;
                    const BlockPlan*     bp_arr = slot_ptr->plans;
                    const size_t         boff   = slot_ptr->batch_off;
                    uint32_t             blk_crc_arr[MB_BLOCKS];
                    uint32_t*            blk_crc_ptr = blk_crc_arr;

                    auto pack_one_block = ^(size_t ib) {
                        const BlockPlan&     bp = bp_arr[ib];
                        const BlockPackInfo& pi = pi_ptr[ib];
                        uint8_t* const blk_start = dst_base + pi.local_off;
                        uint8_t* p = blk_start;

                        auto write_sub = [&](int32_t sidx, uint32_t raw_off, uint32_t raw_len, uint8_t dm) {
                            if (sidx == -1) {
                                uint8_t h[4] = { HX_SUB_RAW, 0u, 0u, 0u };
                                memcpy(p, h, 4u); p += 4u;
                                return;
                            }
                            if (sidx == -2) {
                                uint8_t h[4] = { HX_SUB_RAW, 0u, 1u, 0u };
                                memcpy(p, h, 4u); p += 4u;
                                memcpy(p, sbuf + raw_off, raw_len); p += raw_len;
                                return;
                            }
                            const HxRansStreamJob& sj = sjobs[sidx];
                            uint8_t h[4] = { dm, (uint8_t)sj.rle_byte, (uint8_t)sj.num_tiles, 0u };
                            memcpy(p, h, 4u); p += 4u;
                            if (dm == HX_SUB_RAW) {
                                memcpy(p, sbuf + sj.data_offset, sj.raw_size); p += sj.raw_size;
                            } else if (dm == HX_SUB_RANS) {
                                p = hx_write_freq_table(p, sym_ptr + (size_t)sidx * HX_N_SYMBOLS, sj.n_nonzero);
                                for (uint32_t t = 0u; t < sj.num_tiles; ++t) {
                                    const HxTileMeta& tm = tm_ptr[sj.first_tile_idx + t];
                                    Rans32TileDiskDesc td;
                                    td.tile_mode     = (uint8_t)tm.mode;
                                    td.rle_byte      = (uint8_t)tm.rle_byte;
                                    td.reserved      = 0u;
                                    td.payload_words = (tm.mode == HX_SUB_RANS) ? tm.word_count :
                                                       (tm.mode == HX_SUB_RAW)  ? tm.raw_size   : 0u;
                                    memcpy(p, &td, sizeof(td)); p += sizeof(td);
                                }
                                for (uint32_t t = 0u; t < sj.num_tiles; ++t) {
                                    uint32_t tgid = sj.first_tile_idx + t;
                                    const HxTileMeta& tm = tm_ptr[tgid];
                                    if (tm.mode == HX_SUB_RAW) {
                                        memcpy(p, sbuf + sj.data_offset + t * HX_TILE_SIZE, tm.raw_size);
                                        p += tm.raw_size;
                                    } else if (tm.mode == HX_SUB_RANS) {
                                        memcpy(p, fs_ptr + (size_t)tgid * HX_LANES, HX_LANES * sizeof(uint32_t));
                                        p += HX_LANES * sizeof(uint32_t);
                                        size_t wb = (size_t)tm.word_count * sizeof(uint16_t);
                                        memcpy(p, w_ptr + (size_t)tgid * HX_TILE_MAX_WORDS, wb);
                                        p += wb;
                                    }
                                }
                            }
                        };

                        if (pi.use_raw) {
                            HxBlockHeader bh = { HX_XFORM_NONE, HX_CODEC_RAW, 0u, 0u, bp.raw_size, bp.crc32c, bp.raw_size };
                            memcpy(p, &bh, sizeof(bh)); p += sizeof(bh);
                            memcpy(p, in_data + boff + ib * HX_BLOCK_SIZE, bp.raw_size);
                        } else {
                            HxBlockHeader bh = { bp.xform_id, bp.codec_id, bp.rle_byte, 0u, bp.raw_size, bp.crc32c, pi.payload_sz };
                            memcpy(p, &bh, sizeof(bh)); p += sizeof(bh);
                            if (bp.codec_id == HX_CODEC_RANS_ONLY) {
                                uint32_t np = hx_rans_only_plane_count(bp.xform_id, bp.raw_size);
                                for (uint32_t k = 0u; k < np; ++k) {
                                    write_sub(bp.sidx[k], bp.raw_off[k], bp.raw_len[k], pi.sub_mode[k]);
                                }
                            } else if (bp.codec_id == HX_CODEC_LZ_RANS) {
                                HxLzBlockMeta lm = { bp.lz_seq_count, bp.lz_lit_size, bp.lz_tail_lit_size, bp.lz_extra_size };
                                memcpy(p, &lm, sizeof(lm)); p += sizeof(lm);
                                for (int s = 0; s < 4; ++s) write_sub(bp.sidx[s], 0u, 0u, pi.sub_mode[s]);
                                if (bp.lz_extra_size > 0u) memcpy(p, ex_pool + bp.lz_extra_off, bp.lz_extra_size);
                            }
                        }
                        // パック直後の L1D キャッシュ上から並列 CRC32C 計算 (write_q 単スレッド走査を完全排除!)
                        if (want_crc) {
                            blk_crc_ptr[ib] = hx_crc32c(blk_start, sizeof(HxBlockHeader) + pi.payload_sz);
                        }
                    };

                    if (batch_total >= 4u * 1024u * 1024u || (want_crc && batch_total >= 1024u * 1024u)) {
                        dispatch_apply(slot_ptr->bc, par_q, pack_one_block);
                    } else {
                        for (uint32_t ib = 0u; ib < slot_ptr->bc; ++ib) pack_one_block(ib);
                    }

                    write_cursor += (uint64_t)batch_total;
                    if (want_raw_crc) {
                        uint32_t rc = raw_acc_crc;
                        for (uint32_t ib = 0u; ib < slot_ptr->bc; ++ib) {
                            rc = hx_crc32c_combine(rc, bp_arr[ib].crc32c, bp_arr[ib].raw_size);
                        }
                        raw_acc_crc = rc;
                    }
                    if (want_crc && batch_total > 0u) {
                        uint32_t c = payload_crc;
                        for (uint32_t ib = 0u; ib < slot_ptr->bc; ++ib) {
                            c = hx_crc32c_combine(c, blk_crc_arr[ib], (uint64_t)sizeof(HxBlockHeader) + pinfo[ib].payload_sz);
                        }
                        payload_crc = c;
                    }
                    if (out_file && batch_total > 0u) {
                        if (fwrite(dst_base, 1, batch_total, out_file) != batch_total) {
                            pipe_failed = true;
                        }
                    }
                }
                dispatch_semaphore_signal(slot_sem);
                dispatch_group_leave(done_grp);
            });
        }

        CFTimeInterval td0 = CACurrentMediaTime();
        dispatch_group_wait(done_grp, DISPATCH_TIME_FOREVER);
        st.pack_s += CACurrentMediaTime() - td0;

        if (pipe_failed) return false;

        // シークテーブルを先頭 (file_base_off + sizeof(hdr)) へ書き戻し、末尾へ戻す
        if (out_vec) {
            memcpy(out_vec->data() + sizeof(hdr), block_offsets.data(), (size_t)num_blocks * sizeof(uint64_t));
        } else if (out_file) {
            if (fseeko(out_file, file_base_off + (off_t)sizeof(hdr), SEEK_SET) != 0) return false;
            if (fwrite(block_offsets.data(), sizeof(uint64_t), num_blocks, out_file) != num_blocks) return false;
            if (fseeko(out_file, file_base_off + (off_t)write_cursor, SEEK_SET) != 0) return false;
        }
        if (out_written_bytes) *out_written_bytes = write_cursor;
        if (out_raw_crc)       *out_raw_crc       = raw_acc_crc;
        if (out_stream_crc) {
            std::vector<uint8_t> hbuf(header_total);
            memcpy(hbuf.data(), &hdr, sizeof(hdr));
            if (num_blocks > 0u) memcpy(hbuf.data() + sizeof(hdr), block_offsets.data(), (size_t)num_blocks * sizeof(uint64_t));
            *out_stream_crc = hx_crc32c(hbuf.data(), header_total) ^ payload_crc;
        }

        st.total_s = CACurrentMediaTime() - t_all0;
        if (stats) *stats = st;
        return true;
    }

    bool compress(const uint8_t* in_data, size_t in_size, uint32_t level,
                  std::vector<uint8_t>& out_archive, HxTimingStats* stats = nullptr) {
        return compress_core(in_data, in_size, level, &out_archive, nullptr, stats);
    }

    bool compress_file(const char* in_path, const char* out_path, uint32_t level, HxTimingStats* stats = nullptr) {
        int fd = open(in_path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st_in;
        if (fstat(fd, &st_in) < 0) { close(fd); return false; }
        size_t sz = (size_t)st_in.st_size;
        const uint8_t* mapped = (const uint8_t*)"";
        if (sz > 0u) {
            void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) { close(fd); return false; }
            madvise(m, sz, MADV_SEQUENTIAL);
            mapped = (const uint8_t*)m;
        }

        std::string tmp_path = std::string(out_path) + ".tmp";
        FILE* fout = fopen(tmp_path.c_str(), "wb");
        if (!fout) {
            if (sz > 0u) munmap((void*)mapped, sz);
            close(fd);
            return false;
        }
        setvbuf(fout, nullptr, _IOFBF, 4 * 1024 * 1024);

        bool ok = compress_core(mapped, sz, level, nullptr, fout, stats);
        if (fclose(fout) != 0) ok = false;
        if (sz > 0u) munmap((void*)mapped, sz);
        close(fd);

        if (!ok || rename(tmp_path.c_str(), out_path) != 0) {
            unlink(tmp_path.c_str());
            return false;
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 解凍コア: GPU 時 2 スロット非同期ダブルバッファ (GPU rANS || CPU LZ+NEON+CRC)
    //           CPU 時 ゼロコピー CPU NEON 32-way + 非同期ダブルバッファ fwrite
    // ═══════════════════════════════════════════════════════════════════════════
    bool decompress_core(
        const uint8_t*        in_archive,
        size_t                in_size,
        std::vector<uint8_t>* out_vec,
        FILE*                 out_file,
        HxTimingStats*        stats = nullptr,
        uint32_t*             inout_raw_crc = nullptr
    ) {
        HxTimingStats st;
        CFTimeInterval t_all0 = CACurrentMediaTime();

        if (in_size < sizeof(HxArchiveHeader)) return false;
        HxArchiveHeader hdr;
        memcpy(&hdr, in_archive, sizeof(hdr));
        if (memcmp(hdr.magic, "ASZ2", 4) != 0 || hdr.version != 1u || hdr.block_size != HX_BLOCK_SIZE)
            return false;

        if (out_vec) out_vec->resize((size_t)hdr.original_size);
        if (hdr.num_blocks == 0u) {
            if (stats) { st.total_s = CACurrentMediaTime() - t_all0; *stats = st; }
            return hdr.original_size == 0u;
        }

        const size_t off_bytes = (size_t)hdr.num_blocks * sizeof(uint64_t);
        if (in_size < sizeof(hdr) + off_bytes) return false;
        std::vector<uint64_t> block_offsets(hdr.num_blocks);
        memcpy(block_offsets.data(), in_archive + sizeof(hdr), off_bytes);

        dispatch_queue_t     par_q   = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        dispatch_queue_t     write_q = out_file ? dispatch_queue_create("aplzx.dec.write", DISPATCH_QUEUE_SERIAL) : nil;
        static const uint32_t OUT_WR_SLOTS = 3u;
        dispatch_semaphore_t out_sem = out_file ? dispatch_semaphore_create(OUT_WR_SLOTS) : nil;
        dispatch_group_t     wrt_grp = out_file ? dispatch_group_create() : nil;

        HxRawBuf file_out_slots[OUT_WR_SLOTS];
        const bool verify_only = (!out_vec && !out_file);
        if (out_file) {
            for (uint32_t s = 0u; s < OUT_WR_SLOTS; ++s) file_out_slots[s].ensure(MB_BYTES);
        } else if (verify_only) {
            file_out_slots[0].ensure(MB_BYTES);
        }
        HxRawBuf* file_out_ptr = file_out_slots;
        __block bool write_failed = false;

        // 解凍は明示的に GPU 初期化済み (eager_gpu / warmup_gpu_decode_async) の場合のみ GPU 2 スロット使用。
        // 単発 CLI では 0ms 起動＋完全ゼロコピー ARM64 NEON 32-way が 1.5 倍高速 (実測 6.4 GB/s Verify / 3.8 GB/s File-to-File)。
        const bool use_gpu_path = gpu_dec_ready_atomic.load(std::memory_order_acquire) && !force_cpu_rans;
        const uint32_t GPU_RANS_MIN_BYTES = 1024u * 1024u;

        // 共通ブロック展開ラムダ (LZ 展開 + NEON 逆変換 + HW CRC32C)
        auto decode_blocks_cpu = ^bool(const DecBlockPlan* dplans_ptr, uint32_t bc, uint8_t* dst_base) {
            __block bool ok = true;
            dispatch_apply(bc, par_q, ^(size_t ib) {
                const DecBlockPlan& dp = dplans_ptr[ib];
                uint8_t* final_dst = dst_base + ib * HX_BLOCK_SIZE;

                if (dp.bh.codec_id == HX_CODEC_RAW) {
                    memcpy(final_dst, dp.payload_ptr, dp.bh.raw_size);
                } else if (dp.bh.codec_id == HX_CODEC_RLE) {
                    memset(final_dst, dp.bh.rle_byte, dp.bh.raw_size);
                } else if (dp.bh.codec_id == HX_CODEC_RANS_ONLY) {
                    if (dp.bh.xform_id == HX_XFORM_SHUFFLE2 && dp.bh.raw_size >= 2u) {
                        hx_neon_unshuffle2_planes(dp.s_ptr[0], dp.s_ptr[1], final_dst, dp.bh.raw_size);
                    } else if (dp.bh.xform_id == HX_XFORM_SHUFFLE4 && dp.bh.raw_size >= 4u) {
                        hx_neon_unshuffle4_planes(dp.s_ptr[0], dp.s_ptr[1], dp.s_ptr[2], dp.s_ptr[3],
                                                  final_dst, dp.bh.raw_size);
                    } else if (dp.bh.xform_id != HX_XFORM_NONE) {
                        hx_neon_transform_inverse(dp.s_ptr[0], final_dst, dp.bh.raw_size, dp.bh.xform_id);
                    } else {
                        memcpy(final_dst, dp.s_ptr[0], dp.bh.raw_size);
                    }
                } else if (dp.bh.codec_id == HX_CODEC_LZ_RANS) {
                    uint8_t* lz_dst = (dp.bh.xform_id != HX_XFORM_NONE)
                                          ? hx_get_thread_lz_state().acquire_xtmp()
                                          : final_dst;
                    const uint8_t* ex_ptr = dp.extra_ptr;
                    bool lz_ok;
                    if (__builtin_expect((size_t)((in_archive + in_size) - ex_ptr) >= (size_t)dp.extra_bytes + 8u, 1)) {
                        lz_ok = hx_lz_decode_block(dp.lm, dp.s_ptr[0], dp.s_ptr[1], dp.s_ptr[2], dp.s_ptr[3],
                                                   ex_ptr, dp.extra_bytes, lz_dst, dp.bh.raw_size);
                    } else {
                        std::vector<uint8_t> ex_pad((size_t)dp.extra_bytes + 8u, 0u);
                        if (dp.extra_bytes > 0u) memcpy(ex_pad.data(), ex_ptr, dp.extra_bytes);
                        lz_ok = hx_lz_decode_block(dp.lm, dp.s_ptr[0], dp.s_ptr[1], dp.s_ptr[2], dp.s_ptr[3],
                                                   ex_pad.data(), dp.extra_bytes, lz_dst, dp.bh.raw_size);
                    }
                    if (!lz_ok) {
                        ok = false;
                        return;
                    }
                    if (dp.bh.xform_id != HX_XFORM_NONE) {
                        hx_neon_transform_inverse(lz_dst, final_dst, dp.bh.raw_size, dp.bh.xform_id);
                    }
                } else {
                    ok = false;
                    return;
                }

                if (hx_crc32c(final_dst, dp.bh.raw_size) != dp.bh.crc32c) {
                    ok = false;
                }
            });
            return ok;
        };

        if (use_gpu_path) {
            // ── GPU 2 スロット完全非同期パイプライン (GPU rANS || CPU LZ+NEON+CRC) ──
            dispatch_queue_t     dec_work_q   = dispatch_queue_create("aplzx.dec.cpu", DISPATCH_QUEUE_SERIAL);
            dispatch_semaphore_t dec_slot_sem = dispatch_semaphore_create(DEC_SLOTS);
            dispatch_group_t     dec_grp      = dispatch_group_create();
            __block bool         dec_failed   = false;
            HxTimingStats*       st_ptr       = &st;
            uint32_t             mb_idx       = 0u;

            for (uint32_t b0 = 0u; b0 < hdr.num_blocks; b0 += MB_BLOCKS, ++mb_idx) {
                CFTimeInterval tw0 = CACurrentMediaTime();
                dispatch_semaphore_wait(dec_slot_sem, DISPATCH_TIME_FOREVER);
                st.rans_s += CACurrentMediaTime() - tw0;

                if (dec_failed || write_failed) {
                    dispatch_semaphore_signal(dec_slot_sem);
                    break;
                }

                const uint32_t s_id        = mb_idx % DEC_SLOTS;
                DecSlot&       ds          = dec_slots[s_id];
                const uint32_t bc          = std::min(MB_BLOCKS, hdr.num_blocks - b0);
                const size_t   batch_off   = (size_t)b0 * HX_BLOCK_SIZE;
                const size_t   batch_bytes = std::min((size_t)bc * HX_BLOCK_SIZE, (size_t)hdr.original_size - batch_off);

                CFTimeInterval tp0 = CACurrentMediaTime();
                HxRansStreamJob* sjobs   = (HxRansStreamJob*)ds.buf_sjobs.contents;
                HxTileMeta*      tm_ptr  = (HxTileMeta*)ds.buf_tmeta.contents;
                uint16_t*        w_ptr   = (uint16_t*)ds.buf_words.contents;
                uint32_t*        fs_ptr  = (uint32_t*)ds.buf_fstates.contents;
                uint32_t*        lut_ptr = (uint32_t*)ds.buf_lut.contents;
                uint8_t*         dec_s   = (uint8_t*)ds.buf_streams.contents;

                uint32_t n_streams        = 0u;
                uint32_t n_tiles          = 0u;
                uint32_t cur_off          = 0u;
                uint32_t total_rans_bytes = 0u;

                struct FreqTask     { uint32_t sidx; const uint8_t* p; const uint8_t* end; };
                struct TileCopyTask { uint32_t tgid; const uint8_t* p; uint32_t wc; };
                FreqTask     freq_tasks[MAX_STREAMS_MB];
                TileCopyTask tile_copies[MAX_TILES_MB];
                uint32_t n_freq  = 0u;
                uint32_t n_tcopy = 0u;

                auto parse_substream = [&](const uint8_t*& p, const uint8_t* end,
                                           uint32_t expected_len, int32_t& out_sidx,
                                           const uint8_t*& out_ptr, uint32_t& out_len) -> bool {
                    out_sidx = -1; out_ptr = nullptr; out_len = expected_len;
                    if ((size_t)(end - p) < 4u) return false;
                    uint8_t smode = p[0], rleb = p[1], ntiles = p[2];
                    p += 4;
                    if (expected_len == 0u) return true;
                    if (smode == HX_SUB_RAW) {
                        if ((size_t)(end - p) < expected_len) return false;
                        out_ptr = p;
                        p += expected_len;
                        return true;
                    }
                    uint32_t idx = n_streams++;
                    uint32_t exp_nt = (expected_len + HX_TILE_SIZE - 1u) / HX_TILE_SIZE;
                    out_sidx = (int32_t)idx;
                    out_ptr  = dec_s + cur_off;

                    if (smode == HX_SUB_RLE) {
                        memset(dec_s + cur_off, rleb, expected_len);
                        sjobs[idx] = { cur_off, expected_len, n_tiles, 0u, HX_SUB_RLE, rleb, 0u, 0u };
                    } else if (smode == HX_SUB_RANS) {
                        if (ntiles != exp_nt || ntiles == 0u || ntiles > HX_MAX_TILES_STR) return false;
                        if ((size_t)(end - p) < 2u) return false;
                        uint8_t fkind = p[0];
                        if (fkind != HX_FREQ_SPARSE && fkind != HX_FREQ_DENSE12) return false;
                        uint32_t n_nz = (uint32_t)p[1] + 1u;
                        size_t fbytes = 2u + ((fkind == HX_FREQ_SPARSE) ? ((size_t)n_nz * 3u) : 384u);
                        if ((size_t)(end - p) < fbytes) return false;
                        freq_tasks[n_freq++] = { idx, p, p + fbytes };
                        p += fbytes;

                        if ((size_t)(end - p) < (size_t)ntiles * sizeof(Rans32TileDiskDesc)) return false;
                        Rans32TileDiskDesc tds[HX_MAX_TILES_STR];
                        memcpy(tds, p, (size_t)ntiles * sizeof(Rans32TileDiskDesc));
                        p += (size_t)ntiles * sizeof(Rans32TileDiskDesc);

                        uint32_t ft = n_tiles;
                        for (uint32_t t = 0u; t < ntiles; ++t) {
                            uint32_t tgid = ft + t;
                            uint32_t tlen = std::min(HX_TILE_SIZE, expected_len - t * HX_TILE_SIZE);
                            if (tds[t].tile_mode == HX_SUB_RAW) {
                                if (tds[t].payload_words != tlen || (size_t)(end - p) < tlen) return false;
                                memcpy(dec_s + cur_off + t * HX_TILE_SIZE, p, tlen);
                                p += tlen;
                                tm_ptr[tgid] = { HX_SUB_RAW, 0u, tlen, tlen };
                            } else if (tds[t].tile_mode == HX_SUB_RLE) {
                                tm_ptr[tgid] = { HX_SUB_RLE, tds[t].rle_byte, tlen, 0u };
                            } else if (tds[t].tile_mode == HX_SUB_RANS) {
                                uint32_t wc = tds[t].payload_words;
                                if (wc > HX_TILE_MAX_WORDS) return false;
                                size_t need = HX_LANES * sizeof(uint32_t) + (size_t)wc * sizeof(uint16_t);
                                if ((size_t)(end - p) < need) return false;
                                tile_copies[n_tcopy++] = { tgid, p, wc };
                                tm_ptr[tgid] = { HX_SUB_RANS, 0u, tlen, wc };
                                p += need;
                                total_rans_bytes += tlen;
                            } else {
                                return false;
                            }
                        }
                        sjobs[idx] = { cur_off, expected_len, ft, ntiles, HX_SUB_RANS, 0u, 0u, 0u };
                        n_tiles += ntiles;
                    } else {
                        return false;
                    }
                    cur_off += (expected_len + 15u) & ~15u;
                    return true;
                };

                bool parse_ok = true;
                for (uint32_t ib = 0u; ib < bc && parse_ok; ++ib) {
                    uint32_t gidx = b0 + ib;
                    if (block_offsets[gidx] >= in_size) { parse_ok = false; break; }
                    const uint8_t* p   = in_archive + block_offsets[gidx];
                    const uint8_t* end = in_archive + in_size;
                    if ((size_t)(end - p) < sizeof(HxBlockHeader)) { parse_ok = false; break; }

                    DecBlockPlan& dp = ds.dplans[ib];
                    memcpy(&dp.bh, p, sizeof(dp.bh));
                    p += sizeof(dp.bh);
                    if ((size_t)(end - p) < dp.bh.comp_payload) { parse_ok = false; break; }
                    const uint8_t* blk_end = p + dp.bh.comp_payload;
                    dp.payload_ptr = p;
                    dp.extra_ptr   = nullptr;
                    dp.extra_bytes = 0u;
                    dp.sidx[0] = dp.sidx[1] = dp.sidx[2] = dp.sidx[3] = -1;
                    dp.s_ptr[0] = dp.s_ptr[1] = dp.s_ptr[2] = dp.s_ptr[3] = nullptr;

                    uint32_t exp_raw = (uint32_t)std::min((uint64_t)HX_BLOCK_SIZE, hdr.original_size - (batch_off + (size_t)ib * HX_BLOCK_SIZE));
                    if (dp.bh.raw_size != exp_raw) { parse_ok = false; break; }

                    if (dp.bh.codec_id == HX_CODEC_RANS_ONLY) {
                        uint32_t np = hx_rans_only_plane_count(dp.bh.xform_id, dp.bh.raw_size);
                        uint32_t p_off[4], p_len[4];
                        hx_rans_only_plane_layout(dp.bh.xform_id, dp.bh.raw_size, p_off, p_len);
                        for (uint32_t k = 0u; k < np; ++k) {
                            if (!parse_substream(p, blk_end, p_len[k], dp.sidx[k], dp.s_ptr[k], dp.s_len[k])) {
                                parse_ok = false; break;
                            }
                        }
                    } else if (dp.bh.codec_id == HX_CODEC_LZ_RANS) {
                        if ((size_t)(blk_end - p) < sizeof(HxLzBlockMeta)) { parse_ok = false; break; }
                        memcpy(&dp.lm, p, sizeof(dp.lm));
                        p += sizeof(dp.lm);
                        uint32_t lens[4] = { dp.lm.lit_size, dp.lm.num_seqs, dp.lm.num_seqs, dp.lm.num_seqs };
                        for (int s = 0; s < 4; ++s) {
                            if (!parse_substream(p, blk_end, lens[s], dp.sidx[s], dp.s_ptr[s], dp.s_len[s])) {
                                parse_ok = false; break;
                            }
                        }
                        if (!parse_ok || (size_t)(blk_end - p) < dp.lm.extra_bytes) { parse_ok = false; break; }
                        dp.extra_ptr   = p;
                        dp.extra_bytes = dp.lm.extra_bytes;
                    }
                }

                __block bool freq_ok = parse_ok;
                if (parse_ok && n_freq > 0u) {
                    const FreqTask* ft_arr = freq_tasks;
                    dispatch_apply(n_freq, par_q, ^(size_t i) {
                        const uint8_t* fp = ft_arr[i].p;
                        if (!rans32_read_freq_table_to_lut(fp, ft_arr[i].end,
                                                           lut_ptr + (size_t)ft_arr[i].sidx * HX_PROB_SCALE)) {
                            freq_ok = false;
                        }
                    });
                }
                if (freq_ok && n_tcopy > 0u) {
                    const TileCopyTask* tc_arr = tile_copies;
                    dispatch_apply(n_tcopy, par_q, ^(size_t i) {
                        const TileCopyTask& tc = tc_arr[i];
                        memcpy(fs_ptr + (size_t)tc.tgid * HX_LANES, tc.p, HX_LANES * sizeof(uint32_t));
                        memcpy(w_ptr + (size_t)tc.tgid * HX_TILE_MAX_WORDS, tc.p + HX_LANES * sizeof(uint32_t),
                               (size_t)tc.wc * sizeof(uint16_t));
                    });
                }
                st.pack_s += CACurrentMediaTime() - tp0;

                if (!freq_ok) {
                    dec_failed = true;
                    dispatch_semaphore_signal(dec_slot_sem);
                    break;
                }

                ds.bc               = bc;
                ds.n_streams        = n_streams;
                ds.n_tiles          = n_tiles;
                ds.total_rans_bytes = total_rans_bytes;
                ds.batch_off        = batch_off;
                ds.batch_bytes      = batch_bytes;
                ds.used_gpu         = false;

                // 非同期 GPU rANS 復号投入 (待たずに次スロットの Parse + GPU 投入へ!)
                if (total_rans_bytes >= GPU_RANS_MIN_BYTES) {
                    ds.used_gpu = true;
                    dispatch_semaphore_t gsem = ds.gpu_done_sem;
                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [cq commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pso_rans_dec];
                        [enc setBuffer:ds.buf_words   offset:0 atIndex:0];
                        [enc setBuffer:ds.buf_fstates offset:0 atIndex:1];
                        [enc setBuffer:ds.buf_lut     offset:0 atIndex:2];
                        [enc setBuffer:ds.buf_sjobs   offset:0 atIndex:3];
                        [enc setBuffer:ds.buf_tmeta   offset:0 atIndex:4];
                        [enc setBuffer:ds.buf_streams offset:0 atIndex:5];
                        [enc setBuffer:ds.buf_streams offset:0 atIndex:6];
                        [enc setBuffer:ds.buf_err     offset:0 atIndex:7];
                        [enc dispatchThreadgroups:MTLSizeMake(n_streams, 1, 1)
                             threadsPerThreadgroup:MTLSizeMake(HX_TG_SIZE, 1, 1)];
                        [enc endEncoding];
                        [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                            if (completed.error) dec_failed = true;
                            dispatch_semaphore_signal(gsem);
                        }];
                        [cb commit];
                    }
                }

                // dec_work_q: GPU 完了待ち → CPU 全コア並列 LZ + NEON 逆変換 + CRC32C → fwrite
                DecSlot* ds_ptr  = &ds;
                uint32_t out_slt = mb_idx % OUT_WR_SLOTS;
                dispatch_group_enter(dec_grp);
                dispatch_async(dec_work_q, ^{
                    if (ds_ptr->used_gpu) {
                        dispatch_semaphore_wait(ds_ptr->gpu_done_sem, DISPATCH_TIME_FOREVER);
                    }
                    if (!dec_failed && !write_failed) {
                        if (ds_ptr->used_gpu) {
                            const uint32_t* errs = (const uint32_t*)ds_ptr->buf_err.contents;
                            for (uint32_t t = 0u; t < ds_ptr->n_tiles; ++t) {
                                if (errs[t] != 0u) { dec_failed = true; break; }
                            }
                        } else if (ds_ptr->total_rans_bytes > 0u) {
                            const HxRansStreamJob* sj_a = (const HxRansStreamJob*)ds_ptr->buf_sjobs.contents;
                            const HxTileMeta*      tm_a = (const HxTileMeta*)ds_ptr->buf_tmeta.contents;
                            const uint16_t*        w_a  = (const uint16_t*)ds_ptr->buf_words.contents;
                            const uint32_t*        fs_a = (const uint32_t*)ds_ptr->buf_fstates.contents;
                            const uint32_t*        lu_a = (const uint32_t*)ds_ptr->buf_lut.contents;
                            uint8_t*               ds_b = (uint8_t*)ds_ptr->buf_streams.contents;
                            __block bool neon_ok = true;
                            dispatch_apply(ds_ptr->n_streams, par_q, ^(size_t si) {
                                const HxRansStreamJob& sj = sj_a[si];
                                if (sj.mode != HX_SUB_RANS) return;
                                const uint32_t* lut = lu_a + si * HX_PROB_SCALE;
                                for (uint32_t t = 0u; t < sj.num_tiles; ++t) {
                                    uint32_t tgid = sj.first_tile_idx + t;
                                    uint8_t* dst = ds_b + sj.data_offset + t * HX_TILE_SIZE;
                                    const HxTileMeta& tm = tm_a[tgid];
                                    if (tm.mode == HX_SUB_RLE) memset(dst, (int)tm.rle_byte, tm.raw_size);
                                    else if (tm.mode == HX_SUB_RANS) {
                                        if (!rans32_decode_tile_neon(w_a + (size_t)tgid * HX_TILE_MAX_WORDS,
                                                                     tm.word_count, fs_a + (size_t)tgid * HX_LANES,
                                                                     lut, dst, tm.raw_size))
                                            neon_ok = false;
                                    }
                                }
                            });
                            if (!neon_ok) dec_failed = true;
                        }

                        if (!dec_failed) {
                            if (out_file) dispatch_semaphore_wait(out_sem, DISPATCH_TIME_FOREVER);
                            uint8_t* dst_base = out_file ? file_out_ptr[out_slt].data()
                                                         : (out_vec ? (out_vec->data() + ds_ptr->batch_off)
                                                                    : file_out_ptr[0].data());
                            CFTimeInterval tlz0 = CACurrentMediaTime();
                            if (!decode_blocks_cpu(ds_ptr->dplans, ds_ptr->bc, dst_base)) {
                                dec_failed = true;
                            }
                            st_ptr->lz_s += CACurrentMediaTime() - tlz0;

                            // 各ブロックで検証済みの bh.crc32c を GF(2) 結合 (再走査ゼロ・0.002 ms!)
                            if (!dec_failed && inout_raw_crc) {
                                uint32_t c = *inout_raw_crc;
                                for (uint32_t ib = 0u; ib < ds_ptr->bc; ++ib) {
                                    c = hx_crc32c_combine(c, ds_ptr->dplans[ib].bh.crc32c, ds_ptr->dplans[ib].bh.raw_size);
                                }
                                *inout_raw_crc = c;
                            }

                            if (out_file) {
                                if (!dec_failed) {
                                    const uint8_t* wbuf = dst_base;
                                    size_t wlen = ds_ptr->batch_bytes;
                                    dispatch_group_enter(wrt_grp);
                                    dispatch_async(write_q, ^{
                                        if (!write_failed && fwrite(wbuf, 1, wlen, out_file) != wlen) write_failed = true;
                                        dispatch_semaphore_signal(out_sem);
                                        dispatch_group_leave(wrt_grp);
                                    });
                                } else {
                                    dispatch_semaphore_signal(out_sem);
                                }
                            }
                        }
                    }
                    dispatch_semaphore_signal(dec_slot_sem);
                    dispatch_group_leave(dec_grp);
                });
            }

            CFTimeInterval td0 = CACurrentMediaTime();
            dispatch_group_wait(dec_grp, DISPATCH_TIME_FOREVER);
            st.rans_s += CACurrentMediaTime() - td0;

            if (out_file) dispatch_group_wait(wrt_grp, DISPATCH_TIME_FOREVER);
            if (dec_failed || write_failed) return false;

            st.total_s = CACurrentMediaTime() - t_all0;
            if (stats) *stats = st;
            return true;
        }

        // ── ゼロコピー CPU NEON 復号パス (Metal 未起動単発 CLI 用) ───────────────
        DecBlockPlan dplans[MB_BLOCKS];
        struct CpuTileRef {
            uint8_t        mode;
            uint8_t        rle_byte;
            uint32_t       raw_size;
            uint32_t       word_count;
            const uint8_t* ptr;
        };
        HxRansStreamJob cpu_sjobs_arr[MAX_STREAMS_MB];
        CpuTileRef      cpu_trefs_arr[MAX_TILES_MB];
        uint32_t mb_idx = 0u;

        for (uint32_t b0 = 0u; b0 < hdr.num_blocks; b0 += MB_BLOCKS, ++mb_idx) {
            if (write_failed) break;
            const uint32_t bc = std::min(MB_BLOCKS, hdr.num_blocks - b0);
            const size_t   batch_off   = (size_t)b0 * HX_BLOCK_SIZE;
            const size_t   batch_bytes = std::min((size_t)bc * HX_BLOCK_SIZE, (size_t)hdr.original_size - batch_off);

            uint32_t out_slot = mb_idx % OUT_WR_SLOTS;
            if (out_file) dispatch_semaphore_wait(out_sem, DISPATCH_TIME_FOREVER);
            uint8_t* batch_dst_base = out_file ? file_out_slots[out_slot].data()
                                               : (out_vec ? (out_vec->data() + batch_off)
                                                          : file_out_slots[0].data());

            CFTimeInterval tp0 = CACurrentMediaTime();
            HxRansStreamJob* sjobs   = cpu_sjobs_arr;
            CpuTileRef*      trefs   = cpu_trefs_arr;
            uint32_t*        lut_ptr = nullptr;
            uint8_t*         dec_s   = nullptr;

            auto ensure_cpu_dec_stream_buf = [&]() {
                if (!dec_s) {
                    cpu_dec_stream_buf.ensure(STREAM_BUF_CAP);
                    dec_s = cpu_dec_stream_buf.data();
                }
            };
            auto ensure_cpu_dec_lut_buf = [&]() {
                if (!lut_ptr) {
                    cpu_dec_lut_buf.ensure((size_t)MAX_STREAMS_MB * HX_PROB_SCALE * sizeof(uint32_t));
                    lut_ptr = (uint32_t*)cpu_dec_lut_buf.data();
                }
            };

            uint32_t n_streams        = 0u;
            uint32_t n_tiles          = 0u;
            uint32_t cur_off          = 0u;
            uint32_t total_rans_bytes = 0u;

            struct FreqTask { uint32_t sidx; const uint8_t* p; const uint8_t* end; };
            FreqTask freq_tasks[MAX_STREAMS_MB];
            const uint8_t* freq_p_by_s[MAX_STREAMS_MB];
            const uint8_t* freq_end_by_s[MAX_STREAMS_MB];
            uint32_t n_freq = 0u;

            auto parse_substream = [&](const uint8_t*& p, const uint8_t* end,
                                       uint32_t expected_len, int32_t& out_sidx,
                                       const uint8_t*& out_ptr, uint32_t& out_len) -> bool {
                out_sidx = -1; out_ptr = nullptr; out_len = expected_len;
                if ((size_t)(end - p) < 4u) return false;
                uint8_t smode = p[0], rleb = p[1], ntiles = p[2];
                p += 4;
                if (expected_len == 0u) return true;
                if (smode == HX_SUB_RAW) {
                    if ((size_t)(end - p) < expected_len) return false;
                    out_ptr = p;
                    p += expected_len;
                    return true;
                }
                ensure_cpu_dec_stream_buf();
                uint32_t idx = n_streams++;
                freq_p_by_s[idx]   = nullptr;
                freq_end_by_s[idx] = nullptr;
                uint32_t exp_nt = (expected_len + HX_TILE_SIZE - 1u) / HX_TILE_SIZE;
                out_sidx = (int32_t)idx;
                out_ptr  = dec_s + cur_off;

                if (smode == HX_SUB_RLE) {
                    memset(dec_s + cur_off, rleb, expected_len);
                    sjobs[idx] = { cur_off, expected_len, n_tiles, 0u, HX_SUB_RLE, rleb, 0u, 0u };
                } else if (smode == HX_SUB_RANS) {
                    if (ntiles != exp_nt || ntiles == 0u || ntiles > HX_MAX_TILES_STR) return false;
                    if ((size_t)(end - p) < 2u) return false;
                    uint8_t fkind = p[0];
                    if (fkind != HX_FREQ_SPARSE && fkind != HX_FREQ_DENSE12) return false;
                    uint32_t n_nz = (uint32_t)p[1] + 1u;
                    size_t fbytes = 2u + ((fkind == HX_FREQ_SPARSE) ? ((size_t)n_nz * 3u) : 384u);
                    if ((size_t)(end - p) < fbytes) return false;
                    freq_tasks[n_freq++] = { idx, p, p + fbytes };
                    freq_p_by_s[idx]     = p;
                    freq_end_by_s[idx]   = p + fbytes;
                    p += fbytes;

                    if ((size_t)(end - p) < (size_t)ntiles * sizeof(Rans32TileDiskDesc)) return false;
                    Rans32TileDiskDesc tds[HX_MAX_TILES_STR];
                    memcpy(tds, p, (size_t)ntiles * sizeof(Rans32TileDiskDesc));
                    p += (size_t)ntiles * sizeof(Rans32TileDiskDesc);

                    uint32_t ft = n_tiles;
                    for (uint32_t t = 0u; t < ntiles; ++t) {
                        uint32_t tgid = ft + t;
                        uint32_t tlen = std::min(HX_TILE_SIZE, expected_len - t * HX_TILE_SIZE);
                        if (tds[t].tile_mode == HX_SUB_RAW) {
                            if (tds[t].payload_words != tlen || (size_t)(end - p) < tlen) return false;
                            memcpy(dec_s + cur_off + t * HX_TILE_SIZE, p, tlen);
                            p += tlen;
                            trefs[tgid] = { HX_SUB_RAW, 0u, tlen, tlen, nullptr };
                        } else if (tds[t].tile_mode == HX_SUB_RLE) {
                            trefs[tgid] = { HX_SUB_RLE, tds[t].rle_byte, tlen, 0u, nullptr };
                        } else if (tds[t].tile_mode == HX_SUB_RANS) {
                            uint32_t wc = tds[t].payload_words;
                            if (wc > HX_TILE_MAX_WORDS) return false;
                            size_t need = HX_LANES * sizeof(uint32_t) + (size_t)wc * sizeof(uint16_t);
                            if ((size_t)(end - p) < need) return false;
                            trefs[tgid] = { HX_SUB_RANS, 0u, tlen, wc, p };
                            p += need;
                            total_rans_bytes += tlen;
                        } else {
                            return false;
                        }
                    }
                    sjobs[idx] = { cur_off, expected_len, ft, ntiles, HX_SUB_RANS, 0u, 0u, 0u };
                    n_tiles += ntiles;
                } else {
                    return false;
                }
                cur_off += (expected_len + 15u) & ~15u;
                return true;
            };

            auto fail_cleanup = [&]() -> bool {
                if (out_file) {
                    dispatch_semaphore_signal(out_sem);
                    dispatch_group_wait(wrt_grp, DISPATCH_TIME_FOREVER);
                }
                return false;
            };

            for (uint32_t ib = 0u; ib < bc; ++ib) {
                uint32_t gidx = b0 + ib;
                if (block_offsets[gidx] >= in_size) return fail_cleanup();
                const uint8_t* p   = in_archive + block_offsets[gidx];
                const uint8_t* end = in_archive + in_size;
                if ((size_t)(end - p) < sizeof(HxBlockHeader)) return fail_cleanup();

                DecBlockPlan& dp = dplans[ib];
                memcpy(&dp.bh, p, sizeof(dp.bh));
                p += sizeof(dp.bh);
                if ((size_t)(end - p) < dp.bh.comp_payload) return fail_cleanup();
                const uint8_t* blk_end = p + dp.bh.comp_payload;
                dp.payload_ptr = p;
                dp.extra_ptr   = nullptr;
                dp.extra_bytes = 0u;
                dp.sidx[0] = dp.sidx[1] = dp.sidx[2] = dp.sidx[3] = -1;
                dp.s_ptr[0] = dp.s_ptr[1] = dp.s_ptr[2] = dp.s_ptr[3] = nullptr;

                uint32_t exp_raw = (uint32_t)std::min((uint64_t)HX_BLOCK_SIZE, hdr.original_size - (batch_off + (size_t)ib * HX_BLOCK_SIZE));
                if (dp.bh.raw_size != exp_raw) return fail_cleanup();

                if (dp.bh.codec_id == HX_CODEC_RANS_ONLY) {
                    uint32_t np = hx_rans_only_plane_count(dp.bh.xform_id, dp.bh.raw_size);
                    uint32_t p_off[4], p_len[4];
                    hx_rans_only_plane_layout(dp.bh.xform_id, dp.bh.raw_size, p_off, p_len);
                    for (uint32_t k = 0u; k < np; ++k) {
                        if (!parse_substream(p, blk_end, p_len[k], dp.sidx[k], dp.s_ptr[k], dp.s_len[k]))
                            return fail_cleanup();
                    }
                } else if (dp.bh.codec_id == HX_CODEC_LZ_RANS) {
                    if ((size_t)(blk_end - p) < sizeof(HxLzBlockMeta)) return fail_cleanup();
                    memcpy(&dp.lm, p, sizeof(dp.lm));
                    p += sizeof(dp.lm);
                    uint32_t lens[4] = { dp.lm.lit_size, dp.lm.num_seqs, dp.lm.num_seqs, dp.lm.num_seqs };
                    for (int s = 0; s < 4; ++s) {
                        if (!parse_substream(p, blk_end, lens[s], dp.sidx[s], dp.s_ptr[s], dp.s_len[s]))
                            return fail_cleanup();
                    }
                    if ((size_t)(blk_end - p) < dp.lm.extra_bytes) return fail_cleanup();
                    dp.extra_ptr   = p;
                    dp.extra_bytes = dp.lm.extra_bytes;
                }
            }

            const bool use_tile_parallel = (total_rans_bytes > 0u && n_tiles > n_streams && n_streams < 64u);
            if (use_tile_parallel) {
                // タイル並列パス (FP16 32ストリーム×2タイル等): 共有 LUT 構築 → タイル粒度並列復号
                __block bool freq_ok = true;
                if (n_freq > 0u) {
                    ensure_cpu_dec_lut_buf();
                    const FreqTask* ft_arr = freq_tasks;
                    uint32_t* lptr = lut_ptr;
                    dispatch_apply(n_freq, par_q, ^(size_t i) {
                        const uint8_t* fp = ft_arr[i].p;
                        if (!rans32_read_freq_table_to_lut(fp, ft_arr[i].end,
                                                           lptr + (size_t)ft_arr[i].sidx * HX_PROB_SCALE)) {
                            freq_ok = false;
                        }
                    });
                }
                st.pack_s += CACurrentMediaTime() - tp0;
                if (!freq_ok) return fail_cleanup();

                CFTimeInterval tr0 = CACurrentMediaTime();
                __block bool neon_ok = true;
                struct DecTileTask { uint32_t si; uint32_t local_t; };
                DecTileTask dtasks[MAX_TILES_MB];
                uint32_t nt_idx = 0u;
                for (uint32_t si = 0u; si < n_streams; ++si) {
                    for (uint32_t t = 0u; t < sjobs[si].num_tiles; ++t) {
                        dtasks[nt_idx++] = { si, t };
                    }
                }
                const DecTileTask* dt_ptr = dtasks;
                uint32_t* lptr = lut_ptr;
                dispatch_apply(nt_idx, par_q, ^(size_t k) {
                    uint32_t si = dt_ptr[k].si;
                    uint32_t t  = dt_ptr[k].local_t;
                    const HxRansStreamJob& sj = sjobs[si];
                    if (sj.mode != HX_SUB_RANS) return;
                    uint32_t tgid = sj.first_tile_idx + t;
                    uint8_t* dst = dec_s + sj.data_offset + t * HX_TILE_SIZE;
                    const CpuTileRef& tr = trefs[tgid];
                    if (tr.mode == HX_SUB_RLE) {
                        memset(dst, (int)tr.rle_byte, tr.raw_size);
                    } else if (tr.mode == HX_SUB_RANS) {
                        uint32_t fstates[HX_LANES];
                        memcpy(fstates, tr.ptr, sizeof(fstates));
                        if (!rans32_decode_tile_neon_bytes(tr.ptr + sizeof(fstates), tr.word_count,
                                                           fstates, lptr + (size_t)si * HX_PROB_SCALE,
                                                           dst, tr.raw_size)) {
                            neon_ok = false;
                        }
                    }
                });
                st.rans_s += CACurrentMediaTime() - tr0;
                if (!neon_ok) return fail_cleanup();
            } else {
                // ストリーム融合パス (LZ 全データ等): L1D ローカル 16 KB LUT 構築 → 即デコード!
                // (2 MiB LUT バッファ確保・L2 退避再読出・dispatch_apply バリア 1 回を完全排除!)
                st.pack_s += CACurrentMediaTime() - tp0;
                CFTimeInterval tr0 = CACurrentMediaTime();
                if (n_freq > 0u || total_rans_bytes > 0u) {
                    __block bool dec_stream_ok = true;
                    const uint8_t* const* fp_arr   = freq_p_by_s;
                    const uint8_t* const* fend_arr = freq_end_by_s;
                    dispatch_apply(n_streams, par_q, ^(size_t si) {
                        const HxRansStreamJob& sj = sjobs[si];
                        if (sj.mode != HX_SUB_RANS) return;
                        uint32_t local_lut[HX_PROB_SCALE];
                        const uint8_t* fp = fp_arr[si];
                        if (!fp || !rans32_read_freq_table_to_lut(fp, fend_arr[si], local_lut)) {
                            dec_stream_ok = false;
                            return;
                        }
                        for (uint32_t t = 0u; t < sj.num_tiles; ++t) {
                            uint32_t tgid = sj.first_tile_idx + t;
                            uint8_t* dst = dec_s + sj.data_offset + t * HX_TILE_SIZE;
                            const CpuTileRef& tr = trefs[tgid];
                            if (tr.mode == HX_SUB_RLE) {
                                memset(dst, (int)tr.rle_byte, tr.raw_size);
                            } else if (tr.mode == HX_SUB_RANS) {
                                uint32_t fstates[HX_LANES];
                                memcpy(fstates, tr.ptr, sizeof(fstates));
                                if (!rans32_decode_tile_neon_bytes(tr.ptr + sizeof(fstates), tr.word_count,
                                                                   fstates, local_lut, dst, tr.raw_size)) {
                                    dec_stream_ok = false;
                                    return;
                                }
                            }
                        }
                    });
                    if (!dec_stream_ok) return fail_cleanup();
                }
                st.rans_s += CACurrentMediaTime() - tr0;
            }

            CFTimeInterval tlz0 = CACurrentMediaTime();
            bool dec_ok = decode_blocks_cpu(dplans, bc, batch_dst_base);
            st.lz_s += CACurrentMediaTime() - tlz0;
            if (!dec_ok) return fail_cleanup();

            // 各ブロック検証済みの bh.crc32c を GF(2) 結合 (write_q 上での単スレッド再走査を完全排除!)
            if (inout_raw_crc) {
                uint32_t c = *inout_raw_crc;
                for (uint32_t ib = 0u; ib < bc; ++ib) {
                    c = hx_crc32c_combine(c, dplans[ib].bh.crc32c, dplans[ib].bh.raw_size);
                }
                *inout_raw_crc = c;
            }

            if (out_file) {
                const uint8_t* wbuf = batch_dst_base;
                size_t wlen = batch_bytes;
                dispatch_group_enter(wrt_grp);
                dispatch_async(write_q, ^{
                    if (!write_failed && fwrite(wbuf, 1, wlen, out_file) != wlen) write_failed = true;
                    dispatch_semaphore_signal(out_sem);
                    dispatch_group_leave(wrt_grp);
                });
            }
        }

        if (out_file) {
            dispatch_group_wait(wrt_grp, DISPATCH_TIME_FOREVER);
            if (write_failed) return false;
        }
        st.total_s = CACurrentMediaTime() - t_all0;
        if (stats) *stats = st;
        return true;
    }

    bool decompress(const uint8_t* in_archive, size_t in_size,
                    std::vector<uint8_t>& out_data, HxTimingStats* stats = nullptr) {
        return decompress_core(in_archive, in_size, &out_data, nullptr, stats);
    }

    // O(1) ブロック単位ランダムアクセス部分解凍 (Phase 3 部分展開用)
    // [range_off, range_off + range_len) に重なる 256 KiB ブロックだけを選択復号・CRC32C 検証!
    bool decompress_range(
        const uint8_t* in_archive,
        size_t         in_size,
        uint64_t       range_off,
        uint64_t       range_len,
        uint8_t*       out_buf,
        uint32_t*      out_blocks_decoded = nullptr
    ) {
        if (out_blocks_decoded) *out_blocks_decoded = 0u;
        if (in_size < sizeof(HxArchiveHeader)) return false;
        HxArchiveHeader hdr;
        memcpy(&hdr, in_archive, sizeof(hdr));
        if (memcmp(hdr.magic, "ASZ2", 4) != 0 || hdr.version != 1u || hdr.block_size != HX_BLOCK_SIZE)
            return false;
        if (range_off > hdr.original_size || range_len > hdr.original_size - range_off)
            return false;
        if (range_len == 0u) return true;

        const size_t off_bytes = (size_t)hdr.num_blocks * sizeof(uint64_t);
        if (in_size < sizeof(hdr) + off_bytes) return false;
        const uint8_t* off_table_ptr = in_archive + sizeof(hdr);

        const uint32_t first_blk = (uint32_t)(range_off / HX_BLOCK_SIZE);
        const uint32_t last_blk  = (uint32_t)((range_off + range_len - 1u) / HX_BLOCK_SIZE);
        if (last_blk >= hdr.num_blocks) return false;
        const uint32_t n_sel = last_blk - first_blk + 1u;
        if (out_blocks_decoded) *out_blocks_decoded = n_sel;

        // 1 ブロック独立復号ヘルパ (CPU NEON ゼロコピー、Metal 未起動で <0.2ms!)
        auto decode_single_block = [&](uint32_t gidx, uint8_t* blk_out, uint32_t& out_raw_size) -> bool {
            uint64_t boff;
            memcpy(&boff, off_table_ptr + (size_t)gidx * sizeof(uint64_t), sizeof(uint64_t));
            if (boff >= in_size) return false;
            const uint8_t* p   = in_archive + boff;
            const uint8_t* end = in_archive + in_size;
            if ((size_t)(end - p) < sizeof(HxBlockHeader)) return false;

            HxBlockHeader bh;
            memcpy(&bh, p, sizeof(bh));
            p += sizeof(bh);
            if ((size_t)(end - p) < bh.comp_payload) return false;
            const uint8_t* blk_end = p + bh.comp_payload;

            uint64_t exp_raw = std::min((uint64_t)HX_BLOCK_SIZE, hdr.original_size - (uint64_t)gidx * HX_BLOCK_SIZE);
            if (bh.raw_size != (uint32_t)exp_raw) return false;
            out_raw_size = bh.raw_size;

            if (bh.codec_id == HX_CODEC_RAW) {
                if (bh.comp_payload < bh.raw_size) return false;
                memcpy(blk_out, p, bh.raw_size);
                return hx_crc32c(blk_out, bh.raw_size) == bh.crc32c;
            }
            if (bh.codec_id == HX_CODEC_RLE) {
                memset(blk_out, bh.rle_byte, bh.raw_size);
                return hx_crc32c(blk_out, bh.raw_size) == bh.crc32c;
            }

            std::vector<uint8_t>  stream_storage;
            std::vector<uint32_t> lut(HX_PROB_SCALE);
            auto decode_sub = [&](const uint8_t*& sp, const uint8_t* send,
                                  uint32_t exp_len, const uint8_t*& out_ptr, size_t& out_storage_off) -> bool {
                out_ptr = nullptr;
                out_storage_off = (size_t)-1;
                if ((size_t)(send - sp) < 4u) return false;
                uint8_t smode = sp[0], rleb = sp[1], ntiles = sp[2];
                sp += 4;
                if (exp_len == 0u) return true;
                if (smode == HX_SUB_RAW) {
                    if ((size_t)(send - sp) < exp_len) return false;
                    out_ptr = sp;
                    sp += exp_len;
                    return true;
                }
                size_t base = stream_storage.size();
                stream_storage.resize(base + exp_len);
                out_storage_off = base;
                uint8_t* dst = stream_storage.data() + base;

                if (smode == HX_SUB_RLE) {
                    memset(dst, rleb, exp_len);
                    return true;
                }
                if (smode != HX_SUB_RANS) return false;
                uint32_t exp_nt = (exp_len + HX_TILE_SIZE - 1u) / HX_TILE_SIZE;
                if (ntiles != exp_nt || ntiles == 0u || ntiles > HX_MAX_TILES_STR) return false;
                if (!rans32_read_freq_table_to_lut(sp, send, lut.data())) return false;
                if ((size_t)(send - sp) < (size_t)ntiles * sizeof(Rans32TileDiskDesc)) return false;
                Rans32TileDiskDesc tds[HX_MAX_TILES_STR];
                memcpy(tds, sp, (size_t)ntiles * sizeof(Rans32TileDiskDesc));
                sp += (size_t)ntiles * sizeof(Rans32TileDiskDesc);

                for (uint32_t t = 0u; t < ntiles; ++t) {
                    uint32_t tlen = std::min(HX_TILE_SIZE, exp_len - t * HX_TILE_SIZE);
                    uint8_t* tdst = dst + t * HX_TILE_SIZE;
                    if (tds[t].tile_mode == HX_SUB_RAW) {
                        if (tds[t].payload_words != tlen || (size_t)(send - sp) < tlen) return false;
                        memcpy(tdst, sp, tlen);
                        sp += tlen;
                    } else if (tds[t].tile_mode == HX_SUB_RLE) {
                        memset(tdst, tds[t].rle_byte, tlen);
                    } else if (tds[t].tile_mode == HX_SUB_RANS) {
                        uint32_t wc = tds[t].payload_words;
                        if (wc > HX_TILE_MAX_WORDS) return false;
                        size_t need = HX_LANES * sizeof(uint32_t) + (size_t)wc * sizeof(uint16_t);
                        if ((size_t)(send - sp) < need) return false;
                        uint32_t fstates[HX_LANES];
                        memcpy(fstates, sp, sizeof(fstates));
                        if (!rans32_decode_tile_neon_bytes(sp + sizeof(fstates), wc, fstates, lut.data(), tdst, tlen))
                            return false;
                        sp += need;
                    } else {
                        return false;
                    }
                }
                return true;
            };

            if (bh.codec_id == HX_CODEC_RANS_ONLY) {
                uint32_t np = hx_rans_only_plane_count(bh.xform_id, bh.raw_size);
                uint32_t p_off[4], p_len[4];
                hx_rans_only_plane_layout(bh.xform_id, bh.raw_size, p_off, p_len);
                const uint8_t* raw_ptrs[4] = {};
                size_t         st_offs[4]  = { (size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1 };
                for (uint32_t k = 0u; k < np; ++k) {
                    if (!decode_sub(p, blk_end, p_len[k], raw_ptrs[k], st_offs[k])) return false;
                }
                const uint8_t* ptrs[4];
                for (uint32_t k = 0u; k < np; ++k) {
                    ptrs[k] = (st_offs[k] == (size_t)-1) ? raw_ptrs[k] : (stream_storage.data() + st_offs[k]);
                }
                if (bh.xform_id == HX_XFORM_SHUFFLE2 && bh.raw_size >= 2u) {
                    hx_neon_unshuffle2_planes(ptrs[0], ptrs[1], blk_out, bh.raw_size);
                } else if (bh.xform_id == HX_XFORM_SHUFFLE4 && bh.raw_size >= 4u) {
                    hx_neon_unshuffle4_planes(ptrs[0], ptrs[1], ptrs[2], ptrs[3], blk_out, bh.raw_size);
                } else if (bh.xform_id != HX_XFORM_NONE) {
                    hx_neon_transform_inverse(ptrs[0], blk_out, bh.raw_size, bh.xform_id);
                } else {
                    memcpy(blk_out, ptrs[0], bh.raw_size);
                }
                return hx_crc32c(blk_out, bh.raw_size) == bh.crc32c;
            }

            if (bh.codec_id == HX_CODEC_LZ_RANS) {
                if ((size_t)(blk_end - p) < sizeof(HxLzBlockMeta)) return false;
                HxLzBlockMeta lm;
                memcpy(&lm, p, sizeof(lm));
                p += sizeof(lm);
                uint32_t lens[4] = { lm.lit_size, lm.num_seqs, lm.num_seqs, lm.num_seqs };
                const uint8_t* raw_ptrs[4] = {};
                size_t         st_offs[4]  = { (size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1 };
                for (int s = 0; s < 4; ++s) {
                    if (!decode_sub(p, blk_end, lens[s], raw_ptrs[s], st_offs[s])) return false;
                }
                if ((size_t)(blk_end - p) < lm.extra_bytes) return false;
                const uint8_t* ptrs[4];
                for (int s = 0; s < 4; ++s) {
                    ptrs[s] = (st_offs[s] == (size_t)-1) ? raw_ptrs[s] : (stream_storage.data() + st_offs[s]);
                }
                const uint8_t* ex_ptr = p;
                std::vector<uint8_t> ex_pad;
                if ((size_t)((in_archive + in_size) - ex_ptr) < (size_t)lm.extra_bytes + 8u) {
                    ex_pad.resize((size_t)lm.extra_bytes + 8u, 0u);
                    if (lm.extra_bytes > 0u) memcpy(ex_pad.data(), ex_ptr, lm.extra_bytes);
                    ex_ptr = ex_pad.data();
                }
                uint8_t* lz_dst = (bh.xform_id != HX_XFORM_NONE)
                                      ? hx_get_thread_lz_state().acquire_xtmp()
                                      : blk_out;
                if (!hx_lz_decode_block(lm, ptrs[0], ptrs[1], ptrs[2], ptrs[3],
                                        ex_ptr, lm.extra_bytes, lz_dst, bh.raw_size))
                    return false;
                if (bh.xform_id != HX_XFORM_NONE) {
                    hx_neon_transform_inverse(lz_dst, blk_out, bh.raw_size, bh.xform_id);
                }
                return hx_crc32c(blk_out, bh.raw_size) == bh.crc32c;
            }
            return false;
        };

        std::vector<uint8_t> blk_tmp(HX_BLOCK_SIZE);
        for (uint32_t b = first_blk; b <= last_blk; ++b) {
            uint32_t b_raw = 0u;
            if (!decode_single_block(b, blk_tmp.data(), b_raw)) return false;

            uint64_t b_start = (uint64_t)b * HX_BLOCK_SIZE;
            uint64_t b_end   = b_start + b_raw;
            uint64_t c_start = std::max(range_off, b_start);
            uint64_t c_end   = std::min(range_off + range_len, b_end);
            if (c_end > c_start) {
                memcpy(out_buf + (c_start - range_off), blk_tmp.data() + (c_start - b_start), (size_t)(c_end - c_start));
            }
        }
        return true;
    }

    bool decompress_file(const char* in_path, const char* out_path, HxTimingStats* stats = nullptr) {
        int fd = open(in_path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st_in;
        if (fstat(fd, &st_in) < 0) { close(fd); return false; }
        size_t sz = (size_t)st_in.st_size;
        if (sz < sizeof(HxArchiveHeader)) { close(fd); return false; }

        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); return false; }
        madvise(m, sz, MADV_SEQUENTIAL);

        FILE* fout = fopen(out_path, "wb");
        if (!fout) { munmap(m, sz); close(fd); return false; }
        setvbuf(fout, nullptr, _IOFBF, 4 * 1024 * 1024);

        bool ok = decompress_core((const uint8_t*)m, sz, nullptr, fout, stats);
        if (fclose(fout) != 0) ok = false;
        munmap(m, sz);
        close(fd);
        if (!ok) unlink(out_path);
        return ok;
    }
};
