# APLZ — Apple Parallel LZ

**English** | [日本語](README.ja.md) | [中文](README.zh.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

A high-speed lossless compression/decompression engine running on Apple Silicon GPU. Combines Metal Compute Shaders (MSL 3.0) with tANS (Asymmetric Numeral Systems) in a fully GPU-accelerated pipeline, achieving **perfect round-trip** compression.

## Highlights

- **Zero-copy I/O**: Leverages Apple Silicon's Unified Memory Architecture (UMA). `mmap` + `newBufferWithBytesNoCopy` eliminates CPU↔GPU input data copies
- **Parallel LZ77**: 128 threads cooperatively process each 64 KB chunk using a 4-gram multiplicative hash (`h4`), 2-way hash table (`atomic_fetch_min`), Run-Skipping, 4-byte parallel match comparison, and parallel prefix-sum compaction
- **GPU Per-chunk 128 Interleaved tANS**: Builds optimal tANS tables per chunk directly on the GPU (`build_tans_tables`). Each thread maintains an independent ANS state, encoding/decoding 128 bitstreams in parallel
- **Fully Parallel LZ77 Decode**: Tile-based parallel prefix-scan for validation and literal writes, followed by unified modulo-indexed parallel copy across all 128 threads — including overlapping matches (`dist < len`), with no serial fallback
- **Async Double Buffering**: GCD-based batch pipeline (`dispatch_semaphore` / `dispatch_group`). Overlaps GPU execution with CPU table prep and async file I/O
- **O(1) Streaming Memory**: Fixed memory usage independent of input size. Processes in mega-batches (32 MB) for stable operation on 5 GB+ files
- **Perfect Round-trip**: Guarantees byte-level exact match after compress → decompress, with GPU-side ANS/LZ stream validation

## Architecture

### Compression Pipeline (`-c`)

```
mmap(input)  ← zero-copy MTLBuffer (Apple Silicon UMA)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Mega-batch streaming loop (512 chunks = 32 MB / batch)   ║
║  (Pass 1 double-buffered prefetch across mega-batches)    ║
║                                                           ║
║  1. GPU Pass 1: compress_chunk                            ║
║     ┌───────────────────────────────────────────────────┐ ║
║     │ 1 threadgroup (128 threads) = 1 chunk (64 KB)     │ ║
║     │   A-1  Hash table init                            │ ║
║     │   A-2  4-gram hash + Run-Skipping (atomic_min x2) │ ║
║     │   A-3  LZ77 4-byte parallel match -> sparse[]     │ ║
║     │   B    Parallel prefix-sum compaction -> compact[]│ ║
║     └───────────────────────────────────────────────────┘ ║
║                          |                                ║
║                          v                                ║
║  2. GPU Table Build: build_tans_tables                    ║
║     Per chunk: Histogram -> Normalize -> Spread -> EncTbl ║
║                          | SymInfo[512] + enc_table[1024] ║
║                          v                                ║
║  3. GPU Pass 2: tans_encode (double-buffered, 32 ch/batch)║
║     128 interleaved ANS streams per chunk                 ║
║     addCompletedHandler -> serial write queue (atomic tmp)║
╚═══════════════════════════════════════════════════════════╝
    | seek table backpatch + atomic rename
    v
  .aplz file
```

### Decompression Pipeline (`-d`)

```
mmap(.aplz) -> parse header + validate chunk_offsets (seek table)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Mega-batch streaming pipeline (double-buffered 32MB out) ║
║  for each mega-batch:                                     ║
║    ┌───────────────────────────────────────────────────┐  ║
║    │ Double-buffered batch pipeline (32 chunks/batch)  │  ║
║    │                                                   │  ║
║    │ CPU (batch N+1, parallel via dispatch_apply):     │  ║
║    │   read per-chunk compact_freq + 128 streams       │  ║
║    │   build DecodeEntry[L] per chunk                  │  ║
║    │                                                   │  ║
║    │ GPU (batch N, 128 threads/chunk):                 │  ║
║    │   tans_decode -> lz77_decode -> mb_out[slot]      │  ║
║    └───────────────────────────────────────────────────┘  ║
║    -> async fwrite(mb_out[slot]) -> reuse buffers         ║
╚═══════════════════════════════════════════════════════════╝
    v
  restored original data
```

## File Format (.aplz)

| Offset | Size | Content |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic `"APLZ"`, `version = 3`, `original_size`, `chunk_size = 65536`, `num_chunks`) |
| 24 | 4 B | `n_streams` = 128 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8×N B | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (N chunks, v3 format) |

Per-chunk data layout (v3 format):
- `token_cnt` (4 B): number of dense tokens
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` pairs (4 B each) — non-zero symbols only
- `stream_sizes[128]` (2 B each = 256 B): byte size of each ANS stream
- stream data: concatenated ANS bitstreams (128 interleaved streams; each match token stores a 16-bit backward distance)

> Empty files (`original_size = 0`, `num_chunks = 0`) are stored as a 32-byte header only (`FileHeader` + `n_streams` + `ans_log_l`).

## Installation

Requires macOS + Xcode Command Line Tools.

```bash
# Install to ~/.local/bin (optimized build with -O3 -flto + precompiled metallib)
./install.sh

# Install to /usr/local/bin (requires sudo)
./install.sh /usr/local

# Uninstall
./install.sh uninstall
```

### Finder Integration (Optional)

```bash
# Add right-click → "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# Remove
./setup_finder.sh uninstall
```

Supports Google Drive and cloud storage files. The service automatically processes files via `/tmp` to work around macOS sandbox restrictions.

## Usage

### aplz command (recommended)

```bash
# Compress a file
aplz compress myfile.txt              → myfile.txt.aplz

# Compress a directory (auto tar)
aplz compress myproject/              → myproject.tar.aplz

# Extract
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# File info
aplz info myfile.txt.aplz
```

### gpu_zip (low-level API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal [--debug]
./gpu_zip -d <input.aplz> <output> compression.metal [--debug]
```

> If a precompiled `compression.metallib` exists next to `compression.metal`, `gpu_zip` loads it automatically and falls back to runtime source compilation otherwise.

### Development Build

```bash
./build.sh        # -O2 build (generates gpu_zip + compression.metallib)
./build.sh clean   # clean artifacts
```

## Performance (Apple M4)

Measured with `-O3 -flto` release build, best of 3 runs (512 MB inputs, verified byte-exact round-trip):

### Compression

| Data | Input | Output | Ratio | Throughput |
|---|---|---|---|---|
| Text (repeating pattern) | 512 MB | 10.3 MB | 2.0% | 297 MB/s |
| Random binary | 512 MB | 535.9 MB | 104.7% | 134 MB/s |

### Decompression

| Data | Input | Speed | Notes |
|---|---|---|---|
| Text (repeating pattern) | 512 MB | 2.6 GB/s | 16 mega-batches, full pipeline overlap |
| Random binary | 512 MB | 232 MB/s | 16 mega-batches, mostly literals |

> Random data is incompressible so output slightly exceeds input (tANS/distance field overhead).
> Memory usage is constant regardless of input size thanks to mega-batch streaming.

## Source Files

| File | Role |
|---|---|
| `APLZ.h` | Shared header (structs & constants for CPU and GPU) |
| `compression.metal` | GPU kernels (MSL 3.0): `compress_chunk`, `build_tans_tables`, `tans_encode`, `tans_decode`, `lz77_decode` |
| `main.mm` | Host driver (Objective-C++): O(1) streaming mega-batch + async double-buffered pipelines |
| `aplz` | User-facing CLI wrapper (Bash): file/directory support, tar integration |
| `install.sh` | Installer: optimized build (`-O3 -flto` + `.metallib`) + system-wide deployment |
| `setup_finder.sh` | macOS Finder Quick Action generator (right-click compress/extract) |
| `build.sh` | Development build script |
| `bench_compress.sh` | Benchmark script comparing APLZ against gzip/zip/zstd/lz4/xz |

## Technical Details

- **LZ77**: 4-gram multiplicative hash (`h4`), 2-way hash table (`atomic_fetch_min`) with Run-Skipping, 4-byte parallel match comparison, parallel prefix-sum compaction (`sparse` → `compact`), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)**: Dynamically builds optimal tANS tables per 64 KB chunk on the GPU (`build_tans_tables`). 128 interleaved ANS streams, 512 symbols (256 literals + 256 match lengths), L=1024, Duda's fast spread, and threadgroup cooperative loading on GPU
- **ANS Encoding**: Forward encode with renormalization, 16-bit match distance fields, sentinel bit for bitstream end detection
- **ANS Decoding**: Reverse playback from final state, sentinel via `clz()`, word-level batch bit extraction, O(1) decode table lookup, and per-chunk stream/state bounds validation
- **Compact Frequency Serialization**: Stores only non-zero entries per chunk (`n_nonzero + [sym_id, freq]` pairs), minimizing file size overhead
- **Fully Parallel LZ77 Decode**: Tile-based parallel prefix-scan for validation and literal writes, followed by unified modulo-indexed parallel copy (`out[off - dist + (j % dist)]`) handling both overlapping and non-overlapping matches without a serial fallback
- **Async Double Buffering**: GCD-based batch pipeline (32 chunks/batch, 2 slots). `dispatch_semaphore` for slot exclusion, `dispatch_apply` for parallel CPU decode-table build, `addCompletedHandler` + serial `dispatch_queue` for ordered async writes, and atomic temp-file rename on compression
- **O(1) Streaming Memory**: Mega-batch architecture (512 chunks = 32 MB per batch). Buffers are reused across mega-batches for constant memory usage regardless of input size

## Known Limitations

- **BS_CAP = 1024 B**: 1 stream max output is 1024 bytes. Highly incompressible data with extreme expansion may truncate bits.
- **Chunk size fixed at 64 KB**: Tunable in `APLZ.h` / `compression.metal` but requires rebuild.

## License

MIT License. See [LICENSE](LICENSE) for details.
