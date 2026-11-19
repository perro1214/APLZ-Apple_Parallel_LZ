# APLZ-X (`.asz`) — Apple Silicon UMA Heterogeneous Archiver

[日本語版 README はこちら](README.ja.md)

**APLZ-X (`asz`)** is a lossless compressor and native archive format engineered from the ground up for **Apple Silicon (M1–M4+)** Unified Memory Architecture (UMA).

Instead of forcing sequential LZ parsing onto a GPU or wrapping directories in opaque `.tar` streams, APLZ-X coordinates **CPU P-cores + E-cores + Apple GPU** across a shared 3-slot asynchronous pipeline, combined with **FastCDC content-defined deduplication**, **type-aware Solid Grouping**, and **O(1) 256 KiB block-level random-access extraction**.

---

## Key Features

- **Apple Silicon 32-way Symmetry**: 32-way SIMD-group Coalesced rANS maps 1:1 to Apple GPU's 32-wide SIMD-group and ARM64's 32×128-bit NEON register file (`uint32x4_t × 8`), producing **bit-identical archives** across GPU and CPU paths.
- **Zero-Cold-Start Hybrid CPU/GPU Dispatch**:
  - Small/medium one-shot CLI files (`< 96 MiB`): runs 100% on CPU NEON + 4-way ILP branchless 32-way rANS with **0 ms Metal startup overhead**.
  - Large files (`>= 96 MiB`): warms up Metal asynchronously in the background while CPU processes initial batches, then seamlessly transitions to the **3-slot async pipeline (`CPU LZ+Xform || Apple GPU rANS || Parallel Pack + fwrite`)**.
- **OpenZL-Style Per-Block Reversible Transforms**: Automatic per-block probe selects `SHUFFLE2` (FP16/BF16 AI weights with per-plane rANS + high-entropy plane RAW bypass), `SHUFFLE4` (FP32/Int32 telemetry & tensors), `ARM64_BCJ` (Apple Silicon Mach-O executables), or `RAW`/`RLE` bypass.
- **Native `.asz` Container (No Temporary `.tar`)**:
  - **Normalized FastCDC** (Gear Hash, 16–256 KiB chunks, 160-bit content fingerprint) deduplicates identical chunks across files, snapshots, and shifted versions.
  - **Type-Aware Solid Grouping** packs small files (`< 64 KiB`) by data class (`TEXT` / `CODE` / `NUM` / `MEDIA` / `BIN`) sorted by extension for strong cross-file LZ compression while capping Solid Packs at 4 MiB.
  - **O(1) Listing & 256 KiB Block Random-Access Extraction**: List any archive in milliseconds from its compressed EOF Central Directory, or extract a single file from a multi-GB archive by decoding only the 1–2 overlapping 256 KiB blocks.
- **Triple-Layer Integrity & Security**: Hardware-accelerated ARM64 CRC32C with $\text{GF}(2)$ `crc32c_combine` verifies Header, Packs, 256 KiB Blocks, Chunks, and Files. Strict path sanitization blocks Zip-Slip and symlink-escape attacks.

---

## Installation (macOS / Apple Silicon)

```bash
# Install to ~/.local/bin/asz (no sudo required)
./install.sh

# Or system-wide (/usr/local/bin/asz)
./install.sh --system

# Register macOS Finder Quick Actions / Services ("APLZ で圧縮" / "APLZ で解凍")
./setup_finder.sh

# Uninstall
./install.sh --uninstall
./setup_finder.sh --uninstall
```

### Build & Test from Source

```bash
./build.sh test
```

---

## CLI Usage

```bash
# Create an archive from files and/or directories (-1 fast, -3 default, -6 high)
asz c -3 -o backup.asz ./my_project
asz c -1 --no-dedup -o fast.asz ./large_file.bin

# List archive contents & dedup summary (O(1) footer read)
asz l backup.asz

# Verify full integrity without writing to disk
asz t backup.asz

# Extract entire archive
asz x backup.asz -C ./restored

# Extract a single file or subfolder via O(1) 256 KiB block random access
asz x backup.asz -C ./restored my_project/src/main.cpp

# Directly decompress a single-file archive
asz d archive.asz restored.bin
```

---

## Benchmarks (Apple M4, End-to-End File-to-File)

All runs verified byte-for-byte with `cmp` / `dircmp`.

### 1. Realistic Mixed Directory (979 files, 90.6 MiB: 3-snapshot repo + FP16 weights + FP32 telemetry + Mach-O)

| Tool | Archive Size | Ratio | Pack | Full Extract | List | 1-File Extract |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`asz -1`** | **25.27 MB** | **26.6%** | **76.9 ms** | **262.4 ms** | **4.2 ms** | **4.73 ms** |
| **`asz -3`** | **24.60 MB** | **25.9%** | **81.8 ms** | **217.7 ms** | **4.1 ms** | **8.94 ms** |
| `tar+zstd -1 -T0` | 53.52 MB | 56.4% | 385.0 ms | 500.6 ms | 69.1 ms | 68.02 ms |
| `tar+zstd -3 -T0` | 50.40 MB | 53.1% | 399.6 ms | 467.2 ms | 70.2 ms | 80.58 ms |
| `tar+lz4 -1` | 69.78 MB | 73.5% | 426.3 ms | 407.5 ms | 49.7 ms | 49.46 ms |

### 2. Large Single File — 1.00 GiB Heterogeneous (`FP32 + FP16 + Mach-O + Mixed`)

| Tool | Ratio | Archive Size | Compress | Decompress | Verify / List |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **`asz -3`** | **37.58%** | **384.8 MiB** | **2,949 MB/s (0.35 s)** | **2,398 MB/s (0.43 s)** | Verify `0.39 s` / List `7.1 ms` |
| `zstd -3 -T0` | 66.19% | 677.8 MiB | 2,248 MB/s (0.46 s) | 1,183 MB/s (0.87 s) | — |

### 3. Single Files — 256 MiB (`asz` vs `zstd -3 -T0` vs `lz4 -1`)

| Dataset | `asz -1` (Ratio / Comp / Dec) | `asz -3` (Ratio / Comp / Dec) | `zstd -3 -T0` (Ratio / Comp / Dec) | `lz4 -1` (Ratio / Comp / Dec) |
| :--- | :--- | :--- | :--- | :--- |
| **Source+Docs** | **3.4%** / 3,722 / 3,689 MB/s | **3.2%** / 3,318 / **4,032 MB/s** | 0.2% / 9,140 / 5,482 MB/s | 36.6% / 1,082 / 1,654 MB/s |
| **FP16 Weights** | **84.1%** / 2,096 / **3,372 MB/s** | **84.1%** / 2,035 / **2,442 MB/s** | 91.6% / 3,268 / 1,195 MB/s | 100.0% / 2,565 / 1,915 MB/s |
| **FP32 Telemetry** | **41.0%** / **2,156** / **4,135 MB/s** | **38.3%** / **1,771** / **3,539 MB/s** | 75.5% / 1,279 / 1,008 MB/s | 99.9% / 1,551 / 1,608 MB/s |
| **Mixed Real** | **48.6%** / **2,410** / **3,238 MB/s** | **48.5%** / **2,363** / **3,190 MB/s** | 57.9% / 1,853 / 1,589 MB/s | 73.5% / 1,016 / 1,098 MB/s |

---

## Project Layout

- [SPEC.md](SPEC.md) — Full mathematical & binary format specification
- [src/rans/](src/rans/) — 32-way Coalesced rANS core (Metal GPU + ARM64 NEON + Scalar reference)
- [src/engine/](src/engine/) — 3-slot UMA heterogeneous block engine (`ASZ2`: LZ + NEON transforms + GPU/CPU rANS + `decompress_range`)
- [src/asz/](src/asz/) — `.asz` native container, FastCDC deduplication, Solid Grouping, and `asz` CLI
- [tests/](tests/) — Automated verification suite and Phase 1 / Phase 2 benchmarks
- [build.sh](build.sh) / [install.sh](install.sh) / [setup_finder.sh](setup_finder.sh) — Build, install, and macOS Finder Services scripts
