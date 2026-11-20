# APLZ v4 (`.aplz`) — Apple Silicon UMA Heterogeneous Archiver

[![CI](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml/badge.svg)](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml)

[日本語版 README はこちら](README.ja.md)

**APLZ (`aplz`)** is a lossless compressor and native archive format engineered from the ground up for **Apple Silicon (M1–M4+)** Unified Memory Architecture (UMA).

Instead of forcing sequential LZ parsing onto a GPU or wrapping directories in opaque `.tar` streams, APLZ v4 coordinates **CPU P-cores + E-cores + Apple GPU** across a shared 3-slot asynchronous pipeline, combined with **FastCDC content-defined deduplication**, **type-aware Solid Grouping**, and **O(1) 256 KiB block-level random-access extraction**.

---

## Key Features

- **Apple Silicon 32-way Symmetry**: 32-way SIMD-group Coalesced rANS maps 1:1 to Apple GPU's 32-wide SIMD-group and ARM64's 32×128-bit NEON register file (`uint32x4_t × 8`), producing **bit-identical archives** across GPU and CPU paths.
- **Zero-Cold-Start Hybrid CPU/GPU Dispatch**:
  - Small/medium one-shot CLI files (`< 384 MiB`): runs 100% on CPU ARM64 NEON (`vqtbl1q_u8` 32-way rANS decode, 64-bit 1-multiply 4-way ILP rANS encode, L1D-fused LUT) with **0 ms Metal startup overhead**.
  - Large files (`>= 384 MiB`): warms up Metal asynchronously in the background while CPU processes initial batches, then seamlessly transitions to the **3-slot async pipeline (`CPU LZ+Xform || Apple GPU rANS || Parallel Pack + fwrite`)**.
- **OpenZL-Style Per-Block Reversible Transforms**: Automatic per-block probe selects `SHUFFLE2` (FP16/BF16 AI weights with per-plane rANS + high-entropy plane RAW bypass), `SHUFFLE4` (FP32/Int32 telemetry & tensors), `ARM64_BCJ` (Apple Silicon Mach-O executables), or `RAW`/`RLE` bypass.
- **Native `.aplz` v4 Container (No Temporary `.tar`)**:
  - **Normalized FastCDC** (Gear Hash, 16–256 KiB chunks, 160-bit content fingerprint) deduplicates identical chunks across files, snapshots, and shifted versions.
  - **Type-Aware Solid Grouping** packs small files (`< 64 KiB`) by data class (`TEXT` / `CODE` / `NUM` / `MEDIA` / `BIN`) sorted by extension for strong cross-file LZ compression while capping Solid Packs at 4 MiB.
  - **O(1) Listing & 256 KiB Block Random-Access Extraction**: List any archive in milliseconds from its compressed EOF Central Directory, or extract a single file from a multi-GB archive by decoding only the 1–2 overlapping 256 KiB blocks.
- **Triple-Layer Integrity & Security**: Hardware-accelerated ARM64 CRC32C with $\text{GF}(2)$ `crc32c_combine` verifies Header, Packs, 256 KiB Blocks, Chunks, and Files. Strict path sanitization blocks Zip-Slip and symlink-escape attacks.

---

## Installation (macOS / Apple Silicon)

```bash
# Install to ~/.local/bin/aplz (no sudo required)
./install.sh

# Or system-wide (/usr/local/bin/aplz)
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
aplz c -3 -o backup.aplz ./my_project
aplz c -1 --no-dedup -o fast.aplz ./large_file.bin

# List archive contents & dedup summary (O(1) footer read)
aplz l backup.aplz

# Verify full integrity without writing to disk
aplz t backup.aplz

# Extract entire archive
aplz x backup.aplz -C ./restored

# Extract a single file or subfolder via O(1) 256 KiB block random access
aplz x backup.aplz -C ./restored my_project/src/main.cpp

# Directly decompress a single-file archive
aplz d archive.aplz restored.bin
```

---

## Benchmarks (Apple M4, End-to-End File-to-File)

Measured on Apple M4 (10-core CPU / 10-core GPU, UMA, macOS, `aplz` v4.0.0). All runs verified byte-for-byte with `cmp` / `dircmp`.

### 1. Realistic Mixed Directory (979 files, 89.4 MiB: 3-snapshot repo + 2-gen FP16 weights + FP32 telemetry + Mach-O)

| Tool | Archive Size | Ratio | Pack | Full Extract | List | 1-File Extract |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`aplz -1`** | **12.39 MiB** | **13.9%** | **43.9 ms** | **61.7 ms** | **3.0 ms** | **3.13 ms** |
| **`aplz -3`** | **12.33 MiB** | **13.8%** | **46.5 ms** | **61.5 ms** | **2.9 ms** | **3.08 ms** |
| `tar+zstd -1 -T0` | 33.07 MiB | 37.0% | 338.9 ms | 355.5 ms | 47.0 ms | 46.03 ms |
| `tar+zstd -3 -T0` | 32.39 MiB | 36.2% | 357.4 ms | 353.3 ms | 58.2 ms | 57.36 ms |
| `tar+lz4 -1` | 47.97 MiB | 53.7% | 371.5 ms | 354.0 ms | 41.8 ms | 41.02 ms |

### 2. Large Single File — 1.00 GiB Heterogeneous (`FP32 + FP16 + Mach-O + Mixed`)

| Tool | Ratio | Archive Size | Compress | Decompress (`d`) | Extract (`x`) | Verify (`t`) / List (`l`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`aplz -1`** | **38.24%** | **391.6 MiB** | **3,297 MB/s (0.31 s)** | **3,566 MB/s (0.29 s)** | **3,109 MB/s (0.33 s)** | Verify `0.12 s` (8,403 MB/s) / List `3.9 ms` |
| **`aplz -3`** | **37.56%** | **384.7 MiB** | **3,207 MB/s (0.32 s)** | **3,438 MB/s (0.30 s)** | **3,657 MB/s (0.28 s)** | Verify `0.12 s` (8,768 MB/s) / List `3.7 ms` |
| `aplz -1 --no-dedup` | 58.12% | 595.2 MiB | 2,642 MB/s (0.39 s) | 3,120 MB/s (0.33 s) | 2,662 MB/s (0.38 s) | Verify `0.14 s` (7,421 MB/s) / List `4.7 ms` |
| `zstd -3 -T0` | 66.20% | 677.9 MiB | 2,003 MB/s (0.51 s) | 1,263 MB/s (0.81 s) | — | — |

### 3. Single Files — 256 MiB (`aplz` vs `zstd -3 -T0` vs `lz4 -1`)

| Dataset | `aplz -1` (Ratio / Comp / Dec) | `aplz -3` (Ratio / Comp / Dec) | `zstd -3 -T0` (Ratio / Comp / Dec) | `lz4 -1` (Ratio / Comp / Dec) |
| :--- | :--- | :--- | :--- | :--- |
| **Source+Docs** | 2.9% / 5,026 / 4,566 MB/s | 2.7% / 4,756 / 4,406 MB/s | 0.4% / 9,656 / 5,600 MB/s | 33.8% / 1,103 / 1,739 MB/s |
| **Mach-O ARM64** | **1.9%** / **7,309** / **4,847 MB/s** | **1.8%** / **7,041** / **5,068 MB/s** | 39.8% / 1,966 / 1,213 MB/s | 57.4% / 737 / 1,575 MB/s |
| **FP16 Weights** | **84.1%** / 2,655 / **4,258 MB/s** | **84.1%** / 2,666 / **4,197 MB/s** | 91.6% / 3,678 / 1,220 MB/s | 100.0% / 2,622 / 2,291 MB/s |
| **FP32 Telemetry** | **41.0%** / **2,487** / **4,677 MB/s** | **38.4%** / **2,000** / **4,718 MB/s** | 75.5% / 1,314 / 1,030 MB/s | 99.9% / 1,613 / 1,945 MB/s |
| **Mixed Real** | **48.5%** / 2,857 / **2,849 MB/s** | **48.3%** / 2,710 / **3,513 MB/s** | 57.9% / 3,149 / 1,964 MB/s | 72.8% / 1,318 / 1,867 MB/s |

> **Tip**: For unique large numeric tensors without cross-chunk redundancy, `--no-dedup` bypasses FastCDC and streams 256 MiB packs directly (e.g., `FP16 Weights` up to **4,338 MB/s** compress / **4,260 MB/s** decompress, `FP32 Telemetry` up to **3,390 MB/s** compress / **4,749 MB/s** decompress).

---

## Project Layout

- [SPEC.md](SPEC.md) — Full mathematical & binary format specification
- [src/rans/](src/rans/) — 32-way Coalesced rANS core (Metal GPU + ARM64 NEON + Scalar reference)
- [src/engine/](src/engine/) — 3-slot UMA heterogeneous block engine (`ASZ2`: LZ + NEON transforms + GPU/CPU rANS + `decompress_range`)
- [src/archive/](src/archive/) — `.aplz` v4 native container, FastCDC deduplication, Solid Grouping, and `aplz` CLI
- [tests/](tests/) — Automated verification suite and Phase 1 / Phase 2 benchmarks
- [build.sh](build.sh) / [install.sh](install.sh) / [setup_finder.sh](setup_finder.sh) — Build, install, and macOS Finder Services scripts
