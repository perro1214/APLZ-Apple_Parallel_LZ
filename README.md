# APLZ v4 (`.aplz`) — Apple Silicon UMA Heterogeneous Archiver

[![CI](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml/badge.svg)](https://github.com/perro1214/APLZ-Apple_Parallel_LZ/actions/workflows/ci.yml)

[日本語版 README はこちら](README.ja.md)

**APLZ (`aplz`)** is a lossless compressor and native archive format engineered from the ground up for **Apple Silicon (M1–M4+)** Unified Memory Architecture (UMA).

Instead of forcing sequential LZ parsing onto a GPU or wrapping directories in opaque `.tar` streams, APLZ v4 coordinates **CPU P-cores + E-cores + Apple GPU** across a shared 3-slot asynchronous pipeline, combined with **FastCDC content-defined deduplication**, **type-aware Solid Grouping**, and **O(1) 256 KiB block-level random-access extraction**.

---

## Key Features

- **Apple Silicon 32-way Symmetry**: 32-way SIMD-group Coalesced rANS maps 1:1 to Apple GPU's 32-wide SIMD-group and ARM64's 32×128-bit NEON register file (`uint32x4_t × 8`), producing **bit-identical archives** across GPU and CPU paths.
- **Zero-Cold-Start Hybrid CPU/GPU Dispatch**:
  - Small/medium one-shot CLI files (`< 96 MiB`): runs 100% on CPU NEON + 4-way ILP branchless 32-way rANS with **0 ms Metal startup overhead**.
  - Large files (`>= 96 MiB`): warms up Metal asynchronously in the background while CPU processes initial batches, then seamlessly transitions to the **3-slot async pipeline (`CPU LZ+Xform || Apple GPU rANS || Parallel Pack + fwrite`)**.
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
| **`aplz -1`** | **12.39 MiB** | **13.9%** | **49.1 ms** | **76.0 ms** | **2.9 ms** | **3.23 ms** |
| **`aplz -3`** | **12.32 MiB** | **13.8%** | **49.8 ms** | **76.3 ms** | **3.0 ms** | **3.19 ms** |
| `tar+zstd -1 -T0` | 33.08 MiB | 37.0% | 346.5 ms | 364.3 ms | 47.0 ms | 46.17 ms |
| `tar+zstd -3 -T0` | 32.39 MiB | 36.2% | 373.1 ms | 368.1 ms | 58.0 ms | 57.63 ms |
| `tar+lz4 -1` | 47.97 MiB | 53.7% | 380.5 ms | 366.3 ms | 42.1 ms | 41.28 ms |

### 2. Large Single File — 1.00 GiB Heterogeneous (`FP32 + FP16 + Mach-O + Mixed`)

| Tool | Ratio | Archive Size | Compress | Decompress (`d`) | Extract (`x`) | Verify (`t`) / List (`l`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`aplz -1`** | **38.25%** | **391.6 MiB** | **3,010 MB/s (0.34 s)** | **3,450 MB/s (0.30 s)** | **3,192 MB/s (0.32 s)** | Verify `0.14 s` (7,066 MB/s) / List `4.6 ms` |
| **`aplz -3`** | **37.55%** | **384.5 MiB** | **2,797 MB/s (0.37 s)** | **3,496 MB/s (0.29 s)** | **3,355 MB/s (0.31 s)** | Verify `0.15 s` (7,020 MB/s) / List `4.4 ms` |
| `aplz -1 --no-dedup` | 58.11% | 595.1 MiB | 2,417 MB/s (0.42 s) | 3,290 MB/s (0.31 s) | 2,852 MB/s (0.36 s) | Verify `0.22 s` (4,692 MB/s) / List `4.1 ms` |
| `zstd -3 -T0` | 66.20% | 677.9 MiB | 2,153 MB/s (0.48 s) | 1,204 MB/s (0.85 s) | — | — |

### 3. Single Files — 256 MiB (`aplz` vs `zstd -3 -T0` vs `lz4 -1`)

| Dataset | `aplz -1` (Ratio / Comp / Dec) | `aplz -3` (Ratio / Comp / Dec) | `zstd -3 -T0` (Ratio / Comp / Dec) | `lz4 -1` (Ratio / Comp / Dec) |
| :--- | :--- | :--- | :--- | :--- |
| **Source+Docs** | **3.0%** / 3,794 / 3,603 MB/s | **2.8%** / 3,650 / **3,826 MB/s** | 0.3% / 9,170 / 4,216 MB/s | 33.7% / 1,054 / 1,589 MB/s |
| **Mach-O ARM64** | **1.9%** / **4,869** / **4,187 MB/s** | **1.8%** / **4,991** / **3,987 MB/s** | 39.8% / 1,792 / 1,184 MB/s | 57.4% / 719 / 1,439 MB/s |
| **FP16 Weights** | **84.1%** / 1,840 / **3,542 MB/s** | **84.1%** / 2,105 / **3,596 MB/s** | 91.6% / 3,189 / 1,211 MB/s | 100.0% / 2,281 / 1,824 MB/s |
| **FP32 Telemetry** | **41.0%** / **2,215** / **3,998 MB/s** | **38.3%** / **1,693** / **3,959 MB/s** | 75.5% / 1,193 / 1,023 MB/s | 99.9% / 1,530 / 1,759 MB/s |
| **Mixed Real** | **48.5%** / 2,188 / **2,984 MB/s** | **48.4%** / 2,118 / **2,968 MB/s** | 57.9% / 2,933 / 1,867 MB/s | 72.7% / 1,082 / 1,747 MB/s |

> **Tip**: For unique large numeric tensors without cross-chunk redundancy, `--no-dedup` bypasses FastCDC and streams 256 MiB packs directly (e.g., `FP16 Weights` up to **3,211 MB/s** compress / **3,905 MB/s** decompress, `FP32 Telemetry` up to **2,710 MB/s** compress / **3,748 MB/s** decompress).

---

## Project Layout

- [SPEC.md](SPEC.md) — Full mathematical & binary format specification
- [src/rans/](src/rans/) — 32-way Coalesced rANS core (Metal GPU + ARM64 NEON + Scalar reference)
- [src/engine/](src/engine/) — 3-slot UMA heterogeneous block engine (`ASZ2`: LZ + NEON transforms + GPU/CPU rANS + `decompress_range`)
- [src/archive/](src/archive/) — `.aplz` v4 native container, FastCDC deduplication, Solid Grouping, and `aplz` CLI
- [tests/](tests/) — Automated verification suite and Phase 1 / Phase 2 benchmarks
- [build.sh](build.sh) / [install.sh](install.sh) / [setup_finder.sh](setup_finder.sh) — Build, install, and macOS Finder Services scripts
