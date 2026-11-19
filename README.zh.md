# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | **中文** | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

基于 Apple Silicon GPU 的高速无损压缩/解压引擎。结合 Metal Compute Shader (MSL 3.0) 与 tANS (Asymmetric Numeral Systems) 的全 GPU 加速流水线，实现**完美的往返压缩 (Round-trip)**。

## 亮点

- **Zero-copy I/O**: 充分利用 Apple Silicon 的统一内存架构 (UMA)。`mmap` + `newBufferWithBytesNoCopy` 消除 CPU↔GPU 输入数据拷贝
- **并行 LZ77**: 128 个线程协同处理每个 64 KB 块。采用 4-gram 乘法哈希 (`h4`)、2-way 哈希表 (`atomic_fetch_min`)、Run-Skipping、4 字节并行匹配比较与 parallel prefix-sum 压缩整理
- **GPU Per-chunk 128 交织 tANS**: 直接在 GPU (`build_tans_tables`) 上为每个块动态构建最优 tANS 表。每个线程维护独立的 ANS 状态，并行编码/解码 128 个比特流
- **全并行 LZ77 解码**: 基于分块 parallel prefix-scan 完成校验与字面量写入，并使用取模索引在 128 线程上统一并行复制（包括 `dist < len` 重叠匹配），无需串行回退
- **异步双缓冲**: 基于 GCD (`dispatch_semaphore` / `dispatch_group`) 的批处理流水线，重叠 GPU 执行、CPU 表构建与异步文件 I/O
- **O(1) 流式内存**: 固定内存用量，与输入大小无关。以 mega-batch (32 MB) 为单位处理，5 GB+ 文件稳定运行
- **完美往返**: 压缩→解压后保证字节级完全一致，并包含 GPU 侧 ANS/LZ 流校验

## 架构

### 压缩流水线 (`-c`)

```
mmap(input)  <- zero-copy MTLBuffer (Apple Silicon UMA)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Mega-batch 流式循环 (512 chunks = 32 MB / batch)         ║
║  (Pass 1 在 mega-batch 间采用双缓冲预取)                   ║
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
║  2. GPU 表构建: build_tans_tables                         ║
║     Per chunk: Histogram -> Normalize -> Spread -> EncTbl ║
║                          | SymInfo[512] + enc_table[1024] ║
║                          v                                ║
║  3. GPU Pass 2: tans_encode (双缓冲, 32 chunks/batch)      ║
║     每块 128 路交织 ANS 流                                ║
║     addCompletedHandler -> serial write queue (原子临时)   ║
╚═══════════════════════════════════════════════════════════╝
    | 回写 seek table + 原子 rename
    v
  .aplz file
```

### 解压流水线 (`-d`)

```
mmap(.aplz) -> 解析头部 + 校验 chunk_offsets (seek table)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Mega-batch 流式流水线 (双缓冲 32 MB 输出)                 ║
║  for each mega-batch:                                     ║
║    ┌───────────────────────────────────────────────────┐  ║
║    │ 双缓冲批处理流水线 (32 chunks/batch)               │  ║
║    │                                                   │  ║
║    │ CPU (batch N+1, 通过 dispatch_apply 并行):        │  ║
║    │   读取每块 compact_freq + 128 路流                │  ║
║    │   为每块构建 DecodeEntry[L]                        │  ║
║    │                                                   │  ║
║    │ GPU (batch N, 128 threads/chunk):                 │  ║
║    │   tans_decode -> lz77_decode -> mb_out[slot]      │  ║
║    └───────────────────────────────────────────────────┘  ║
║    -> 异步 fwrite(mb_out[slot]) -> 复用缓冲区              ║
╚═══════════════════════════════════════════════════════════╝
    v
  恢复原始数据
```

## 文件格式 (.aplz)

| 偏移 | 大小 | 内容 |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic `"APLZ"`, `version = 3`, `original_size`, `chunk_size = 65536`, `num_chunks`) |
| 24 | 4 B | `n_streams` = 128 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8×N B | `chunk_offsets[N]` (seek table) |
| ... | 可变 | Chunk data (N 个块，v3 格式) |

Per-chunk 数据布局 (v3 格式):
- `token_cnt` (4 B): dense token 数量
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` 对 (各 4 B) — 仅非零符号
- `stream_sizes[128]` (各 2 B = 256 B): 每个 ANS 流的字节大小
- stream data: 拼接的 ANS 比特流 (128 路交织流；每个匹配 token 存储 16-bit 后向距离)

> 空文件 (`original_size = 0`, `num_chunks = 0`) 仅写入 32 字节头部 (`FileHeader` + `n_streams` + `ans_log_l`)。

## 安装

需要 macOS + Xcode Command Line Tools。

```bash
# 安装到 ~/.local/bin (使用 -O3 -flto 优化构建 + 预编译 metallib)
./install.sh

# 安装到 /usr/local/bin (需要 sudo)
./install.sh /usr/local

# 卸载
./install.sh uninstall
```

### Finder 集成 (可选)

```bash
# 添加右键菜单 -> "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# 移除
./setup_finder.sh uninstall
```

支持 Google Drive 和云存储文件。服务运行时自动通过 `/tmp` 处理，绕过 macOS 沙盒限制。

## 使用方法

### aplz 命令 (推荐)

```bash
# 压缩文件
aplz compress myfile.txt              → myfile.txt.aplz

# 压缩目录 (自动 tar)
aplz compress myproject/              → myproject.tar.aplz

# 解压
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# 文件信息
aplz info myfile.txt.aplz
```

### gpu_zip (底层 API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal [--debug]
./gpu_zip -d <input.aplz> <output> compression.metal [--debug]
```

> 若同目录下存在预编译的 `compression.metallib`，`gpu_zip` 会优先自动加载；否则回退至运行时源码编译。

### 开发构建

```bash
./build.sh        # -O2 构建 (生成 gpu_zip + compression.metallib)
./build.sh clean   # 清理产物
```

## 性能 (Apple M4)

使用 `-O3 -flto` 发布构建，3 次运行取最佳值（512 MB 输入，均通过字节级完全往返校验）：

### 压缩

| 数据 | 输入 | 输出 | 压缩率 | 吞吐量 |
|---|---|---|---|---|
| 文本 (重复模式) | 512 MB | 10.3 MB | 2.0% | 297 MB/s |
| 随机二进制 | 512 MB | 535.9 MB | 104.7% | 134 MB/s |

### 解压

| 数据 | 输入 | 速度 | 备注 |
|---|---|---|---|
| 文本 (重复模式) | 512 MB | 2.6 GB/s | 16 mega-batches，完整流水线重叠 |
| 随机二进制 | 512 MB | 232 MB/s | 16 mega-batches，大部分为字面量 |

> 随机数据不可压缩，因此输出略大于输入 (tANS/distance 字段开销)。
> 得益于 mega-batch 流式处理，内存用量恒定，与输入大小无关。

## 源文件结构

| 文件 | 作用 |
|---|---|
| `APLZ.h` | 共享头文件 (CPU/GPU 通用结构体与常量) |
| `compression.metal` | GPU 内核 (MSL 3.0): `compress_chunk`, `build_tans_tables`, `tans_encode`, `tans_decode`, `lz77_decode` |
| `main.mm` | 主机驱动 (Objective-C++): O(1) 流式 mega-batch + 异步双缓冲流水线 |
| `aplz` | 用户 CLI 封装 (Bash): 文件/目录支持, tar 集成 |
| `install.sh` | 安装器: 优化构建 (`-O3 -flto` + `.metallib`) + 系统级部署 |
| `setup_finder.sh` | macOS Finder Quick Action 生成器 (右键压缩/解压) |
| `build.sh` | 开发构建脚本 |
| `bench_compress.sh` | 与 gzip/zip/zstd/lz4/xz 对比的基准测试脚本 |

## 技术细节

- **LZ77**: 4-gram 乘法哈希 (`h4`)，2-way 哈希表 (`atomic_fetch_min`) + Run-Skipping，4 字节并行匹配比较，parallel prefix-sum 压缩整理 (`sparse` → `compact`)，max match 255，min match 3，distance up to 65534
- **Per-chunk tANS (v3)**: 在 GPU (`build_tans_tables`) 上为每个 64 KB 块动态构建最优 tANS 表。128 路交织 ANS 流，512 symbols (256 literals + 256 match lengths)，L=1024，Duda's fast spread，Threadgroup cooperative loading 实现快速表访问
- **ANS Encoding**: Forward encode with renormalization，16-bit 匹配距离字段，sentinel bit 用于比特流结束检测
- **ANS Decoding**: Reverse playback from final state，通过 `clz()` 检测 sentinel，按字批量提取比特，O(1) 解码表查表，以及每块流/状态边界校验
- **Compact Frequency Serialization**: 每块频率表仅存储非零条目 (`n_nonzero + [sym_id, freq]` 对)，最小化文件大小开销
- **全并行 LZ77 解码**: 分块 parallel prefix-scan 校验并写入字面量，再通过取模索引 (`out[off - dist + (j % dist)]`) 统一并行复制重叠与非重叠匹配（无串行回退）
- **Async Double Buffering**: 基于 GCD 的批处理流水线 (32 chunks/batch, 2 slots)。`dispatch_semaphore` 控制槽位互斥，`dispatch_apply` 并行构建解码表，`addCompletedHandler` + 串行 `dispatch_queue` 保证顺序异步写入，压缩完成后原子重命名
- **O(1) Streaming Memory**: Mega-batch 架构 (512 chunks = 32 MB/batch)。缓冲区跨 mega-batch 复用，内存占用与输入大小无关

## 已知限制

- **BS_CAP = 1024 B**: 单个流最大输出 1024 字节。极端膨胀的不可压缩数据可能会截断比特。
- **块大小固定为 64 KB**: 可在 `APLZ.h` / `compression.metal` 中调整，但需要重新构建。

## License

MIT License. See [LICENSE](LICENSE) for details.
