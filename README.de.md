# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Español](README.es.md) | [Français](README.fr.md) | **Deutsch**

Hochgeschwindigkeits-Engine für verlustfreie Kompression/Dekompression auf Apple Silicon GPU. Kombiniert Metal Compute Shaders (MSL 3.0) mit tANS (Asymmetric Numeral Systems) in einer vollständig GPU-beschleunigten Pipeline und erreicht **perfekte Round-trip-Kompression**.

## Highlights

- **Zero-copy I/O**: Nutzt die Unified Memory Architecture (UMA) von Apple Silicon. `mmap` + `newBufferWithBytesNoCopy` eliminiert CPU↔GPU-Eingabedatenkopien
- **Paralleles LZ77**: 128 Threads verarbeiten kooperativ jeden 64-KB-Block mit 4-Gramm-Multiplikationshash (`h4`), 2-Wege-Hash-Tabelle (`atomic_fetch_min`), Run-Skipping, parallelem 4-Byte-Vergleich und Parallel-Prefix-Sum-Kompaktierung
- **GPU Per-chunk 128 verschränktes tANS**: Baut optimale tANS-Tabellen pro Block direkt auf der GPU auf (`build_tans_tables`). Jeder Thread hält einen unabhängigen ANS-Zustand und kodiert/dekodiert 128 Bitstreams parallel
- **Vollständig parallele LZ77-Dekodierung**: Kachelbasierter Parallel-Prefix-Scan für Validierung und Literal-Schreibvorgänge, gefolgt von einheitlicher modulo-indizierter Parallelkopie über alle 128 Threads — einschließlich überlappender Matches (`dist < len`), ohne seriellen Rückfall
- **Asynchrones Doppelpuffern**: GCD-basierte Batch-Pipeline (`dispatch_semaphore` / `dispatch_group`). Überlappt GPU-Ausführung mit CPU-Tabellenerstellung und asynchroner Datei-E/A
- **O(1) Streaming-Speicher**: Fester Speicherverbrauch unabhängig von der Eingabegröße. Verarbeitet in Mega-Batches (32 MB) für stabilen Betrieb bei 5 GB+ Dateien
- **Perfekter Round-trip**: Garantiert Byte-genaue Übereinstimmung nach Kompression → Dekompression mit GPU-seitiger ANS/LZ-Stream-Validierung

## Architektur

### Kompressionspipeline (`-c`)

```
mmap(input)  <- zero-copy MTLBuffer (Apple Silicon UMA)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Mega-Batch-Streaming-Schleife (512 Chunks = 32 MB/Batch) ║
║  (Pass-1 Doppelpuffer-Prefetch zwischen Mega-Batches)     ║
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
║  2. GPU-Tabellenbau: build_tans_tables                    ║
║     Per chunk: Histogram -> Normalize -> Spread -> EncTbl ║
║                          | SymInfo[512] + enc_table[1024] ║
║                          v                                ║
║  3. GPU Pass 2: tans_encode (Doppelpuffer, 32 Ch/Batch)   ║
║     128 verschränkte ANS-Streams pro Chunk                ║
║     addCompletedHandler -> serial write queue (atom. tmp) ║
╚═══════════════════════════════════════════════════════════╝
    | Seek-Table-Rückschreibung + atomares rename
    v
  .aplz file
```

### Dekompressionspipeline (`-d`)

```
mmap(.aplz) -> Header lesen + chunk_offsets validieren (seek table)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Mega-Batch-Streaming-Pipeline (32 MB Doppelpuffer-Ausg.) ║
║  for each mega-batch:                                     ║
║    ┌───────────────────────────────────────────────────┐  ║
║    │ Doppelpuffer-Batch-Pipeline (32 Chunks/Batch)     │  ║
║    │                                                   │  ║
║    │ CPU (batch N+1, parallel via dispatch_apply):     │  ║
║    │   pro Chunk compact_freq + 128 Streams lesen      │  ║
║    │   DecodeEntry[L] pro Chunk aufbauen               │  ║
║    │                                                   │  ║
║    │ GPU (batch N, 128 threads/chunk):                 │  ║
║    │   tans_decode -> lz77_decode -> mb_out[slot]      │  ║
║    └───────────────────────────────────────────────────┘  ║
║    -> asynchrones fwrite(mb_out[slot]) -> Puffer recyceln ║
╚═══════════════════════════════════════════════════════════╝
    v
  wiederhergestellte Originaldaten
```

## Dateiformat (.aplz)

| Offset | Größe | Inhalt |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic `"APLZ"`, `version = 3`, `original_size`, `chunk_size = 65536`, `num_chunks`) |
| 24 | 4 B | `n_streams` = 128 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8×N B | `chunk_offsets[N]` (seek table) |
| ... | variabel | Chunk data (N Chunks, v3-Format) |

Per-chunk Datenstruktur (v3-Format):
- `token_cnt` (4 B): Anzahl der dichten Tokens
- `compact_freq`: `n_nonzero` (2 B) + `[sym_id, freq]` Paare (je 4 B) — nur Nicht-Null-Symbole
- `stream_sizes[128]` (je 2 B = 256 B): Bytegröße jedes ANS-Streams
- stream data: verkettete ANS-Bitstreams (128 verschränkte Streams; jedes Match-Token speichert eine 16-Bit-Rückwärtsdistanz)

> Leere Dateien (`original_size = 0`, `num_chunks = 0`) werden nur mit dem 32-Byte-Header (`FileHeader` + `n_streams` + `ans_log_l`) gespeichert.

## Installation

Erfordert macOS + Xcode Command Line Tools.

```bash
# In ~/.local/bin installieren (optimierter Build mit -O3 -flto + vorkompilierter metallib)
./install.sh

# In /usr/local/bin installieren (erfordert sudo)
./install.sh /usr/local

# Deinstallieren
./install.sh uninstall
```

### Finder-Integration (optional)

```bash
# Rechtsklick -> "Compress with APLZ" / "Extract with APLZ" hinzufügen
./setup_finder.sh

# Entfernen
./setup_finder.sh uninstall
```

Unterstützt Google Drive und Cloud-Speicher-Dateien. Der Dienst verarbeitet Dateien automatisch über `/tmp`, um die macOS-Sandbox-Einschränkungen zu umgehen.

## Verwendung

### aplz-Befehl (empfohlen)

```bash
# Datei komprimieren
aplz compress myfile.txt              → myfile.txt.aplz

# Verzeichnis komprimieren (automatisches tar)
aplz compress myproject/              → myproject.tar.aplz

# Extrahieren
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# Dateiinformationen
aplz info myfile.txt.aplz
```

### gpu_zip (Low-Level-API)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal [--debug]
./gpu_zip -d <input.aplz> <output> compression.metal [--debug]
```

> Befindet sich eine vorkompilierte `compression.metallib` neben `compression.metal`, lädt `gpu_zip` diese automatisch und fällt andernfalls auf die Laufzeit-Quellkompilierung zurück.

### Entwicklungs-Build

```bash
./build.sh        # -O2 Build (erzeugt gpu_zip + compression.metallib)
./build.sh clean   # Artefakte bereinigen
```

## Leistung (Apple M4)

Gemessen mit `-O3 -flto` Release-Build, Bestwert aus 3 Durchläufen (512 MB Eingabe, Byte-genauer Round-trip verifiziert):

### Kompression

| Daten | Eingabe | Ausgabe | Ratio | Durchsatz |
|---|---|---|---|---|
| Text (wiederholendes Muster) | 512 MB | 10.3 MB | 2.0% | 297 MB/s |
| Zufällige Binärdaten | 512 MB | 535.9 MB | 104.7% | 134 MB/s |

### Dekompression

| Daten | Eingabe | Geschwindigkeit | Anmerkungen |
|---|---|---|---|
| Text (wiederholendes Muster) | 512 MB | 2.6 GB/s | 16 Mega-Batches, vollständige Pipeline-Überlappung |
| Zufällige Binärdaten | 512 MB | 232 MB/s | 16 Mega-Batches, überwiegend Literale |

> Zufällige Daten sind inkompressibel, daher übersteigt die Ausgabe leicht die Eingabe (tANS/distance-Feld-Overhead).
> Der Speicherverbrauch ist dank Mega-Batch-Streaming unabhängig von der Eingabegröße konstant.

## Quelldateien

| Datei | Rolle |
|---|---|
| `APLZ.h` | Gemeinsamer Header (Structs und Konstanten für CPU und GPU) |
| `compression.metal` | GPU-Kernel (MSL 3.0): `compress_chunk`, `build_tans_tables`, `tans_encode`, `tans_decode`, `lz77_decode` |
| `main.mm` | Host-Treiber (Objective-C++): O(1)-Streaming-Mega-Batch + asynchrone Doppelpuffer-Pipelines |
| `aplz` | Benutzer-CLI-Wrapper (Bash): Datei-/Verzeichnisunterstützung, tar-Integration |
| `install.sh` | Installer: optimierter Build (`-O3 -flto` + `.metallib`) + systemweite Bereitstellung |
| `setup_finder.sh` | macOS Finder Quick Action Generator (Rechtsklick-Kompression/-Extraktion) |
| `build.sh` | Entwicklungs-Build-Skript |
| `bench_compress.sh` | Benchmark-Skript zum Vergleich von APLZ mit gzip/zip/zstd/lz4/xz |

## Technische Details

- **LZ77**: 4-Gramm-Multiplikationshash (`h4`), 2-Wege-Hash-Tabelle (`atomic_fetch_min`) mit Run-Skipping, paralleler 4-Byte-Vergleich, Parallel-Prefix-Sum-Kompaktierung (`sparse` → `compact`), max match 255, min match 3, Distanz bis 65534
- **Per-chunk tANS (v3)**: Baut optimale tANS-Tabellen pro 64-KB-Block dynamisch auf der GPU auf (`build_tans_tables`). 128 verschränkte ANS-Streams, 512 Symbole (256 Literale + 256 Match-Längen), L=1024, Duda's fast spread und kooperatives Laden in den Threadgroup-Speicher
- **ANS Encoding**: Vorwärtskodierung mit Renormalisierung, 16-Bit-Distanzfelder, Sentinel-Bit zur Bitstream-Ende-Erkennung
- **ANS Decoding**: Rückwärtsdekodierung vom Endzustand, Sentinel über `clz()`, wortweise Batch-Bitextraktion, O(1)-Tabellenzugriff und Stream-/Zustands-Grenzvalidierung pro Chunk
- **Compact Frequency Serialization**: Speichert nur Nicht-Null-Einträge pro Block (`n_nonzero + [sym_id, freq]` Paare), minimiert den Dateigrößen-Overhead
- **Vollständig parallele LZ77-Dekodierung**: Kachelbasierter Parallel-Prefix-Scan für Validierung und Literale, gefolgt von einheitlicher modulo-indizierter Parallelkopie (`out[off - dist + (j % dist)]`) für überlappende und nicht überlappende Matches ohne seriellen Rückfall
- **Asynchrones Doppelpuffern**: GCD-basierte Batch-Pipeline (32 Chunks/Batch, 2 Slots). `dispatch_semaphore` für Slot-Synchronisation, `dispatch_apply` für parallelen CPU-Dekodiertabellenbau, `addCompletedHandler` + serielle `dispatch_queue` für geordnete asynchrone Schreibvorgänge und atomares Umbenennen bei Kompression
- **O(1) Streaming-Speicher**: Mega-Batch-Architektur (512 Chunks = 32 MB pro Batch). Puffer werden über Mega-Batches hinweg wiederverwendet

## Bekannte Einschränkungen

- **BS_CAP = 1024 B**: Maximale Ausgabe eines Streams beträgt 1024 Bytes. Extrem expandierende inkompressible Daten können Bits abschneiden.
- **Blockgröße auf 64 KB festgelegt**: In `APLZ.h` / `compression.metal` anpassbar, erfordert aber Neukompilierung.

## License

MIT License. See [LICENSE](LICENSE) for details.
