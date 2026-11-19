# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | [中文](README.zh.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md)

Motor de compresión/descompresión sin pérdida de alta velocidad que funciona en GPU Apple Silicon. Combina Metal Compute Shaders (MSL 3.0) con tANS (Asymmetric Numeral Systems) en un pipeline completamente acelerado por GPU, logrando una **compresión Round-trip perfecta**.

## Características

- **Zero-copy I/O**: Aprovecha la Unified Memory Architecture (UMA) de Apple Silicon. `mmap` + `newBufferWithBytesNoCopy` elimina copias de entrada CPU↔GPU
- **LZ77 paralelo**: 128 hilos procesan cooperativamente cada bloque de 64 KB mediante hash multiplicativo de 4-gramas (`h4`), tabla hash de 2 vías (`atomic_fetch_min`), Run-Skipping, comparación paralela de 4 bytes y compactación con parallel prefix-sum
- **tANS intercalado por bloque en GPU (128 flujos)**: Construye dinámicamente tablas tANS óptimas por bloque directamente en la GPU (`build_tans_tables`). Cada hilo mantiene un estado ANS independiente, codificando/decodificando 128 flujos de bits en paralelo
- **Decodificación LZ77 completamente paralela**: Prefix-scan paralelo por bloques para validación y escritura de literales, seguido de copia paralela unificada con índice módulo en los 128 hilos — incluidas coincidencias superpuestas (`dist < len`), sin respaldo serial
- **Doble buffer asíncrono**: Pipeline por lotes basado en GCD (`dispatch_semaphore` / `dispatch_group`). Superpone la ejecución GPU con la preparación de tablas en CPU y la E/S asíncrona de archivos
- **Memoria O(1) en streaming**: Uso de memoria fijo independiente del tamaño de entrada. Procesa en mega-batches (32 MB) para operación estable con archivos de 5 GB+
- **Round-trip perfecto**: Garantiza coincidencia exacta a nivel de bytes tras comprimir → descomprimir, con validación de flujos ANS/LZ en GPU

## Arquitectura

### Pipeline de compresión (`-c`)

```
mmap(input)  <- zero-copy MTLBuffer (Apple Silicon UMA)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Bucle streaming mega-batch (512 chunks = 32 MB / batch)  ║
║  (Prefetch con doble buffer de Pass 1 entre mega-batches) ║
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
║  2. Construcción de tablas en GPU: build_tans_tables      ║
║     Per chunk: Histogram -> Normalize -> Spread -> EncTbl ║
║                          | SymInfo[512] + enc_table[1024] ║
║                          v                                ║
║  3. GPU Pass 2: tans_encode (doble buffer, 32 ch/batch)   ║
║     128 flujos ANS intercalados por bloque                ║
║     addCompletedHandler -> serial write queue (tmp atóm.) ║
╚═══════════════════════════════════════════════════════════╝
    | actualización de seek table + rename atómico
    v
  .aplz file
```

### Pipeline de descompresión (`-d`)

```
mmap(.aplz) -> leer header + validar chunk_offsets (seek table)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Pipeline streaming mega-batch (salida 32 MB doble buf.)  ║
║  for each mega-batch:                                     ║
║    ┌───────────────────────────────────────────────────┐  ║
║    │ Pipeline por lotes con doble buffer (32 ch/batch) │  ║
║    │                                                   │  ║
║    │ CPU (batch N+1, paralelo vía dispatch_apply):     │  ║
║    │   leer compact_freq + 128 flujos por bloque       │  ║
║    │   construir DecodeEntry[L] por bloque             │  ║
║    │                                                   │  ║
║    │ GPU (batch N, 128 threads/chunk):                 │  ║
║    │   tans_decode -> lz77_decode -> mb_out[slot]      │  ║
║    └───────────────────────────────────────────────────┘  ║
║    -> fwrite asíncrono(mb_out[slot]) -> reutilizar buffers║
╚═══════════════════════════════════════════════════════════╝
    v
  datos originales restaurados
```

## Formato de archivo (.aplz)

| Offset | Tamaño | Contenido |
|---|---|---|
| 0 | 24 B | `FileHeader` (magic `"APLZ"`, `version = 3`, `original_size`, `chunk_size = 65536`, `num_chunks`) |
| 24 | 4 B | `n_streams` = 128 |
| 28 | 4 B | `ans_log_l` = 10 |
| 32 | 8×N B | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (N bloques, formato v3) |

Estructura de datos por bloque (formato v3):
- `token_cnt` (4 B): número de tokens densos
- `compact_freq`: `n_nonzero` (2 B) + pares `[sym_id, freq]` (4 B cada) — solo símbolos no nulos
- `stream_sizes[128]` (2 B cada = 256 B): tamaño en bytes de cada flujo ANS
- stream data: flujos de bits ANS concatenados (128 flujos intercalados; cada token de coincidencia almacena una distancia hacia atrás de 16 bits)

> Los archivos vacíos (`original_size = 0`, `num_chunks = 0`) se almacenan solo con un encabezado de 32 bytes (`FileHeader` + `n_streams` + `ans_log_l`).

## Instalación

Requiere macOS + Xcode Command Line Tools.

```bash
# Instalar en ~/.local/bin (compilación optimizada con -O3 -flto + metallib precompilado)
./install.sh

# Instalar en /usr/local/bin (requiere sudo)
./install.sh /usr/local

# Desinstalar
./install.sh uninstall
```

### Integración con Finder (opcional)

```bash
# Agregar clic derecho -> "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# Eliminar
./setup_finder.sh uninstall
```

Compatible con Google Drive y archivos de almacenamiento en la nube. El servicio procesa automáticamente los archivos vía `/tmp` para eludir las restricciones del sandbox de macOS.

## Uso

### Comando aplz (recomendado)

```bash
# Comprimir un archivo
aplz compress myfile.txt              → myfile.txt.aplz

# Comprimir un directorio (tar automático)
aplz compress myproject/              → myproject.tar.aplz

# Extraer
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# Información del archivo
aplz info myfile.txt.aplz
```

### gpu_zip (API de bajo nivel)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal [--debug]
./gpu_zip -d <input.aplz> <output> compression.metal [--debug]
```

> Si existe un `compression.metallib` precompilado junto a `compression.metal`, `gpu_zip` lo carga automáticamente y recurre a la compilación desde fuente en tiempo de ejecución en caso contrario.

### Compilación de desarrollo

```bash
./build.sh        # compilación -O2 (genera gpu_zip + compression.metallib)
./build.sh clean   # limpiar artefactos
```

## Rendimiento (Apple M4)

Medido con compilación release `-O3 -flto`, mejor de 3 ejecuciones (entradas de 512 MB, round-trip exacto verificado):

### Compresión

| Datos | Entrada | Salida | Ratio | Rendimiento |
|---|---|---|---|---|
| Texto (patrón repetitivo) | 512 MB | 10.3 MB | 2.0% | 297 MB/s |
| Binario aleatorio | 512 MB | 535.9 MB | 104.7% | 134 MB/s |

### Descompresión

| Datos | Entrada | Velocidad | Notas |
|---|---|---|---|
| Texto (patrón repetitivo) | 512 MB | 2.6 GB/s | 16 mega-batches, superposición completa del pipeline |
| Binario aleatorio | 512 MB | 232 MB/s | 16 mega-batches, mayormente literales |

> Los datos aleatorios son incompresibles, por lo que la salida supera ligeramente la entrada (overhead de campos tANS/distance).
> El uso de memoria es constante independientemente del tamaño de entrada gracias al streaming por mega-batch.

## Archivos fuente

| Archivo | Función |
|---|---|
| `APLZ.h` | Header compartido (structs y constantes para CPU y GPU) |
| `compression.metal` | Kernels GPU (MSL 3.0): `compress_chunk`, `build_tans_tables`, `tans_encode`, `tans_decode`, `lz77_decode` |
| `main.mm` | Driver del host (Objective-C++): mega-batch O(1) + pipelines asíncronos con doble buffer |
| `aplz` | CLI wrapper para usuario (Bash): soporte de archivos/directorios, integración tar |
| `install.sh` | Instalador: compilación optimizada (`-O3 -flto` + `.metallib`) + despliegue a nivel de sistema |
| `setup_finder.sh` | Generador de Quick Actions para macOS Finder (compresión/extracción con clic derecho) |
| `build.sh` | Script de compilación para desarrollo |
| `bench_compress.sh` | Script de benchmark comparando APLZ con gzip/zip/zstd/lz4/xz |

## Detalles técnicos

- **LZ77**: 4-gram multiplicative hash (`h4`), 2-way hash table (`atomic_fetch_min`) con Run-Skipping, comparación paralela de 4 bytes, compactación parallel prefix-sum (`sparse` → `compact`), max match 255, min match 3, distance up to 65534
- **Per-chunk tANS (v3)**: Construye dinámicamente tablas tANS óptimas por bloque de 64 KB en la GPU (`build_tans_tables`). 128 flujos ANS intercalados, 512 símbolos (256 literales + 256 longitudes de coincidencia), L=1024, Duda's fast spread y carga cooperativa en memoria threadgroup
- **ANS Encoding**: Codificación directa con renormalización, campos de distancia de 16 bits, bit centinela para detección del fin del flujo
- **ANS Decoding**: Reproducción inversa desde el estado final, centinela vía `clz()`, extracción de bits por palabra, búsqueda O(1) en tabla de decodificación y validación de límites por bloque
- **Compact Frequency Serialization**: Almacena solo entradas no nulas por bloque (pares `n_nonzero + [sym_id, freq]`), minimizando el overhead del tamaño del archivo
- **Decodificación LZ77 completamente paralela**: Prefix-scan paralelo por bloques para validación y literales, seguido de copia paralela unificada con índice módulo (`out[off - dist + (j % dist)]`) para coincidencias superpuestas y no superpuestas sin respaldo serial
- **Async Double Buffering**: Pipeline por lotes basado en GCD (32 chunks/batch, 2 slots). `dispatch_semaphore` para exclusión de slots, `dispatch_apply` para construcción paralela de tablas de decodificación en CPU, `addCompletedHandler` + `dispatch_queue` serial para escrituras asíncronas ordenadas y renombrado atómico en compresión
- **O(1) Streaming Memory**: Arquitectura mega-batch (512 chunks = 32 MB por lote). Los buffers se reutilizan entre mega-batches manteniendo memoria constante

## Limitaciones conocidas

- **BS_CAP = 1024 B**: La salida máxima de 1 flujo es 1024 bytes. Los datos incompresibles con expansión extrema pueden truncar bits.
- **Tamaño de bloque fijo en 64 KB**: Configurable en `APLZ.h` / `compression.metal` pero requiere recompilación.

## License

MIT License. See [LICENSE](LICENSE) for details.
