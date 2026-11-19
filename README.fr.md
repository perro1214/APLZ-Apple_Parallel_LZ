# APLZ — Apple Parallel LZ

[English](README.md) | [日本語](README.ja.md) | [中文](README.zh.md) | [Español](README.es.md) | **Français** | [Deutsch](README.de.md)

Moteur de compression/décompression sans perte à haute vitesse fonctionnant sur GPU Apple Silicon. Combine les Metal Compute Shaders (MSL 3.0) avec tANS (Asymmetric Numeral Systems) dans un pipeline entièrement accéléré par GPU, réalisant une **compression Round-trip parfaite**.

## Points forts

- **Zero-copy I/O** : Exploite l'Unified Memory Architecture (UMA) d'Apple Silicon. `mmap` + `newBufferWithBytesNoCopy` élimine les copies d'entrée CPU↔GPU
- **LZ77 parallèle** : 128 threads traitent coopérativement chaque bloc de 64 Ko avec un hachage multiplicatif 4-grammes (`h4`), une table de hachage à 2 voies (`atomic_fetch_min`), Run-Skipping, comparaison parallèle 4 octets et compactage par parallel prefix-sum
- **tANS entrelacé par bloc sur GPU (128 flux)** : Construit dynamiquement des tables tANS optimales par bloc directement sur GPU (`build_tans_tables`). Chaque thread maintient un état ANS indépendant, encodant/décodant 128 flux de bits en parallèle
- **Décodage LZ77 entièrement parallèle** : Prefix-scan parallèle par tuiles pour validation et écriture des littéraux, suivi d'une copie parallèle unifiée par modulo sur 128 threads — y compris pour les correspondances chevauchantes (`dist < len`), sans repli séquentiel
- **Double tampon asynchrone** : Pipeline par lots basé sur GCD (`dispatch_semaphore` / `dispatch_group`). Recouvre l'exécution GPU avec la préparation des tables CPU et les E/S fichiers asynchrones
- **Mémoire O(1) en streaming** : Utilisation mémoire fixe indépendante de la taille d'entrée. Traite en mega-batches (32 Mo) pour un fonctionnement stable sur des fichiers de 5 Go+
- **Round-trip parfait** : Garantit une correspondance exacte au niveau octet après compression → décompression, avec validation GPU des flux ANS/LZ

## Architecture

### Pipeline de compression (`-c`)

```
mmap(input)  <- zero-copy MTLBuffer (Apple Silicon UMA)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Boucle streaming mega-batch (512 chunks = 32 Mo / batch) ║
║  (Préchargement double tampon de Pass 1 inter-batches)    ║
║                                                           ║
║  1. GPU Pass 1 : compress_chunk                           ║
║     ┌───────────────────────────────────────────────────┐ ║
║     │ 1 threadgroup (128 threads) = 1 chunk (64 Ko)     │ ║
║     │   A-1  Hash table init                            │ ║
║     │   A-2  4-gram hash + Run-Skipping (atomic_min x2) │ ║
║     │   A-3  LZ77 4-byte parallel match -> sparse[]     │ ║
║     │   B    Parallel prefix-sum compaction -> compact[]│ ║
║     └───────────────────────────────────────────────────┘ ║
║                          |                                ║
║                          v                                ║
║  2. Construction des tables GPU : build_tans_tables       ║
║     Per chunk: Histogram -> Normalize -> Spread -> EncTbl ║
║                          | SymInfo[512] + enc_table[1024] ║
║                          v                                ║
║  3. GPU Pass 2 : tans_encode (double tampon, 32 ch/batch) ║
║     128 flux ANS entrelacés par bloc                      ║
║     addCompletedHandler -> serial write queue (tmp atom.) ║
╚═══════════════════════════════════════════════════════════╝
    | mise à jour de la seek table + rename atomique
    v
  .aplz file
```

### Pipeline de décompression (`-d`)

```
mmap(.aplz) -> lecture header + validation chunk_offsets (seek table)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║  Pipeline streaming mega-batch (sortie 32 Mo double buf.) ║
║  for each mega-batch:                                     ║
║    ┌───────────────────────────────────────────────────┐  ║
║    │ Pipeline par lots double tampon (32 chunks/batch) │  ║
║    │                                                   │  ║
║    │ CPU (batch N+1, parallèle via dispatch_apply) :   │  ║
║    │   lecture compact_freq + 128 flux par bloc        │  ║
║    │   construction de DecodeEntry[L] par bloc         │  ║
║    │                                                   │  ║
║    │ GPU (batch N, 128 threads/chunk) :                │  ║
║    │   tans_decode -> lz77_decode -> mb_out[slot]      │  ║
║    └───────────────────────────────────────────────────┘  ║
║    -> fwrite asynchrone(mb_out[slot]) -> réutilisation    ║
╚═══════════════════════════════════════════════════════════╝
    v
  données originales restaurées
```

## Format de fichier (.aplz)

| Offset | Taille | Contenu |
|---|---|---|
| 0 | 24 o | `FileHeader` (magic `"APLZ"`, `version = 3`, `original_size`, `chunk_size = 65536`, `num_chunks`) |
| 24 | 4 o | `n_streams` = 128 |
| 28 | 4 o | `ans_log_l` = 10 |
| 32 | 8×N o | `chunk_offsets[N]` (seek table) |
| ... | variable | Chunk data (N blocs, format v3) |

Structure des données par bloc (format v3) :
- `token_cnt` (4 o) : nombre de tokens denses
- `compact_freq` : `n_nonzero` (2 o) + paires `[sym_id, freq]` (4 o chaque) — symboles non nuls uniquement
- `stream_sizes[128]` (2 o chaque = 256 o) : taille en octets de chaque flux ANS
- stream data : flux de bits ANS concaténés (128 flux entrelacés ; chaque token de correspondance stocke une distance arrière de 16 bits)

> Les fichiers vides (`original_size = 0`, `num_chunks = 0`) sont enregistrés uniquement avec un en-tête de 32 octets (`FileHeader` + `n_streams` + `ans_log_l`).

## Installation

Nécessite macOS + Xcode Command Line Tools.

```bash
# Installer dans ~/.local/bin (compilation optimisée avec -O3 -flto + metallib précompilé)
./install.sh

# Installer dans /usr/local/bin (nécessite sudo)
./install.sh /usr/local

# Désinstaller
./install.sh uninstall
```

### Intégration Finder (optionnelle)

```bash
# Ajouter clic droit -> "Compress with APLZ" / "Extract with APLZ"
./setup_finder.sh

# Supprimer
./setup_finder.sh uninstall
```

Compatible avec Google Drive et les fichiers de stockage cloud. Le service traite automatiquement les fichiers via `/tmp` pour contourner les restrictions du sandbox macOS.

## Utilisation

### Commande aplz (recommandée)

```bash
# Compresser un fichier
aplz compress myfile.txt              → myfile.txt.aplz

# Compresser un répertoire (tar automatique)
aplz compress myproject/              → myproject.tar.aplz

# Extraire
aplz extract myfile.txt.aplz          → myfile.txt
aplz extract myproject.tar.aplz       → myproject/

# Informations sur le fichier
aplz info myfile.txt.aplz
```

### gpu_zip (API bas niveau)

```bash
./gpu_zip -c <input> <output.aplz> compression.metal [--debug]
./gpu_zip -d <input.aplz> <output> compression.metal [--debug]
```

> Si un fichier `compression.metallib` précompilé se trouve à côté de `compression.metal`, `gpu_zip` le charge automatiquement et se replie sinon sur la compilation à la volée.

### Compilation de développement

```bash
./build.sh        # compilation -O2 (génère gpu_zip + compression.metallib)
./build.sh clean   # nettoyer les artefacts
```

## Performances (Apple M4)

Mesuré avec le build release `-O3 -flto`, meilleur sur 3 exécutions (entrées de 512 Mo, round-trip exact vérifié) :

### Compression

| Données | Entrée | Sortie | Ratio | Débit |
|---|---|---|---|---|
| Texte (motif répétitif) | 512 Mo | 10.3 Mo | 2.0% | 297 Mo/s |
| Binaire aléatoire | 512 Mo | 535.9 Mo | 104.7% | 134 Mo/s |

### Décompression

| Données | Entrée | Vitesse | Notes |
|---|---|---|---|
| Texte (motif répétitif) | 512 Mo | 2.6 Go/s | 16 mega-batches, recouvrement complet du pipeline |
| Binaire aléatoire | 512 Mo | 232 Mo/s | 16 mega-batches, principalement des littéraux |

> Les données aléatoires sont incompressibles, donc la sortie dépasse légèrement l'entrée (overhead des champs tANS/distance).
> L'utilisation mémoire est constante quelle que soit la taille d'entrée grâce au streaming par mega-batch.

## Fichiers sources

| Fichier | Rôle |
|---|---|
| `APLZ.h` | Header partagé (structs et constantes pour CPU et GPU) |
| `compression.metal` | Kernels GPU (MSL 3.0) : `compress_chunk`, `build_tans_tables`, `tans_encode`, `tans_decode`, `lz77_decode` |
| `main.mm` | Pilote hôte (Objective-C++) : mega-batch O(1) + pipelines asynchrones à double tampon |
| `aplz` | CLI wrapper utilisateur (Bash) : support fichiers/répertoires, intégration tar |
| `install.sh` | Installateur : compilation optimisée (`-O3 -flto` + `.metallib`) + déploiement système |
| `setup_finder.sh` | Générateur de Quick Actions macOS Finder (compression/extraction par clic droit) |
| `build.sh` | Script de compilation de développement |
| `bench_compress.sh` | Script de benchmark comparant APLZ à gzip/zip/zstd/lz4/xz |

## Détails techniques

- **LZ77** : Hachage multiplicatif 4-grammes (`h4`), table de hachage 2-way (`atomic_fetch_min`) avec Run-Skipping, comparaison parallèle 4 octets, compactage parallel prefix-sum (`sparse` → `compact`), max match 255, min match 3, distance jusqu'à 65534
- **Per-chunk tANS (v3)** : Construit dynamiquement des tables tANS optimales par bloc de 64 Ko sur GPU (`build_tans_tables`). 128 flux ANS entrelacés, 512 symboles (256 littéraux + 256 longueurs de correspondance), L=1024, Duda's fast spread et chargement coopératif en mémoire threadgroup
- **ANS Encoding** : Encodage direct avec renormalisation, champs de distance 16 bits, bit sentinelle pour la fin de flux
- **ANS Decoding** : Lecture inverse depuis l'état final, sentinelle via `clz()`, extraction de bits par mot, accès O(1) à la table de décodage et validation des bornes par bloc
- **Compact Frequency Serialization** : Stocke uniquement les entrées non nulles par bloc (paires `n_nonzero + [sym_id, freq]`), minimisant l'overhead de taille du fichier
- **Décodage LZ77 entièrement parallèle** : Prefix-scan parallèle par tuiles pour validation et littéraux, suivi d'une copie parallèle unifiée par modulo (`out[off - dist + (j % dist)]`) gérant correspondances chevauchantes et non chevauchantes sans repli séquentiel
- **Async Double Buffering** : Pipeline par lots GCD (32 chunks/batch, 2 slots). `dispatch_semaphore` pour l'exclusion de slots, `dispatch_apply` pour la construction parallèle CPU des tables de décodage, `addCompletedHandler` + `dispatch_queue` série pour les écritures asynchrones ordonnées et renommage atomique en compression
- **O(1) Streaming Memory** : Architecture mega-batch (512 chunks = 32 Mo par lot). Les tampons sont réutilisés d'un mega-batch à l'autre pour une mémoire constante

## Limitations connues

- **BS_CAP = 1024 o** : La sortie maximale d'un flux est de 1024 octets. Les données incompressibles à expansion extrême peuvent tronquer des bits.
- **Taille de bloc fixée à 64 Ko** : Configurable dans `APLZ.h` / `compression.metal`, mais nécessite une recompilation.

## License

MIT License. See [LICENSE](LICENSE) for details.
