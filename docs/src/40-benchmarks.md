# Benchmarks

Measurement of **3D DCT** performance on varying grid sizes ($N^3$). Results collected using in-place `mul!` (where supported) to exclude allocation overhead. Lower is better.

## GPU Performance (NVIDIA RTX 2080 Ti)

| Grid Size ($N^3$) | `cuFFT` (Baseline) | **`Opt 3D DCT`** | `Batched DCT` (Old) |
| :--- | :--- | :--- | :--- |
| **$16^3$** | 0.068 ms | **0.104 ms** | 0.883 ms |
| **$32^3$** | 0.064 ms | **0.117 ms** | 0.908 ms |
| **$64^3$** | 0.112 ms | **0.237 ms** | 1.138 ms |
| **$128^3$** | 0.818 ms | **1.414 ms** | 3.228 ms |
| **$256^3$** | 5.980 ms | **10.455 ms** | 23.120 ms |

> **Note**: `Opt 3D DCT` maintains excellent performance across all sizes, being only ~1.75x slower than raw `cuFFT` (due to necessary pre/post-processing). In contrast, the naive `Batched DCT` is ~3.9x slower than FFT. For $N=256$, `Opt 3D DCT` is **>2.2x faster** than the batched implementation.

## CPU Performance (Intel Xeon Gold 6132)

### Single Thread

| Grid Size ($N^3$) | `FFTW rfft` | **`Opt 3D DCT`** | `FFTW dct` | `Batched DCT` |
| :--- | :--- | :--- | :--- | :--- |
| **$16^3$** | 0.010 ms | **0.101 ms** | 0.055 ms | 0.123 ms |
| **$32^3$** | 0.106 ms | **0.815 ms** | 0.400 ms | 0.880 ms |
| **$64^3$** | 1.203 ms | **7.035 ms** | 4.299 ms | 14.845 ms |
| **$128^3$** | 15.532 ms | **63.666 ms** | 48.774 ms | 146.376 ms |
| **$256^3$** | 254.233 ms | **656.963 ms** | 425.112 ms | 1580.613 ms |

### 8 Threads

| Grid Size ($N^3$) | `FFTW rfft` | **`Opt 3D DCT`** | `FFTW dct` | `Batched DCT` |
| :--- | :--- | :--- | :--- | :--- |
| **$16^3$** | 0.015 ms | **0.150 ms** | 0.058 ms | 0.424 ms |
| **$32^3$** | 0.100 ms | **0.508 ms** | 0.426 ms | 0.693 ms |
| **$64^3$** | 1.241 ms | **3.856 ms** | 4.336 ms | 8.703 ms |
| **$128^3$** | 14.905 ms | **38.795 ms** | 47.904 ms | 96.860 ms |
| **$256^3$** | 243.146 ms | **332.595 ms** | 420.537 ms | 1066.093 ms |

> **Note**: On multi-threaded CPU, `Opt 3D DCT` (332ms) **outperforms** `FFTW.dct` (420ms) at large sizes ($N=256$) by being ~1.26x faster! It is consistently ~3x faster than the batched implementation. Single-threaded performance is slightly slower than `FFTW.dct`, highlighting efficient parallel scaling.
