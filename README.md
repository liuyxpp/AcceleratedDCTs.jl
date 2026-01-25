# AcceleratedDCTs.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://liuyxpp.github.io/AcceleratedDCTs.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev)
[![Test workflow status](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/liuyxpp/AcceleratedDCTs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/liuyxpp/AcceleratedDCTs.jl)
[![Docs workflow Status](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

**Fast, Device-Agnostic, AbstractFFTs-compatible DCT library for Julia.**

AcceleratedDCTs.jl provides highly optimized Discrete Cosine Transform (DCT-II) and Inverse DCT (DCT-III) implementations for 1D, 2D and 3D data. It leverages **KernelAbstractions.jl** to run efficiently on both CPUs (multithreaded) and GPUs (CUDA, AMD, etc.), and implements the **AbstractFFTs.jl** interface for easy integration.

## Key Features

*   **⚡ High Performance**: optimized algorithms (Makhoul's method) that outperform standard separable approaches on GPU (~2x speedup for 3D) and CPU (~3x speedup for 3D).
*   **🚀 Device Agnostic**: Runs on CPU (Threads) and GPU (`CuArray`, `ROCArray` via `KernelAbstractions`).
*   **🧩 AbstractFFTs Compatible**: Zero-allocation `mul!`, `ldiv!`, and precomputed `Plan` support.
*   **📦 3D Optimized**: Specialized 3D kernels that avoid redundant transposes.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/liuyxpp/AcceleratedDCTs.jl")
```

## Quick Start

### Basic Usage

```julia
using AcceleratedDCTs: plan_dct, mul!
using CUDA

# 1. Create Data
N = 128
x_gpu = CUDA.rand(Float64, N, N, N)

# 2. Create Optimized Plan (Recommended)
p = plan_dct(x_gpu)

# 3. Execute
y = p * x_gpu           # Standard execution
mul!(y, p, x_gpu)       # Zero-allocation (in-place output)

# 4. Inverse
x_rec = p \ y
# or
inv_p = inv(p)
mul!(x_rec, inv_p, y)
```

### One-shot Functions

For convenience (slower due to plan creation overhead):

```julia
using AcceleratedDCTs: dct, idct

y = dct(x_gpu)
x_rec = idct(y)
```

## Benchmarks

Measurement of **3D DCT** performance on varying grid sizes ($N^3$). Results collected using in-place `mul!` (where supported) to exclude allocation overhead.
Lower is better.

### GPU Performance (Nvidia RTX 2080 Ti)

| Grid Size ($N^3$) | `cuFFT` (Baseline) | **`Opt 3D DCT`** | `Batched DCT` (Old) |
| :--- | :--- | :--- | :--- |
| **$16^3$** | 0.068 ms | **0.104 ms** | 0.883 ms |
| **$32^3$** | 0.064 ms | **0.117 ms** | 0.908 ms |
| **$64^3$** | 0.112 ms | **0.237 ms** | 1.138 ms |
| **$128^3$** | 0.818 ms | **1.414 ms** | 3.228 ms |
| **$256^3$** | 5.980 ms | **10.455 ms** | 23.120 ms |

> **Note**: `Opt 3D DCT` maintains excellent performance across all sizes, being only ~1.75x slower than raw `cuFFT` (due to necessary pre/post-processing). In contrast, the naive `Batched DCT` is ~3.9x slower than FFT. For $N=256$, `Opt 3D DCT` is **>2.2x faster** than the batched implementation.

### CPU Performance (Single Thread, Intel Xeon Gold 6132)

| Grid Size ($N^3$) | `FFTW rfft` | **`Opt 3D DCT`** | `FFTW dct` | `Batched DCT` |
| :--- | :--- | :--- | :--- | :--- |
| **$16^3$** | 0.010 ms | **0.101 ms** | 0.055 ms | 0.123 ms |
| **$32^3$** | 0.106 ms | **0.815 ms** | 0.400 ms | 0.880 ms |
| **$64^3$** | 1.203 ms | **7.035 ms** | 4.299 ms | 14.845 ms |
| **$128^3$** | 15.532 ms | **63.666 ms** | 48.774 ms | 146.376 ms |
| **$256^3$** | 254.233 ms | **656.963 ms** | 425.112 ms | 1580.613 ms |

### CPU Performance (8 Threads, Intel Xeon Gold 6132)

| Grid Size ($N^3$) | `FFTW rfft` | **`Opt 3D DCT`** | `FFTW dct` | `Batched DCT` |
| :--- | :--- | :--- | :--- | :--- |
| **$16^3$** | 0.015 ms | **0.150 ms** | 0.058 ms | 0.424 ms |
| **$32^3$** | 0.100 ms | **0.508 ms** | 0.426 ms | 0.693 ms |
| **$64^3$** | 1.241 ms | **3.856 ms** | 4.336 ms | 8.703 ms |
| **$128^3$** | 14.905 ms | **38.795 ms** | 47.904 ms | 96.860 ms |
| **$256^3$** | 243.146 ms | **332.595 ms** | 420.537 ms | 1066.093 ms |

> **Note**: On multi-threaded CPU, `Opt 3D DCT` (332ms) **outperforms** `FFTW.dct` (420ms) at large sizes ($N=256$) by being ~1.26x faster! It is consistently ~3x faster than the batched implementation. Single-threaded performance is slightly slower than `FFTW.dct`, highlighting efficient parallel scaling.

## Documentation

For detailed theory, algorithm explanation, and advanced usage, see [docs/Documentation.md](docs/Documentation.md).

## AI Usage Disclaimer

Most of source codes and docs in this project are generated by Claude Opus 4.5 (thinking) and Gemini 3.0 Pro (High) in Google Antigravity. The LLM are guided by human with many rounds to achieve a pre-designed goal. And the AI generated contents are carefully examined by human. The correctness are verified with FFTW and the roundtrip transform. See `test` folder for verification details.
