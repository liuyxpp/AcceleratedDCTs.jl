# AcceleratedDCTs.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://liuyxpp.github.io/AcceleratedDCTs.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev)
[![Test workflow status](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/liuyxpp/AcceleratedDCTs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/liuyxpp/AcceleratedDCTs.jl)
[![Docs workflow Status](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/liuyxpp/AcceleratedDCTs.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

**Fast, Device-Agnostic, AbstractFFTs-compatible DCT library for Julia.**

AcceleratedDCTs.jl provides highly optimized Discrete Cosine Transform implementations for 1D, 2D and 3D data:
*   **DCT-II** (Standard "DCT") and **DCT-III** (Inverse DCT)
*   **DCT-I** and **IDCT-I** (symmetric boundary conditions)

It leverages **KernelAbstractions.jl** to run efficiently on both CPUs (multithreaded) and GPUs (CUDA, AMD, etc.), and implements the **AbstractFFTs.jl** interface for easy integration.

## Key Features

*   **⚡ High Performance**: optimized algorithms (Makhoul's method) that outperform standard separable approaches.
*   **🚀 Device Agnostic**: Runs on CPU (Threads) and GPU (`CuArray`, `ROCArray` via `KernelAbstractions`).
*   **🔥 VkDCT Backend**: Optional experimental C++/Vulkan backend for DCT-I offering **~15x speedup** on GPU compared to the pure Julia implementation.
*   **🧩 AbstractFFTs Compatible**: Zero-allocation `mul!`, `ldiv!`, and precomputed `Plan` support.
*   **📦 3D Optimized**: Specialized 3D kernels that avoid redundant transposes.

## Installation

```julia
using Pkg
Pkg.add("AcceleratedDCTs")
```

## Quick Start

### Basic Usage

```julia
using AcceleratedDCTs: plan_dct, mul!
using CUDA

# 1. Create Data
N = 128
x_gpu = CUDA.rand(Float64, N, N, N)  # can be any Real, e.g. Float32

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

### DCT-I (Symmetric Boundary)

```julia
using AcceleratedDCTs: dct1, idct1, plan_dct1

# One-shot
y = dct1(x_gpu)
x_rec = idct1(y)

# Plan-based (recommended for repeated use)
p = plan_dct1(x_gpu)
y = p * x_gpu
x_rec = p \ y
```

## Benchmarks

Measurement of **3D DCT** performance on varying grid sizes ($N^3$). Results collected using in-place `mul!` (where supported) to exclude allocation overhead.
Lower is better.

### DCT-II (`dct`) Performance (GPU, NVIDIA RTX 2080 Ti)

| Grid Size ($N^3$) | `cuFFT` (Baseline) | **`Opt 3D DCT`** | `Batched DCT` (Old) |
| :--- | :--- | :--- | :--- |
| **$16^3$** | 0.080 ms | **0.113 ms** | 1.041 ms |
| **$32^3$** | 0.076 ms | **0.131 ms** | 0.946 ms |
| **$64^3$** | 0.116 ms | **0.246 ms** | 1.165 ms |
| **$128^3$** | 0.833 ms | **1.423 ms** | 3.302 ms |
| **$256^3$** | 5.945 ms | **10.417 ms** | 26.019 ms |

> **Note**: `Opt 3D DCT` maintains excellent performance across all sizes, being only ~1.75x slower than raw `cuFFT` (due to necessary pre/post-processing). In contrast, the naive `Batched DCT` is ~3.9x slower than FFT. For $N=256$, `Opt 3D DCT` is **>2.2x faster** than the batched implementation.

### DCT-I (`dct1`) Performance (GPU, NVIDIA RTX 2080 Ti)

Measurement of **3D DCT-I** performance. Compares `Opt DCT-I` against raw `cuFFT rfft` of size $(2M-2)^3$ to measure overhead (since CUDA has no native DCT-I).

| Grid Size ($M^3$) | `cuFFT rfft` (Baseline) | **`Opt DCT-I`** | Overhead |
| :--- | :--- | :--- | :--- |
| **$16^3$** | 0.079 ms | **0.108 ms** | ~1.36x |
| **$32^3$** | 0.245 ms | **0.313 ms** | ~1.27x |
| **$64^3$** | 1.204 ms | **1.323 ms** | ~1.10x |
| **$128^3$** | 23.289 ms | **23.951 ms** | ~1.03x |
| **$256^3$** | 88.519 ms | **92.446 ms** | ~1.04x |

> **Note**: Our optimized DCT-I implementation adds minimal overhead (<5% at large sizes) over the raw FFT, demonstrating extremely efficient kernel implementation.
 
## VkDCT Extension (High Performance GPU DCT-I)

For maximum performance on NVIDIA GPUs (providing **7x-15x speedup** over the device-agnostic backend), you can compile the optional `VkDCT` library.

**Setup**:
1.  Ensure you have `cmake` and `nvcc` (CUDA Toolkit) installed.
2.  Run the compilation script:
    ```bash
    cd lib/VkDCT
    ./compile.sh
    ```
3.  Load `CUDA` in your Julia session. The extension `VkDCTExt` will automatically load and accelerate `plan_dct1` for `CuArray`.

```julia
using AcceleratedDCTs
using CUDA

# Automatically uses VkDCT if compiled
p = plan_dct1(CuArray(rand(128, 128, 128))) 
```

## Documentation

Comprehensive documentation is available at [https://liuyxpp.github.io/AcceleratedDCTs.jl/dev/](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev/).

The documentation includes:
*   [**Quick Start & Tutorial**](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev/10-tutorial/): Usage examples and the plan-based API.
*   [**Theory & Algorithms**](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev/20-theory/): Mathematical background of Makhoul's algorithm.
*   [**Implementation Details**](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev/30-implementation/): Insights into `KernelAbstractions` and buffer management.
*   [**Benchmarks**](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev/40-benchmarks/): In-depth performance analysis on CPU and GPU.
*   [**API Reference**](https://liuyxpp.github.io/AcceleratedDCTs.jl/dev/95-reference/): Detailed function documentation.


## AI Usage Disclaimer

Most of source codes and docs in this project are generated by Claude Opus 4.5 (thinking) and Gemini 3.0 Pro (High) in Google Antigravity. The LLM are guided by human with many rounds to achieve a pre-designed goal. And the AI generated contents are carefully examined by human. The correctness are verified with FFTW and the roundtrip transform. See `test` folder for verification details.
