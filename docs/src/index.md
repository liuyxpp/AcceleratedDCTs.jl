# AcceleratedDCTs.jl

Documentation for [AcceleratedDCTs](https://github.com/lyx/AcceleratedDCTs.jl).

## Introduction

AcceleratedDCTs.jl aims to provide the fastest possible Discrete Cosine Transform (DCT) for Julia, running on both CPUs and GPUs. It supports:
*   **DCT-II** (Standard "DCT") and **DCT-III** (Inverse DCT) — commonly used in signal processing and solving PDEs
*   **DCT-I** and **IDCT-I** — for symmetric boundary conditions

The core innovation of this package is the implementation of **Algorithm 2 (2D)** and **Algorithm 3 (3D)**, which reduce $N$-dimensional DCTs to $N$-dimensional Real-to-Complex (R2C) FFTs with $O(N)$ pre/post-processing steps, avoiding the overhead of separable 1D transforms (which require redundant transposes).

## Key Features

*   **⚡ High Performance**: optimized algorithms (Makhoul's method) that outperform standard separable approaches.
*   **🧠 Efficient DCT-I**: New separable split-radix algorithm for DCT-I that avoids memory expansion ($O(M)$ vs old $O(2M)$).
*   **🚀 Device Agnostic**: Runs on CPU (Threads) and GPU (`CuArray`, `ROCArray` via `KernelAbstractions`).
*   **🔥 VkDCT Backend**: Pre-compiled [VkFFT](https://github.com/DTolm/VkFFT)-based CUDA library (`VkDCT_jll`) for DCT-I offering **~15x speedup** on GPU. Zero-setup: just `using CUDA, AcceleratedDCTs`.
*   **🧩 AbstractFFTs Compatible**: Zero-allocation `mul!`, `ldiv!`, and precomputed `Plan` support.
*   **📦 3D Optimized**: Specialized 3D kernels that avoid redundant transposes.

