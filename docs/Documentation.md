# AcceleratedDCTs.jl Documentation

## 1. Introduction

AcceleratedDCTs.jl aims to provide the fastest possible Discrete Cosine Transform (DCT) for Julia, running on both CPUs and GPUs. It focuses on the **DCT-II** (Standard "DCT") and **DCT-III** (Inverse DCT), commonly used in signal processing and solving partial differential equations (PDEs).

The core innovation of this package is the implementation of **Algorithm 2 (2D)** and **Algorithm 3 (3D)**, which reduce $N$-dimensional DCTs to $N$-dimensional Real-to-Complex (R2C) FFTs with $O(N)$ pre/post-processing steps, avoiding the overhead of separable 1D transforms (which require redundant transposes).

---

## 2. Theory & Algorithms

### 2.1 The General Idea (Makhoul's Algorithm)

- See https://arxiv.org/abs/2110.01172 for detailed algorithms for 1D & 2D DCTs.
- CUDA C implementation: https://github.com/JeremieMelo/dct_cuda

A standard DCT-II of length $N$ can be computed by:
1.  **Preprocessing**: Permuting the input sequence $x$ into a new sequence $x'$.
    *   $x'$ takes even indices $0, 2, 4...$ from the front.
    *   $x'$ takes odd indices $1, 3, 5...$ from the back (reversed).
2.  **FFT**: Computing the Real FFT of $x'$.
3.  **Postprocessing**: Applying complex weights (twiddle factors) to the FFT output to recover DCT coefficients.

This approach is faster than $O(N^2)$ direct matrix multiplication and often faster than other $O(N \log N)$ approaches because highly optimized FFT libraries (FFTW, cuFFT) can be leveraged.

### 2.2 Algorithm 2 (2D)

For a 2D $N_1 \times N_2$ grid, we apply the permutation logic independently to both rows and columns.
*   **Input**: $x(n_1, n_2)$
*   **Permutation**: $x'(n_1, n_2) = x(\tau(n_1), \tau(n_2))$
*   **Transform**: $X = \text{RFFT}(x')$ (2D Real FFT)
*   **Reconstruction**: $y(n_1, n_2) = 2 \operatorname{Re}(\dots)$ involving sums of 4 symmetric points from $X$.

### 2.3 Algorithm 3 (3D)

We extend this to 3D.
*   **Separable Permutation**: $x'(n_1, n_2, n_3) = x(\tau(n_1), \tau(n_2), \tau(n_3))$.
    *   This scatters the spatial correlation but allows us to use a single 3D FFT.
*   **3D RFFT**: $X = \text{3D\_RFFT}(x')$.
*   **Recursive Reconstruction**:
    The post-processing extracts the cosine components.
    $$
    y = 2 \operatorname{Re} \{ W_3 \cdot [ W_2 \cdot ( W_1 \cdot X + \dots ) + \dots ] \}
    $$
    This is implemented efficiently in a single kernel pass in `dct_optimized.jl`.

---

## 3. Implementation Details

### 3.1 KernelAbstractions.jl
The code is written using `KernelAbstractions`, meaning the **exact same code** runs on:
*   CPU (using `Base.Threads`)
*   NVIDIA GPU (using `CUDA.jl`)
*   AMD GPU (using `AMDGPU.jl`) - *conceptual support, untested*

We strictly avoid "scalar indexing" (accessing single elements from the host), ensuring high performance on GPUs.

### 3.2 Plan-Based API (AbstractFFTs)
To maximize performance, we separate **resource allocation** (cheap on CPU, expensive on GPU) from **execution**.
*   **`plan_dct_opt(x)`**:
    *   Allocates temporary buffers (`tmp_real`, `tmp_comp`).
    *   Creates an internal FFT plan (`plan_rfft`).
    *   Pre-calculates twiddle factors (`cispi(...)`) on the device.
*   **`mul!(y, p, x)`**:
    *   Reuses all buffers.
    *   Zero memory allocation during execution.

---

## 4. Benchmarks

Measured on **NVIDIA RTX 2080 Ti**, grid size **384x384x384**.

| Implementation | Description | Time | Speedup vs Batched |
| :--- | :--- | :--- | :--- |
| **`dct_3d_opt`** | **This Package (Algorithm 3)** | **42 ms** | **1.8x** |
| `dct_fast` | Batched 1D Separable (Reference) | 77 ms | 1.0x |
| `cuFFT rfft` | Theoretical Lower Bound (FFT only) | 26 ms | - |

On CPU (Single Thread 128^3), `dct_3d_opt` is **~2.5x faster** than the batched approach.

---

## 5. API Reference

### 5.1 Plan Creation

#### `plan_dct_opt(x::AbstractArray, region=1:ndims(x))`
Creates an optimized DCT-II plan.
*   **x**: Input array (CPU Array or CuArray).
*   **Returns**: `DCTOptPlan`.

#### `plan_idct_opt(x::AbstractArray, region=1:ndims(x))`
Creates an optimized IDCT-III plan.
*   **Returns**: `IDCTOptPlan`.

### 5.2 Execution

#### `*(plan, x)`
Computes the transform, allocating a new output array.
```julia
y = p * x
```

#### `mul!(y, plan, x)`
Computes the transform in-place (updates `y`), utilizing pre-allocated plan buffers. **Zero allocation.**
```julia
mul!(y, p, x)
```

#### `\(plan, y)`
Computes the inverse transform (IDCT) using the cached inverse plan.
```julia
x_rec = p \ y
```

### 5.3 Convenience Functions
*   `dct_2d_opt(x)` / `idct_2d_opt(x)`
*   `dct_3d_opt(x)` / `idct_3d_opt(x)`

*Note: These functions create a new plan every call. Use `plan_dct_opt` for loops.*
