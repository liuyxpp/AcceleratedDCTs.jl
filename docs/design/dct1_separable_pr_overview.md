# feat: Optimized Separable DCT-I & IDCT-I (CPU & GPU)

## Summary
This PR introduces a major overhaul of the DCT-I and IDCT-I implementations in `AcceleratedDCTs.jl`. It replaces the memory-intensive "Mirroring" approach with a new **Separable Split-Radix** algorithm, optimized for both CPU (via FFTW) and GPU (via a novel Permuted Strategy).

## Key Comparison

| Feature | New Implementation (Separable) | Old Implementation (Mirroring) |
| :--- | :--- | :--- |
| **Algorithm** | Separable Split-Radix ($N=M-1$) | Mirroring ($N=2M-2$) |
| **Memory Complexity** | $O(M^D)$ (Efficient) | $O((2M)^D)$ (Excessive) |
| **GPU Strategy** | **Permuted Unit-Stride FFT** | standard `rfft` |
| **CPU backend** | Native `FFTW.REDFT00` | `FFTW.rfft` wrapper |
| **Performance ($129^3$)**| **5.71 ms** (GPU) | 7.83 ms (GPU) |

## Detailed Changes

### 1. Separable DCT-I (Generic/GPU)
- Implemented `src/dct1_separable.jl` as the default backend for GPU and Generic arrays.
- **GPU Optimization**: Uses a **Permuted Strategy** that dynamically permutes dimensions within kernels to ensure all internal FFTs are performed with **Unit Stride** along the first dimension. This bypasses the stride penalty of cuFFT on higher dimensions.
- **Performance**:
    - **~1.4x faster** than the Mirroring approach at optimal sizes (e.g., $129^3$).
    - Beats the raw throughput of a "Naive Complex FFT" of comparable complexity ($2M-2$).

### 2. Native FFTW DCT-I (CPU)
- Integrated `src/dct1_fftw.jl` to use `FFTW.r2r(..., REDFT00)` directly for `Array` types.
- Achieves parity with FFTW's highly optimized performance on CPU.

### 3. Legacy Mirroring
- Renamed the old implementation to `src/dct1_mirror.jl` (`plan_dct1_mirror`).
- Kept as a fallback and for validation purposes.

## Benchmarks (RTX 2080 Ti)

**Grid Size: $257^3$**
- **Complex FFT ($2M-2$)**: 80.73 ms (Baseline)
- **Mirror DCT-I**: 54.89 ms
- **New Separable DCT-I**: **43.50 ms**

The new implementation is **1.3x faster** than the previous method and significantly more memory efficient.

## Verification
- Added `test/test_dct1_separable.jl` covering 1D, 2D, 3D cases for consistency with FFTW and roundtrip accuracy.
- Verified generic fallback and GPU execution.
