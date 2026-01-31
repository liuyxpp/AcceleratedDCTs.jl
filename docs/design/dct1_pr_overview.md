# Feature: Optimized N-dimensional DCT-I / IDCT-I Support

## Summary

This PR introduces comprehensive support for **DCT-I (Discrete Cosine Transform Type-I)** and its inverse (**IDCT-I**) across 1D, 2D, and 3D arrays. The implementation is fully integrated into the existing plan-based API and supports both CPU and GPU execution.

Significantly, for CPU arrays, we implement an automatic dispatch to **FFTW's native `REDFT00`**, ensuring our performance strictly matches the state-of-the-art FFTW implementation. For GPUs (and generic AbstractArrays), we provide a highly optimized fallback using "Mirroring + R2C FFT", enabling DCT-I support on devices (like CUDA) that lack native DCT-I solvers.

## Key Changes

### 1. New API
- Added `dct1`, `idct1` for immediate execution.
- Added `plan_dct1`, `plan_idct1` for performance-critical code.
- Added `DCT1Plan`, `IDCT1Plan` types compatible with `AbstractFFTs`.

### 2. Implementation Details (`src/`)
- **Generic / GPU**: Implemented in [src/dct1_optimized.jl](cci:7://file:///home/lyx/Develop/AcceleratedDCTs.jl/src/dct1_optimized.jl:0:0-0:0) using a "Mirroring + Real-to-Complex FFT" approach via `KernelAbstractions.jl`. This enables DCT-I on NVIDIA, AMD, and Intel GPUs.
- **CPU Optimized**: Implemented in [src/dct1_fftw.jl](cci:7://file:///home/lyx/Develop/AcceleratedDCTs.jl/src/dct1_fftw.jl:0:0-0:0) using `FFTW.REDFT00`. The system automatically dispatches standard `Array` inputs to this path for maximum performance.

### 3. Dependencies
- Added `FFTW.jl` as a direct dependency to support the optimized CPU path.

### 4. Tests (`test/`)
- Added [test_dct1_optimized.jl](cci:7://file:///home/lyx/Develop/AcceleratedDCTs.jl/test/test_dct1_optimized.jl:0:0-0:0) and [test_idct1_optimized.jl](cci:7://file:///home/lyx/Develop/AcceleratedDCTs.jl/test/test_idct1_optimized.jl:0:0-0:0).
- Validated:
    - Consistency with FFTW reference.
    - Roundtrip accuracy (`idct1(dct1(x)) ≈ x`).
    - Linearity and Float32 support.
    - Correct dispatch for CPU plans.

### 5. Documentation & Benchmarks (`docs/`, `benchmark/`)
- Updated **Theory**: Added DCT-I mathematical definition.
- Updated **Tutorial**: Added "Symmetric DCT" section.
- Updated **Benchmarks**:
    - **CPU**: Confirmed matching performance with native FFTW.
    - **GPU**: Benchmarked low-overhead performance (<5% overhead vs raw FFT).

## Performance Highlights

| Hardware | Implementation | Comparison |
| :--- | :--- | :--- |
| **CPU** | Native FFTW Wrapper | **~1x** vs FFTW `r2r` (Identical) |
| **GPU** | Mirroring + FFT | **~1.05x** overhead vs raw `cuFFT` |

## Checklist

- [x] All tests passed.
- [x] Documentation updated (Tutorial, Implementation, Benchmarks).
- [x] Benchmarks expanded.
- [x] `Project.toml` updated (added FFTW).