# PR: VkDCT Integration (High-Performance GPU DCT-I)

## Summary

This PR integrates `VkDCT`, a high-performance C++/CUDA backend for 3D DCT-I transforms, into `AcceleratedDCTs.jl`. It leverages `VkFFT` (via a custom shim) to achieve massive performance gains over the existing generic separable implementation on NVIDIA GPUs.

The integration is implemented as a **Julia Package Extension** (`VkDCTExt`), which automatically activates when `CUDA.jl` is loaded.

## Key Highlights

-   **Backend**: Introduced `VkDCTExt`, wrapping a minimal C++ shim (`libvkfft_dct.so`) that calls `VkFFT`.
-   **Performance**: Achieves **~15x speedup** (Float32) and **~2x speedup** (Float64) compared to the native Julia separable implementation on a standard 3D grid.
-   **Architecture**:
    -   Uses `Package Extensions` (Julia 1.9+) to keep the core package lightweight.
    -   Adds `CUDA` as a weak dependency.
    -   Only overloads `plan_dct1` / `plan_idct1` for `CuArray{T, 3}`.

## ⚠️ Breaking Changes

This PR changes the default behavior for `plan_dct1` with `CuArray` inputs.

-   **Constraint**: The pure Julia/KernelAbstractions implementation for GPU is **replaced** by the `VkDCT` backend when `CUDA` is loaded.
-   **Requirement**: Users **MUST** manually compile the underlying C++ library (`libvkfft_dct.so`) for the function to work.
    -   If the library is missing, `plan_dct1(x::CuArray)` will now throw an error instructing the user to compile it, whereas previously it would have run (slower) using the generic implementation.
    -   _Mitigation_: Users can still force the old path by strictly not creating the library, but the error message is explicit about the expectation.

## Detailed Changes

### 1. New Components
-   **`lib/VkDCT/`**: Contains the C++ source code (`dct_shim.cu`) and build script (`compile.sh`).
-   **`ext/VkDCTExt.jl`**: The extension module that implements `VkFFTDCT1Plan` and overloads `AcceleratedDCTs.plan_dct1`.

### 2. Configuration
-   **`Project.toml`**:
    -   Added `CUDA` and `Libdl` to `[extras]` / `[weakdeps]`.
    -   Defined `VkDCTExt` extension.

### 3. Documentation
-   Updated `README.md` with:
    -   "VkDCT Backend" feature highlight.
    -   Setup instructions for compiling the library.
-   Updated `docs/src/30-implementation.md` and `docs/src/40-benchmarks.md` with detailed architecture and performance data.

### 4. Tests
-   Added `test/test_vkdct_float64.jl`: Verifies the extension integration, float64 precision, and IDCT correctness.

## Performance Benchmark (Float32, $129^3$)

| Implementation | Time | Speedup vs Baseline |
| :--- | :--- | :--- |
| **AcceleratedDCTs (Separable)** | 2.765 ms | 1.0x (Baseline) |
| **VkDCT (Extension)** | **0.176 ms** | **15.7x** |
| **cuFFT (Ref)** | 1.063 ms | - |

## How to Test

1.  Check out this branch.
2.  Compile the library:
    ```bash
    cd lib/VkDCT
    ./compile.sh
    ```
3.  Run tests:
    ```julia
    using Pkg; Pkg.test("AcceleratedDCTs")
    ```
    (Note: Requires `CUDA` environment)
