# VkDCT Development Guide

This document explains the internal architecture of `VkDCT`, how it integrates VkFFT, and the logic behind the C++ shim and Julia wrapper.

## Architecture Overview

```mermaid
graph TD
    Julia[Julia Application] -->|VkDCT.jl| Wrapper[VkDCT Wrapper]
    Wrapper -->|ccall| Shim[libvkfft_dct.so (C++ Shim)]
    Shim -->|VkFFT API| VkFFT[VkFFT Library]
    VkFFT -->|Direct Calls| CUDA[CUDA Driver / NVRTC]
    CUDA --> GPU[NVIDIA GPU]
```

## 1. VkFFT Compilation Strategy

VkFFT is a header-only library (`vkFFT.h`), but it requires backend-specific configuration at compile time.

- **Backend selection**: Defined via preprocessor macros. We use `#define VKFFT_BACKEND 1` for CUDA.
- **Dependencies**:
  - `cuda_runtime.h`: For CUDA runtime types (e.g., `cudaStream_t`).
  - `nvrtc`: NVIDIA Runtime Compilation library. VkFFT compiles kernels at runtime for optimal performance on the specific GPU and problem size.
  - `cuda` (driver): For using the CUDA driver API (`cuInit`, `cuDeviceGet`).

Our `compile.sh` script handles this:
```bash
nvcc -O3 --shared -Xcompiler -fPIC \
     -arch=sm_75 \
     -o libvkfft_dct.so dct_shim.cu \
     -I. -lcuda -lnvrtc
```
- `-shared -fPIC`: Generates a dynamic shared object (`.so`) loadable by Julia's `Libdl`.
- `-lcuda -lnvrtc`: Links essential CUDA libraries required by VkFFT.

## 2. dct_shim.cu Details

The C++ shim acts as a bridge between Julia's C interface (`ccall`) and VkFFT's C-like configuration structs.

### Data Structures
- **`VkDCTContext`**: Persists state across calls. It holds:
  - `VkFFTApplication`: The compiled VkFFT plan/app.
  - `VkFFTConfiguration`: Configuration parameters (dims, type, etc.).
  - `CUdevice`: Handle to the CUDA device.

### Key Functions

1.  **`create_dct3d_plan(nx, ny, nz, precision)`**:
    -   Initializes CUDA Driver API (`cuInit`).
    -   Configures `VkFFTConfiguration`:
        -   `FFTdim = 3`: 3D transform.
        -   `performDCT = 1`: Enables DCT-I ("REDFT00").
        -   `doublePrecision = 1`: Enabled if `precision == 1` (Float64).
        -   `normalize = 0`: Disables internal 1/N scaling. We handle normalization in Julia for flexibility.
    -   Calls `initializeVkFFT` to compile kernels.

2.  **`exec_dct3d(plan, buffer, stream, inverse)`**:
    -   Updates `app.configuration.stream` to the current Julia CUDA stream.
    -   Sets up `VkFFTLaunchParams` pointing to the I/O `buffer`.
    -   Calls `VkFFTAppend` to execute the kernel.
    -   **Note**: DCT-I is its own inverse (up to a scale factor), so `inverse` flag primarily affects internal normalization checks (which we disabled), but we pass it correctly for semantic correctness.

## 3. VkDCT.jl Wrapper

The Julia wrapper provides a high-level, idiomatic interface.

- **`VkDCTPlan{T}`**: Wraps the `void*` context pointer.
    -   **Finalizer**: Registers `destroy_plan!` to avoid GPU memory leaks when the plan goes out of scope.
    -   **Type Parameter `T`**: Ensures type safety (`Float32` vs `Float64`) matching the C++ plan configuration.
- **`mul!(Y, p, X)`**:
    -   Checks size compatibility.
    -   Handles in-place vs. out-of-place (via `copyto!`).
    -   Obtains the current CUDA stream via `CUDA.stream()`.
    -   Calls `exec_dct3d` using `ccall` with `CuPtr{T}` to pass the device pointer.
- **`ldiv!(Y, p, X)`**:
    -   Inverse transform implementation.
    -   Calls `exec_dct3d` with `inverse=1`.
    -   Applies scalar normalization: $1 / (8(N_x-1)(N_y-1)(N_z-1))$.
    -   Uses in-place broadcasting (`.*=`) for efficiency.
