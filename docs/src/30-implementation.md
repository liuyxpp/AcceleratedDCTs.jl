# Implementation Details

## KernelAbstractions.jl
The code is written using `KernelAbstractions`, meaning the **exact same code** runs on:
*   CPU (using `Base.Threads`)
*   NVIDIA GPU (using `CUDA.jl`)
*   AMD GPU (using `AMDGPU.jl`) - *conceptual support, untested*

We strictly avoid "scalar indexing" (accessing single elements from the host), ensuring high performance on GPUs.

## Plan-Based API (AbstractFFTs)
To maximize performance, we separate **resource allocation** (cheap on CPU, expensive on GPU) from **execution**.

### DCT-II/DCT-III Plans
*   **`plan_dct(x)`** / **`plan_idct(x)`**:
    *   Allocates temporary buffers (`tmp_real`, `tmp_comp`).
    *   Creates an internal FFT plan (`plan_rfft`).
    *   Pre-calculates twiddle factors (`cispi(...)`) on the device.

### DCT-I Plans (VkDCT / Extension)
*   **`plan_dct1(x)`** via **`VkDCTExt`**:
    *   **Backend**: Uses custom `VkFFT` based C++ library (`libvkfft_dct.so`) loaded via `VkDCTExt` extension.
    *   **Availability**: Triggered automatically when `CUDA.jl` is loaded and the library is available.
    *   **Performance**: Extremely fast (~7x-15x faster than Separable) due to hand-tuned kernels and reduced memory traffic.
    *   **Features**: Supports `Float32`/`Float64`, full 3D transforms.

### DCT-I Plans (Generic / Separable)
*   **`plan_dct1(x)`** / **`plan_idct1(x)`**:
    *   **Default (Generic GPU)**: Uses **Separable Split-Radix** algorithm.
        *   Maps $M$ points to size $N=M-1$ Complex FFT.
        *   **GPU Strategy**: Uses a **Permuted** approach to ensure Unit Stride memory access for internal FFTs across all dimensions, maximizing performance.
        *   Memory scaling: $O(M^D)$ (efficient for N-D).
    *   **CPU (`Array`)**: Uses FFTW's native `REDFT00`.
    *   Dispatch is automatic.

### DCT-I Plans (Mirroring / Legacy)
*   **`plan_dct1_mirror(x)`** / **`plan_idct1_mirror(x)`**:
    *   Uses **Mirroring** strategy.
    *   Maps $M$ points to size $2M-2$ Real FFT.
    *   Memory scaling: $O((2M)^D)$ (expensive for N-D).
    *   Available as an alternative for validation.

### Execution
*   **`mul!(y, p, x)`**:
    *   Reuses all buffers.
    *   Zero memory allocation during execution.
