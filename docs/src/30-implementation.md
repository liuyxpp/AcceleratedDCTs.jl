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

### DCT-I Plans
*   **`plan_dct1(x)`** / **`plan_idct1(x)`**:
    *   **CPU (`Array`)**: Uses FFTW's native `REDFT00` for optimal performance.
    *   **GPU (`CuArray`)**: Uses mirroring buffer (size `2M-2`) + R2C FFT.
    *   Dispatch is automatic based on array type.

### Execution
*   **`mul!(y, p, x)`**:
    *   Reuses all buffers.
    *   Zero memory allocation during execution.
