# Implementation Details

## KernelAbstractions.jl
The code is written using `KernelAbstractions`, meaning the **exact same code** runs on:
*   CPU (using `Base.Threads`)
*   NVIDIA GPU (using `CUDA.jl`)
*   AMD GPU (using `AMDGPU.jl`) - *conceptual support, untested*

We strictly avoid "scalar indexing" (accessing single elements from the host), ensuring high performance on GPUs.

## Plan-Based API (AbstractFFTs)
To maximize performance, we separate **resource allocation** (cheap on CPU, expensive on GPU) from **execution**.
*   **`plan_dct(x)`**:
    *   Allocates temporary buffers (`tmp_real`, `tmp_comp`).
    *   Creates an internal FFT plan (`plan_rfft`).
    *   Pre-calculates twiddle factors (`cispi(...)`) on the device.
*   **`mul!(y, p, x)`**:
    *   Reuses all buffers.
    *   Zero memory allocation during execution.
