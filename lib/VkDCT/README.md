# VkDCT.jl - High-Performance GPU DCT-I Library

VkDCT.jl provides a Julia interface to a high-performance 3D DCT-I implementation powered by [VkFFT](https://github.com/DTolm/VkFFT). It supports generic GPU backends (currently optimized for CUDA) and offers significant performance improvements over generic FFT-based approaches.

## Features

- **High Performance**: ~10x-15x faster than generic CUDA FFT approaches for Float32, ~2x faster for Float64.
- **Support for Arbitrary Sizes**: Handles non-power-of-2, prime, and combined sizes efficiently (e.g., $129^3$).
- **Double Precision**: Supports both `Float32` and `Float64`.
- **In-place Operations**: Optimized memory usage with `mul!` and `ldiv!`.
- **Standard Interface**: Follows `AbstractFFTs`/`LinearAlgebra` patterns.

## Installation & Compilation

VkDCT requires compilation of the C++ shim library `libvkfft_dct.so` before use.

### Prerequisites
- CUDA Toolkit (nvcc)
- CMake (optional, for VkFFT build system, currently using direct `compile.sh`)
- Julia 1.9+ with `CUDA.jl`

### Compile
1. Navigate to the library directory:
   ```bash
   cd lib/VkDCT
   ```
2. Run the compilation script:
   ```bash
   bash compile.sh
   # This generates libvkfft_dct.so in the same directory
   ```

## Usage

```julia
using CUDA
include("lib/VkDCT/VkDCT.jl")
using .VkDCT

# 1. Create Data (3D Array)
nx, ny, nz = 129, 129, 129
data = CUDA.rand(Float32, nx, ny, nz) # or Float64

# 2. Create Plan
p = plan_dct(data) 
# or p = VkDCT.plan_dct((nx, ny, nz), Float32)

# 3. Forward Transform (Unnormalized by default)
out = p * data
# or in-place:
mul!(data, p, data)

# 4. Inverse Transform (Normalized)
# idct(x) performs the inverse transform and applies 
# the 1/(8*(Nx-1)(Ny-1)(Nz-1)) scaling factor.
recovered = p \ out
# or explicit:
idct_out = idct(out)

# 5. Clean up
VkDCT.destroy_plan!(p) # Optional, handled by GC finalizer
```

## Performance Tips

- **Reuse Plans**: Creating a plan involves compiling CUDA kernels (runtime compilation via NVRTC). Always reuse plans for the same size/type.
- **In-place**: Use `mul!(A, p, A)` to avoid allocating new GPU memory.
- **Precision**: Use `Float32` unless `Float64` is strictly required. `Float32` on consumer GPUs (like RTX series) is significantly faster.
