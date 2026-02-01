# Tutorial & Usage

## Getting Started

```julia
using AcceleratedDCTs
using FFTW  # Required for CPU execution
using CUDA  # Optional: for GPU execution

# Create random data (1D, 2D, or 3D)
x = rand(100) 
```

## High-Level API (Convenience)

The simplest way to use the package. Note that these functions creating a new plan every call, which has some overhead.

### Standard DCT (DCT-II / DCT-III)

Used for general signal processing and half-sample symmetric boundaries.

```julia
using AcceleratedDCTs: dct, idct

# Forward (DCT-II)
y = dct(x)

# Inverse (DCT-III)
x_rec = idct(y)
```

### Symmetric DCT (DCT-I / IDCT-I)

Used for whole-sample symmetric boundary conditions.

```julia
using AcceleratedDCTs: dct1, idct1

# Forward (DCT-I)
y = dct1(x)

# Inverse (IDCT-I)
x_rec = idct1(y)
```

> **Note**: `dct1` uses an efficient **Separable** algorithm by default ($O(M)$ memory). If you need the legacy **Mirroring** algorithm ($O(2M)$ memory), use `dct1_mirror` / `plan_dct1_mirror`.

---

## Performance API (Plan-Based)

For production code (e.g., inside loops), use the plan-based API to separate **resource allocation** from **execution**. This allows you to pre-allocate buffers and reuse them, achieving zero-allocation execution.

### 1. Create a Plan

Plans pre-calculate twiddle factors and allocate necessary temporary buffers.

```julia
using AcceleratedDCTs: plan_dct, plan_dct1

# For Standard DCT-II/III
p = plan_dct(x)

# For Symmetric DCT-I
p1 = plan_dct1(x)
```

### 2. Execute the Plan

Once you have a plan, you can execute it in multiple ways.

#### Out-of-Place (Allocating)
Creates a new output array.
```julia
y = p * x
```

#### In-Place (Zero Allocation)
Writes result directly to `y`. **Fastest method.**
```julia
using LinearAlgebra: mul!

y = similar(x)
mul!(y, p, x)
```

#### Inverse Transform
You can use the same plan to compute the inverse.

```julia
# Allocating Inverse
x_rec = p \ y

# Zero-Allocation Inverse (using ldiv!)
using LinearAlgebra: ldiv!
ldiv!(x_rec, p, y)

# Explicit Inverse Plan
pinv = inv(p)
mul!(x_rec, pinv, y)
```

---

## Advanced Usage

### GPU Support
Simply pass a `CuArray` (or `ROCArray`) to the functions. The package automatically selects the appropriate GPU kernel.

```julia
using CUDA
x_gpu = CUDA.rand(128, 128, 128)
p = plan_dct(x_gpu)
y_gpu = p * x_gpu
```

### Precision (Float32 vs Float64)
The package supports any `AbstractFloat` type. For maximum performance on GPUs, use `Float32`.

```julia
x_f32 = rand(Float32, 1024)
p = plan_dct(x_f32) # Creates a Float32 plan
```
