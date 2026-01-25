# Tutorial & Usage

## Overview
This package provides both a high-level `dct`/`idct` interface and a performance-oriented plan-based interface.

## Quick Start
```julia
using AcceleratedDCTs
using FFTW # Required for CPU

x = rand(100)
y = dct(x)
x_rec = idct(y)
```

## Plan-Based API (Recommended)

To maximize performance, especially for repeated transforms, we separate **resource allocation** (cheap on CPU, expensive on GPU) from **execution**.

### 1. Plan Creation

#### `plan_dct(x::AbstractArray, region=1:ndims(x))`
Creates an optimized DCT-II plan.
*   **x**: Input array (CPU Array or CuArray).
*   **Returns**: `DCTPlan`.

#### `plan_idct(x::AbstractArray, region=1:ndims(x))`
Creates an optimized IDCT-III plan.
*   **Returns**: `IDCTPlan`.

### 2. Execution

#### Out-of-place: `*(plan, x)`
Computes the transform, allocating a new output array.
```julia
using AcceleratedDCTs: plan_dct
p = plan_dct(x)
y = p * x
```

#### In-place: `mul!(y, plan, x)`
Computes the transform in-place (updates `y`), utilizing pre-allocated plan buffers. **Zero allocation.**
```julia
using LinearAlgebra: mul!
mul!(y, p, x)
```

#### Inverse: `\(plan, y)`
Computes the inverse transform (IDCT) using the cached inverse plan.
```julia
x_rec = p \ y
```

### 3. Convenience Functions
*   `dct(x)`
*   `idct(x)`

*Note: These functions create a new plan every call. Use `plan_dct` for loops.*
