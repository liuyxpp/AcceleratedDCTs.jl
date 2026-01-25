# Optimized N-D DCT/IDCT Implementation
# 
# Features:
# - Uses R2C FFT (rfft) for memory efficiency
# - Separable implementation using Batched 1D DCT on first dimension
# - Uses permutations for other dimensions
# - Device agnostic via KernelAbstractions.jl
# - Optimized with plan caching, buffer reuse, and precomputed twiddles

# ============================================================================
# Batched 1D DCT/IDCT Kernels (operate on first dimension)
# ============================================================================

"""
Preprocess batch of 1D signals along first dimension.
x: (N, Batch), v: (N, Batch)
"""
@kernel function preprocess_batch_dim1_kernel!(v, @Const(x), N, halfN, Batch)
    I = @index(Global, Cartesian)
    n = I[1] - 1  # 0-based index in N
    b = I[2]      # 1-based index in Batch
    
    if n < N
        src = n < halfN ? 2*n : 2*N - 2*n - 1
        @inbounds v[n + 1, b] = x[src + 1, b]
    end
end

"""
Postprocess batch of 1D signals with precomputed twiddles.
V: (halfN+1, Batch) complex, y: (N, Batch) real
twiddles: (N,) complex
"""
@kernel function postprocess_batch_dim1_kernel!(y, @Const(V), @Const(twiddles), N, halfN, Batch)
    I = @index(Global, Cartesian)
    k = I[1] - 1  # 0-based
    b = I[2]
    
    if k < N
        # Get V[k] using Hermitian symmetry along first dim
        if k <= halfN
            Vk = V[k + 1, b]
        else
            Vk = conj(V[N - k + 1, b])
        end
        
        # Apply precomputed twiddle factor
        # expk = cispi(-k / (2*N))
        @inbounds expk = twiddles[k + 1]
        
        @inbounds y[k + 1, b] = real(Vk * expk)
    end
end

"""
IDCT Preprocess batch with precomputed twiddles.
y: (N, Batch) real, V: (halfN+1, Batch) complex
twiddles_inv: (halfN+1,) complex (cispi(k/(2N)))
"""
@kernel function idct_preprocess_batch_dim1_kernel!(V, @Const(y), @Const(twiddles_inv), N, halfN, Batch)
    I = @index(Global, Cartesian)
    k = I[1] - 1
    b = I[2]
    
    if k <= halfN
        if k == 0
            @inbounds V[1, b] = Complex(y[1, b], zero(eltype(y)))
        elseif k == halfN
            @inbounds V[k + 1, b] = Complex(y[k + 1, b] * sqrt(eltype(y)(2)), zero(eltype(y)))
        else
            @inbounds expk_inv = twiddles_inv[k + 1]
            @inbounds V[k + 1, b] = (y[k + 1, b] - im * y[N - k + 1, b]) * expk_inv
        end
    end
end

"""
IDCT Postprocess batch (inverse reorder).
v: (N, Batch), x: (N, Batch)
"""
@kernel function idct_postprocess_batch_dim1_kernel!(x, @Const(v), N, halfN, Batch)
    I = @index(Global, Cartesian)
    n = I[1] - 1
    b = I[2]
    
    if n < N
        dst = n < halfN ? 2*n : 2*N - 2*n - 1
        @inbounds x[dst + 1, b] = v[n + 1, b]
    end
end

# ============================================================================
# Helpers
# ============================================================================

function get_twiddles(N::Int, ::Type{T}, backend) where T
    # Compute cispi(-k / (2N)) for k in 0:N-1
    # We compute this on CPU and move to backend, or compute on backend if possible.
    # Since N is small (e.g. 256), CPU compute + copy is fine.
    
    # Forward twiddles: cispi(-k / (2N))
    data = Vector{Complex{T}}(undef, N)
    for k in 0:N-1
        data[k+1] = cispi(-T(k) / (2N))
    end
    
    # Move to backend
    dest = KernelAbstractions.allocate(backend, Complex{T}, N)
    copyto!(dest, data)
    return dest
end

function get_inv_twiddles(N::Int, ::Type{T}, backend) where T
    # Inverse twiddles: cispi(k / (2N)) for k in 0:halfN
    halfN = N ÷ 2
    data = Vector{Complex{T}}(undef, halfN + 1)
    for k in 0:halfN
        data[k+1] = cispi(T(k) / (2N))
    end
    
    dest = KernelAbstractions.allocate(backend, Complex{T}, halfN + 1)
    copyto!(dest, data)
    return dest
end

# ============================================================================
# In-place permutedims kernels (GPU-compatible)
# ============================================================================

"""
3D permutedims! kernel: dst[i,j,k] = src[perm_inverse...]
perm specifies the output dimension order from input dimensions.
"""
@kernel function permutedims_3d_kernel!(dst, @Const(src), perm1, perm2, perm3)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    
    # Build source index based on permutation
    # perm = (p1, p2, p3) means dst[i,j,k] = src[idx[p1], idx[p2], idx[p3]]
    # where idx = (i, j, k)
    idx = (i, j, k)
    @inbounds dst[i, j, k] = src[idx[perm1], idx[perm2], idx[perm3]]
end

"""
    permutedims_3d!(dst, src, perm)

In-place 3D permutedims using KernelAbstractions. 
Zero allocation after initial kernel compilation.
"""
function permutedims_3d!(dst::AbstractArray{T,3}, src::AbstractArray{T,3}, perm::NTuple{3,Int}) where T
    be = get_backend(dst)
    ndrange = size(dst)
    
    # Convert perm to inverse permutation for indexing
    # If perm = (2,1,3), then dst[i,j,k] = src[j,i,k]
    # We need inv_perm such that inv_perm[perm[d]] = d
    inv_perm = ntuple(d -> findfirst(==(d), perm), 3)
    
    permutedims_3d_kernel!(be)(dst, src, inv_perm[1], inv_perm[2], inv_perm[3]; ndrange=ndrange)
    KernelAbstractions.synchronize(be)
    return dst
end

"""
2D permutedims! kernel
"""
@kernel function permutedims_2d_kernel!(dst, @Const(src))
    I = @index(Global, Cartesian)
    i, j = I[1], I[2]
    @inbounds dst[i, j] = src[j, i]
end

"""
    permutedims_2d!(dst, src)

In-place 2D transpose using KernelAbstractions.
"""
function permutedims_2d!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}) where T
    be = get_backend(dst)
    ndrange = size(dst)
    permutedims_2d_kernel!(be)(dst, src; ndrange=ndrange)
    KernelAbstractions.synchronize(be)
    return dst
end

# ============================================================================
# DCTPlan: Precomputed plans, twiddles, and buffers for N-D DCT/IDCT
# ============================================================================

"""
    DCTBatchedPlan{T, N}

Precomputed plan for N-dimensional DCT/IDCT operations (Batched).

Contains cached:
- FFT plans for each dimension
- Twiddle factors for each dimension
- Work buffers to avoid allocations
- Temp buffer for zero-allocation mul! operations

# Usage
```julia
plan = plan_dct_batched(x)    # Create plan
y = plan * x                  # Compute DCT
mul!(y, plan, x)              # Compute DCT (zero allocation)
x_rec = plan \\ y              # Compute IDCT
ldiv!(x, plan, y)             # Compute IDCT (zero allocation)
```
"""
struct DCTBatchedPlan{T, N}
    # Original array size
    dims::NTuple{N, Int}
    
    # Twiddles for each dimension (indexed by dim size)
    twiddles::Dict{Int, Any}
    twiddles_inv::Dict{Int, Any}
    
    # FFT plans for each (N, Batch) configuration
    # Key: (N, Batch)
    fft_plans::Dict{Tuple{Int, Int}, Any}
    ifft_plans::Dict{Tuple{Int, Int}, Any}
    
    # Work buffers for each (N, Batch) configuration
    # v_buffers: real buffer (N, Batch)
    # V_buffers: complex buffer (N÷2+1, Batch)
    v_buffers::Dict{Tuple{Int, Int}, Any}
    V_buffers::Dict{Tuple{Int, Int}, Any}
    
    # Temp buffer for ping-pong operations in mul!
    # Only 1 buffer needed - output y is used as the other buffer
    temp_buffer::Any
    
    # Backend
    backend::Any
end

"""
    plan_dct_batched(x::AbstractArray{T, N}) where {T <: Real, N}

Create a batched DCT plan for arrays with the same size and element type as `x`.

# Example
```julia
x = rand(256, 256, 256)
plan = plan_dct_batched(x)
y = plan * x   # DCT
mul!(y, plan, x)  # DCT (zero allocation)
x2 = plan \\ y  # IDCT
```
"""
function plan_dct_batched(x::AbstractArray{T, N}) where {T <: Real, N}
    dims = size(x)
    backend = get_backend(x)
    
    # Initialize dictionaries with Any value type for simplicity
    twiddles = Dict{Int, Any}()
    twiddles_inv = Dict{Int, Any}()
    fft_plans = Dict{Tuple{Int, Int}, Any}()
    ifft_plans = Dict{Tuple{Int, Int}, Any}()
    v_buffers = Dict{Tuple{Int, Int}, Any}()
    V_buffers = Dict{Tuple{Int, Int}, Any}()
    
    # Precompute twiddles for all unique dimensions
    unique_dims = unique(dims)
    for dim_size in unique_dims
        if !haskey(twiddles, dim_size)
            twiddles[dim_size] = get_twiddles(dim_size, T, backend)
            twiddles_inv[dim_size] = get_inv_twiddles(dim_size, T, backend)
        end
    end
    
    # Precompute plans and buffers for each dimension's batch configuration
    # For N-D array, transforming dim d means:
    #   - Permute so dim d is first
    #   - Batch size = prod of other dims
    for d in 1:N
        dim_size = dims[d]
        batch_size = prod(dims) ÷ dim_size
        key = (dim_size, batch_size)
        
        if !haskey(fft_plans, key)
            v_buf = similar(x, (dim_size, batch_size))
            V_buf = similar(x, Complex{T}, (dim_size ÷ 2 + 1, batch_size))
            
            v_buffers[key] = v_buf
            V_buffers[key] = V_buf
            fft_plans[key] = plan_rfft(v_buf, 1)
            ifft_plans[key] = plan_irfft(V_buf, dim_size, 1)
        end
    end
    
    # Create temp buffer for mul! ping-pong operations
    # Only 1 buffer needed - output y acts as the second buffer
    temp_buffer = similar(x)
    
    return DCTBatchedPlan{T, N}(
        dims, twiddles, twiddles_inv, fft_plans, ifft_plans, v_buffers, V_buffers, temp_buffer, backend
    )
end

# ============================================================================
# Core Batched Functions with Buffers
# ============================================================================

function dct_batch_dim1!(y, x, v_buf, V_buf, twiddles, plan)
    # x: input (N, Batch) or reshaped view
    # y: output (N, Batch) or reshaped view
    # v_buf: (N, Batch) real buffer
    # V_buf: (N÷2+1, Batch) complex buffer
    # twiddles: (N,) complex precomputed
    
    dims = size(x)
    N = dims[1]
    Batch = prod(dims[2:end])
    halfN = N ÷ 2
    
    be = get_backend(x)
    
    # 1. Preprocess: x -> v_buf
    preprocess_batch_dim1_kernel!(be)(v_buf, x, N, halfN, Batch; ndrange=(N, Batch))
    KernelAbstractions.synchronize(be)
    
    # 2. R2C FFT: v_buf -> V_buf
    # Use mul! with plan if available, otherwise regular rfft
    if plan !== nothing
        mul!(V_buf, plan, v_buf)
    else
        # If no plan provided, we have to assume V_buf is large enough
        # But standard rfft allocates. So we can't use V_buf easily without a plan or in-place fft.
        # Fallback to allocation if no plan (should be avoided in fast path)
        V_out = rfft(v_buf, 1)
        # copy to V_buf to maintain interface? Or just use V_out
        # For simplicity, optimize strictly for plan case.
        copyto!(V_buf, V_out)
    end
    
    # 3. Postprocess: V_buf -> y
    postprocess_batch_dim1_kernel!(be)(y, V_buf, twiddles, N, halfN, Batch; ndrange=(N, Batch))
    KernelAbstractions.synchronize(be)
    
    return y
end

function idct_batch_dim1!(x, y, v_buf_N, V_buf, twiddles_inv, plan_inv)
    # y: input (N, Batch)
    # x: output (N, Batch)
    # v_buf_N: (N, Batch) real buffer (for IRFFT output)
    # V_buf: (N÷2+1, Batch) complex buffer (for Preprocess output)
    
    dims = size(y)
    N = dims[1]
    Batch = prod(dims[2:end])
    halfN = N ÷ 2
    
    be = get_backend(y)
    
    # 1. Preprocess: y -> V_buf
    idct_preprocess_batch_dim1_kernel!(be)(V_buf, y, twiddles_inv, N, halfN, Batch; ndrange=(halfN+1, Batch))
    KernelAbstractions.synchronize(be)
    
    # 2. IRFFT: V_buf -> v_buf_N
    if plan_inv !== nothing
        mul!(v_buf_N, plan_inv, V_buf)
    else
        v_out = irfft(V_buf, N, 1)
        copyto!(v_buf_N, v_out)
    end
    
    # 3. Postprocess: v_buf_N -> x
    idct_postprocess_batch_dim1_kernel!(be)(x, v_buf_N, N, halfN, Batch; ndrange=(N, Batch))
    KernelAbstractions.synchronize(be)
    
    return x
end

# ============================================================================
# DCTPlan Operations (using cached resources)
# ============================================================================

"""
    Base.:*(plan::DCTBatchedPlan, x::AbstractArray) -> y

Compute the N-dimensional DCT of `x` using the precomputed plan.
"""
function Base.:*(plan::DCTBatchedPlan{T, 1}, x::AbstractVector{T}) where T
    N = plan.dims[1]
    Batch = 1
    key = (N, Batch)
    
    v_buf = plan.v_buffers[key]
    V_buf = plan.V_buffers[key]
    twiddles = plan.twiddles[N]
    fft_plan = plan.fft_plans[key]
    
    y = similar(x)
    x_reshaped = reshape(x, N, 1)
    y_reshaped = reshape(y, N, 1)
    
    dct_batch_dim1!(y_reshaped, x_reshaped, v_buf, V_buf, twiddles, fft_plan)
    return y
end

function Base.:*(plan::DCTBatchedPlan{T, 2}, x::AbstractMatrix{T}) where T
    N1, N2 = plan.dims
    
    # Step 1: DCT along dim 1
    Batch1 = N2
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles1 = plan.twiddles[N1]
    fft_plan1 = plan.fft_plans[key1]
    
    t1 = similar(x)
    dct_batch_dim1!(reshape(t1, N1, Batch1), reshape(x, N1, Batch1), v_buf1, V_buf1, twiddles1, fft_plan1)
    
    # Step 2: DCT along dim 2 (permute, transform, permute back)
    t1_p = permutedims(t1, (2, 1))  # (N2, N1)
    
    Batch2 = N1
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles2 = plan.twiddles[N2]
    fft_plan2 = plan.fft_plans[key2]
    
    t2 = similar(t1_p)
    dct_batch_dim1!(reshape(t2, N2, Batch2), reshape(t1_p, N2, Batch2), v_buf2, V_buf2, twiddles2, fft_plan2)
    
    return permutedims(t2, (2, 1))
end

function Base.:*(plan::DCTBatchedPlan{T, 3}, x::AbstractArray{T, 3}) where T
    N1, N2, N3 = plan.dims
    
    # Step 1: DCT along dim 1
    Batch1 = N2 * N3
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles1 = plan.twiddles[N1]
    fft_plan1 = plan.fft_plans[key1]
    
    t1 = similar(x)
    dct_batch_dim1!(reshape(t1, N1, Batch1), reshape(x, N1, Batch1), v_buf1, V_buf1, twiddles1, fft_plan1)
    
    # Step 2: DCT along dim 2
    t1_p = permutedims(t1, (2, 1, 3))  # (N2, N1, N3)
    Batch2 = N1 * N3
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles2 = plan.twiddles[N2]
    fft_plan2 = plan.fft_plans[key2]
    
    t2 = similar(t1_p)
    dct_batch_dim1!(reshape(t2, N2, Batch2), reshape(t1_p, N2, Batch2), v_buf2, V_buf2, twiddles2, fft_plan2)
    
    # Step 3: DCT along dim 3
    t2_p = permutedims(t2, (3, 2, 1))  # (N3, N1, N2) after reordering from (N2, N1, N3)
    Batch3 = N1 * N2
    key3 = (N3, Batch3)
    v_buf3 = plan.v_buffers[key3]
    V_buf3 = plan.V_buffers[key3]
    twiddles3 = plan.twiddles[N3]
    fft_plan3 = plan.fft_plans[key3]
    
    t3 = similar(t2_p)
    dct_batch_dim1!(reshape(t3, N3, Batch3), reshape(t2_p, N3, Batch3), v_buf3, V_buf3, twiddles3, fft_plan3)
    
    # Permute back to original order
    return permutedims(t3, (2, 3, 1))
end

"""
    Base.:\\(plan::DCTBatchedPlan, y::AbstractArray) -> x

Compute the N-dimensional IDCT of `y` using the precomputed plan.
"""
function Base.:\(plan::DCTBatchedPlan{T, 1}, y::AbstractVector{T}) where T
    N = plan.dims[1]
    Batch = 1
    key = (N, Batch)
    
    v_buf = plan.v_buffers[key]
    V_buf = plan.V_buffers[key]
    twiddles_inv = plan.twiddles_inv[N]
    ifft_plan = plan.ifft_plans[key]
    
    x = similar(y)
    x_reshaped = reshape(x, N, 1)
    y_reshaped = reshape(y, N, 1)
    
    idct_batch_dim1!(x_reshaped, y_reshaped, v_buf, V_buf, twiddles_inv, ifft_plan)
    return x
end

function Base.:\(plan::DCTBatchedPlan{T, 2}, y::AbstractMatrix{T}) where T
    N1, N2 = plan.dims
    
    # Step 1: IDCT along dim 2 (reverse order from DCT)
    y_p = permutedims(y, (2, 1))  # (N2, N1)
    
    Batch2 = N1
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles_inv2 = plan.twiddles_inv[N2]
    ifft_plan2 = plan.ifft_plans[key2]
    
    t1 = similar(y_p)
    idct_batch_dim1!(reshape(t1, N2, Batch2), reshape(y_p, N2, Batch2), v_buf2, V_buf2, twiddles_inv2, ifft_plan2)
    
    t1_p = permutedims(t1, (2, 1))  # (N1, N2)
    
    # Step 2: IDCT along dim 1
    Batch1 = N2
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles_inv1 = plan.twiddles_inv[N1]
    ifft_plan1 = plan.ifft_plans[key1]
    
    x = similar(t1_p)
    idct_batch_dim1!(reshape(x, N1, Batch1), reshape(t1_p, N1, Batch1), v_buf1, V_buf1, twiddles_inv1, ifft_plan1)
    
    return x
end

function Base.:\(plan::DCTBatchedPlan{T, 3}, y::AbstractArray{T, 3}) where T
    N1, N2, N3 = plan.dims
    
    # Reverse order: IDCT dim 3, then dim 2, then dim 1
    
    # Step 1: IDCT along dim 3
    y_p = permutedims(y, (3, 1, 2))  # (N3, N1, N2)
    Batch3 = N1 * N2
    key3 = (N3, Batch3)
    v_buf3 = plan.v_buffers[key3]
    V_buf3 = plan.V_buffers[key3]
    twiddles_inv3 = plan.twiddles_inv[N3]
    ifft_plan3 = plan.ifft_plans[key3]
    
    t1 = similar(y_p)
    idct_batch_dim1!(reshape(t1, N3, Batch3), reshape(y_p, N3, Batch3), v_buf3, V_buf3, twiddles_inv3, ifft_plan3)
    
    t1_p = permutedims(t1, (3, 2, 1))  # (N2, N1, N3)
    
    # Step 2: IDCT along dim 2
    Batch2 = N1 * N3
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles_inv2 = plan.twiddles_inv[N2]
    ifft_plan2 = plan.ifft_plans[key2]
    
    t2 = similar(t1_p)
    idct_batch_dim1!(reshape(t2, N2, Batch2), reshape(t1_p, N2, Batch2), v_buf2, V_buf2, twiddles_inv2, ifft_plan2)
    
    t2_p = permutedims(t2, (2, 1, 3))  # (N1, N2, N3)
    
    # Step 3: IDCT along dim 1
    Batch1 = N2 * N3
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles_inv1 = plan.twiddles_inv[N1]
    ifft_plan1 = plan.ifft_plans[key1]
    
    x = similar(t2_p)
    idct_batch_dim1!(reshape(x, N1, Batch1), reshape(t2_p, N1, Batch1), v_buf1, V_buf1, twiddles_inv1, ifft_plan1)
    
    return x
end

# ============================================================================
# Zero-allocation mul! and ldiv! implementations
# ============================================================================

"""
    LinearAlgebra.mul!(y, plan::DCTBatchedPlan{T, 3}, x) -> y

Compute 3D DCT with zero allocation using ping-pong buffer strategy.
Uses y and plan.temp_buffer alternately to avoid allocations.

# Flow (ping-pong between buf and y):
1. x    → DCT dim1 → buf     # buf = (1',2,3)
2. buf  → perm(2,1,3) → y    # y   = (2,1',3)
3. y    → DCT dim1 → buf     # buf = (2',1',3)
4. buf  → perm(3,2,1) → y    # y   = (3,2',1')
5. y    → DCT dim1 → buf     # buf = (3',2',1')
6. buf  → perm(2,3,1) → y    # y   = (1',2',3') ✓
"""
function LinearAlgebra.mul!(y::AbstractArray{T, 3}, plan::DCTBatchedPlan{T, 3}, x::AbstractArray{T, 3}) where T
    N1, N2, N3 = plan.dims
    buf = plan.temp_buffer
    
    # Step 1: x → DCT dim1 → buf
    Batch1 = N2 * N3
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles1 = plan.twiddles[N1]
    fft_plan1 = plan.fft_plans[key1]
    dct_batch_dim1!(reshape(buf, N1, Batch1), reshape(x, N1, Batch1), v_buf1, V_buf1, twiddles1, fft_plan1)
    
    # Step 2: buf → perm(2,1,3) → y
    # After perm: (N2, N1, N3)
    y_perm = reshape(y, (N2, N1, N3))
    buf_orig = reshape(buf, (N1, N2, N3))
    permutedims_3d!(y_perm, buf_orig, (2, 1, 3))
    
    # Step 3: y → DCT dim1 → buf (buf now shaped as (N2, N1, N3))
    Batch2 = N1 * N3
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles2 = plan.twiddles[N2]
    fft_plan2 = plan.fft_plans[key2]
    buf_2 = reshape(buf, (N2, N1, N3))
    dct_batch_dim1!(reshape(buf_2, N2, Batch2), reshape(y_perm, N2, Batch2), v_buf2, V_buf2, twiddles2, fft_plan2)
    
    # Step 4: buf → perm(3,2,1) → y
    # From (N2, N1, N3) → (N3, N1, N2)
    y_perm2 = reshape(y, (N3, N1, N2))
    permutedims_3d!(y_perm2, buf_2, (3, 2, 1))
    
    # Step 5: y → DCT dim1 → buf (buf now shaped as (N3, N1, N2))
    Batch3 = N1 * N2
    key3 = (N3, Batch3)
    v_buf3 = plan.v_buffers[key3]
    V_buf3 = plan.V_buffers[key3]
    twiddles3 = plan.twiddles[N3]
    fft_plan3 = plan.fft_plans[key3]
    buf_3 = reshape(buf, (N3, N1, N2))
    dct_batch_dim1!(reshape(buf_3, N3, Batch3), reshape(y_perm2, N3, Batch3), v_buf3, V_buf3, twiddles3, fft_plan3)
    
    # Step 6: buf → perm(2,3,1) → y
    # From (N3, N1, N2) → (N1, N2, N3) = final output shape
    permutedims_3d!(y, buf_3, (2, 3, 1))
    
    return y
end

"""
    LinearAlgebra.ldiv!(x, plan::DCTBatchedPlan{T, 3}, y) -> x

Compute 3D IDCT with zero allocation using ping-pong buffer strategy.

# Flow (reverse of DCT):
1. y    → perm(3,1,2) → buf  # buf = (3',1',2')
2. buf  → IDCT dim1 → x     # x   = (3,1',2')
3. x    → perm(3,2,1) → buf  # buf = (2',1',3)
4. buf  → IDCT dim1 → x     # x   = (2,1',3)
5. x    → perm(2,1,3) → buf  # buf = (1',2,3)
6. buf  → IDCT dim1 → x     # x   = (1,2,3) ✓
"""
function LinearAlgebra.ldiv!(x::AbstractArray{T, 3}, plan::DCTBatchedPlan{T, 3}, y::AbstractArray{T, 3}) where T
    N1, N2, N3 = plan.dims
    buf = plan.temp_buffer
    
    # Step 1: y → perm(3,1,2) → buf
    # From (N1, N2, N3) → (N3, N1, N2)
    buf_1 = reshape(buf, (N3, N1, N2))
    permutedims_3d!(buf_1, y, (3, 1, 2))
    
    # Step 2: buf → IDCT dim1 → x
    Batch3 = N1 * N2
    key3 = (N3, Batch3)
    v_buf3 = plan.v_buffers[key3]
    V_buf3 = plan.V_buffers[key3]
    twiddles_inv3 = plan.twiddles_inv[N3]
    ifft_plan3 = plan.ifft_plans[key3]
    x_1 = reshape(x, (N3, N1, N2))
    idct_batch_dim1!(reshape(x_1, N3, Batch3), reshape(buf_1, N3, Batch3), v_buf3, V_buf3, twiddles_inv3, ifft_plan3)
    
    # Step 3: x → perm(3,2,1) → buf
    # From (N3, N1, N2) → (N2, N1, N3)
    buf_2 = reshape(buf, (N2, N1, N3))
    permutedims_3d!(buf_2, x_1, (3, 2, 1))
    
    # Step 4: buf → IDCT dim1 → x
    Batch2 = N1 * N3
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles_inv2 = plan.twiddles_inv[N2]
    ifft_plan2 = plan.ifft_plans[key2]
    x_2 = reshape(x, (N2, N1, N3))
    idct_batch_dim1!(reshape(x_2, N2, Batch2), reshape(buf_2, N2, Batch2), v_buf2, V_buf2, twiddles_inv2, ifft_plan2)
    
    # Step 5: x → perm(2,1,3) → buf
    # From (N2, N1, N3) → (N1, N2, N3)
    buf_3 = reshape(buf, (N1, N2, N3))
    permutedims_3d!(buf_3, x_2, (2, 1, 3))
    
    # Step 6: buf → IDCT dim1 → x
    Batch1 = N2 * N3
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles_inv1 = plan.twiddles_inv[N1]
    ifft_plan1 = plan.ifft_plans[key1]
    idct_batch_dim1!(reshape(x, N1, Batch1), reshape(buf_3, N1, Batch1), v_buf1, V_buf1, twiddles_inv1, ifft_plan1)
    
    return x
end

# 1D and 2D mul!/ldiv! - simpler versions
function LinearAlgebra.mul!(y::AbstractVector{T}, plan::DCTBatchedPlan{T, 1}, x::AbstractVector{T}) where T
    N = plan.dims[1]
    Batch = 1
    key = (N, Batch)
    
    v_buf = plan.v_buffers[key]
    V_buf = plan.V_buffers[key]
    twiddles = plan.twiddles[N]
    fft_plan = plan.fft_plans[key]
    
    dct_batch_dim1!(reshape(y, N, 1), reshape(x, N, 1), v_buf, V_buf, twiddles, fft_plan)
    return y
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, plan::DCTBatchedPlan{T, 1}, y::AbstractVector{T}) where T
    N = plan.dims[1]
    Batch = 1
    key = (N, Batch)
    
    v_buf = plan.v_buffers[key]
    V_buf = plan.V_buffers[key]
    twiddles_inv = plan.twiddles_inv[N]
    ifft_plan = plan.ifft_plans[key]
    
    idct_batch_dim1!(reshape(x, N, 1), reshape(y, N, 1), v_buf, V_buf, twiddles_inv, ifft_plan)
    return x
end

function LinearAlgebra.mul!(y::AbstractMatrix{T}, plan::DCTBatchedPlan{T, 2}, x::AbstractMatrix{T}) where T
    N1, N2 = plan.dims
    buf = plan.temp_buffer
    
    # Step 1: x → DCT dim1 → buf
    Batch1 = N2
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles1 = plan.twiddles[N1]
    fft_plan1 = plan.fft_plans[key1]
    dct_batch_dim1!(reshape(buf, N1, Batch1), reshape(x, N1, Batch1), v_buf1, V_buf1, twiddles1, fft_plan1)
    
    # Step 2: buf → perm(2,1) → y
    y_perm = reshape(y, (N2, N1))
    permutedims_2d!(y_perm, buf)
    
    # Step 3: y → DCT dim1 → buf
    Batch2 = N1
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles2 = plan.twiddles[N2]
    fft_plan2 = plan.fft_plans[key2]
    buf_2 = reshape(buf, (N2, N1))
    dct_batch_dim1!(reshape(buf_2, N2, Batch2), reshape(y_perm, N2, Batch2), v_buf2, V_buf2, twiddles2, fft_plan2)
    
    # Step 4: buf → perm(2,1) → y
    permutedims_2d!(y, buf_2)
    
    return y
end

function LinearAlgebra.ldiv!(x::AbstractMatrix{T}, plan::DCTBatchedPlan{T, 2}, y::AbstractMatrix{T}) where T
    N1, N2 = plan.dims
    buf = plan.temp_buffer
    
    # Step 1: y → perm(2,1) → buf
    buf_1 = reshape(buf, (N2, N1))
    permutedims_2d!(buf_1, y)
    
    # Step 2: buf → IDCT dim1 → x
    Batch2 = N1
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles_inv2 = plan.twiddles_inv[N2]
    ifft_plan2 = plan.ifft_plans[key2]
    x_1 = reshape(x, (N2, N1))
    idct_batch_dim1!(reshape(x_1, N2, Batch2), reshape(buf_1, N2, Batch2), v_buf2, V_buf2, twiddles_inv2, ifft_plan2)
    
    # Step 3: x → perm(2,1) → buf
    buf_2 = reshape(buf, (N1, N2))
    permutedims_2d!(buf_2, x_1)
    
    # Step 4: buf → IDCT dim1 → x
    Batch1 = N2
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles_inv1 = plan.twiddles_inv[N1]
    ifft_plan1 = plan.ifft_plans[key1]
    idct_batch_dim1!(reshape(x, N1, Batch1), reshape(buf_2, N1, Batch1), v_buf1, V_buf1, twiddles_inv1, ifft_plan1)
    
    return x
end

# ============================================================================
# Backward compatible simple versions (allocating)
# ============================================================================

function dct_batch_dim1(x::AbstractArray{T}) where T<:Real
    dims = size(x)
    N = dims[1]
    Batch = prod(dims[2:end])
    y = similar(x)
    
    x_reshaped = reshape(x, (N, Batch))
    y_reshaped = reshape(y, (N, Batch))
    
    v = similar(x, (N, Batch))
    V = similar(x, Complex{T}, (N÷2+1, Batch))
    
    be = get_backend(x)
    twiddles = get_twiddles(N, T, be)
    plan = plan_rfft(v, 1)
    
    dct_batch_dim1!(y_reshaped, x_reshaped, v, V, twiddles, plan)
    
    return y
end

function idct_batch_dim1(y::AbstractArray{T}) where T<:Real
    dims = size(y)
    N = dims[1]
    Batch = prod(dims[2:end])
    x = similar(y)
    
    x_reshaped = reshape(x, (N, Batch))
    y_reshaped = reshape(y, (N, Batch))
    
    # Note: For IDCT, V_buf is input to IRFFT, v_buf is output of IRFFT
    V = similar(y, Complex{T}, (N÷2+1, Batch))
    v = similar(y, (N, Batch))
    
    be = get_backend(y)
    twiddles_inv = get_inv_twiddles(N, T, be)
    plan_inv = plan_irfft(V, N, 1)
    
    idct_batch_dim1!(x_reshaped, y_reshaped, v, V, twiddles_inv, plan_inv)
    
    return x
end

# ============================================================================
# Public API (Backward Compatible)
# ============================================================================

# 1D
function dct_batched(x::AbstractVector{T}) where T <: Real
    plan = plan_dct_batched(x)
    return plan * x
end

function idct_batched(y::AbstractVector{T}) where T <: Real
    plan = plan_dct_batched(y)
    return plan \ y
end

# 2D
function dct_batched(x::AbstractMatrix{T}) where T <: Real
    plan = plan_dct_batched(x)
    return plan * x
end

function idct_batched(y::AbstractMatrix{T}) where T <: Real
    plan = plan_dct_batched(y)
    return plan \ y
end

# 3D
function dct_batched(x::AbstractArray{T, 3}) where T <: Real
    plan = plan_dct_batched(x)
    return plan * x
end

function idct_batched(y::AbstractArray{T, 3}) where T <: Real
    plan = plan_dct_batched(y)
    return plan \ y
end

# ============================================================================
# In-place versions (for maximum performance with pre-allocated output)
# ============================================================================

"""
    dct_batched!(y, x, plan::DCTBatchedPlan)

Compute the N-dimensional DCT of `x` and store the result in `y`.
Uses preallocated buffers from the plan.
"""
function dct_batched!(y::AbstractVector{T}, x::AbstractVector{T}, plan::DCTBatchedPlan{T, 1}) where T
    N = plan.dims[1]
    Batch = 1
    key = (N, Batch)
    
    v_buf = plan.v_buffers[key]
    V_buf = plan.V_buffers[key]
    twiddles = plan.twiddles[N]
    fft_plan = plan.fft_plans[key]
    
    x_reshaped = reshape(x, N, 1)
    y_reshaped = reshape(y, N, 1)
    
    dct_batch_dim1!(y_reshaped, x_reshaped, v_buf, V_buf, twiddles, fft_plan)
    return y
end

function dct_batched!(y::AbstractMatrix{T}, x::AbstractMatrix{T}, plan::DCTBatchedPlan{T, 2}) where T
    N1, N2 = plan.dims
    
    # Step 1: DCT along dim 1
    Batch1 = N2
    key1 = (N1, Batch1)
    v_buf1 = plan.v_buffers[key1]
    V_buf1 = plan.V_buffers[key1]
    twiddles1 = plan.twiddles[N1]
    fft_plan1 = plan.fft_plans[key1]
    
    # Use y as intermediate storage for step 1
    dct_batch_dim1!(reshape(y, N1, Batch1), reshape(x, N1, Batch1), v_buf1, V_buf1, twiddles1, fft_plan1)
    
    # Step 2: DCT along dim 2 (permute, transform, permute back)
    # Need temporary for permutation
    t1_p = permutedims(y, (2, 1))  # (N2, N1)
    
    Batch2 = N1
    key2 = (N2, Batch2)
    v_buf2 = plan.v_buffers[key2]
    V_buf2 = plan.V_buffers[key2]
    twiddles2 = plan.twiddles[N2]
    fft_plan2 = plan.fft_plans[key2]
    
    t2 = similar(t1_p)
    dct_batch_dim1!(reshape(t2, N2, Batch2), reshape(t1_p, N2, Batch2), v_buf2, V_buf2, twiddles2, fft_plan2)
    
    # Copy result back
    copyto!(y, permutedims(t2, (2, 1)))
    return y
end

function dct_batched!(y::AbstractArray{T, 3}, x::AbstractArray{T, 3}, plan::DCTBatchedPlan{T, 3}) where T
    # Just use the non-in-place version and copy
    result = plan * x
    copyto!(y, result)
    return y
end

"""
    idct_batched!(x, y, plan::DCTBatchedPlan)

Compute the N-dimensional IDCT of `y` and store the result in `x`.
Uses preallocated buffers from the plan.
"""
function idct_batched!(x::AbstractVector{T}, y::AbstractVector{T}, plan::DCTBatchedPlan{T, 1}) where T
    N = plan.dims[1]
    Batch = 1
    key = (N, Batch)
    
    v_buf = plan.v_buffers[key]
    V_buf = plan.V_buffers[key]
    twiddles_inv = plan.twiddles_inv[N]
    ifft_plan = plan.ifft_plans[key]
    
    x_reshaped = reshape(x, N, 1)
    y_reshaped = reshape(y, N, 1)
    
    idct_batch_dim1!(x_reshaped, y_reshaped, v_buf, V_buf, twiddles_inv, ifft_plan)
    return x
end

function idct_batched!(x::AbstractMatrix{T}, y::AbstractMatrix{T}, plan::DCTBatchedPlan{T, 2}) where T
    # Use the non-in-place version and copy
    result = plan \ y
    copyto!(x, result)
    return x
end

function idct_batched!(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}, plan::DCTBatchedPlan{T, 3}) where T
    # Use the non-in-place version and copy
    result = plan \ y
    copyto!(x, result)
    return x
end

public dct_batched!, idct_batched!
