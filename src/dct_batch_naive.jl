# Optimized N-D DCT/IDCT Implementation
# 
# Features:
# - Uses R2C FFT (rfft) for memory efficiency
# - Separable implementation using Batched 1D DCT on first dimension
# - Uses permutations for other dimensions
# - Device agnostic via KernelAbstractions.jl

using KernelAbstractions
using AbstractFFTs

export dct_fast, idct_fast

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
Postprocess batch of 1D signals.
V: (halfN+1, Batch) complex, y: (N, Batch) real
"""
@kernel function postprocess_batch_dim1_kernel!(y, @Const(V), N, halfN, Batch)
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
        
        # Apply twiddle factor
        expk = cispi(-k / (2*N))
        @inbounds y[k + 1, b] = real(Vk * expk)
    end
end

"""
IDCT Preprocess batch.
y: (N, Batch) real, V: (halfN+1, Batch) complex
"""
@kernel function idct_preprocess_batch_dim1_kernel!(V, @Const(y), N, halfN, Batch)
    I = @index(Global, Cartesian)
    k = I[1] - 1
    b = I[2]
    
    if k <= halfN
        if k == 0
            @inbounds V[1, b] = Complex(y[1, b], zero(eltype(y)))
        elseif k == halfN
            @inbounds V[k + 1, b] = Complex(y[k + 1, b] * sqrt(eltype(y)(2)), zero(eltype(y)))
        else
            expk_inv = cispi(k / (2*N))
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
# Core Batched Functions
# ============================================================================

function dct_batch_dim1(x::AbstractArray{T}) where T<:Real
    # Reshape to (N, Batch)
    dims = size(x)
    N = dims[1]
    halfN = N ÷ 2
    Batch = prod(dims[2:end])
    
    x_reshaped = reshape(x, (N, Batch))
    v = similar(x, (N, Batch))
    y = similar(x, (N, Batch))
    
    be = get_backend(x)
    
    # 1. Preprocess
    preprocess_batch_dim1_kernel!(be)(v, x_reshaped, N, halfN, Batch; ndrange=(N, Batch))
    KernelAbstractions.synchronize(be)
    
    # 2. R2C FFT along dim 1
    V = rfft(v, 1)
    
    # 3. Postprocess
    postprocess_batch_dim1_kernel!(be)(y, V, N, halfN, Batch; ndrange=(N, Batch))
    KernelAbstractions.synchronize(be)
    
    return reshape(y, dims)
end

function idct_batch_dim1(y::AbstractArray{T}) where T<:Real
    dims = size(y)
    N = dims[1]
    halfN = N ÷ 2
    Batch = prod(dims[2:end])
    
    y_reshaped = reshape(y, (N, Batch))
    # V size: (halfN+1, Batch)
    V = similar(y, Complex{T}, (halfN + 1, Batch))
    x = similar(y, (N, Batch))
    
    be = get_backend(y)
    
    # 1. Preprocess
    idct_preprocess_batch_dim1_kernel!(be)(V, y_reshaped, N, halfN, Batch; ndrange=(halfN+1, Batch))
    KernelAbstractions.synchronize(be)
    
    # 2. IRFFT along dim 1
    # Note: irfft needs original length N
    v = irfft(V, N, 1)
    
    # 3. Postprocess
    idct_postprocess_batch_dim1_kernel!(be)(x, v, N, halfN, Batch; ndrange=(N, Batch))
    KernelAbstractions.synchronize(be)
    
    return reshape(x, dims)
end

# ============================================================================
# Public API
# ============================================================================

function dct_fast(x::AbstractVector)
    return dct_batch_dim1(x)
end

function idct_fast(y::AbstractVector)
    return idct_batch_dim1(y)
end

function dct_fast(x::AbstractMatrix)
    # DCT rows (dim 1)
    t = dct_batch_dim1(x)
    # DCT cols (dim 2) -> permute, dct dim1, permute back
    tp = permutedims(t, (2, 1))
    tp = dct_batch_dim1(tp)
    return permutedims(tp, (2, 1))
end

function idct_fast(y::AbstractMatrix)
    # IDCT cols (dim 2)
    yp = permutedims(y, (2, 1))
    yp = idct_batch_dim1(yp)
    t = permutedims(yp, (2, 1))
    # IDCT rows (dim 1)
    return idct_batch_dim1(t)
end

function dct_fast(x::AbstractArray{T,3}) where T
    # DCT dim 1
    t = dct_batch_dim1(x)
    
    # DCT dim 2: permute (2, 1, 3) -> dim 2 is now dim 1
    t = permutedims(t, (2, 1, 3))
    t = dct_batch_dim1(t)
    t = permutedims(t, (2, 1, 3)) # restore (1, 2, 3)
    
    # DCT dim 3: permute (3, 2, 1) -> dim 3 is now dim 1
    t = permutedims(t, (3, 2, 1))
    t = dct_batch_dim1(t)
    return permutedims(t, (3, 2, 1)) # restore (1, 2, 3)
end

function idct_fast(y::AbstractArray{T,3}) where T
    # IDCT dim 3
    t = permutedims(y, (3, 2, 1))
    t = idct_batch_dim1(t)
    t = permutedims(t, (3, 2, 1))
    
    # IDCT dim 2
    t = permutedims(t, (2, 1, 3))
    t = idct_batch_dim1(t)
    t = permutedims(t, (2, 1, 3))
    
    # IDCT dim 1
    return idct_batch_dim1(t)
end
