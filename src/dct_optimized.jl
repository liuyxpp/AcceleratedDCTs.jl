# Optimized N-D DCT/IDCT Implementation
# 
# Features:
# - Uses R2C FFT (rfft) for memory efficiency
# - Multi-dimensional FFT in one pass
# - Device agnostic via KernelAbstractions.jl
#
# Based on Algorithm 1 from the reference paper with optimizations.

using KernelAbstractions
using AbstractFFTs

export dct_fast, idct_fast

# ============================================================================
# Preprocessing Kernels (reorder input for DCT)
# ============================================================================

"""
1D preprocessing: reorder x[n] -> v[n] where
  v[n] = x[2n]       for n < N/2
  v[n] = x[2N-2n-1]  for n >= N/2
"""
@kernel function preprocess_kernel_1d!(v, @Const(x), N, halfN)
    I = @index(Global, Cartesian)
    n = I[1] - 1  # 0-based
    
    if n < N
        if n < halfN
            src = 2*n
        else
            src = 2*N - 2*n - 1
        end
        @inbounds v[n + 1] = x[src + 1]
    end
end

"""
2D preprocessing: reorder x[n1,n2] -> v[n1,n2]
"""
@kernel function preprocess_kernel_2d!(v, @Const(x), M, N, halfM, halfN)
    I = @index(Global, Cartesian)
    n1, n2 = I[1] - 1, I[2] - 1  # 0-based
    
    if n1 < M && n2 < N
        # Determine source indices
        src1 = n1 < halfM ? 2*n1 : 2*M - 2*n1 - 1
        src2 = n2 < halfN ? 2*n2 : 2*N - 2*n2 - 1
        @inbounds v[n1 + 1, n2 + 1] = x[src1 + 1, src2 + 1]
    end
end

"""
3D preprocessing: reorder x[n1,n2,n3] -> v[n1,n2,n3]
"""
@kernel function preprocess_kernel_3d!(v, @Const(x), L, M, N, halfL, halfM, halfN)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1] - 1, I[2] - 1, I[3] - 1  # 0-based
    
    if n1 < L && n2 < M && n3 < N
        src1 = n1 < halfL ? 2*n1 : 2*L - 2*n1 - 1
        src2 = n2 < halfM ? 2*n2 : 2*M - 2*n2 - 1
        src3 = n3 < halfN ? 2*n3 : 2*N - 2*n3 - 1
        @inbounds v[n1 + 1, n2 + 1, n3 + 1] = x[src1 + 1, src2 + 1, src3 + 1]
    end
end

# ============================================================================
# Postprocessing Kernels (apply twiddle factors after FFT)
# ============================================================================

"""
1D postprocessing: y[k] = Re(V[k] * exp(-jπk/(2N)))
V has size N/2+1 (Hermitian half from rfft)
"""
@kernel function postprocess_kernel_1d!(y, @Const(V), N, halfN)
    I = @index(Global, Cartesian)
    k = I[1] - 1  # 0-based
    
    if k < N
        # Get V[k] using Hermitian symmetry
        if k <= halfN
            Vk = V[k + 1]
        else
            Vk = conj(V[N - k + 1])
        end
        
        # Apply twiddle factor
        expk = cispi(-k / (2*N))  # exp(-jπk/(2N))
        @inbounds y[k + 1] = real(Vk * expk)
    end
end

"""
2D postprocessing: y[k1,k2] from V of size M×(N/2+1)
"""
@kernel function postprocess_kernel_2d!(y, @Const(V), M, N, halfN)
    I = @index(Global, Cartesian)
    k1, k2 = I[1] - 1, I[2] - 1  # 0-based
    
    if k1 < M && k2 < N
        # Get V[k1, k2] using Hermitian symmetry in last dimension
        if k2 <= halfN
            Vk = V[k1 + 1, k2 + 1]
        else
            # V[k1, k2] = conj(V[M-k1, N-k2]) for k2 > N/2
            k1_mirror = k1 == 0 ? 0 : M - k1
            k2_mirror = N - k2
            Vk = conj(V[k1_mirror + 1, k2_mirror + 1])
        end
        
        # Apply twiddle factors for both dimensions
        expk1 = cispi(-k1 / (2*M))
        expk2 = cispi(-k2 / (2*N))
        @inbounds y[k1 + 1, k2 + 1] = real(Vk * expk1 * expk2)
    end
end

"""
3D postprocessing: y[k1,k2,k3] from V of size L×M×(N/2+1)
"""
@kernel function postprocess_kernel_3d!(y, @Const(V), L, M, N, halfN)
    I = @index(Global, Cartesian)
    k1, k2, k3 = I[1] - 1, I[2] - 1, I[3] - 1  # 0-based
    
    if k1 < L && k2 < M && k3 < N
        # Get V[k1, k2, k3] using Hermitian symmetry in last dimension
        if k3 <= halfN
            Vk = V[k1 + 1, k2 + 1, k3 + 1]
        else
            k1_mirror = k1 == 0 ? 0 : L - k1
            k2_mirror = k2 == 0 ? 0 : M - k2
            k3_mirror = N - k3
            Vk = conj(V[k1_mirror + 1, k2_mirror + 1, k3_mirror + 1])
        end
        
        # Apply twiddle factors
        expk1 = cispi(-k1 / (2*L))
        expk2 = cispi(-k2 / (2*M))
        expk3 = cispi(-k3 / (2*N))
        @inbounds y[k1 + 1, k2 + 1, k3 + 1] = real(Vk * expk1 * expk2 * expk3)
    end
end

# ============================================================================
# IDCT Preprocessing Kernels (reconstruct Hermitian half from DCT coeffs)
# ============================================================================

"""
1D IDCT preprocessing: reconstruct V[k] for k = 0..N/2 from y[k]
V[k] = (y[k] - j*y[N-k]) * exp(jπk/(2N))
"""
@kernel function idct_preprocess_kernel_1d!(V, @Const(y), N, halfN)
    I = @index(Global, Cartesian)
    k = I[1] - 1  # 0-based
    
    if k <= halfN
        if k == 0
            @inbounds V[1] = Complex(y[1], zero(eltype(y)))
        elseif k == halfN
            # Nyquist: y[N/2] = V[N/2] * cos(π/4), V is real
            @inbounds V[k + 1] = Complex(y[k + 1] * sqrt(eltype(y)(2)), zero(eltype(y)))
        else
            expk_inv = cispi(k / (2*N))  # exp(jπk/(2N))
            @inbounds V[k + 1] = (y[k + 1] - im * y[N - k + 1]) * expk_inv
        end
    end
end

"""
2D IDCT preprocessing: reconstruct V of size M×(N/2+1) from y
"""
@kernel function idct_preprocess_kernel_2d!(V, @Const(y), M, N, halfN)
    I = @index(Global, Cartesian)
    k1, k2 = I[1] - 1, I[2] - 1  # 0-based
    
    if k1 < M && k2 <= halfN
        expk1_inv = cispi(k1 / (2*M))
        expk2_inv = cispi(k2 / (2*N))
        
        # Handle Hermitian symmetry reconstruction
        if k2 == 0
            # First column: no Hermitian pair in k2 direction
            if k1 == 0
                @inbounds V[1, 1] = Complex(y[1, 1], zero(eltype(y)))
            else
                k1_mirror = M - k1
                @inbounds V[k1 + 1, 1] = (y[k1 + 1, 1] - im * y[k1_mirror + 1, 1]) * expk1_inv
            end
        elseif k2 == halfN && iseven(N)
            # Nyquist column
            if k1 == 0
                @inbounds V[1, k2 + 1] = Complex(y[1, k2 + 1] * sqrt(eltype(y)(2)), zero(eltype(y))) * expk2_inv
            else
                k1_mirror = M - k1
                val = (y[k1 + 1, k2 + 1] - im * y[k1_mirror + 1, k2 + 1]) * expk1_inv
                @inbounds V[k1 + 1, k2 + 1] = val * sqrt(eltype(y)(2)) * expk2_inv
            end
        else
            # Interior
            k2_mirror = N - k2
            if k1 == 0
                @inbounds V[1, k2 + 1] = (y[1, k2 + 1] - im * y[1, k2_mirror + 1]) * expk2_inv
            else
                k1_mirror = M - k1
                # Combine k1 and k2 Hermitian pairs
                val = (y[k1 + 1, k2 + 1] - im * y[k1_mirror + 1, k2_mirror + 1])
                @inbounds V[k1 + 1, k2 + 1] = val * expk1_inv * expk2_inv
            end
        end
    end
end

# ============================================================================
# IDCT Postprocessing Kernels (inverse reorder)
# ============================================================================

"""
1D inverse reorder: v[n] -> x where x[2n] = v[n] for n < N/2, etc.
"""
@kernel function idct_postprocess_kernel_1d!(x, @Const(v), N, halfN)
    I = @index(Global, Cartesian)
    n = I[1] - 1  # 0-based
    
    if n < N
        if n < halfN
            dst = 2*n
        else
            dst = 2*N - 2*n - 1
        end
        @inbounds x[dst + 1] = v[n + 1]
    end
end

"""
2D inverse reorder
"""
@kernel function idct_postprocess_kernel_2d!(x, @Const(v), M, N, halfM, halfN)
    I = @index(Global, Cartesian)
    n1, n2 = I[1] - 1, I[2] - 1  # 0-based
    
    if n1 < M && n2 < N
        dst1 = n1 < halfM ? 2*n1 : 2*M - 2*n1 - 1
        dst2 = n2 < halfN ? 2*n2 : 2*N - 2*n2 - 1
        @inbounds x[dst1 + 1, dst2 + 1] = v[n1 + 1, n2 + 1]
    end
end

"""
3D inverse reorder
"""
@kernel function idct_postprocess_kernel_3d!(x, @Const(v), L, M, N, halfL, halfM, halfN)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1] - 1, I[2] - 1, I[3] - 1  # 0-based
    
    if n1 < L && n2 < M && n3 < N
        dst1 = n1 < halfL ? 2*n1 : 2*L - 2*n1 - 1
        dst2 = n2 < halfM ? 2*n2 : 2*M - 2*n2 - 1
        dst3 = n3 < halfN ? 2*n3 : 2*N - 2*n3 - 1
        @inbounds x[dst1 + 1, dst2 + 1, dst3 + 1] = v[n1 + 1, n2 + 1, n3 + 1]
    end
end

# ============================================================================
# Main API: dct and idct (unified interface for 1D, 2D, 3D)
# ============================================================================

"""
    dct(x::AbstractArray{T}) where T<:Real

Compute the N-dimensional Discrete Cosine Transform (DCT-II).

Supports 1D, 2D, and 3D arrays. Uses R2C FFT for efficiency.
Device agnostic - works on CPU and GPU arrays.

# Arguments
- `x`: Input array (1D, 2D, or 3D) with even dimensions

# Returns
- DCT coefficients array of same size as input
"""
function dct_fast(x::AbstractVector{T}) where T<:Real
    N = length(x)
    @assert iseven(N) "Length must be even"
    halfN = N ÷ 2
    be = get_backend(x)
    
    # Allocate
    v = similar(x)
    y = similar(x)
    
    # Step 1: Preprocess
    preprocess_kernel_1d!(be)(v, x, N, halfN; ndrange=(N,))
    KernelAbstractions.synchronize(be)
    
    # Step 2: R2C FFT
    V = rfft(v)
    
    # Step 3: Postprocess
    postprocess_kernel_1d!(be)(y, V, N, halfN; ndrange=(N,))
    KernelAbstractions.synchronize(be)
    
    return y
end

function dct_fast(x::AbstractMatrix{T}) where T<:Real
    M, N = size(x)
    @assert iseven(M) && iseven(N) "Dimensions must be even"
    halfM, halfN = M ÷ 2, N ÷ 2
    be = get_backend(x)
    
    # Allocate
    v = similar(x)
    y = similar(x)
    
    # Step 1: Preprocess
    preprocess_kernel_2d!(be)(v, x, M, N, halfM, halfN; ndrange=(M, N))
    KernelAbstractions.synchronize(be)
    
    # Step 2: 2D R2C FFT (on last dimension)
    V = rfft(v, [1,2])
    
    # Step 3: Postprocess
    postprocess_kernel_2d!(be)(y, V, M, N, halfN; ndrange=(M, N))
    KernelAbstractions.synchronize(be)
    
    return y
end

function dct_fast(x::AbstractArray{T,3}) where T<:Real
    L, M, N = size(x)
    @assert iseven(L) && iseven(M) && iseven(N) "Dimensions must be even"
    halfL, halfM, halfN = L ÷ 2, M ÷ 2, N ÷ 2
    be = get_backend(x)
    
    # Allocate
    v = similar(x)
    y = similar(x)
    
    # Step 1: Preprocess
    preprocess_kernel_3d!(be)(v, x, L, M, N, halfL, halfM, halfN; ndrange=(L, M, N))
    KernelAbstractions.synchronize(be)
    
    # Step 2: 3D R2C FFT (on last dimension)
    V = rfft(v, [1,2,3])
    
    # Step 3: Postprocess
    postprocess_kernel_3d!(be)(y, V, L, M, N, halfN; ndrange=(L, M, N))
    KernelAbstractions.synchronize(be)
    
    return y
end

"""
    idct(y::AbstractArray{T}) where T<:Real

Compute the N-dimensional Inverse Discrete Cosine Transform (IDCT, DCT-III).

Supports 1D, 2D, and 3D arrays. Uses C2R IFFT for efficiency.
Device agnostic - works on CPU and GPU arrays.

# Arguments
- `y`: DCT coefficients array (1D, 2D, or 3D) with even dimensions

# Returns
- Reconstructed signal array of same size as input
"""
function idct_fast(y::AbstractVector{T}) where T<:Real
    N = length(y)
    @assert iseven(N) "Length must be even"
    halfN = N ÷ 2
    be = get_backend(y)
    
    # Allocate
    V = similar(y, Complex{T}, halfN + 1)
    x = similar(y)
    
    # Step 1: Preprocess (reconstruct Hermitian half)
    idct_preprocess_kernel_1d!(be)(V, y, N, halfN; ndrange=(halfN+1,))
    KernelAbstractions.synchronize(be)
    
    # Step 2: C2R IFFT
    v = irfft(V, N)
    
    # Step 3: Postprocess (inverse reorder)
    idct_postprocess_kernel_1d!(be)(x, v, N, halfN; ndrange=(N,))
    KernelAbstractions.synchronize(be)
    
    return x
end

function idct_fast(y::AbstractMatrix{T}) where T<:Real
    M, N = size(y)
    @assert iseven(M) && iseven(N) "Dimensions must be even"
    halfM, halfN = M ÷ 2, N ÷ 2
    be = get_backend(y)
    
    # Allocate
    V = similar(y, Complex{T}, M, halfN + 1)
    x = similar(y)
    
    # Step 1: Preprocess
    idct_preprocess_kernel_2d!(be)(V, y, M, N, halfN; ndrange=(M, halfN+1))
    KernelAbstractions.synchronize(be)
    
    # Step 2: C2R IFFT (on last dimension)
    v = irfft(V, N, [1,2])
    
    # Step 3: Postprocess
    idct_postprocess_kernel_2d!(be)(x, v, M, N, halfM, halfN; ndrange=(M, N))
    KernelAbstractions.synchronize(be)
    
    return x
end

function idct_fast(y::AbstractArray{T,3}) where T<:Real
    L, M, N = size(y)
    @assert iseven(L) && iseven(M) && iseven(N) "Dimensions must be even"
    halfL, halfM, halfN = L ÷ 2, M ÷ 2, N ÷ 2
    be = get_backend(y)
    
    # Allocate
    V = similar(y, Complex{T}, L, M, halfN + 1)
    x = similar(y)
    
    # Step 1: Preprocess
    # Note: 3D IDCT preprocess kernel is more complex, using separable approach for now
    # TODO: Implement optimized 3D IDCT preprocess kernel
    
    # For now, use the reference implementation for 3D IDCT
    return idct3d(y)
end
