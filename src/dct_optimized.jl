using KernelAbstractions
using AbstractFFTs
using LinearAlgebra

# ============================================================================
# 2D Kernels
# ============================================================================

@kernel function dct_2d_preprocess_kernel!(x_prime, @Const(x), N1, N2, limit1, limit2)
    I = @index(Global, Cartesian)
    n1 = I[1] - 1
    n2 = I[2] - 1
    
    # 2D permutation: x'[n1, n2] = x[p1[n1], p2[n2]]
    # p[n] = (n <= limit) ? 2n : 2N - 2n - 1
    
    if n1 < N1 && n2 < N2
        # Compute source index for n1
        src1 = (n1 <= limit1) ? 2 * n1 : 2 * N1 - 2 * n1 - 1
        
        # Compute source index for n2
        src2 = (n2 <= limit2) ? 2 * n2 : 2 * N2 - 2 * n2 - 1
        
        @inbounds x_prime[n1+1, n2+1] = x[src1+1, src2+1]
    end
end

@kernel function dct_2d_postprocess_kernel!(y, @Const(X), @Const(w1), @Const(w2), N1, N2)
    I = @index(Global, Cartesian)
    n1 = I[1] - 1
    n2 = I[2] - 1
    
    if n1 < N1 && n2 < N2
        W2 = w2[n2+1]
        W1 = w1[n1+1]
        
        # Access X with Hermitian symmetry handling for n1 (first dimension of RFFT)
        # val1 = X[n1, n2]
        if n1 <= (N1 ÷ 2)
            val1 = X[n1+1, n2+1]
        else
            # X[n1, n2] = conj(X[N1-n1, N2-n2])
            n1_sym = N1 - n1
            n2_sym = (n2 == 0) ? 0 : N2 - n2
            val1 = conj(X[n1_sym+1, n2_sym+1])
        end
        
        # val2 = X[N1-n1, n2]
        # logic for N1-n1:
        # if n1=0 -> N1 -> sym to 0. (X(0) is real, X(N)=X(0) is periodic assumption from derivation)
        n1_b = (n1 == 0) ? 0 : N1 - n1
        
        if n1_b <= (N1 ÷ 2)
            val2 = X[n1_b+1, n2+1]
        else
            n1_sym_b = N1 - n1_b
            n2_sym_b = (n2 == 0) ? 0 : N2 - n2
            val2 = conj(X[n1_sym_b+1, n2_sym_b+1])
        end
        
        term = W1 * val1 + conj(W1) * val2
        y[n1+1, n2+1] = 2 * real(W2 * term)
    end
end

@kernel function idct_2d_preprocess_kernel!(X_prime, @Const(x), @Const(w1), @Const(w2), N1, N2, limit_n1)
    # Output is X_prime size (limit_n1+1, N2) where limit_n1 = N1 ÷ 2
    I = @index(Global, Cartesian)
    n1 = I[1] - 1
    n2 = I[2] - 1
    
    # We iterate n1 from 0 to N1÷2 (inclusive)
    if n1 <= limit_n1 && n2 < N2
        W1 = w1[n1+1]
        W2 = w2[n2+1]
        
        # Need 4 points from x
        # indices for x are modulo N/periodic or zero padded?
        # IDCT formulation used zero-padding for N-n indices if n=0.
        # But we need to be careful about x access.
        # x is N1xN2.
        
        idx1_a = n1
        idx1_b = (n1 == 0) ? N1 : N1 - n1
        
        idx2_a = n2
        idx2_b = (n2 == 0) ? N2 : N2 - n2
        
        v1 = _get_val_safe(x, idx1_a, idx2_a, N1, N2)
        v2 = _get_val_safe(x, idx1_b, idx2_b, N1, N2)
        v3 = _get_val_safe(x, idx1_b, idx2_a, N1, N2)
        v4 = _get_val_safe(x, idx1_a, idx2_b, N1, N2)
        
        term_real = v1 - v2
        term_imag = v3 + v4
        
        # Conjugate twiddles
        val = conj(W1) * conj(W2) * Complex(term_real, -term_imag)
        
        X_prime[n1+1, n2+1] = val
    end
end

@kernel function idct_2d_postprocess_kernel!(y, @Const(v), N1, N2)
    I = @index(Global, Cartesian)
    n1 = I[1] - 1
    n2 = I[2] - 1
    
    if n1 < N1 && n2 < N2
        # Inverse permutation
        # p[n] = (n even) ? n/2 : N - (n+1)/2
        
        idx1 = iseven(n1) ? (n1 ÷ 2) : (N1 - (n1 + 1) ÷ 2)
        idx2 = iseven(n2) ? (n2 ÷ 2) : (N2 - (n2 + 1) ÷ 2)
        
        # 0.25 scaling
        y[n1+1, n2+1] = 0.25f0 * v[idx1+1, idx2+1]
    end
end

# ============================================================================
# 3D Kernels
# ============================================================================

@kernel function dct_3d_preprocess_kernel!(x_prime, @Const(x), N1, N2, N3, limit1, limit2, limit3)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1]-1, I[2]-1, I[3]-1
    
    if n1 < N1 && n2 < N2 && n3 < N3
        src1 = (n1 <= limit1) ? 2 * n1 : 2 * N1 - 2 * n1 - 1
        src2 = (n2 <= limit2) ? 2 * n2 : 2 * N2 - 2 * n2 - 1
        src3 = (n3 <= limit3) ? 2 * n3 : 2 * N3 - 2 * n3 - 1
        
        @inbounds x_prime[n1+1, n2+1, n3+1] = x[src1+1, src2+1, src3+1]
    end
end

@kernel function dct_3d_postprocess_kernel!(y, @Const(X), @Const(w1), @Const(w2), @Const(w3), N1, N2, N3)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1]-1, I[2]-1, I[3]-1
    
    if n1 < N1 && n2 < N2 && n3 < N3
        W1 = w1[n1+1]; W2 = w2[n2+1]; W3 = w3[n3+1]
        
        # Recursive phi1 reconstruction
        # Needs 4 points from X
        
        n1_src = (n1 == 0) ? 0 : N1 - n1
        n2_src = (n2 == 0) ? 0 : N2 - n2
        
        # To get 3D X values with symmetry
        # (n1, n2, n3)
        v1 = _get_X3_val(X, n1, n2, n3, N1, N2, N3)
        # (N1-n1, n2, n3)
        v2 = _get_X3_val(X, n1_src, n2, n3, N1, N2, N3)
        # (n1, N2-n2, n3)
        v3 = _get_X3_val(X, n1, n2_src, n3, N1, N2, N3)
        # (N1-n1, N2-n2, n3)
        v4 = _get_X3_val(X, n1_src, n2_src, n3, N1, N2, N3)
        
        phi1_A = W1 * v1 + conj(W1) * v2
        phi1_B = W1 * v3 + conj(W1) * v4
        
        inner = W2 * phi1_A + conj(W2) * phi1_B
        y[n1+1, n2+1, n3+1] = 2 * real(W3 * inner)
    end
end

@kernel function idct_3d_preprocess_kernel!(X_prime, @Const(x), @Const(w1), @Const(w2), @Const(w3), N1, N2, N3, limit_n1)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1]-1, I[2]-1, I[3]-1
    
    if n1 <= limit_n1 && n2 < N2 && n3 < N3
        W1 = w1[n1+1]; W2 = w2[n2+1]; W3 = w3[n3+1]
        
        idx1_a = n1; idx1_b = (n1 == 0) ? N1 : N1 - n1
        idx2_a = n2; idx2_b = (n2 == 0) ? N2 : N2 - n2
        idx3_a = n3; idx3_b = (n3 == 0) ? N3 : N3 - n3
        
        v000 = _get_val_safe_3d(x, idx1_a, idx2_a, idx3_a, N1, N2, N3)
        v100 = _get_val_safe_3d(x, idx1_b, idx2_a, idx3_a, N1, N2, N3)
        v010 = _get_val_safe_3d(x, idx1_a, idx2_b, idx3_a, N1, N2, N3)
        v001 = _get_val_safe_3d(x, idx1_a, idx2_a, idx3_b, N1, N2, N3)
        v110 = _get_val_safe_3d(x, idx1_b, idx2_b, idx3_a, N1, N2, N3)
        v101 = _get_val_safe_3d(x, idx1_b, idx2_a, idx3_b, N1, N2, N3)
        v011 = _get_val_safe_3d(x, idx1_a, idx2_b, idx3_b, N1, N2, N3)
        v111 = _get_val_safe_3d(x, idx1_b, idx2_b, idx3_b, N1, N2, N3)
        
        term_R = v000 - (v110 + v101 + v011)
        term_I = (v100 + v010 + v001) - v111
        
        val = conj(W1) * conj(W2) * conj(W3) * Complex(term_R, -term_I)
        X_prime[n1+1, n2+1, n3+1] = val
    end
end

@kernel function idct_3d_postprocess_kernel!(y, @Const(v), N1, N2, N3)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1]-1, I[2]-1, I[3]-1
    
    if n1 < N1 && n2 < N2 && n3 < N3
        idx1 = iseven(n1) ? (n1 ÷ 2) : (N1 - (n1 + 1) ÷ 2)
        idx2 = iseven(n2) ? (n2 ÷ 2) : (N2 - (n2 + 1) ÷ 2)
        idx3 = iseven(n3) ? (n3 ÷ 2) : (N3 - (n3 + 1) ÷ 2)
        
        # 0.125 scaling
        y[n1+1, n2+1, n3+1] = 0.125f0 * v[idx1+1, idx2+1, idx3+1]
    end
end

# ============================================================================
# Inline Helpers for Kernels
# ============================================================================

@inline function _get_val_safe(x, n1, n2, N1, N2)
    if n1 >= N1 || n2 >= N2
        return zero(eltype(x))
    end
    @inbounds return x[n1+1, n2+1]
end

@inline function _get_val_safe_3d(x, n1, n2, n3, N1, N2, N3)
    if n1 >= N1 || n2 >= N2 || n3 >= N3
        return zero(eltype(x))
    end
    @inbounds return x[n1+1, n2+1, n3+1]
end

@inline function _get_X3_val(X, n1, n2, n3, N1, N2, N3)
    # Symmetry: X[n1, n2, n3] = conj(X[N1-n1, N2-n2, N3-n3])
    # RFFT dim 1 range: 0 to N1/2
    
    if n1 <= (N1 ÷ 2)
        @inbounds return X[n1+1, n2+1, n3+1]
    else
        n1_s = N1 - n1
        n2_s = (n2 == 0) ? 0 : N2 - n2
        n3_s = (n3 == 0) ? 0 : N3 - n3
        @inbounds return conj(X[n1_s+1, n2_s+1, n3_s+1])
    end
end

# ============================================================================
# Main Functions
# ============================================================================

function dct_2d_opt(x::AbstractMatrix{T}) where T <: Real
    N1, N2 = size(x)
    backend = get_backend(x)
    
    # 1. Preprocess
    x_prime = similar(x)
    
    dct_2d_preprocess_kernel!(backend)(
        x_prime, x, N1, N2, (N1-1)÷2, (N2-1)÷2; 
        ndrange=(N1, N2)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. FFT
    X = rfft(x_prime) 
    # Note: X will be on the same backend if rfft supports it (CUDA.CUFFT / FFTW)
    
    # 3. Postprocess
    y = similar(x)
    w1 = _get_twiddles_gpu(N1, T, backend)
    w2 = _get_twiddles_gpu(N2, T, backend)
    
    dct_2d_postprocess_kernel!(backend)(
        y, X, w1, w2, N1, N2;
        ndrange=(N1, N2)
    )
    KernelAbstractions.synchronize(backend)
    
    return y
end

function idct_2d_opt(x::AbstractMatrix{T}) where T <: Real
    N1, N2 = size(x)
    backend = get_backend(x)
    
    # 1. Preprocess
    limit_n1 = N1 ÷ 2
    # Output complex array for IRFFT
    # Size: (N1÷2 + 1, N2)
    X_prime = KernelAbstractions.allocate(backend, Complex{T}, limit_n1+1, N2)
    
    w1 = _get_twiddles_gpu(N1, T, backend)
    w2 = _get_twiddles_gpu(N2, T, backend)
    
    idct_2d_preprocess_kernel!(backend)(
        X_prime, x, w1, w2, N1, N2, limit_n1;
        ndrange=(limit_n1+1, N2)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. IRFFT
    x_spatial = irfft(X_prime, N1)
    
    # 3. Postprocess
    y = similar(x)
    idct_2d_postprocess_kernel!(backend)(
        y, x_spatial, N1, N2;
        ndrange=(N1, N2)
    )
    KernelAbstractions.synchronize(backend)
    
    return y
end

function dct_3d_opt(x::AbstractArray{T, 3}) where T <: Real
    N1, N2, N3 = size(x)
    backend = get_backend(x)
    
    # 1. Preprocess
    x_prime = similar(x)
    dct_3d_preprocess_kernel!(backend)(
        x_prime, x, N1, N2, N3, (N1-1)÷2, (N2-1)÷2, (N3-1)÷2;
        ndrange=(N1, N2, N3)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. FFT
    X = rfft(x_prime)
    
    # 3. Postprocess
    y = similar(x)
    w1 = _get_twiddles_gpu(N1, T, backend)
    w2 = _get_twiddles_gpu(N2, T, backend)
    w3 = _get_twiddles_gpu(N3, T, backend)
    
    dct_3d_postprocess_kernel!(backend)(
        y, X, w1, w2, w3, N1, N2, N3;
        ndrange=(N1, N2, N3)
    )
    KernelAbstractions.synchronize(backend)
    
    return y
end

function idct_3d_opt(x::AbstractArray{T, 3}) where T <: Real
    N1, N2, N3 = size(x)
    backend = get_backend(x)
    
    # 1. Preprocess
    limit_n1 = N1 ÷ 2
    X_prime = KernelAbstractions.allocate(backend, Complex{T}, limit_n1+1, N2, N3)
    
    w1 = _get_twiddles_gpu(N1, T, backend)
    w2 = _get_twiddles_gpu(N2, T, backend)
    w3 = _get_twiddles_gpu(N3, T, backend)
    
    idct_3d_preprocess_kernel!(backend)(
        X_prime, x, w1, w2, w3, N1, N2, N3, limit_n1;
        ndrange=(limit_n1+1, N2, N3)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. IRFFT
    x_spatial = irfft(X_prime, N1)
    
    # 3. Postprocess
    y = similar(x)
    idct_3d_postprocess_kernel!(backend)(
        y, x_spatial, N1, N2, N3;
        ndrange=(N1, N2, N3)
    )
    KernelAbstractions.synchronize(backend)
    
    return y
end

# ============================================================================
# Utilities
# ============================================================================

function _get_twiddles_gpu(N::Int, ::Type{T}, backend) where T
    # Check if we should cache this? For now, recompute/copy.
    # To optimize, we could add a caching mechanism like DCTPlan.
    
    # Alloc on CPU
    w_cpu = [cis(-T(π) * n / (2N)) for n in 0:(N-1)]
    
    # Alloc on Device and Copy
    w_dev = KernelAbstractions.allocate(backend, Complex{T}, N)
    copyto!(w_dev, w_cpu)
    
    return w_dev
end
