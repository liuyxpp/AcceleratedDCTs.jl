# 1D DCT and IDCT implementation using FFT
# Based on Algorithm 1 from the reference paper
#
# DCT (1a): X_k = Σ_{n=0}^{N-1} x_n cos(π/N * (n + 1/2) * k)
# IDCT (1b): X_k = (1/2)x_0 + Σ_{n=1}^{N-1} x_n cos(π/N * n * (k + 1/2))
#
# The FFT-based algorithm:
# 1. Preprocess: reorder input
# 2. Apply FFT
# 3. Postprocess: apply twiddle factors

using AbstractFFTs

"""
    dct1d(x::AbstractVector{T}) where T<:Real

Compute the 1D Discrete Cosine Transform (DCT-II) of a real vector.

# Algorithm
Uses FFT-based computation:
1. Reorder input: v[n] = x[2n] for n < N/2, v[n] = x[2N-2n-1] for n >= N/2
2. Compute FFT: V = fft(v)
3. Apply twiddle factors: y[k] = Re(V[k] * exp(-jπk/(2N)))

# Arguments
- `x`: Input vector of length N (must be even)

# Returns
- DCT coefficients vector of length N
"""
function dct1d(x::AbstractVector{T}) where T<:Real
    N = length(x)
    @assert iseven(N) "Length must be even"
    halfN = N ÷ 2
    
    # Step 1: Preprocess (reorder)
    v = similar(x)
    @inbounds for n in 0:N-1
        if n < halfN
            v[n+1] = x[2*n + 1]      # x[2n] in 0-indexed
        else
            v[n+1] = x[2*N - 2*n]    # x[2N-2n-1] in 0-indexed, +1 for Julia
        end
    end
    
    # Step 2: FFT
    V = fft(v)
    
    # Step 3: Apply twiddle factors and take real part
    # y[k] = Re(V[k] * exp(-jπk/(2N)))
    y = similar(x)
    @inbounds for k in 0:N-1
        expk = cis(-T(π) * k / (2*N))  # exp(-jπk/(2N))
        y[k+1] = real(V[k+1] * expk)
    end
    
    return y
end

"""
    idct1d(y::AbstractVector{T}) where T<:Real

Compute the 1D Inverse Discrete Cosine Transform (IDCT, DCT-III) of DCT coefficients.

# Algorithm
Uses FFT-based computation:
1. Reconstruct FFT coefficients V from DCT coefficients y
2. Apply inverse FFT: v = ifft(V)
3. Inverse reorder to get original signal

# Arguments
- `y`: DCT coefficients vector of length N (must be even)

# Returns
- Reconstructed signal vector of length N
"""
function idct1d(y::AbstractVector{T}) where T<:Real
    N = length(y)
    @assert iseven(N) "Length must be even"
    halfN = N ÷ 2
    
    # Step 1: Reconstruct V from y
    # For k in 1:halfN-1: V[k] = (y[k] - j*y[N-k]) * conj(expk[k])
    # where expk[k] = exp(-jπk/(2N))
    V = zeros(Complex{T}, N)
    
    # DC component (k=0)
    V[1] = Complex{T}(y[1], zero(T))
    
    # k = 1 to halfN-1
    @inbounds for k in 1:halfN-1
        expk_inv = cis(T(π) * k / (2*N))  # conj(exp(-jπk/(2N)))
        # Z[k] = y[k] - j*y[N-k], then V[k] = Z[k] * expk_inv
        V[k+1] = (y[k+1] - im*y[N-k+1]) * expk_inv
    end
    
    # Nyquist component (k = N/2)
    # V[N/2] is real, and y[N/2] = V[N/2] * cos(π/4) = V[N/2]/√2
    V[halfN+1] = Complex{T}(y[halfN+1] * sqrt(T(2)), zero(T))
    
    # Hermitian symmetry: V[N-k] = conj(V[k])
    @inbounds for k in halfN+1:N-1
        V[k+1] = conj(V[N-k+1])
    end
    
    # Step 2: Inverse FFT
    v = ifft(V)
    
    # Step 3: Inverse reorder
    x = similar(y)
    @inbounds for n in 0:N-1
        if n < halfN
            x[2*n + 1] = real(v[n+1])
        else
            x[2*N - 2*n] = real(v[n+1])
        end
    end
    
    return x
end

# 2D DCT and IDCT implementation
# Uses separable 1D transforms: first on rows, then on columns
#
# This provides a simple, correct implementation.
# For GPU optimization, use KernelAbstractions kernels.

"""
    dct2d(x::AbstractMatrix{T}) where T<:Real

Compute the 2D Discrete Cosine Transform (DCT-II) of a real matrix.

# Algorithm
Uses separable 1D DCT transforms:
1. Apply 1D DCT to each row
2. Apply 1D DCT to each column

# Arguments
- `x`: Input matrix of size M×N (M and N must be even)

# Returns
- DCT coefficients matrix of size M×N
"""
function dct2d(x::AbstractMatrix{T}) where T<:Real
    M, N = size(x)
    tmp = similar(x)
    y = similar(x)
    
    # DCT on rows
    for i in 1:M
        tmp[i, :] = dct1d(x[i, :])
    end
    # DCT on columns
    for j in 1:N
        y[:, j] = dct1d(tmp[:, j])
    end
    
    return y
end

"""
    idct2d(y::AbstractMatrix{T}) where T<:Real

Compute the 2D Inverse Discrete Cosine Transform (IDCT, DCT-III) of DCT coefficients.

# Algorithm
Uses separable 1D IDCT transforms:
1. Apply 1D IDCT to each column
2. Apply 1D IDCT to each row

# Arguments
- `y`: DCT coefficients matrix of size M×N (M and N must be even)

# Returns
- Reconstructed signal matrix of size M×N
"""
function idct2d(y::AbstractMatrix{T}) where T<:Real
    M, N = size(y)
    tmp = similar(y)
    x = similar(y)
    
    # IDCT on columns
    for j in 1:N
        tmp[:, j] = idct1d(y[:, j])
    end
    # IDCT on rows
    for i in 1:M
        x[i, :] = idct1d(tmp[i, :])
    end
    
    return x
end

# 3D DCT and IDCT implementation
# Uses separable 1D transforms along each dimension

"""
    dct3d(x::AbstractArray{T,3}) where T<:Real

Compute the 3D Discrete Cosine Transform (DCT-II) of a real 3D array.

# Algorithm
Uses separable 1D DCT transforms:
1. Apply 1D DCT along dimension 1 (rows)
2. Apply 1D DCT along dimension 2 (columns)
3. Apply 1D DCT along dimension 3 (depth)

# Arguments
- `x`: Input array of size L×M×N (all dimensions must be even)

# Returns
- DCT coefficients array of size L×M×N
"""
function dct3d(x::AbstractArray{T,3}) where T<:Real
    L, M, N = size(x)
    
    # Allocate intermediate arrays
    tmp1 = similar(x)
    tmp2 = similar(x)
    y = similar(x)
    
    # DCT along dimension 1
    for j in 1:M, k in 1:N
        tmp1[:, j, k] = dct1d(x[:, j, k])
    end
    
    # DCT along dimension 2
    for i in 1:L, k in 1:N
        tmp2[i, :, k] = dct1d(tmp1[i, :, k])
    end
    
    # DCT along dimension 3
    for i in 1:L, j in 1:M
        y[i, j, :] = dct1d(tmp2[i, j, :])
    end
    
    return y
end

"""
    idct3d(y::AbstractArray{T,3}) where T<:Real

Compute the 3D Inverse Discrete Cosine Transform (IDCT, DCT-III) of DCT coefficients.

# Algorithm
Uses separable 1D IDCT transforms:
1. Apply 1D IDCT along dimension 3 (depth)
2. Apply 1D IDCT along dimension 2 (columns)
3. Apply 1D IDCT along dimension 1 (rows)

# Arguments
- `y`: DCT coefficients array of size L×M×N (all dimensions must be even)

# Returns
- Reconstructed signal array of size L×M×N
"""
function idct3d(y::AbstractArray{T,3}) where T<:Real
    L, M, N = size(y)
    
    # Allocate intermediate arrays
    tmp1 = similar(y)
    tmp2 = similar(y)
    x = similar(y)
    
    # IDCT along dimension 3
    for i in 1:L, j in 1:M
        tmp1[i, j, :] = idct1d(y[i, j, :])
    end
    
    # IDCT along dimension 2
    for i in 1:L, k in 1:N
        tmp2[i, :, k] = idct1d(tmp1[i, :, k])
    end
    
    # IDCT along dimension 1
    for j in 1:M, k in 1:N
        x[:, j, k] = idct1d(tmp2[:, j, k])
    end
    
    return x
end


