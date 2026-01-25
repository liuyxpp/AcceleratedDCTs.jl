using AbstractFFTs
using LinearAlgebra

"""
    dct_2d_opt(x)

Compute the 2D Discrete Cosine Transform (DCT-II) of `x` using Algorithm 2.
"""
function dct_2d_opt(x::AbstractMatrix{T}) where T <: Real
    N1, N2 = size(x)
    
    # 1. Preprocess: Permutation (Eq 13)
    x_prime = Matrix{T}(undef, N1, N2)
    
    # We use 0-based indexing logic for the permutation formula, 
    # but 1-based for Julia arrays.
    
    # Precompute 1D permutations for rows and cols (separable)
    p1 = _dct_perm_indices(N1)
    p2 = _dct_perm_indices(N2)
    
    # Apply separable permutation
    # x_prime[n1+1, n2+1] = x[p1[n1+1], p2[n2+1]]
    # Optimized loop order for cache friendliness (col-major)
    @inbounds for j in 1:N2
        pj = p2[j]
        for i in 1:N1
            x_prime[i, j] = x[p1[i], pj]
        end
    end
    
    # 2. 2D Real FFT
    X = rfft(x_prime)
    
    # 3. Postprocess (Eq 14)
    # y(n1, n2) = 2 * Re(...)
    y = Matrix{T}(undef, N1, N2)
    
    # Compute twiddle factors
    # e^{-j * pi * n / (2N)}
    w1 = _get_dct_twiddles(N1)
    w2 = _get_dct_twiddles(N2)
    
    _dct_2d_postprocess_opt!(y, X, w1, w2, N1, N2)
    
    return y
end


"""
    idct_2d_opt(x)

Compute the 2D Inverse Discrete Cosine Transform (DCT-III) of `x` using Algorithm 2.
"""
function idct_2d_opt(x::AbstractMatrix{T}) where T <: Real
    # x is real DCT coefficients
    N1, N2 = size(x)
    
    # 1. IDCT Preprocess (Eq 15)
    # Produces input for IRFFT 
    # We construct x' of size (N1÷2 + 1, N2) for irfft
    
    X_prime = Matrix{Complex{T}}(undef, N1÷2 + 1, N2)
    
    w1 = _get_dct_twiddles(N1)
    w2 = _get_dct_twiddles(N2)
    
    _idct_2d_preprocess_opt!(X_prime, x, w1, w2, N1, N2)
    
    # 2. 2D Inverse Real FFT
    # Note: irfft expects first dimension to be N1÷2 + 1, and we must specify full length N1
    x_spatial = irfft(X_prime, N1)
    
    # 3. Postprocess: Permutation (Eq 16)
    y = Matrix{T}(undef, N1, N2)
    
    p1 = _idct_perm_indices(N1)
    p2 = _idct_perm_indices(N2)
    
    @inbounds for j in 1:N2
        pj = p2[j]
        for i in 1:N1
            # Apply 0.25 scaling as per Markdown definition of IDCT
            y[i, j] = 0.25 * x_spatial[p1[i], pj]
        end
    end
    
    return y
end

# ============================================================================
# Helpers
# ============================================================================

function _dct_perm_indices(N::Int)
    # Eq (13) logic for one dimension
    # 0 <= n <= floor((N-1)/2)  => 2n
    # floor((N+1)/2) <= n < N   => 2N - 2n - 1
    
    p = Vector{Int}(undef, N)
    limit = (N - 1) ÷ 2
    
    for n in 0:limit
        p[n+1] = 2n + 1
    end
    for n in (limit+1):(N-1)
        p[n+1] = (2N - 2n - 1) + 1
    end
    return p
end

function _idct_perm_indices(N::Int)
    # Eq (16) logic
    # n even => n/2
    # n odd  => N - (n+1)/2
    
    p = Vector{Int}(undef, N)
    for n in 0:(N-1)
        if iseven(n)
            # n = 2k -> k. Index: n+1 -> k+1 = n/2 + 1
            p[n+1] = (n ÷ 2) + 1
        else
            # n = 2k+1 -> N - (n+1)/2
            p[n+1] = (N - (n - 1) ÷ 2 - 1) + 1 # Wait, (n+1)/2 for n odd is (2k+2)/2 = k+1?
            # Formula: N - (n+1)/2?
            # n=1 (odd) -> N - 1
            # n=3 -> N - 2
            # Let's check:
            # If n is odd, n=2k+1. formula says N - (n+1)/2.Integer division?
            # Julia integer division: 
            # If n=1, (n+1)/2 = 1. Src = N-1. (Correct? Last element)
            p[n+1] = N - (n + 1) ÷ 2 + 1 # Convert to 1-based?
            # Wait, 0-based indices in formula.
            # n=1 -> index N-1. 1-based: n=2 -> N.
            # 0-based result: val = N - (n+1)/2. 
            # 1-based result: val + 1 = N - (n+1)/2 + 1.
            
            # Let's use float div to be sure or carefully check integer div
            val = N - (n + 1) ÷ 2 
            p[n+1] = val + 1
        end
    end
    return p
end

function _get_dct_twiddles(N::Int)
    # e^{-j * pi * n / (2N)} for n in 0:N-1
    return [cis(-π * n / (2N)) for n in 0:(N-1)]
end

function _dct_2d_postprocess_opt!(y, X, w1, w2, N1, N2)
    # Eq (14)
    # y(n1, n2) = 2 Re( w2[n2] * ( w1[n1] * X(n1, n2) + conj(w1[n1]) * X(N1-n1, n2) ) )
    
    # We iterate over result coordinates n1, n2
    @inbounds for n2 in 0:(N2-1)
        W2 = w2[n2+1]
        for n1 in 0:(N1-1)
            W1 = w1[n1+1]
            
            # Fetch X(n1, n2) and X(N1-n1, n2)
            # Handle RFFT symmetry
            
            # term1: X(n1, n2)
            val1 = _get_X_rfft(X, n1, n2, N1, N2)
            
            # term2: X(N1-n1, n2)
            # n1' = (N1 - n1) % N1. 
            # If the user crossed out "X(N1)=0", and X is FFT result, it is periodic. X(N1)=X(0).
            n1_src = (n1 == 0) ? 0 : N1 - n1
            val2 = _get_X_rfft(X, n1_src, n2, N1, N2)
            
            # e^{...} is w1
            # e^{+...} is conj(w1)
            
            term = W1 * val1 + conj(W1) * val2
            
            y[n1+1, n2+1] = 2 * real(W2 * term)
        end
    end
end

@inline function _get_X_rfft(X, n1, n2, N1, N2)
    # Access X corresponding to full FFT indices (n1, n2)
    # using rfft output (0:N1/2, 0:N2-1)
    
    # Normalize n2 to range 0..N2-1 if strictly needed, but loop guarantees it.
    
    if n1 <= (N1 ÷ 2)
        return X[n1+1, n2+1]
    else
        # Conjugate symmetry: X[n1, n2] = conj(X[N1-n1, N2-n2])
        # n1 > N1/2 implies N1-n1 < N1/2
        
        n1_src = N1 - n1
        # n2 wrap: (N2 - n2) % N2. 
        # If n2=0 -> 0. Else N2-n2.
        n2_src = (n2 == 0) ? 0 : N2 - n2
        
        return conj(X[n1_src+1, n2_src+1])
    end
end


function _idct_2d_preprocess_opt!(X_prime, x, w1, w2, N1, N2)
    # Eq (15)
    # x'(n1, n2) = w1[n1] * w2[n2] * ( (x[n1,n2] - x[N1-n1, N2-n2]) - j*(x[N1-n1, n2] + x[n1, N2-n2]) )
    # Output X_prime is complex, size (N1÷2 + 1, N2)
    # Iterate n1 from 0 to N1÷2
    
    limit_n1 = N1 ÷ 2
    
    @inbounds for n2 in 0:(N2-1)
        W2 = w2[n2+1]
        
        # Wrapping logic for N2-n2? 
        # For DCT coeffs (x), we assume zero outside 0..N-1 boundary.
        # So x(N2, ...) = 0.
        idx2_a = n2
        idx2_b = (n2 == 0) ? N2 : N2 - n2
        
        for n1 in 0:limit_n1
            W1 = w1[n1+1]
            
            # Zero-padding logic for N1-n1
            idx1_a = n1
            idx1_b = (n1 == 0) ? N1 : N1 - n1
            
            # Get values
            v1 = _get_x_val(x, idx1_a, idx2_a, N1, N2) # x(n1, n2)
            v2 = _get_x_val(x, idx1_b, idx2_b, N1, N2) # x(N1-n1, N2-n2)
            v3 = _get_x_val(x, idx1_b, idx2_a, N1, N2) # x(N1-n1, n2)
            v4 = _get_x_val(x, idx1_a, idx2_b, N1, N2) # x(n1, N2-n2)
            
            term_real = v1 - v2
            term_imag = v3 + v4
            
            # Result = conj(w1) * conj(w2) * (term_real - j * term_imag)
            # Use conjugate to invert the shift
            
            val = conj(W1) * conj(W2) * Complex(term_real, -term_imag)
            
            X_prime[n1+1, n2+1] = val
        end
    end
end

@inline function _get_x_val(x, n1, n2, N1, N2)
    # Returns x[n1+1, n2+1] if valid, else 0
    # Condition: x(N1, n2) = 0, x(n1, N2) = 0
    
    if n1 >= N1 || n2 >= N2
        return zero(eltype(x))
    end
    @inbounds return x[n1+1, n2+1]
end
