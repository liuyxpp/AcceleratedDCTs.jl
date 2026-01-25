# ============================================================================
# AbstractFFTs Plan Definitions
# ============================================================================

"""
    DCTPlan

Optimized DCT Plan for device-agnostic execution.
"""
struct DCTPlan{T, N, P, Twiddles, BReal, BComp, Region} <: AbstractFFTs.Plan{T}
    complex_plan::P      # RFFT plan
    twiddles::Twiddles   # Tuple of twiddles (w1, w2, ...)
    tmp_real::BReal      # Real buffer for Permutation
    tmp_comp::BComp      # Complex buffer for FFT
    region::Region       # Dimensions
    pinv::Base.RefValue{Any} # Cache for inverse plan
end

"""
    IDCTPlan

Optimized IDCT Plan for device-agnostic execution.
"""
struct IDCTPlan{T, N, P, Twiddles, BReal, BComp, Region} <: AbstractFFTs.Plan{T}
    complex_plan::P      # IRFFT plan
    twiddles::Twiddles   # Tuple of twiddles
    tmp_real::BReal      # Real buffer
    tmp_comp::BComp      # Complex buffer for Preprocess
    region::Region
    pinv::Base.RefValue{Any}
end

# Properties
Base.ndims(::DCTPlan{T, N}) where {T, N} = N
Base.ndims(::IDCTPlan{T, N}) where {T, N} = N

# Properties
Base.size(p::DCTPlan) = size(p.tmp_real)
Base.size(p::IDCTPlan) = size(p.tmp_real)
Base.eltype(::DCTPlan{T}) where T = T
Base.eltype(::IDCTPlan{T}) where T = T

# Plan creation
function plan_dct(x::AbstractArray{T, N}, region=1:N) where {T <: Real, N}
    # Currently only full transform supported
    if region != 1:N && region != 1:ndims(x) && region != (1:ndims(x)...,)
        error("Partial DCT optimization not yet implemented. Use region=1:ndims(x)")
    end
    
    dims = size(x)
    backend = get_backend(x)
    
    # 1. Buffers
    tmp_real = similar(x)
    
    # Complex buffer size calculation (RFFT)
    # RFFT on real array of size (N1, N2, ...) -> (N1÷2+1, N2, ...)
    cdims = ntuple(i -> ifelse(i == 1, dims[1] ÷ 2 + 1, dims[i]), N)
    tmp_comp = KernelAbstractions.allocate(backend, Complex{T}, cdims...)
    
    # 2. Plan RFFT
    # We plan on the buffers to avoid destroying input x during planning if possible
    # AbstractFFTs.plan_rfft typically plans out-of-place or in-place.
    # We use out-of-place RFFT plan: tmp_real -> tmp_comp
    complex_plan = plan_rfft(tmp_real)
    
    # 3. Twiddles
    # Generate on CPU, copy to GPU
    twiddles = ntuple(i -> _get_twiddles_gpu(dims[i], T, backend), N)
    
    return DCTPlan{T, N, typeof(complex_plan), typeof(twiddles), typeof(tmp_real), typeof(tmp_comp), typeof(region)}(
        complex_plan, twiddles, tmp_real, tmp_comp, region, Ref{Any}(nothing)
    )
end

function plan_idct(x::AbstractArray{T, N}, region=1:N) where {T <: Real, N}
    # IDCT Plan
    if region != 1:N && region != 1:ndims(x) && region != (1:ndims(x)...,)
        error("Partial IDCT optimization not yet implemented. Use region=1:ndims(x)")
    end
    
    dims = size(x)
    backend = get_backend(x)
    
    # 1. Buffers
    tmp_real = similar(x)
    cdims = ntuple(i -> ifelse(i == 1, dims[1] ÷ 2 + 1, dims[i]), N)
    tmp_comp = KernelAbstractions.allocate(backend, Complex{T}, cdims...)
    
    # 2. Plan IRFFT
    # irfft plan: tmp_comp -> tmp_real.
    # Note: irfft requires length (d) argument for first dimension.
    complex_plan = plan_irfft(tmp_comp, dims[1])
    
    # 3. Twiddles
    twiddles = ntuple(i -> _get_twiddles_gpu(dims[i], T, backend), N)
    
    return IDCTPlan{T, N, typeof(complex_plan), typeof(twiddles), typeof(tmp_real), typeof(tmp_comp), typeof(region)}(
        complex_plan, twiddles, tmp_real, tmp_comp, region, Ref{Any}(nothing)
    )
end

# Inversion (caching)
function Base.inv(p::DCTPlan{T, N}) where {T, N}
    if p.pinv[] === nothing
        # create inverse plan matching p
        p.pinv[] = plan_idct(p.tmp_real, p.region)
    end
    return p.pinv[]
end

function Base.inv(p::IDCTPlan{T, N}) where {T, N}
    if p.pinv[] === nothing
        p.pinv[] = plan_dct(p.tmp_real, p.region)
    end
    return p.pinv[]
end

# Execution: *
function Base.:*(p::DCTPlan, x::AbstractArray)
    y = similar(x)
    mul!(y, p, x)
    return y
end

function Base.:*(p::IDCTPlan, x::AbstractArray)
    y = similar(x)
    mul!(y, p, x)
    return y
end

# Execution: \ (ldiv)
function Base.:\(p::DCTPlan, x::AbstractArray)
    # plan \ x == inv(plan) * x
    inv_p = inv(p)
    return inv_p * x
end

function Base.:\(p::IDCTPlan, x::AbstractArray)
    inv_p = inv(p)
    return inv_p * x
end

# Execution: mul! and ldiv!

# dct!
function LinearAlgebra.mul!(y::AbstractArray, p::DCTPlan{T, 2}, x::AbstractArray) where T
    backend = get_backend(x)
    N1, N2 = size(x)
    
    # 1. Preprocess: x -> p.tmp_real
    dct_2d_preprocess_kernel!(backend)(
        p.tmp_real, x, N1, N2, (N1-1)÷2, (N2-1)÷2; 
        ndrange=(N1, N2),
        workgroupsize=(16, 16)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. FFT: p.tmp_real -> p.tmp_comp
    # Use mul! for the internal FFT plan
    mul!(p.tmp_comp, p.complex_plan, p.tmp_real)
    
    # 3. Postprocess: p.tmp_comp -> y
    w1, w2 = p.twiddles
    dct_2d_postprocess_kernel!(backend)(
        y, p.tmp_comp, w1, w2, N1, N2;
        ndrange=(N1, N2),
        workgroupsize=(16, 16)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, p::DCTPlan{T, 3}, x::AbstractArray) where T
    backend = get_backend(x)
    N1, N2, N3 = size(x)
    
    # 1. Preprocess
    dct_3d_preprocess_kernel!(backend)(
        p.tmp_real, x, N1, N2, N3, (N1-1)÷2, (N2-1)÷2, (N3-1)÷2;
        ndrange=(N1, N2, N3),
        workgroupsize=(8, 8, 4)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. FFT
    mul!(p.tmp_comp, p.complex_plan, p.tmp_real)
    
    # 3. Postprocess
    w1, w2, w3 = p.twiddles
    dct_3d_postprocess_kernel!(backend)(
        y, p.tmp_comp, w1, w2, w3, N1, N2, N3;
        ndrange=(N1, N2, N3),
        workgroupsize=(8, 8, 4)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

# idct! (via mul!(y, inv_plan, x))
function LinearAlgebra.mul!(y::AbstractArray, p::IDCTPlan{T, 2}, x::AbstractArray) where T
    backend = get_backend(x)
    N1, N2 = size(x)
    limit_n1 = N1 ÷ 2
    
    # 1. Preprocess: x -> p.tmp_comp
    w1, w2 = p.twiddles
    idct_2d_preprocess_kernel!(backend)(
        p.tmp_comp, x, w1, w2, N1, N2, limit_n1;
        ndrange=(limit_n1+1, N2),
        workgroupsize=(16, 16)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. IRFFT: p.tmp_comp -> p.tmp_real
    mul!(p.tmp_real, p.complex_plan, p.tmp_comp)
    
    # 3. Postprocess: p.tmp_real -> y
    idct_2d_postprocess_kernel!(backend)(
        y, p.tmp_real, N1, N2;
        ndrange=(N1, N2),
        workgroupsize=(16, 16)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, p::IDCTPlan{T, 3}, x::AbstractArray) where T
    backend = get_backend(x)
    N1, N2, N3 = size(x)
    limit_n1 = N1 ÷ 2
    
    # 1. Preprocess
    w1, w2, w3 = p.twiddles
    idct_3d_preprocess_kernel!(backend)(
        p.tmp_comp, x, w1, w2, w3, N1, N2, N3, limit_n1;
        ndrange=(limit_n1+1, N2, N3),
        workgroupsize=(8, 8, 4)
    )
    KernelAbstractions.synchronize(backend)
    
    # 2. IRFFT
    mul!(p.tmp_real, p.complex_plan, p.tmp_comp)
    
    # 3. Postprocess
    idct_3d_postprocess_kernel!(backend)(
        y, p.tmp_real, N1, N2, N3;
        ndrange=(N1, N2, N3),
        workgroupsize=(8, 8, 4)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

# Support ldiv!(y, plan, x) => mul!(y, inv(plan), x)
function LinearAlgebra.ldiv!(y::AbstractArray, p::DCTPlan, x::AbstractArray)
    inv_p = inv(p)
    mul!(y, inv_p, x)
end

function LinearAlgebra.ldiv!(y::AbstractArray, p::IDCTPlan, x::AbstractArray)
    inv_p = inv(p)
    mul!(y, inv_p, x)
end


# ============================================================================
# 2D Kernels
# ============================================================================

@kernel function dct_2d_preprocess_kernel!(x_prime, @Const(x), N1, N2, limit1, limit2)
    # Scatter-based Optimization for Coalesced Reads
    # Thread (k1, k2) reads linear x[k1, k2] and scatters to x_prime[dest]
    
    I = @index(Global, Cartesian)
    k1 = I[1] - 1
    k2 = I[2] - 1
    
    if k1 < N1 && k2 < N2
        # Read x (Coalesced)
        @inbounds val = x[k1+1, k2+1]
        
        # Calculate destinations based on inverse permutation
        # p[n] = k.
        # If k is even: k = 2n => n = k/2.
        # If k is odd: k = 2N - 2n - 1 => 2n = 2N - 1 - k => n = N - (k+1)/2.
        
        dest1 = ifelse(iseven(k1), k1 ÷ 2, N1 - (k1 + 1) ÷ 2)
        dest2 = ifelse(iseven(k2), k2 ÷ 2, N2 - (k2 + 1) ÷ 2)
        
        @inbounds x_prime[dest1+1, dest2+1] = val
    end
end

@kernel function dct_2d_postprocess_kernel!(y, @Const(X), @Const(w1), @Const(w2), N1, N2)
    I = @index(Global, Cartesian)
    n1 = I[1] - 1
    n2 = I[2] - 1
    
    if n1 < N1 && n2 < N2
        W2 = w2[n2+1]
        W1 = w1[n1+1]
        
        # val1 = X[n1, n2] with Hermitian symmetry
        val1 = _get_X2_val(X, n1, n2, N1, N2)
        
        # val2 = X[N1-n1, n2]
        n1_b = ifelse(n1 == 0, 0, N1 - n1)
        val2 = _get_X2_val(X, n1_b, n2, N1, N2)
        
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
        idx1_b = ifelse(n1 == 0, N1, N1 - n1)
        
        idx2_a = n2
        idx2_b = ifelse(n2 == 0, N2, N2 - n2)
        
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
        
        idx1 = ifelse(iseven(n1), n1 ÷ 2, N1 - (n1 + 1) ÷ 2)
        idx2 = ifelse(iseven(n2), n2 ÷ 2, N2 - (n2 + 1) ÷ 2)
        
        # 0.25 scaling (type-correct)
        T = eltype(y)
        y[n1+1, n2+1] = T(0.25) * v[idx1+1, idx2+1]
    end
end

# ============================================================================
# 3D Kernels
# ============================================================================

@kernel function dct_3d_preprocess_kernel!(x_prime, @Const(x), N1, N2, N3, limit1, limit2, limit3)
    # Scatter-based Optimization for Coalesced Reads
    I = @index(Global, Cartesian)
    k1, k2, k3 = I[1]-1, I[2]-1, I[3]-1
    
    if k1 < N1 && k2 < N2 && k3 < N3
        @inbounds val = x[k1+1, k2+1, k3+1]
        
        dest1 = ifelse(iseven(k1), k1 ÷ 2, N1 - (k1 + 1) ÷ 2)
        dest2 = ifelse(iseven(k2), k2 ÷ 2, N2 - (k2 + 1) ÷ 2)
        dest3 = ifelse(iseven(k3), k3 ÷ 2, N3 - (k3 + 1) ÷ 2)
        
        @inbounds x_prime[dest1+1, dest2+1, dest3+1] = val
    end
end

@kernel function dct_3d_postprocess_kernel!(y, @Const(X), @Const(w1), @Const(w2), @Const(w3), N1, N2, N3)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1]-1, I[2]-1, I[3]-1
    
    if n1 < N1 && n2 < N2 && n3 < N3
        W1 = w1[n1+1]; W2 = w2[n2+1]; W3 = w3[n3+1]
        
        # Recursive phi1 reconstruction
        # Needs 4 points from X
        
        n1_src = ifelse(n1 == 0, 0, N1 - n1)
        n2_src = ifelse(n2 == 0, 0, N2 - n2)
        
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
        
        idx1_a = n1; idx1_b = ifelse(n1 == 0, N1, N1 - n1)
        idx2_a = n2; idx2_b = ifelse(n2 == 0, N2, N2 - n2)
        idx3_a = n3; idx3_b = ifelse(n3 == 0, N3, N3 - n3)
        
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
        idx1 = ifelse(iseven(n1), n1 ÷ 2, N1 - (n1 + 1) ÷ 2)
        idx2 = ifelse(iseven(n2), n2 ÷ 2, N2 - (n2 + 1) ÷ 2)
        idx3 = ifelse(iseven(n3), n3 ÷ 2, N3 - (n3 + 1) ÷ 2)
        
        # 0.125 scaling (type-correct)
        T = eltype(y)
        y[n1+1, n2+1, n3+1] = T(0.125) * v[idx1+1, idx2+1, idx3+1]
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

@inline function _get_X2_val(X, n1, n2, N1, N2)
    # Branchless Hermitian symmetry reconstruction for 2D RFFT
    # Avoids warp divergence on GPU by using predicated execution
    half_N1 = N1 ÷ 2
    use_sym = n1 > half_N1
    
    # Direct indices
    i1_d = n1 + 1
    i2_d = n2 + 1
    
    # Symmetric indices
    i1_s = N1 - n1 + 1
    i2_s = ifelse(n2 == 0, 1, N2 - n2 + 1)
    
    # Select indices
    i1 = ifelse(use_sym, i1_s, i1_d)
    i2 = ifelse(use_sym, i2_s, i2_d)
    
    @inbounds val = X[i1, i2]
    return ifelse(use_sym, conj(val), val)
end

@inline function _get_X3_val(X, n1, n2, n3, N1, N2, N3)
    # Branchless Hermitian symmetry reconstruction
    # Avoids warp divergence on GPU by using predicated execution
    half_N1 = N1 ÷ 2
    use_sym = n1 > half_N1
    
    # Direct indices (always computed)
    i1_d = n1 + 1
    i2_d = n2 + 1
    i3_d = n3 + 1
    
    # Symmetric indices (always computed, selected conditionally)
    i1_s = N1 - n1 + 1
    i2_s = ifelse(n2 == 0, 1, N2 - n2 + 1)
    i3_s = ifelse(n3 == 0, 1, N3 - n3 + 1)
    
    # Select indices based on whether we need symmetry
    i1 = ifelse(use_sym, i1_s, i1_d)
    i2 = ifelse(use_sym, i2_s, i2_d)
    i3 = ifelse(use_sym, i3_s, i3_d)
    
    @inbounds val = X[i1, i2, i3]
    return ifelse(use_sym, conj(val), val)
end

# ============================================================================
# Convenience Functions (using One-Shot plan)
# ============================================================================

function dct(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_dct(x)
    return p * x
end

function idct(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_idct(x)
    return p * x
end

# In-place convenience
function dct!(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_dct(x)
    # Note: Optimization only safe if input can be destroyed.
    # Our optimized mul!(y, p, x) uses tmp buffers in plan.
    # If we want literal in-place dct!(x), we need y=x?
    # mul!(y, p, x) handles x->preprocess->fft->postprocess->y.
    # If x===y, we need to ensure safety.
    # Logic: x -> tmp_real ... -> tmp_comp ... -> x.
    # Since tmp_real copy happens first, x can be overwritten in final step.
    # Yes, typically safer to write:
    mul!(x, p, x) 
    return x
end

function idct!(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_idct(x)
    mul!(x, p, x)
    return x
end

# ============================================================================
# Utilities
# ============================================================================

function _get_twiddles_gpu(N::Int, ::Type{T}, backend) where T
    # Check if we should cache this? For now, recompute/copy.
    # To optimize, we could add a caching mechanism like DCTPlan.
    
    # Alloc on CPU
    w_cpu = [cispi(-T(n) / (2N)) for n in 0:(N-1)]
    
    # Alloc on Device and Copy
    w_dev = KernelAbstractions.allocate(backend, Complex{T}, N)
    copyto!(w_dev, w_cpu)
    
    return w_dev
end
