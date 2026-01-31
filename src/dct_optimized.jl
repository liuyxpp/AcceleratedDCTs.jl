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
    twiddles::Twiddles   # Tuple of twiddles (w1, w2, ...)
    tmp_real::BReal      # Real buffer for Permutation
    tmp_comp::BComp      # Complex buffer for FFT
    region::Region       # Dimensions
    pinv::Base.RefValue{Any} # Cache for forward plan
end

"""
    DCT1Plan

Optimized DCT-I Plan for device-agnostic execution.
"""
struct DCT1Plan{T, N, P, BReal, BComp, Region} <: AbstractFFTs.Plan{T}
    complex_plan::P      # RFFT plan
    tmp_real::BReal      # Real buffer for Mirroring (size 2M-2)
    tmp_comp::BComp      # Complex buffer for FFT
    region::Region       # Dimensions
    pinv::Base.RefValue{Any} # Cache for inverse plan
end

"""
    IDCT1Plan

Optimized IDCT-I Plan for device-agnostic execution.
"""
struct IDCT1Plan{T, N, P, BReal, BComp, Region} <: AbstractFFTs.Plan{T}
    complex_plan::P      # RFFT plan (IDCT-I is same as DCT-I with scaling)
    tmp_real::BReal      # Real buffer for Mirroring
    tmp_comp::BComp      # Complex buffer for FFT
    region::Region       # Dimensions
    pinv::Base.RefValue{Any} # Cache for forward plan
end

# Properties
Base.ndims(::DCTPlan{T, N}) where {T, N} = N
Base.ndims(::IDCTPlan{T, N}) where {T, N} = N
Base.ndims(::DCT1Plan{T, N}) where {T, N} = N
Base.ndims(::IDCT1Plan{T, N}) where {T, N} = N

# Properties
Base.size(p::DCTPlan) = size(p.tmp_real)
Base.size(p::IDCTPlan) = size(p.tmp_real)
Base.eltype(::DCTPlan{T}) where T = T
Base.eltype(::IDCTPlan{T}) where T = T
Base.eltype(::DCT1Plan{T}) where T = T
Base.eltype(::IDCT1Plan{T}) where T = T

# Size for DCT1Plan: tmp_real has size (N1, N2, ...) where Ni = 2Mi-2.
# We return (M1, M2, ...)
function Base.size(p::Union{DCT1Plan{T, N}, IDCT1Plan{T, N}}) where {T, N}
    return ntuple(i -> (size(p.tmp_real, i) ÷ 2) + 1, N)
end

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

function Base.inv(p::DCT1Plan{T, N}) where {T, N}
    if p.pinv[] === nothing
        x = KernelAbstractions.allocate(get_backend(p.tmp_real), T, size(p)...)
        p.pinv[] = plan_idct1(x, p.region)
    end
    return p.pinv[]
end

function Base.inv(p::IDCT1Plan{T, N}) where {T, N}
    if p.pinv[] === nothing
        x = KernelAbstractions.allocate(get_backend(p.tmp_real), T, size(p)...)
        p.pinv[] = plan_dct1(x, p.region)
    end
    return p.pinv[]
end

# Execution: *
function Base.:*(p::Union{DCTPlan, DCT1Plan}, x::AbstractArray)
    y = similar(x)
    mul!(y, p, x)
    return y
end

function Base.:*(p::Union{IDCTPlan, IDCT1Plan}, x::AbstractArray)
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
function LinearAlgebra.mul!(y::AbstractArray, p::DCTPlan{T, 1}, x::AbstractArray) where T
    backend = get_backend(x)
    N = size(x, 1)

    # 1. Preprocess: x -> p.tmp_real
    dct_1d_preprocess_kernel!(backend)(
        p.tmp_real, x, N;
        ndrange=(N,),
        workgroupsize=(256,)
    )
    KernelAbstractions.synchronize(backend)

    # 2. FFT: p.tmp_real -> p.tmp_comp
    mul!(p.tmp_comp, p.complex_plan, p.tmp_real)

    # 3. Postprocess: p.tmp_comp -> y
    (w1,) = p.twiddles
    dct_1d_postprocess_kernel!(backend)(
        y, p.tmp_comp, w1, N;
        ndrange=(N,),
        workgroupsize=(256,)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

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
function LinearAlgebra.mul!(y::AbstractArray, p::IDCTPlan{T, 1}, x::AbstractArray) where T
    backend = get_backend(x)
    N = size(x, 1)
    limit_n = N ÷ 2

    # 1. Preprocess: x -> p.tmp_comp
    (w1,) = p.twiddles
    idct_1d_preprocess_kernel!(backend)(
        p.tmp_comp, x, w1, N, limit_n;
        ndrange=(limit_n+1,),
        workgroupsize=(256,)
    )
    KernelAbstractions.synchronize(backend)

    # 2. IRFFT: p.tmp_comp -> p.tmp_real
    mul!(p.tmp_real, p.complex_plan, p.tmp_comp)

    # 3. Postprocess: p.tmp_real -> y
    idct_1d_postprocess_kernel!(backend)(
        y, p.tmp_real, N;
        ndrange=(N,),
        workgroupsize=(256,)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

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
    idct_3d_postprocess_kernel!(y, p.tmp_real, N1, N2, N3;
        ndrange=(N1, N2, N3),
        workgroupsize=(8, 8, 4)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

# Support ldiv!(y, plan, x) => mul!(y, inv(plan), x)
function LinearAlgebra.ldiv!(y::AbstractArray, p::Union{DCTPlan, IDCTPlan, DCT1Plan, IDCT1Plan}, x::AbstractArray)
    inv_p = inv(p)
    mul!(y, inv_p, x)
end

# ============================================================================
# DCT-I Implementation Logic
# ============================================================================

function plan_dct1(x::AbstractArray{T, N}, region=1:N) where {T <: Real, N}
    if region != 1:N && region != 1:ndims(x) && region != (1:ndims(x)...,)
        error("Partial DCT1 optimization not yet implemented.")
    end

    M = size(x)
    backend = get_backend(x)
    
    # 1. Buffers
    # Mirroring size N = 2M - 2
    N_full = ntuple(i -> 2M[i] - 2, N)
    tmp_real = KernelAbstractions.allocate(backend, T, N_full...)

    # Complex buffer size (RFFT)
    # result is (N1/2+1, N2, N3...) = (M1, 2M2-2, 2M3-2...)
    cdims = ntuple(i -> ifelse(i == 1, N_full[1] ÷ 2 + 1, N_full[i]), N)
    tmp_comp = KernelAbstractions.allocate(backend, Complex{T}, cdims...)

    # 2. Plan RFFT
    complex_plan = plan_rfft(tmp_real)

    return DCT1Plan{T, N, typeof(complex_plan), typeof(tmp_real), typeof(tmp_comp), typeof(region)}(
        complex_plan, tmp_real, tmp_comp, region, Ref{Any}(nothing)
    )
end

function plan_idct1(x::AbstractArray{T, N}, region=1:N) where {T <: Real, N}
    p = plan_dct1(x, region)
    return IDCT1Plan{T, N, typeof(p.complex_plan), typeof(p.tmp_real), typeof(p.tmp_comp), typeof(region)}(
        p.complex_plan, p.tmp_real, p.tmp_comp, p.region, Ref{Any}(nothing)
    )
end

function LinearAlgebra.mul!(y::AbstractArray, p::DCT1Plan{T, 1}, x::AbstractArray) where T
    backend = get_backend(x)
    M = size(x, 1)
    N = 2M - 2

    # 1. Preprocess: Mirror x -> tmp_real
    dct1_1d_preprocess_kernel!(backend)(
        p.tmp_real, x, M, N;
        ndrange=(N,),
        workgroupsize=(256,)
    )
    KernelAbstractions.synchronize(backend)

    # 2. FFT: tmp_real -> tmp_comp
    mul!(p.tmp_comp, p.complex_plan, p.tmp_real)

    # 3. Postprocess: real(tmp_comp) -> y
    dct1_1d_postprocess_kernel!(backend)(
        y, p.tmp_comp, M;
        ndrange=(M,),
        workgroupsize=(256,)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, p::DCT1Plan{T, 2}, x::AbstractArray) where T
    backend = get_backend(x)
    M1, M2 = size(x)
    N1, N2 = 2M1-2, 2M2-2

    # 1. Preprocess: Mirror 2D
    dct1_2d_preprocess_kernel!(backend)(
        p.tmp_real, x, M1, M2, N1, N2;
        ndrange=(N1, N2),
        workgroupsize=(16, 16)
    )
    KernelAbstractions.synchronize(backend)

    # 2. FFT
    mul!(p.tmp_comp, p.complex_plan, p.tmp_real)

    # 3. Postprocess
    dct1_2d_postprocess_kernel!(backend)(
        y, p.tmp_comp, M1, M2, N1, N2;
        ndrange=(M1, M2),
        workgroupsize=(16, 16)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, p::DCT1Plan{T, 3}, x::AbstractArray) where T
    backend = get_backend(x)
    M1, M2, M3 = size(x)
    N1, N2, N3 = 2M1-2, 2M2-2, 2M3-2

    # 1. Preprocess: Mirror 3D
    dct1_3d_preprocess_kernel!(backend)(
        p.tmp_real, x, M1, M2, M3, N1, N2, N3;
        ndrange=(N1, N2, N3),
        workgroupsize=(8, 8, 4)
    )
    KernelAbstractions.synchronize(backend)

    # 2. FFT
    mul!(p.tmp_comp, p.complex_plan, p.tmp_real)

    # 3. Postprocess
    dct1_3d_postprocess_kernel!(backend)(
        y, p.tmp_comp, M1, M2, M3, N1, N2, N3;
        ndrange=(M1, M2, M3),
        workgroupsize=(8, 8, 4)
    )
    KernelAbstractions.synchronize(backend)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, p::IDCT1Plan{T, 1}, x::AbstractArray) where T
    p_fwd = DCT1Plan{T, 1, typeof(p.complex_plan), typeof(p.tmp_real), typeof(p.tmp_comp), typeof(p.region)}(p.complex_plan, p.tmp_real, p.tmp_comp, p.region, p.pinv)
    mul!(y, p_fwd, x)
    M = size(x, 1)
    y ./= (2M - 2)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, p::IDCT1Plan{T, 2}, x::AbstractArray) where T
    p_fwd = DCT1Plan{T, 2, typeof(p.complex_plan), typeof(p.tmp_real), typeof(p.tmp_comp), typeof(p.region)}(p.complex_plan, p.tmp_real, p.tmp_comp, p.region, p.pinv)
    mul!(y, p_fwd, x)
    M1, M2 = size(x)
    y ./= (2M1 - 2) * (2M2 - 2)
    return y
end

function LinearAlgebra.mul!(y::AbstractArray, p::IDCT1Plan{T, 3}, x::AbstractArray) where T
    p_fwd = DCT1Plan{T, 3, typeof(p.complex_plan), typeof(p.tmp_real), typeof(p.tmp_comp), typeof(p.region)}(p.complex_plan, p.tmp_real, p.tmp_comp, p.region, p.pinv)
    mul!(y, p_fwd, x)
    M1, M2, M3 = size(x)
    y ./= (2M1 - 2) * (2M2 - 2) * (2M3 - 2)
    return y
end


# ============================================================================
# 1D Kernels
# ============================================================================

@kernel function dct_1d_preprocess_kernel!(x_prime, @Const(x), N)
    I = @index(Global, Cartesian)
    k = I[1] - 1

    if k < N
        @inbounds val = x[k+1]
        dest = ifelse(iseven(k), k ÷ 2, N - (k + 1) ÷ 2)
        @inbounds x_prime[dest+1] = val
    end
end

@kernel function dct_1d_postprocess_kernel!(y, @Const(X), @Const(w), N)
    I = @index(Global, Cartesian)
    n = I[1] - 1

    if n < N
        W = w[n+1]

        # val1 = X[n]
        val1 = _get_X1_val(X, n, N)

        # Eq 10: y(n) = 2Re(W * X(n))

        y[n+1] = real(W * val1)
    end
end

@kernel function idct_1d_preprocess_kernel!(X_prime, @Const(x), @Const(w), N, limit_n)
    I = @index(Global, Cartesian)
    n = I[1] - 1

    if n <= limit_n
        W = w[n+1]

        idx_a = n
        idx_b = ifelse(n == 0, N, N - n)

        # x is real input (y in math notation)
        v1 = _get_val_safe_1d(x, idx_a, N)
        v2 = _get_val_safe_1d(x, idx_b, N)

        # Inverse Logic:
        # X[n] = 0.5 * conj(W) * (y[n] - j*y[N-n])
        # (0.5 factor is applied in postprocess)

        val = conj(W) * Complex(v1, -v2)
        X_prime[n+1] = val
    end
end

@kernel function idct_1d_postprocess_kernel!(y, @Const(v), N)
    I = @index(Global, Cartesian)
    k = I[1] - 1

    if k < N
        # Inverse permutation: y[k] corresponds to v[dest]
        # Same mapping logic as forward preprocess

        dest = ifelse(iseven(k), k ÷ 2, N - (k + 1) ÷ 2)

        # 0.5 scaling
        y[k+1] = v[dest+1]
    end
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
        T = eltype(y)
        y[n1+1, n2+1] = T(0.5) * real(W2 * term)
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

        v1 = _get_val_safe_2d(x, idx1_a, idx2_a, N1, N2)
        v2 = _get_val_safe_2d(x, idx1_b, idx2_b, N1, N2)
        v3 = _get_val_safe_2d(x, idx1_b, idx2_a, N1, N2)
        v4 = _get_val_safe_2d(x, idx1_a, idx2_b, N1, N2)

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
        y[n1+1, n2+1] = v[idx1+1, idx2+1]
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
        T = eltype(y)
        y[n1+1, n2+1, n3+1] = T(0.25) * real(W3 * inner)
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
        y[n1+1, n2+1, n3+1] = v[idx1+1, idx2+1, idx3+1]
    end
end

# ============================================================================
# DCT-I Kernels
# ============================================================================

@kernel function dct1_1d_preprocess_kernel!(g, @Const(f), M, N)
    I = @index(Global, Cartesian)
    n = I[1] - 1
    if n < N
        # Mirroring: g[n] = f[n] if n < M, else f[N-n]
        idx = ifelse(n < M, n, N - n)
        g[n+1] = f[idx+1]
    end
end

@kernel function dct1_1d_postprocess_kernel!(y, @Const(V), M)
    I = @index(Global, Cartesian)
    k = I[1] - 1
    if k < M
        # Only take real part
        y[k+1] = real(V[k+1])
    end
end

@kernel function dct1_2d_preprocess_kernel!(g, @Const(f), M1, M2, N1, N2)
    I = @index(Global, Cartesian)
    n1, n2 = I[1]-1, I[2]-1
    
    if n1 < N1 && n2 < N2
        i1 = ifelse(n1 < M1, n1, N1 - n1)
        i2 = ifelse(n2 < M2, n2, N2 - n2)
        g[n1+1, n2+1] = f[i1+1, i2+1]
    end
end

@kernel function dct1_2d_postprocess_kernel!(y, @Const(V), M1, M2, N1, N2)
    I = @index(Global, Cartesian)
    k1, k2 = I[1]-1, I[2]-1
    
    if k1 < M1 && k2 < M2
        val = _get_X2_val(V, k1, k2, N1, N2)
        y[k1+1, k2+1] = real(val)
    end
end

@kernel function dct1_3d_preprocess_kernel!(g, @Const(f), M1, M2, M3, N1, N2, N3)
    I = @index(Global, Cartesian)
    n1, n2, n3 = I[1]-1, I[2]-1, I[3]-1
    
    if n1 < N1 && n2 < N2 && n3 < N3
        i1 = ifelse(n1 < M1, n1, N1 - n1)
        i2 = ifelse(n2 < M2, n2, N2 - n2)
        i3 = ifelse(n3 < M3, n3, N3 - n3)
        g[n1+1, n2+1, n3+1] = f[i1+1, i2+1, i3+1]
    end
end

@kernel function dct1_3d_postprocess_kernel!(y, @Const(V), M1, M2, M3, N1, N2, N3)
    I = @index(Global, Cartesian)
    k1, k2, k3 = I[1]-1, I[2]-1, I[3]-1
    
    if k1 < M1 && k2 < M2 && k3 < M3
        # V is (M1, N2, N3).
        # We need (k1, k2, k3).
        # Because the original input g was symmetric in all dims,
        # V satisfies Hermitian symmetry in N2, N3 dims too.
        
        # v_val = V[k1, k2, k3] with symmetry
        val = _get_X3_val(V, k1, k2, k3, N1, N2, N3)
        y[k1+1, k2+1, k3+1] = real(val)
    end
end

# ============================================================================
# Inline Helpers for Kernels
# ============================================================================

@inline function _get_val_safe_1d(x, n, N)
    if n >= N
        return zero(eltype(x))
    end
    @inbounds return x[n+1]
end

@inline function _get_val_safe_2d(x, n1, n2, N1, N2)
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

@inline function _get_X1_val(X, n, N)
    # Branchless Hermitian symmetry reconstruction for 1D RFFT
    half_N = N ÷ 2
    use_sym = n > half_N

    # Direct index
    i_d = n + 1

    # Symmetric index (if n > N/2 => N-n)
    i_s = N - n + 1

    idx = ifelse(use_sym, i_s, i_d)

    @inbounds val = X[idx]
    return ifelse(use_sym, conj(val), val)
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

function dct1(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_dct1(x)
    return p * x
end

function idct1(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_idct1(x)
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
    # Alloc on CPU
    w_cpu = [cispi(-T(n) / (2N)) for n in 0:(N-1)]

    # Alloc on Device and Copy
    w_dev = KernelAbstractions.allocate(backend, Complex{T}, N)
    copyto!(w_dev, w_cpu)

    return w_dev
end
