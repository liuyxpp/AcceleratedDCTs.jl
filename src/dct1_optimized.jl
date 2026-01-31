# ============================================================================
# DCT-I / IDCT-I Optimized Implementation
# ============================================================================

# ============================================================================
# DCT-I Plan Definitions
# ============================================================================

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
Base.ndims(::DCT1Plan{T, N}) where {T, N} = N
Base.ndims(::IDCT1Plan{T, N}) where {T, N} = N
Base.eltype(::DCT1Plan{T}) where T = T
Base.eltype(::IDCT1Plan{T}) where T = T

# Size for DCT1Plan: tmp_real has size (N1, N2, ...) where Ni = 2Mi-2.
# We return (M1, M2, ...)
function Base.size(p::Union{DCT1Plan{T, N}, IDCT1Plan{T, N}}) where {T, N}
    return ntuple(i -> (size(p.tmp_real, i) ÷ 2) + 1, N)
end

# ============================================================================
# Plan Inversion
# ============================================================================

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
function Base.:*(p::DCT1Plan, x::AbstractArray)
    y = similar(x)
    mul!(y, p, x)
    return y
end

function Base.:*(p::IDCT1Plan, x::AbstractArray)
    y = similar(x)
    mul!(y, p, x)
    return y
end

# Execution: \ (ldiv)
function Base.:\(p::DCT1Plan, x::AbstractArray)
    inv_p = inv(p)
    return inv_p * x
end

function Base.:\(p::IDCT1Plan, x::AbstractArray)
    inv_p = inv(p)
    return inv_p * x
end

# Support ldiv!(y, plan, x) => mul!(y, inv(plan), x)
function LinearAlgebra.ldiv!(y::AbstractArray, p::Union{DCT1Plan, IDCT1Plan}, x::AbstractArray)
    inv_p = inv(p)
    mul!(y, inv_p, x)
end

# ============================================================================
# Plan Creation
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

# ============================================================================
# Execution (mul!)
# ============================================================================

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
# Convenience Functions
# ============================================================================

function dct1(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_dct1(x)
    return p * x
end

function idct1(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_idct1(x)
    return p * x
end
