# ============================================================================
# DCT-I / IDCT-I FFTW-Based Implementation for CPU
# ============================================================================
#
# This file provides optimized DCT-I/IDCT-I for CPU using FFTW's native
# REDFT00 transform, which is significantly faster than the mirroring + FFT
# approach used for GPU.

import FFTW

# ============================================================================
# FFTW-Based Plan Definitions
# ============================================================================

"""
    FFTWBasedDCT1Plan

DCT-I Plan for CPU arrays using FFTW's native REDFT00.
"""
struct FFTWBasedDCT1Plan{T, N, P} <: AbstractFFTs.Plan{T}
    fftw_plan::P         # FFTW r2r plan (REDFT00)
    sz::NTuple{N, Int}   # Input size
    region::UnitRange{Int}
    pinv::Base.RefValue{Any}
end

"""
    FFTWBasedIDCT1Plan

IDCT-I Plan for CPU arrays using FFTW's native REDFT00 with scaling.
"""
struct FFTWBasedIDCT1Plan{T, N, P} <: AbstractFFTs.Plan{T}
    fftw_plan::P         # FFTW r2r plan (same as DCT-I, self-inverse)
    sz::NTuple{N, Int}   # Input size
    scale::T             # Normalization factor: 1 / ∏ 2*(Mi-1)
    region::UnitRange{Int}
    pinv::Base.RefValue{Any}
end

# Properties
Base.ndims(::FFTWBasedDCT1Plan{T, N}) where {T, N} = N
Base.ndims(::FFTWBasedIDCT1Plan{T, N}) where {T, N} = N
Base.eltype(::FFTWBasedDCT1Plan{T}) where T = T
Base.eltype(::FFTWBasedIDCT1Plan{T}) where T = T
Base.size(p::FFTWBasedDCT1Plan) = p.sz
Base.size(p::FFTWBasedIDCT1Plan) = p.sz

# ============================================================================
# Scaling Factor Computation
# ============================================================================

"""
Compute the normalization scale for IDCT-I.
For DCT-I, the inverse is DCT-I itself scaled by 1/∏(2*(Mi-1)) for each dimension.
"""
function _compute_idct1_scale(sz::NTuple{N, Int}, region) where N
    scale = one(Float64)
    for d in region
        scale *= 2 * (sz[d] - 1)
    end
    return 1 / scale
end

# ============================================================================
# Plan Creation (CPU-specific dispatch)
# ============================================================================

function plan_dct1(x::Array{T, N}, region=1:N; flags=FFTW.ESTIMATE) where {T <: AbstractFloat, N}
    if region != 1:N
        error("FFTW-based DCT1: partial region not yet supported. Use full region 1:$N.")
    end
    
    # FFTW r2r with REDFT00 for all dimensions
    kinds = ntuple(_ -> FFTW.REDFT00, N)
    fftw_plan = FFTW.plan_r2r(x, kinds; flags=flags)
    
    return FFTWBasedDCT1Plan{T, N, typeof(fftw_plan)}(
        fftw_plan, size(x), region, Ref{Any}(nothing)
    )
end

function plan_idct1(x::Array{T, N}, region=1:N; flags=FFTW.ESTIMATE) where {T <: AbstractFloat, N}
    if region != 1:N
        error("FFTW-based IDCT1: partial region not yet supported. Use full region 1:$N.")
    end
    
    # FFTW r2r with REDFT00 (DCT-I is its own inverse up to scaling)
    kinds = ntuple(_ -> FFTW.REDFT00, N)
    fftw_plan = FFTW.plan_r2r(x, kinds; flags=flags)
    
    scale = T(_compute_idct1_scale(size(x), region))
    
    return FFTWBasedIDCT1Plan{T, N, typeof(fftw_plan)}(
        fftw_plan, size(x), scale, region, Ref{Any}(nothing)
    )
end

# ============================================================================
# Plan Inversion
# ============================================================================

function Base.inv(p::FFTWBasedDCT1Plan{T, N}) where {T, N}
    if p.pinv[] === nothing
        x = zeros(T, p.sz...)
        p.pinv[] = plan_idct1(x, p.region)
    end
    return p.pinv[]
end

function Base.inv(p::FFTWBasedIDCT1Plan{T, N}) where {T, N}
    if p.pinv[] === nothing
        x = zeros(T, p.sz...)
        p.pinv[] = plan_dct1(x, p.region)
    end
    return p.pinv[]
end

# ============================================================================
# Execution: *
# ============================================================================

function Base.:*(p::FFTWBasedDCT1Plan, x::Array)
    y = similar(x)
    mul!(y, p, x)
    return y
end

function Base.:*(p::FFTWBasedIDCT1Plan, x::Array)
    y = similar(x)
    mul!(y, p, x)
    return y
end

# ============================================================================
# Execution: \ (ldiv)
# ============================================================================

function Base.:\(p::FFTWBasedDCT1Plan, x::Array)
    inv_p = inv(p)
    return inv_p * x
end

function Base.:\(p::FFTWBasedIDCT1Plan, x::Array)
    inv_p = inv(p)
    return inv_p * x
end

# Support ldiv!(y, plan, x) => mul!(y, inv(plan), x)
function LinearAlgebra.ldiv!(y::Array, p::Union{FFTWBasedDCT1Plan, FFTWBasedIDCT1Plan}, x::Array)
    inv_p = inv(p)
    mul!(y, inv_p, x)
end

# ============================================================================
# Execution: mul! (Forward DCT-I)
# ============================================================================

function LinearAlgebra.mul!(y::Array{T}, p::FFTWBasedDCT1Plan{T}, x::Array{T}) where T
    # FFTW r2r computes DCT-I directly
    mul!(y, p.fftw_plan, x)
    return y
end

# ============================================================================
# Execution: mul! (Inverse DCT-I with scaling)
# ============================================================================

function LinearAlgebra.mul!(y::Array{T}, p::FFTWBasedIDCT1Plan{T}, x::Array{T}) where T
    # DCT-I is self-inverse, just needs scaling
    mul!(y, p.fftw_plan, x)
    y .*= p.scale
    return y
end
