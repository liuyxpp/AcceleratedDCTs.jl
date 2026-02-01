module VkDCTExt

using AcceleratedDCTs
using CUDA
using Libdl
using LinearAlgebra
using AbstractFFTs

# ============================================================================
# Configuration
# ============================================================================

# Library path: assumes libvkfft_dct.so is in ../lib/VkDCT/ relative to package root
# We need to find the package root.
# Since this is an extension, @__DIR__ is inside /ext/
const LIB_MKDCT_PATH = joinpath(@__DIR__, "..", "lib", "VkDCT", "libvkfft_dct.so")

# ============================================================================
# Plan Definition
# ============================================================================

mutable struct VkFFTDCT1Plan{T, N} <: AbstractFFTs.Plan{T}
    handle::Ptr{Cvoid}      # C++ Context pointer
    sz::NTuple{N, Int}      # Input Dimensions
    region::UnitRange{Int}  # Region (currently only supports 1:N)
    pinv::Base.RefValue{Any} # Cache for inverse plan
    
    function VkFFTDCT1Plan{T, N}(sz::NTuple{N, Int}, region) where {T, N}
        # Check library validity
        if !isfile(LIB_MKDCT_PATH)
            # Warn once? Or error?
            error("VkDCT Library not found at $(abspath(LIB_MKDCT_PATH)). Please compile it using `lib/VkDCT/compile.sh`.")
        end

        # Validate Region: VkFFT shim currently hardcoded for 3D full transform
        if N != 3
             throw(ArgumentError("VkFFT-DCT currently only supports 3D arrays."))
        end
        if region != 1:N
             throw(ArgumentError("VkFFT-DCT currently only supports full 3D transforms."))
        end
        
        # Determine strict types for C call
        nx, ny, nz = sz
        precision = (T == Float64) ? 1 : 0
        
        if T != Float32 && T != Float64
            throw(ArgumentError("VkFFT-DCT only supports Float32 and Float64."))
        end
        
        # Create Plan via C++ Shim
        handle = ccall((:create_dct3d_plan, LIB_MKDCT_PATH),
                       Ptr{Cvoid},
                       (UInt64, UInt64, UInt64, Cint),
                       UInt64(nx), UInt64(ny), UInt64(nz), Cint(precision))
                       
        if handle == C_NULL
            error("VkFFT Plan initialization failed.")
        end
        
        obj = new{T, N}(handle, sz, region, Ref{Any}(nothing))
        finalizer(destroy_vk_plan!, obj)
        return obj
    end
end

function destroy_vk_plan!(p::VkFFTDCT1Plan)
    if p.handle != C_NULL
        if isfile(LIB_MKDCT_PATH) 
             ccall((:destroy_dct3d_plan, LIB_MKDCT_PATH), Cvoid, (Ptr{Cvoid},), p.handle)
        end
        p.handle = C_NULL
    end
end

# Properties
Base.ndims(::VkFFTDCT1Plan{T, N}) where {T, N} = N
Base.eltype(::VkFFTDCT1Plan{T}) where T = T
Base.size(p::VkFFTDCT1Plan) = p.sz

# ============================================================================
# Plan Creation Dispatch (Overloading AcceleratedDCTs.plan_dct1)
# ============================================================================

function AcceleratedDCTs.plan_dct1(x::CuArray{T, 3}, region=1:3) where T <: AbstractFloat
    return VkFFTDCT1Plan{T, 3}(size(x), region)
end

# IDCT Plan
mutable struct VkFFTIDCT1Plan{T, N} <: AbstractFFTs.Plan{T}
    forward_plan::VkFFTDCT1Plan{T, N}
    scale::T
    pinv::Base.RefValue{Any}
end

Base.ndims(::VkFFTIDCT1Plan{T, N}) where {T, N} = N
Base.eltype(::VkFFTIDCT1Plan{T}) where T = T
Base.size(p::VkFFTIDCT1Plan) = p.forward_plan.sz

function AcceleratedDCTs.plan_idct1(x::CuArray{T, 3}, region=1:3) where T <: AbstractFloat
    fwd = AcceleratedDCTs.plan_dct1(x, region)
    
    # Compute scale: 1 / (8 * (Nx-1)(Ny-1)(Nz-1))
    nx, ny, nz = size(x)
    scale = one(T) / (8 * (nx-1)*(ny-1)*(nz-1))
    
    p = VkFFTIDCT1Plan{T, 3}(fwd, scale, Ref{Any}(nothing))
    p.pinv[] = fwd
    fwd.pinv[] = p 
    return p
end

# ============================================================================
# Plan Inversion
# ============================================================================

function Base.inv(p::VkFFTDCT1Plan{T, N}) where {T, N}
    if p.pinv[] === nothing
        nx, ny, nz = p.sz
        scale = one(T) / (8 * (nx-1)*(ny-1)*(nz-1))
        inv_p = VkFFTIDCT1Plan{T, N}(p, scale, Ref{Any}(p))
        p.pinv[] = inv_p
    end
    return p.pinv[]
end

function Base.inv(p::VkFFTIDCT1Plan)
    return p.forward_plan
end

# ============================================================================
# Execution: *
# ============================================================================

function Base.:*(p::VkFFTDCT1Plan, x::CuArray)
    y = similar(x)
    mul!(y, p, x)
    return y
end

function Base.:*(p::VkFFTIDCT1Plan, x::CuArray)
    y = similar(x)
    mul!(y, p, x)
    return y
end

# ============================================================================
# Execution: \ (ldiv)
# ============================================================================

Base.:\(p::VkFFTDCT1Plan, x::CuArray) = inv(p) * x
Base.:\(p::VkFFTIDCT1Plan, x::CuArray) = inv(p) * x

function LinearAlgebra.ldiv!(y::CuArray, p::Union{VkFFTDCT1Plan, VkFFTIDCT1Plan}, x::CuArray)
    inv_p = inv(p)
    mul!(y, inv_p, x)
end

# ============================================================================
# Execution: mul!
# ============================================================================

function LinearAlgebra.mul!(y::CuArray{T, 3}, p::VkFFTDCT1Plan{T, 3}, x::CuArray{T, 3}) where T
    if size(x) != p.sz || size(y) != p.sz
        throw(DimensionMismatch("Array size does not match plan"))
    end
    
    if x !== y
        copyto!(y, x)
    end
    
    st = CUDA.stream()
    
    # exec_dct3d(plan, buffer, stream, inverse)
    res = ccall((:exec_dct3d, LIB_MKDCT_PATH),
                Cint,
                (Ptr{Cvoid}, CuPtr{T}, Ptr{Nothing}, Cint),
                p.handle, y, st.handle, 0) # 0 = Forward
                
    if res != 0
        error("VkFFT Exec Failed: $res")
    end
    
    return y
end

function LinearAlgebra.mul!(y::CuArray{T, 3}, p::VkFFTIDCT1Plan{T, 3}, x::CuArray{T, 3}) where T
    if size(x) != Base.size(p) || size(y) != Base.size(p)
        throw(DimensionMismatch("Array size does not match plan"))
    end
    
    if x !== y
        copyto!(y, x)
    end
    
    st = CUDA.stream()
    
    # Inverse execution (inverse=1)
    # Note: DCT-I is its own inverse (unscaled). 
    # Calling with inverse=0 is safer as we know Forward works and matches FFTW.
    # We handle scaling manually.
    res = ccall((:exec_dct3d, LIB_MKDCT_PATH),
                Cint,
                (Ptr{Cvoid}, CuPtr{T}, Ptr{Nothing}, Cint),
                p.forward_plan.handle, y, st.handle, 0)  
                
    if res != 0
        error("VkFFT Exec Failed: $res")
    end
    
    # Apply Scaling
    y .*= p.scale
    
    return y
end

end # module
