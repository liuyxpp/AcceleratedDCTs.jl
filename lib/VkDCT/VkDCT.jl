module VkDCT

using CUDA
using Libdl
using LinearAlgebra

# Dynamic library path (current directory)
const LIB_PATH = joinpath(@__DIR__, "libvkfft_dct.so")

# --- Define Plan Struct ---
mutable struct VkDCTPlan{T}
    handle::Ptr{Cvoid}      # C++ Context pointer
    sz::NTuple{3, Int}      # Save dimensions (Nx, Ny, Nz)
    
    function VkDCTPlan{T}(nx::Integer, ny::Integer, nz::Integer) where T
        # Check library file
        if !isfile(LIB_PATH)
            error("Library file not found: $LIB_PATH. Please compile dct_shim.cu first")
        end
        
        # Determine precision flag (0: Float32, 1: Float64)
        precision = (T == Float64) ? 1 : 0
        if T != Float32 && T != Float64
            error("VkDCT only supports Float32 and Float64")
        end

        # Call C++ to create Plan
        # Note: Argument types must strictly match: (UInt64, UInt64, UInt64, Cint)
        handle = ccall((:create_dct3d_plan, LIB_PATH), 
                       Ptr{Cvoid}, 
                       (UInt64, UInt64, UInt64, Cint), 
                       UInt64(nx), UInt64(ny), UInt64(nz), Cint(precision))
        
        if handle == C_NULL
            error("VkFFT Plan creation failed (possibly insufficient VRAM or unsupported size)")
        end

        obj = new{T}(handle, (Int(nx), Int(ny), Int(nz)))

        # Register GC hook to automatically destroy C++ object
        finalizer(destroy_plan!, obj)
        return obj
    end
end

# --- Helper Constructors ---

# 1. Create from array: plan_dct(A)
# 1. Create from array: plan_dct(A)
function plan_dct(A::CuArray{T, 3}) where T
    return VkDCTPlan{T}(size(A, 1), size(A, 2), size(A, 3))
end

# 2. Create from dimensions and type: plan_dct(dims, type)
plan_dct(dims::NTuple{3, <:Integer}, ::Type{T}) where T = VkDCTPlan{T}(dims[1], dims[2], dims[3])

# 3. Default float32: plan_dct(dims)
plan_dct(dims::NTuple{3, <:Integer}) = plan_dct(dims, Float32)

# --- IDCT Functions ---
plan_idct(A::CuArray{T, 3}) where T = plan_dct(A) # Exact same plan for IDCT
plan_idct(dims::NTuple{3, <:Integer}, ::Type{T}) where T = plan_dct(dims, T)
plan_idct(dims::NTuple{3, <:Integer}) = plan_dct(dims)

# --- Destructor ---
function destroy_plan!(p::VkDCTPlan)
    if p.handle != C_NULL
        ccall((:destroy_dct3d_plan, LIB_PATH), Cvoid, (Ptr{Cvoid},), p.handle)
        p.handle = C_NULL
    end
end

# --- Core Execution Function mul! ---

"""
    mul!(Y, P::VkDCTPlan, X)

Execute 3D DCT-I transform.
- Supports In-place (Y === X)
- If Y !== X, automatically executes copyto!(Y, X) then computes in-place on Y
"""
function LinearAlgebra.mul!(Y::CuArray{T, 3}, p::VkDCTPlan{T}, X::CuArray{T, 3}) where T
    # Size validation
    if size(X) != p.sz
        throw(DimensionMismatch("Input size $(size(X)) does not match Plan size $(p.sz)"))
    end
    if size(Y) != p.sz
        throw(DimensionMismatch("Output size $(size(Y)) does not match Plan size $(p.sz)"))
    end

    # Handle non-in-place operations: copy first
    if Y !== X
        copyto!(Y, X)
    end

    # Get current CUDA stream (implement asynchronous execution)
    st = stream()

    # Call C++ execution (inverse=0 for Forward)
    # Using generic CuPtr{T} to support Float32/Float64
    ret = ccall((:exec_dct3d, LIB_PATH), 
                Cint, 
                (Ptr{Cvoid}, CuPtr{T}, Ptr{Nothing}, Cint), 
                p.handle, Y, st.handle, 0)

    if ret != 0
        error("VkFFT Execution Error Code: $ret")
    end

    return Y
end

"""
    ldiv!(Y, P::VkDCTPlan, X)

Execute 3D Inverse DCT-I transform (normalized).
Matches definition: IDCT(DCT(x)) = x
"""
function LinearAlgebra.ldiv!(Y::CuArray{T, 3}, p::VkDCTPlan{T}, X::CuArray{T, 3}) where T
    # 1. Execute unnormalized transform (inverse=1)
    # Note: For DCT-I, forward and inverse algorithms are identical in VkFFT/FFTW (except normalization)
    # But we pass inverse=1 to be semantically correct for the shim
    
    # Size validation & copy
    if size(X) != p.sz
        throw(DimensionMismatch("Input size $(size(X)) does not match Plan size $(p.sz)"))
    end
    if size(Y) != p.sz
        throw(DimensionMismatch("Output size $(size(Y)) does not match Plan size $(p.sz)"))
    end
    if Y !== X; copyto!(Y, X); end

    st = stream()
    
    ret = ccall((:exec_dct3d, LIB_PATH), 
                Cint, 
                (Ptr{Cvoid}, CuPtr{T}, Ptr{Nothing}, Cint), 
                p.handle, Y, st.handle, 1) # inverse=1

    if ret != 0; error("VkFFT Execution Error Code: $ret"); end

    # 2. Normalize
    # Scaling factor for 3D DCT-I: 1 / (8 * (Nx-1)(Ny-1)(Nz-1))
    nx, ny, nz = p.sz
    scale = 1.0 / (8.0 * (nx-1) * (ny-1) * (nz-1))
    
    # Broadcast multiplication by scalar (highly optimized in CUDA.jl)
    Y .*= T(scale)

    return Y
end

# --- Operator Overloading ---

# P * x
Base.:*(p::VkDCTPlan, x::CuArray) = mul!(similar(x), p, x)

# P \ x (Inverse transform with normalization)
Base.:\(p::VkDCTPlan, x::CuArray) = ldiv!(similar(x), p, x)

# Helper functions
dct(x::CuArray) = plan_dct(x) * x
idct(x::CuArray) = plan_idct(x) \ x

export plan_dct, plan_idct, dct, idct, mul!, ldiv!

end