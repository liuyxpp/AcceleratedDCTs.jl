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
struct DCT1Plan{T, N, P, BComp, Region} <: AbstractFFTs.Plan{T}
    complex_plans::P     # Tuple of 1D FFT plans (one for each dimension in region)
    tmp_comp::BComp      # Complex buffer for Separable FFT (size M)
    region::Region       # Dimensions to transform
    pinv::Base.RefValue{Any} # Cache for inverse plan
end

"""
    IDCT1Plan

Optimized IDCT-I Plan for device-agnostic execution.
"""
struct IDCT1Plan{T, N, P, BComp, Region} <: AbstractFFTs.Plan{T}
    complex_plans::P     # Tuple of 1D FFT plans (same as DCT-I)
    tmp_comp::BComp      # Complex buffer
    region::Region       # Dimensions
    pinv::Base.RefValue{Any} # Cache for forward plan
end

# Properties
Base.ndims(::DCT1Plan{T, N}) where {T, N} = N
Base.ndims(::IDCT1Plan{T, N}) where {T, N} = N
Base.eltype(::DCT1Plan{T}) where T = T
Base.eltype(::IDCT1Plan{T}) where T = T
Base.size(p::DCT1Plan) = size(p.tmp_comp)
Base.size(p::IDCT1Plan) = size(p.tmp_comp)

# ============================================================================
# Plan Inversion
# ============================================================================

function Base.inv(p::DCT1Plan{T, N}) where {T, N}
    if p.pinv[] === nothing
        # IDCT-I uses the same resource, just different dispatch
        # We can share the tmp_comp and plans if appropriate, but plans usually bind to array
        # Effectively IDCT1 is normalized DCT1, structure is identical.
        # We construct a new IDCTPlan with the SAME complex_plans and buffer?
        # Safe if plans are modifying tmp_comp in place and we don't alias?
        # Yes, we can share.
        p.pinv[] = IDCT1Plan{T, N, typeof(p.complex_plans), typeof(p.tmp_comp), typeof(p.region)}(
            p.complex_plans, p.tmp_comp, p.region, Ref{Any}(p)
        )
    end
    return p.pinv[]
end

function Base.inv(p::IDCT1Plan{T, N}) where {T, N}
    if p.pinv[] === nothing
        p.pinv[] = DCT1Plan{T, N, typeof(p.complex_plans), typeof(p.tmp_comp), typeof(p.region)}(
            p.complex_plans, p.tmp_comp, p.region, Ref{Any}(p)
        )
    end
    return p.pinv[]
end

# Execution: *
function Base.:*(p::DCT1Plan, x::AbstractArray)
    y = copy(x) # DCT-I is in-place capable, but * operator returns new array
    mul!(y, p, y) # Assume x is source? No, mul!(y, p, x)
    return y
end

function Base.:*(p::IDCT1Plan, x::AbstractArray)
    y = copy(x)
    mul!(y, p, y)
    return y
end

function Base.:*(p::DCT1Plan, x::AbstractArray, y::AbstractArray)
    # This signature is not standard for *
    mul!(y, p, x)
end

# Execution: \ (ldiv)
function Base.:\(p::DCT1Plan, x::AbstractArray)
    y = similar(x)
    ldiv!(y, p, x)
    return y
end

function Base.:\(p::IDCT1Plan, x::AbstractArray)
    y = similar(x)
    ldiv!(y, p, x)
    return y
end

function LinearAlgebra.ldiv!(y::AbstractArray, p::Union{DCT1Plan, IDCT1Plan}, x::AbstractArray)
    inv_p = inv(p)
    mul!(y, inv_p, x)
end

# ============================================================================
# Plan Creation
# ============================================================================


function plan_dct1(x::AbstractArray{T, N}, region=1:N) where {T <: Real, N}
    # Separable Plan
    backend = get_backend(x)
    
    # 1. Complex Buffer (same size as x)
    tmp_comp = KernelAbstractions.allocate(backend, Complex{T}, size(x))
    
    # 2. Plan 1D FFTs reduction
    # We use FFT of size N = M-1.
    # We must plan on a view of tmp_comp that trims the last element along dimension i.
    
    complex_plans = ntuple(i -> begin
        if i in region
            M_dim = size(x, i)
            N_dim = M_dim - 1
            if N_dim < 1
                nothing
            else
                # STRATEGY: Permuted Packed Layout.
                # We arrange tmp_comp such that the transform dimension 'i' becomes dimension 1.
                # Shape: (N_dim, size(x, 1), ..., size(x, i-1), size(x, i+1), ...)
                
                sz_perm = ntuple(N) do k
                    if k == 1
                        N_dim
                    elseif k <= i
                        size(x, k-1)
                    else
                        size(x, k)
                    end
                end
                
                len_packed = prod(sz_perm)
                v_linear = view(reshape(tmp_comp, :), 1:len_packed)
                v_permuted = reshape(v_linear, sz_perm)
                
                # Always FFT along dimension 1 (Unit Stride)
                plan_fft!(v_permuted, 1)
            end
        else
            nothing
        end
    end, N)
    
    return DCT1Plan{T, N, typeof(complex_plans), typeof(tmp_comp), typeof(region)}(
        complex_plans, tmp_comp, region, Ref{Any}(nothing)
    )
end

function plan_idct1(x::AbstractArray{T, N}, region=1:N) where {T <: Real, N}
    p = plan_dct1(x, region)
    return inv(p)
end

# ============================================================================
# Execution (mul!)
# ============================================================================

function LinearAlgebra.mul!(y::AbstractArray{T, N}, p::DCT1Plan{T, N}, x::AbstractArray{T, N}) where {T, N}
    if x !== y
        copyto!(y, x)
    end
    
    backend = get_backend(y)
    
    for d in p.region
        dim_len = size(y, d)
        N_dim = dim_len - 1
        
        # Prepare permuted packed view
        # Size: (N_dim, M1, ..., M_d-1, M_d+1, ...)
        sz_perm = ntuple(N) do k
            if k == 1
                N_dim
            elseif k <= d
                size(y, k-1)
            else
                size(y, k)
            end
        end
        len_packed = prod(sz_perm)
        v_linear = view(reshape(p.tmp_comp, :), 1:len_packed)
        v_permuted = reshape(v_linear, sz_perm)
        
        # 1. Preprocess: Real M -> Buffer N (Permuted: Dim d -> Dim 1)
        dct1_sep_preprocess_permuted_kernel!(backend)(
            v_permuted, y, d, dim_len;
            ndrange=size(y),
            workgroupsize=_pick_workgroup(backend, size(y))
        )
        KernelAbstractions.synchronize(backend)
        
        # 2. FFT: Size N (Unit Stride along Dim 1)
        plan_d = p.complex_plans[d]
        if plan_d !== nothing
             mul!(v_permuted, plan_d, v_permuted)
        end
        
        # 3. Postprocess: Buffer N (Permuted) -> Real M
        dct1_sep_postprocess_permuted_kernel!(backend)(
            y, v_permuted, d, dim_len;
            ndrange=size(y),
            workgroupsize=_pick_workgroup(backend, size(y))
        )
        KernelAbstractions.synchronize(backend)
    end
    
    return y
end


function LinearAlgebra.mul!(y::AbstractArray{T, N}, p::IDCT1Plan{T, N}, x::AbstractArray{T, N}) where {T, N}
    if x !== y
        copyto!(y, x)
    end
    
    backend = get_backend(y)
    
    for d in p.region
        dim_len = size(y, d)
        N_dim = dim_len - 1
        
        # Prepare permuted packed view
        sz_perm = ntuple(N) do k
            if k == 1
                N_dim
            elseif k <= d
                size(y, k-1)
            else
                size(y, k)
            end
        end
        len_packed = prod(sz_perm)
        v_linear = view(reshape(p.tmp_comp, :), 1:len_packed)
        v_permuted = reshape(v_linear, sz_perm)
        
        # 1. Preprocess
        idct1_sep_preprocess_permuted_kernel!(backend)(
            v_permuted, y, d, dim_len;
            ndrange=size(y),
            workgroupsize=_pick_workgroup(backend, size(y))
        )
        KernelAbstractions.synchronize(backend)
        
        # 2. FFT (Unit Stride)
        plan_d = p.complex_plans[d]
        if plan_d !== nothing
             mul!(v_permuted, plan_d, v_permuted)
        end
        
        # 3. Postprocess
        idct1_sep_postprocess_permuted_kernel!(backend)(
            y, v_permuted, d, dim_len;
            ndrange=size(y),
            workgroupsize=_pick_workgroup(backend, size(y))
        )
        KernelAbstractions.synchronize(backend)
    end
    
    return y
end

# ============================================================================
# Kernels (N-Dimensional Separable via Cartesian Indexing)
# ============================================================================

# Helper to get index modified along one dimension
@inline function _set_idx(I::CartesianIndex{N}, dim, val) where N
    # This is slow if not carefully implemented. 
    # Use tuple manipulation.
    t = Tuple(I)
    Base.setindex(t, val, dim)
    return CartesianIndex(t)
end
# Optimized simpler one:
@inline function _get_val_dim(A, I, dim, idx_val)
    # Construct index where I[dim] = idx_val
    # I is CartesianIndex.
    # We can assume A is generic AbstractArray.
    # GPU friendly?
    # Ideally we avoid allocation.
    
    # Using `CartesianIndex` construction inside kernel is usually fine on modern KA.
    # Let's trust KA.
    
    # Better: linear index stride calculation? Too complex for generic.
    
    # Tuple-based construction
    args = ntuple(i -> ifelse(i == dim, idx_val, I[i]), Val(ndims(A)))
    return A[args...]
    # args... splatting is efficient with tuples.
end

@inline function _set_val_dim!(A, val, I, dim, idx_val)
    args = ntuple(i -> ifelse(i == dim, idx_val, I[i]), Val(ndims(A)))
    A[args...] = val
end



@inline function _set_val_permuted!(Z, val, I, dim, idx_val)
    # Z has permuted dimensions: (dim, 1, ..., dim-1, dim+1, ...)
    # I is from original space (1, ..., dim, ...)
    
    # Map index I to J for Z
    # J[1] = idx_val
    # J[k] = I[k-1] for k <= dim
    # J[k] = I[k] for k > dim
    
    # Faster: build tuple
    # If N=3, dim=2. I=(i,j,k).
    # Z dims: (dim 2, dim 1, dim 3).
    # Z indices: (j_new, i, k).
    
    args = ntuple(Val(ndims(Z))) do k
        if k == 1
            idx_val
        elseif k <= dim
            I[k-1]
        else
            I[k]
        end
    end
    Z[args...] = val
end

@inline function _get_val_permuted(Z, I, dim, idx_val)
    # Inverse of set
    args = ntuple(Val(ndims(Z))) do k
        if k == 1
            idx_val
        elseif k <= dim
            I[k-1]
        else
            I[k]
        end
    end
    return Z[args...]
end

# ------------------------------------------------------------------
# Forward DCT-I Kernels
# ------------------------------------------------------------------

@kernel function dct1_sep_preprocess_permuted_kernel!(Z, @Const(x), dim, M)
    I = @index(Global, Cartesian)
    n = I[dim] - 1
    N = M - 1
    
    if n < N
        idx_even = 2n
        real_even = 0.0
        if idx_even <= N
            real_even = Float64(_get_val_dim(x, I, dim, idx_even + 1))
        else
            k = 2N - idx_even
            real_even = Float64(_get_val_dim(x, I, dim, k + 1))
        end
        
        idx_odd = 2n + 1
        real_odd = 0.0
        if idx_odd <= N
            real_odd = Float64(_get_val_dim(x, I, dim, idx_odd + 1))
        else
            k = 2N - idx_odd
            real_odd = Float64(_get_val_dim(x, I, dim, k + 1))
        end
        
        val = Complex(real_even, real_odd)
        # Write PERMUTED
        _set_val_permuted!(Z, val, I, dim, n + 1)
    end
end

@kernel function dct1_sep_postprocess_permuted_kernel!(y, @Const(Z), dim, M)
    I = @index(Global, Cartesian)
    k = I[dim] - 1
    N = M - 1
    
    if k <= N
        # Logic: G[k] = Re [ 0.5(Z[k] + Z[N-k]*) - 0.5i e^{-i pi k/N} (Z[k] - Z[N-k]*) ]
        idx_k = k % N
        # Read PERMUTED
        Zk = _get_val_permuted(Z, I, dim, idx_k + 1)
        
        idx_mk = (N - k) % N
        Zmk = _get_val_permuted(Z, I, dim, idx_mk + 1)
        
        term1 = 0.5 * (Zk + conj(Zmk))
        term2 = 0.5 * (Zk - conj(Zmk))
        
        angle = -Float64(pi) * k / N
        W = cis(angle)
        
        val = term1 - im * W * term2
        
        _set_val_dim!(y, real(val), I, dim, k + 1)
    end
end

# ------------------------------------------------------------------
# Inverse DCT-I Kernels
# ------------------------------------------------------------------

@kernel function idct1_sep_preprocess_permuted_kernel!(Z, @Const(y), dim, M)
    # Inverse logic.
    # Same as forward but with scaling and slightly different phase?
    # Actually DCT-I is exactly its own inverse multiplied by 2/(N) ?
    # Standard: DCT-I * DCT-I = (2N) * Identity ?
    
    # Forward: y -> Y
    # Inverse: Y -> y * Scale
    # If we use the SAME reduction (y is even symmetric),
    # DCT-I(y) = Forward Algorithm(y)
    
    # The scaling factor for DCT-I (N+1 points):
    # D_I * D_I = (2N) * I (approx, boundaries have weights)
    # Actually:
    # y[0], y[N] have weight 1?
    # Standard definition:
    # X_k = 0.5 (x_0 + (-1)^k x_N) + sum_{n=1}^{N-1} x_n cos(pi n k / N)
    # This definition is symmetric.
    # Inverse is X_k -> x_n * (2/N).
    # Except for k=0,N?
    # FFTW REDFT00:
    # Y_k = x_0 + (-1)^k x_N + 2 sum_{n=1}^{N-1} x_n cos(...)
    # This is 2 * Standard.
    # Its inverse is REDFT00 / (2N).
    
    # So we can just run the EXACT SAME Forward Algorithm, and then scale by 1/(2N).
    # Wait, 2N = 2(M-1).
    
    # So we re-use forward kernels!
    # Just call `dct1_sep_preprocess_kernel!`
    # And then postprocess with scaling.
    
    # To reduce code duplication, we can just call the same kernel logic.
    # But we need to apply scaling.
    # Can we apply scaling in postprocess?
    # Yes.
    
    # So IDCT preprocess == DCT preprocess.
    I = @index(Global, Cartesian)
    n = I[dim] - 1
    N = M - 1
    
    if n < N
         # Copy paste logic from forward or call inline?
         # Inline logic duplication for safety.
        idx_even = 2n
        real_even = 0.0
        if idx_even <= N
            real_even = Float64(_get_val_dim(y, I, dim, idx_even + 1))
        else
            k = 2N - idx_even
            real_even = Float64(_get_val_dim(y, I, dim, k + 1))
        end
        
        idx_odd = 2n + 1
        real_odd = 0.0
        if idx_odd <= N
            real_odd = Float64(_get_val_dim(y, I, dim, idx_odd + 1))
        else
            k = 2N - idx_odd
            real_odd = Float64(_get_val_dim(y, I, dim, k + 1))
        end
        val = Complex(real_even, real_odd)
        # Write PERMUTED
        _set_val_permuted!(Z, val, I, dim, n + 1)
    # elseif n == N
    #    _set_val_dim!(Z, zero(eltype(Z)), I, dim, n + 1)
    end
end

@kernel function idct1_sep_postprocess_permuted_kernel!(y, @Const(Z), dim, M)
    I = @index(Global, Cartesian)
    k = I[dim] - 1
    N = M - 1
    
    if k <= N
        idx_k = k % N
        # Read PERMUTED
        Zk = _get_val_permuted(Z, I, dim, idx_k + 1)
        idx_mk = (N - k) % N
        Zmk = _get_val_permuted(Z, I, dim, idx_mk + 1)
        
        term1 = 0.5 * (Zk + conj(Zmk))
        term2 = 0.5 * (Zk - conj(Zmk))
        angle = -Float64(pi) * k / N
        W = cis(angle)
        val = term1 - im * W * term2
        
        # Scaling for IDCT
        # Factor is 1 / (2N)
        scale = 1.0 / (2*N)
        
        _set_val_dim!(y, real(val) * scale, I, dim, k + 1)
    end
end

# Utilities
@inline function _pick_workgroup(backend, sz)
    # Heuristic for workgroup size
    # 256 for 1D, (16,16) for 2D, ...
    # Simplified: (256, 1, 1...) clipped
    return (256,) # KA auto-pads?
    # Better: let KA decide or simple defaults.
    # If ndims=1: (256,)
    # If ndims=2: (16,16)
    len = length(sz)
    if len == 1
        return (256,)
    elseif len == 2
        return (16, 16)
    elseif len == 3
        return (8, 8, 4)
    else
        return (256,) # Fallback
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

function dct1!(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_dct1(x)
    mul!(x, p, x)
    return x
end

function idct1!(x::AbstractArray{T, N}) where {T <: Real, N}
    p = plan_idct1(x)
    mul!(x, p, x)
    return x
end
