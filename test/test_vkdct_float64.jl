
using Test
using CUDA
using LinearAlgebra
using FFTW

# Include VkDCT module via AcceleratedDCTs extension (implicitly loaded by using AcceleratedDCTs, CUDA)
using AcceleratedDCTs
using AcceleratedDCTs: plan_dct1, dct1, idct1, plan_idct1, ldiv!

# Ground truth function
function fftw_dct1(x::AbstractArray{T}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

@testset "VkDCT Float64 and IDCT Verification" begin
    if !CUDA.functional()
        @warn "CUDA not available, skipping tests"
        return
    end

    @testset "Float64 Support" begin
        sz = (8, 8, 8) # check small size first

        # 1. Float64 Forward
        h_x = rand(Float64, sz...)
        h_y_ref = fftw_dct1(h_x)

        d_x = CuArray(h_x)
        p = plan_dct1(d_x)
        @test p isa AcceleratedDCTs.AbstractFFTs.Plan{Float64}

        d_y = p * d_x
        @test eltype(d_y) == Float64

        h_y_vk = Array(d_y)
        @test h_y_vk ≈ h_y_ref rtol=1e-12 # Higher precision check
    end

    @testset "IDCT Correctness" begin
        sz = (16, 16, 16)
        x = rand(Float64, sz...)
        d_x = CuArray(x)

        # 1. Test dct -> idct roundtrip (normalized)
        d_y = dct1(d_x)
        d_rec = idct1(d_y)

        h_rec = Array(d_rec)
        @test h_rec ≈ x rtol=1e-12

        # 2. Test explicit plan_idct
        pid = plan_idct1(d_x)
        d_rec2 = pid * d_y # Proper usage: IDCT plan * data = result
        @test Array(d_rec2) ≈ x rtol=1e-12

        # 3. Test mul! with IDCT plan
        out = similar(d_x)
        mul!(out, pid, d_y)
        @test Array(out) ≈ x rtol=1e-12

        # 4. Test ldiv! on Forward plan (should produce IDCT)
        p_fwd = plan_dct1(d_x)
        ldiv!(out, p_fwd, d_y)
        @test Array(out) ≈ x rtol=1e-12
    end

    @testset "Performance/Type Stability Check" begin
        # Just ensure it runs for Float32 as well with new generic code
        x32 = rand(Float32, 8, 8, 8)
        d_x32 = CuArray(x32)
        d_rec32 = idct1(dct1(d_x32))
        @test Array(d_rec32) ≈ x32 rtol=1e-5
    end
end
