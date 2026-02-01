import FFTW
import AbstractFFTs
using Test
using AcceleratedDCTs
using AcceleratedDCTs: dct1_mirror, idct1_mirror, plan_dct1_mirror, plan_idct1_mirror, DCT1MirrorPlan, IDCT1MirrorPlan
using LinearAlgebra: mul!, ldiv!

# Helper function to compute FFTW's DCT-I (REDFT00)
function fftw_dct1(x::AbstractVector{T}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

function fftw_dct1(x::AbstractMatrix{T}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

function fftw_dct1(x::AbstractArray{T, 3}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

@testset "Optimized DCT-I Mirror (1D)" begin
    @testset "Consistency with FFTW DCT-I" begin
        for M in [4, 5, 8, 9, 16, 17, 32, 33, 64]
            x = rand(Float64, M)
            y_fftw = fftw_dct1(x)
            y_opt = dct1_mirror(x)
            @test y_opt ≈ y_fftw atol=1e-12
        end
    end

    @testset "Roundtrip Accuracy" begin
        for M in [4, 5, 8, 9, 16, 17, 32, 33, 64]
            x = rand(M)
            y = dct1_mirror(x)
            x_rec = idct1_mirror(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "Core API Functionality" begin
        M = 16
        x = rand(Float64, M)

        # Plan creation
        p = plan_dct1_mirror(x)
        @test p isa AbstractFFTs.Plan{Float64}
        @test p isa DCT1MirrorPlan

        # Forward transform
        y = p * x
        @test dct1_mirror(x) ≈ y

        # Non-allocating mul!
        y_out = similar(x)
        mul!(y_out, p, x)
        @test y_out ≈ y

        # Inverse transform
        x_rec = idct1_mirror(y)
        @test x_rec ≈ x atol=1e-12

        # Plan inverse via \
        x_ldiv = p \ y
        @test x_ldiv ≈ x atol=1e-12

        # Explicit inverse plan
        p_inv = inv(p)
        @test p_inv isa AbstractFFTs.Plan{Float64}
        @test p_inv isa IDCT1MirrorPlan
        x_inv = p_inv * y
        @test x_inv ≈ x atol=1e-12

        # ldiv!
        x_ldiv2 = similar(x)
        ldiv!(x_ldiv2, p, y)
        @test x_ldiv2 ≈ x atol=1e-12
    end
end

@testset "Optimized DCT-I Mirror (2D)" begin
    @testset "Consistency with FFTW DCT-I" begin
        sizes = [(4, 4), (5, 5), (8, 8), (8, 9), (9, 8), (16, 16), (4, 8)]
        for (M1, M2) in sizes
            x = rand(Float64, M1, M2)
            y_fftw = fftw_dct1(x)
            y_opt = dct1_mirror(x)
            @test y_opt ≈ y_fftw atol=1e-11
        end
    end

    @testset "Roundtrip Accuracy" begin
        sizes = [(4, 4), (5, 5), (8, 8), (8, 9), (9, 8), (16, 16), (4, 8)]
        for (M1, M2) in sizes
            x = rand(M1, M2)
            y = dct1_mirror(x)
            x_rec = idct1_mirror(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "Core API Functionality" begin
        M1, M2 = 8, 8
        x = rand(Float64, M1, M2)

        # Plan creation
        p = plan_dct1_mirror(x)
        @test p isa AbstractFFTs.Plan{Float64}

        # Forward transform
        y = p * x
        @test dct1_mirror(x) ≈ y

        # Non-allocating mul!
        y_out = similar(x)
        mul!(y_out, p, x)
        @test y_out ≈ y

        # Inverse transform
        x_rec = idct1_mirror(y)
        @test x_rec ≈ x atol=1e-12
    end
end

@testset "Optimized DCT-I Mirror (3D)" begin
    @testset "Consistency with FFTW DCT-I" begin
        sizes = [(4, 4, 4), (5, 5, 5), (4, 5, 6), (8, 8, 8), (4, 8, 4)]
        for sz in sizes
            x = rand(Float64, sz...)
            y_fftw = fftw_dct1(x)
            y_opt = dct1_mirror(x)
            @test y_opt ≈ y_fftw atol=1e-10
        end
    end

    @testset "Roundtrip Accuracy" begin
        sizes = [(4, 4, 4), (5, 5, 5), (4, 5, 6), (8, 8, 8), (4, 8, 4)]
        for sz in sizes
            x = rand(sz...)
            y = dct1_mirror(x)
            x_rec = idct1_mirror(y)
            @test x_rec ≈ x atol=1e-11
        end
    end

    @testset "Core API Functionality" begin
        M1, M2, M3 = 4, 4, 4
        x = rand(Float64, M1, M2, M3)

        # Plan creation
        p = plan_dct1_mirror(x)
        @test p isa AbstractFFTs.Plan{Float64}

        # Forward transform
        y = p * x
        @test dct1_mirror(x) ≈ y

        # Non-allocating mul!
        y_out = similar(x)
        mul!(y_out, p, x)
        @test y_out ≈ y

        # Inverse transform
        x_rec = idct1_mirror(y)
        @test x_rec ≈ x atol=1e-12
    end
end
