import FFTW
import AbstractFFTs
using Test
using AcceleratedDCTs
using AcceleratedDCTs: dct1, idct1, plan_dct1, plan_idct1, DCT1Plan, IDCT1Plan
using LinearAlgebra: mul!, ldiv!

# Helper function to compute FFTW's IDCT-I
# For DCT-I (REDFT00), the inverse is simply DCT-I again, scaled by 1/(2*(M-1))
function fftw_idct1(x::AbstractVector{T}) where T
    M = length(x)
    return FFTW.r2r(x, FFTW.REDFT00) ./ (2 * (M - 1))
end

function fftw_idct1(x::AbstractMatrix{T}) where T
    M1, M2 = size(x)
    return FFTW.r2r(x, FFTW.REDFT00) ./ (2 * (M1 - 1) * 2 * (M2 - 1))
end

function fftw_idct1(x::AbstractArray{T, 3}) where T
    M1, M2, M3 = size(x)
    return FFTW.r2r(x, FFTW.REDFT00) ./ (2 * (M1 - 1) * 2 * (M2 - 1) * 2 * (M3 - 1))
end

@testset "Optimized IDCT-I (1D)" begin
    @testset "Consistency with FFTW IDCT-I" begin
        for M in [4, 5, 8, 9, 16, 17, 32, 33, 64]
            x = rand(Float64, M)
            y_fftw = fftw_idct1(x)
            y_opt = idct1(x)
            @test y_opt ≈ y_fftw atol=1e-12
        end
    end

    @testset "Roundtrip Accuracy (IDCT -> DCT)" begin
        for M in [4, 5, 8, 9, 16, 17, 32, 33, 64]
            x = rand(M)
            y = idct1(x)
            x_rec = dct1(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "Core API Functionality" begin
        M = 16
        x = rand(Float64, M)

        # Plan creation
        p = plan_idct1(x)
        @test p isa AbstractFFTs.Plan{Float64}

        # Inverse transform
        y = p * x
        @test idct1(x) ≈ y

        # Non-allocating mul!
        y_out = similar(x)
        mul!(y_out, p, x)
        @test y_out ≈ y

        # Forward transform recovers original
        x_rec = dct1(y)
        @test x_rec ≈ x atol=1e-12

        # Plan inverse via \
        x_ldiv = p \ y
        @test x_ldiv ≈ x atol=1e-12

        # Explicit inverse plan
        p_inv = inv(p)
        @test p_inv isa AbstractFFTs.Plan{Float64}
        x_inv = p_inv * y
        @test x_inv ≈ x atol=1e-12

        # ldiv!
        x_ldiv2 = similar(x)
        ldiv!(x_ldiv2, p, y)
        @test x_ldiv2 ≈ x atol=1e-12
    end

    @testset "Float32" begin
        M = 16
        x = rand(Float32, M)
        y = idct1(x)
        x_rec = dct1(y)
        @test x_rec ≈ x atol=1e-5

        # Consistency with FFTW
        y_fftw = fftw_idct1(x)
        @test y ≈ y_fftw atol=1e-5
    end
end

@testset "Optimized IDCT-I (2D)" begin
    @testset "Consistency with FFTW IDCT-I" begin
        sizes = [(4, 4), (5, 5), (8, 8), (8, 9), (9, 8), (16, 16), (4, 8)]
        for (M1, M2) in sizes
            x = rand(Float64, M1, M2)
            y_fftw = fftw_idct1(x)
            y_opt = idct1(x)
            @test y_opt ≈ y_fftw atol=1e-11
        end
    end

    @testset "Roundtrip Accuracy (IDCT -> DCT)" begin
        sizes = [(4, 4), (5, 5), (8, 8), (8, 9), (9, 8), (16, 16), (4, 8)]
        for (M1, M2) in sizes
            x = rand(M1, M2)
            y = idct1(x)
            x_rec = dct1(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "Core API Functionality" begin
        M1, M2 = 8, 8
        x = rand(Float64, M1, M2)

        # Plan creation
        p = plan_idct1(x)
        @test p isa AbstractFFTs.Plan{Float64}

        # Inverse transform
        y = p * x
        @test idct1(x) ≈ y

        # Non-allocating mul!
        y_out = similar(x)
        mul!(y_out, p, x)
        @test y_out ≈ y

        # Forward transform recovers original
        x_rec = dct1(y)
        @test x_rec ≈ x atol=1e-12

        # Plan inverse via \
        x_ldiv = p \ y
        @test x_ldiv ≈ x atol=1e-12

        # Explicit inverse plan
        p_inv = inv(p)
        @test p_inv isa AbstractFFTs.Plan{Float64}
        x_inv = p_inv * y
        @test x_inv ≈ x atol=1e-12
    end

    @testset "Linearity" begin
        M1, M2 = 8, 8
        x1 = rand(M1, M2)
        x2 = rand(M1, M2)
        a, b = 2.5, -1.5

        y_comb = idct1(a .* x1 .+ b .* x2)
        y_sep = a .* idct1(x1) .+ b .* idct1(x2)

        @test y_comb ≈ y_sep atol=1e-12
    end

    @testset "Float32" begin
        M1, M2 = 8, 8
        x = rand(Float32, M1, M2)
        y = idct1(x)
        x_rec = dct1(y)
        @test x_rec ≈ x atol=1e-5

        # Consistency with FFTW
        y_fftw = fftw_idct1(x)
        @test y ≈ y_fftw atol=1e-4
    end
end

@testset "Optimized IDCT-I (3D)" begin
    @testset "Consistency with FFTW IDCT-I" begin
        sizes = [(4, 4, 4), (5, 5, 5), (4, 5, 6), (8, 8, 8), (4, 8, 4)]
        for sz in sizes
            x = rand(Float64, sz...)
            y_fftw = fftw_idct1(x)
            y_opt = idct1(x)
            @test y_opt ≈ y_fftw atol=1e-10
        end
    end

    @testset "Roundtrip Accuracy (IDCT -> DCT)" begin
        sizes = [(4, 4, 4), (5, 5, 5), (4, 5, 6), (8, 8, 8), (4, 8, 4)]
        for sz in sizes
            x = rand(sz...)
            y = idct1(x)
            x_rec = dct1(y)
            @test x_rec ≈ x atol=1e-11
        end
    end

    @testset "Core API Functionality" begin
        M1, M2, M3 = 4, 4, 4
        x = rand(Float64, M1, M2, M3)

        # Plan creation
        p = plan_idct1(x)
        @test p isa AbstractFFTs.Plan{Float64}

        # Inverse transform
        y = p * x
        @test idct1(x) ≈ y

        # Non-allocating mul!
        y_out = similar(x)
        mul!(y_out, p, x)
        @test y_out ≈ y

        # Forward transform recovers original
        x_rec = dct1(y)
        @test x_rec ≈ x atol=1e-12

        # Plan inverse via \
        x_ldiv = p \ y
        @test x_ldiv ≈ x atol=1e-12

        # Explicit inverse plan
        p_inv = inv(p)
        @test p_inv isa AbstractFFTs.Plan{Float64}
        x_inv = p_inv * y
        @test x_inv ≈ x atol=1e-12
    end

    @testset "Float32" begin
        M1, M2, M3 = 4, 4, 4
        x = rand(Float32, M1, M2, M3)
        y = idct1(x)
        x_rec = dct1(y)
        @test x_rec ≈ x atol=1e-4

        # Consistency with FFTW
        y_fftw = fftw_idct1(x)
        @test y ≈ y_fftw atol=1e-4
    end
end
