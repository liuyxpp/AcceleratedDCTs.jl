import FFTW
using Test
using AcceleratedDCTs
using AcceleratedDCTs: dct, idct, plan_dct, DCTPlan, dct!, idct!
using AcceleratedDCTs: dct1d, dct2d, dct3d # Reference implementations
using LinearAlgebra: mul!, ldiv!
using Statistics

@testset "Optimized DCT-II (1D)" begin
    N = 16
    x = rand(Float64, N)

    @testset "Core API Functionality" begin
        p = plan_dct(x)
        y = p * x

        # Convenience
        @test dct(x) ≈ y

        # In-place
        x_copy = copy(x)
        dct!(x_copy)
        @test x_copy ≈ y

        # Inverse
        x_rec = idct(y)
        @test x_rec ≈ x atol=1e-12

        # Inverse plan
        x_ldiv = similar(x)
        ldiv!(x_ldiv, p, y)
        @test x_ldiv ≈ x atol=1e-12
    end

    @testset "Roundtrip Accuracy" begin
        for N in [8, 9, 15, 16, 32, 63, 64]
            x = rand(N)
            y = dct(x)
            x_rec = idct(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "Consistency with Reference DCT" begin
        x = rand(16)
        y_ref = dct1d(x)
        y_opt = dct(x)
        ratio = y_opt ./ y_ref

        # Check if ratio is consistently 2.0
        @test all(isapprox.(ratio, 2.0, atol=1e-5))
    end

    @testset "Float32" begin
        N = 16
        x = rand(Float32, N)
        y = dct(x)
        x_rec = idct(y)
        @test x_rec ≈ x atol=1e-5
    end
end

@testset "Optimized DCT-II (2D)" begin
    # ------------------------------------------------------------------
    # Optimized DCT-II (2D)
    # ------------------------------------------------------------------

    # Define N1, N2 for the tests within this block
    N1, N2 = 8, 8

    @testset "Core API Functionality" begin
        x = rand(Float64, N1, N2)

        # 1. Plan creation
        p = plan_dct(x)
        @test p isa DCTPlan

        # 2. Forward (allocating)
        y = p * x

        # 3. Convenience function
        y_conv = dct(x)
        @test y_conv ≈ y

        # 4. Inverse
        x_rec = idct(y)
        @test x_rec ≈ x atol=1e-12

        # 5. Non-allocating execution
        y_out = similar(x)
        mul!(y_out, p, x)
        @test y_out ≈ y

        # 6. In-place convenience
        x_copy = copy(x)
        dct!(x_copy)
        @test x_copy ≈ y

        # 7. In-place Inverse convenience
        idct!(x_copy)
        @test x_copy ≈ x atol=1e-12

        # 8. Plan Inverse (Allocating)
        x_inv = p \ y
        @test x_inv ≈ x atol=1e-12

        # 9. Plan Inverse (Non-allocating ldiv!)
        x_ldiv = similar(x)
        ldiv!(x_ldiv, p, y)
        @test x_ldiv ≈ x atol=1e-12

        # 10. Explicit Inverse Plan
        p_inv = inv(p)
        x_inv_p = p_inv * y
        @test x_inv_p ≈ x atol=1e-12

        # 11. Inverse Plan reuse
        x_inv_p2 = similar(x)
        mul!(x_inv_p2, p_inv, y)
        @test x_inv_p2 ≈ x atol=1e-12
    end

    @testset "Roundtrip Accuracy" begin
        # Test square and non-square matrices
        sizes = [(4, 4), (8, 9), (16, 16), (4, 8), (9, 15), (32, 8)]

        for (N1_rt, N2_rt) in sizes
            x = rand(N1_rt, N2_rt)

            # Forward
            y = dct(x)

            # Inverse
            x_rec = idct(y)

            # Verify exact reconstruction
            @test x_rec ≈ x atol=1e-14
        end
    end

    @testset "Linearity" begin
        N1, N2 = 8, 8
        x1 = rand(N1, N2)
        x2 = rand(N1, N2)
        a, b = 2.5, -1.5

        y_comb = dct(a .* x1 .+ b .* x2)
        y_sep = a .* dct(x1) .+ b .* dct(x2)

        @test y_comb ≈ y_sep atol=1e-12
    end

    @testset "Consistency with Reference DCT" begin
        # Algorithm 2 implementation has a scaling factor compared to standard DCT-II
        # We observed a factor of 4.0 in manual verification

        N1, N2 = 8, 8
        x = rand(N1, N2)

        # Reference (slow) implementation
        y_ref = dct2d(x)

        # Optimized implementation
        y_opt = dct(x)

        # Check scaling factor is consistently 4.0
        ratio = y_opt ./ y_ref

        # Allow some epsilon variance due to floating point
        @test all(isapprox.(ratio, 4.0, atol=1e-5))
    end

    @testset "Float32" begin
        N1, N2 = 8, 8
        x = rand(Float32, N1, N2)
        y = dct(x)
        x_rec = idct(y)
        @test x_rec ≈ x atol=1e-5
    end

end

@testset "Optimized 3D DCT (Algorithm 3)" begin
    @testset "Roundtrip Accuracy" begin
        sizes = [(4, 4, 4), (6, 8, 10), (4, 8, 4), (4, 8, 9), (9, 8, 15)]
        for sz in sizes
            x = rand(sz...)
            y = dct(x)
            x_rec = idct(y)
            @test x_rec ≈ x atol=1e-13
        end
    end

    @testset "3D API Coverage" begin
        # Verify 3D specific API points
        N = 8
        x = rand(N, N, N)
        p = plan_dct(x)
        y = p * x

        # In-place 3D
        x_copy = copy(x)
        dct!(x_copy)
        @test x_copy ≈ y

        # Inverse 3D in-place
        idct!(x_copy)
        @test x_copy ≈ x atol=1e-12

        # ldiv! 3D
        x_rec = similar(x)
        ldiv!(x_rec, p, y)
        @test x_rec ≈ x atol=1e-12
    end

    @testset "Consistency with Reference DCT" begin
        # 2D has factor 4. 3D should have factor 8?
        N1, N2, N3 = 4, 4, 4
        x = rand(N1, N2, N3)
        y_ref = dct3d(x)
        y_opt = dct(x)
        ratio = y_opt ./ y_ref
        @test all(isapprox.(ratio, 8.0, atol=1e-5))
    end

    @testset "Float32" begin
        N1, N2, N3 = 4, 4, 4
        x = rand(Float32, N1, N2, N3)
        y = dct(x)
        x_rec = idct(y)
        @test x_rec ≈ x atol=1e-5
    end
end