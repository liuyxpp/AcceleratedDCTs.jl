using FFTW
using Test
using AcceleratedDCTs
using LinearAlgebra
using Statistics

@testset "Optimized 2D DCT (Algorithm 2)" begin
    
    @testset "Roundtrip Accuracy" begin
        # Test square and non-square matrices
        sizes = [(4, 4), (8, 8), (16, 16), (4, 8), (8, 16), (32, 8)]
        
        for (N1, N2) in sizes
            x = rand(N1, N2)
            
            # Forward
            y = dct_2d_opt(x)
            
            # Inverse
            x_rec = idct_2d_opt(y)
            
            # Verify exact reconstruction
            @test x_rec ≈ x atol=1e-14
        end
    end
    
    @testset "Linearity" begin
        N1, N2 = 8, 8
        x1 = rand(N1, N2)
        x2 = rand(N1, N2)
        a, b = 2.5, -1.5
        
        y_comb = dct_2d_opt(a .* x1 .+ b .* x2)
        y_sep = a .* dct_2d_opt(x1) .+ b .* dct_2d_opt(x2)
        
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
        y_opt = dct_2d_opt(x)
        
        # Check scaling factor is consistently 4.0
        ratio = y_opt ./ y_ref
        
        # Allow some epsilon variance due to floating point
        @test all(isapprox.(ratio, 4.0, atol=1e-5))
    end
    
    @testset "Optimized 3D DCT (Algorithm 3)" begin
        @testset "Roundtrip Accuracy" begin
            sizes = [(4, 4, 4), (8, 8, 8), (4, 8, 4)]
            for sz in sizes
                x = rand(sz...)
                y = dct_3d_opt(x)
                x_rec = idct_3d_opt(y)
                @test x_rec ≈ x atol=1e-13
            end
        end
        
        @testset "Consistency with Reference DCT" begin
             # 2D has factor 4. 3D should have factor 8?
             N1, N2, N3 = 4, 4, 4
             x = rand(N1, N2, N3)
             y_ref = dct3d(x)
             y_opt = dct_3d_opt(x)
             ratio = y_opt ./ y_ref
             @test all(isapprox.(ratio, 8.0, atol=1e-5))
        end
    end
end
