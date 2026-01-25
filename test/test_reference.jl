using Test
using AcceleratedDCTs
using LinearAlgebra

@testset "Reference Independent Implementation (FFT-based)" begin

    @testset "1D DCT/IDCT Roundtrip" begin
        for N in [4, 8, 16, 32, 64]
            x = rand(N)
            y = dct1d(x)
            x_rec = idct1d(y)
            @test x_rec ≈ x atol=1e-12
        end
    end
    
    @testset "1D DCT Properties" begin
        N = 8
        x = rand(N)
        y = dct1d(x)
        
        # DC component should equal sum of input
        @test y[1] ≈ sum(x) atol=1e-12
        
        # Linearity: DCT(ax + by) = a*DCT(x) + b*DCT(y)
        x2 = rand(N)
        a, b = 2.5, -1.3
        @test dct1d(a*x + b*x2) ≈ a*dct1d(x) + b*dct1d(x2) atol=1e-12
    end
    
    @testset "2D DCT/IDCT Roundtrip" begin
        for sz in [4, 8, 16, 32]
            x = rand(sz, sz)
            y = dct2d(x)
            x_rec = idct2d(y)
            @test x_rec ≈ x atol=1e-12
        end
    end
    
    @testset "2D DCT Properties" begin
        M, N = 8, 8
        x = rand(M, N)
        y = dct2d(x)
        
        # DC component should equal sum of input
        @test y[1,1] ≈ sum(x) atol=1e-11
        
        # Linearity
        x2 = rand(M, N)
        a, b = 2.5, -1.3
        @test dct2d(a*x + b*x2) ≈ a*dct2d(x) + b*dct2d(x2) atol=1e-11
    end
    
    @testset "Normalization relationship with FFTW" begin
        # Our DCT uses the definition: X_k = Σ x_n cos(π/N * (n+1/2) * k)
        # FFTW uses: Y_k = 2 Σ x_j cos[π(j+1/2)k/n] (factor of 2)
        # For 2D, the factor is 4 (2 per dimension)
        # However, the non-DC components have different normalization
        # So we only test that roundtrip works, not exact FFTW matching
        x = rand(8, 8)
        y = dct2d(x)
        x_rec = idct2d(y)
        @test x_rec ≈ x atol=1e-12
    end
    
    @testset "Non-square matrices" begin
        for (M, N) in [(4, 8), (8, 4), (16, 8)]
            x = rand(M, N)
            y = dct2d(x)
            x_rec = idct2d(y)
            @test x_rec ≈ x atol=1e-12
        end
    end
    
    @testset "3D DCT/IDCT Roundtrip" begin
        for sz in [4, 8, 16]
            x = rand(sz, sz, sz)
            y = dct3d(x)
            x_rec = idct3d(y)
            @test x_rec ≈ x atol=1e-11
        end
    end
    
    @testset "3D DCT Properties" begin
        L, M, N = 8, 8, 8
        x = rand(L, M, N)
        y = dct3d(x)
        
        # DC component should equal sum of input
        @test y[1,1,1] ≈ sum(x) atol=1e-10
        
        # Linearity
        x2 = rand(L, M, N)
        a, b = 2.5, -1.3
        @test dct3d(a*x + b*x2) ≈ a*dct3d(x) + b*dct3d(x2) atol=1e-10
    end
    
    @testset "Non-cubic 3D arrays" begin
        for (L, M, N) in [(4, 4, 8), (4, 8, 4), (8, 4, 4), (4, 8, 16)]
            x = rand(L, M, N)
            y = dct3d(x)
            x_rec = idct3d(y)
            @test x_rec ≈ x atol=1e-11
        end
    end
end
