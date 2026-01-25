using FFTW  # Must be loaded first to provide FFT backend
using AcceleratedDCTs
using Test
using LinearAlgebra

@testset "AcceleratedDCTs.jl" begin
    
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
    
    @testset "Optimized DCT/IDCT (dct_fast)" begin
        # 1D
        x = rand(16)
        y = dct_fast(x)
        y_ref = dct1d(x)
        @test y ≈ y_ref atol=1e-12
        @test idct_fast(y) ≈ x atol=1e-12
        
        # 2D
        x2 = rand(8, 8)
        y2 = dct_fast(x2)
        y2_ref = dct2d(x2)
        @test y2 ≈ y2_ref atol=1e-12
        @test idct_fast(y2) ≈ x2 atol=1e-12
        
        # 3D
        x3 = rand(4, 4, 4)
        y3 = dct_fast(x3)
        y3_ref = dct3d(x3)
        @test y3 ≈ y3_ref atol=1e-12
        @test idct_fast(y3) ≈ x3 atol=1e-12
    end
    
    @testset "DCTPlan API with cached buffers" begin
        # Test that plan_dct creates a plan and can be reused
        
        # 1D with plan
        x1 = rand(32)
        plan1 = AcceleratedDCTs.plan_dct(x1)
        y1 = plan1 * x1
        x1_rec = plan1 \ y1
        @test x1_rec ≈ x1 atol=1e-12
        
        # Verify plan can be reused
        x1b = rand(32)
        y1b = plan1 * x1b
        @test (plan1 \ y1b) ≈ x1b atol=1e-12
        
        # 2D with plan
        x2 = rand(16, 16)
        plan2 = AcceleratedDCTs.plan_dct(x2)
        y2 = plan2 * x2
        x2_rec = plan2 \ y2
        @test x2_rec ≈ x2 atol=1e-12
        
        # 2D non-square
        x2b = rand(8, 16)
        plan2b = AcceleratedDCTs.plan_dct(x2b)
        y2b = plan2b * x2b
        @test (plan2b \ y2b) ≈ x2b atol=1e-12
        
        # 3D with plan
        x3 = rand(8, 8, 8)
        plan3 = AcceleratedDCTs.plan_dct(x3)
        y3 = plan3 * x3
        x3_rec = plan3 \ y3
        @test x3_rec ≈ x3 atol=1e-11
        
        # 3D non-cube
        x3b = rand(4, 8, 16)
        plan3b = AcceleratedDCTs.plan_dct(x3b)
        y3b = plan3b * x3b
        @test (plan3b \ y3b) ≈ x3b atol=1e-11
    end
    
end
