using Test
using AcceleratedDCTs
using AcceleratedDCTs: dct1d, idct1d, dct2d, idct2d, dct3d, idct3d
using LinearAlgebra

@testset "Reference Independent Implementation (FFT-based)" begin

    @testset "1D DCT/IDCT Roundtrip" begin
        for N in [4, 5, 8, 9, 15, 16, 32, 63, 64]
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
        for sz in [4, 5, 8, 9, 15, 16, 32]
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

    @testset "Normalization relationship with FFTW (1D)" begin
        # Verify 1D DCT matches FFTW with correct scaling factors
        N = 16
        x = rand(N)
        
        # Our implementation (unnormalized sum)
        y_ours = dct1d(x)
        
        # FFTW (orthogonal DCT-II)
        y_fftw = FFTW.dct(x)
        
        # Apply scaling to our result to match FFTW
        # DC term: scale by sqrt(1/N)
        # AC terms: scale by sqrt(2/N)
        y_scaled = similar(y_ours)
        y_scaled[1] = y_ours[1] * sqrt(1/N)
        y_scaled[2:end] = y_ours[2:end] .* sqrt(2/N)
        
        @test y_scaled ≈ y_fftw atol=1e-12
    end

    @testset "Normalization relationship with FFTW (2D)" begin
        N = 8
        x = rand(N, N)
        y_ours = dct2d(x)
        y_fftw = FFTW.dct(x)
        
        y_scaled = similar(y_ours)
        
        # Factors
        f1 = sqrt(1/N) # k=0
        f2 = sqrt(2/N) # k>0
        
        # (0,0) -> f1 * f1 = 1/N
        y_scaled[1,1] = y_ours[1,1] * (f1 * f1)
        
        # (0, k) -> f1 * f2
        y_scaled[1, 2:end] = y_ours[1, 2:end] .* (f1 * f2)
        
        # (k, 0) -> f2 * f1
        y_scaled[2:end, 1] = y_ours[2:end, 1] .* (f2 * f1)
        
        # (k, l) -> f2 * f2 = 2/N
        y_scaled[2:end, 2:end] = y_ours[2:end, 2:end] .* (f2 * f2)
        
        @test y_scaled ≈ y_fftw atol=1e-12
    end

    @testset "Normalization relationship with FFTW (3D)" begin
        N = 8
        x = rand(N, N, N)
        y_ours = dct3d(x)
        y_fftw = FFTW.dct(x)
        
        y_scaled = similar(y_ours)
        
        f1 = sqrt(1/N)
        f2 = sqrt(2/N)
        
        for k in 1:N, j in 1:N, i in 1:N
            # Determine factor for each dimension
            # Index 1 is DC (k=0), Index >1 is AC (k>0)
            fi = i==1 ? f1 : f2
            fj = j==1 ? f1 : f2
            fk = k==1 ? f1 : f2
            
            y_scaled[i,j,k] = y_ours[i,j,k] * (fi * fj * fk)
        end
        
        @test y_scaled ≈ y_fftw atol=1e-11
    end

    @testset "Non-square matrices" begin
        for (M, N) in [(4, 8), (8, 4), (16, 8), (3, 4), (5, 7)]
            x = rand(M, N)
            y = dct2d(x)
            x_rec = idct2d(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "3D DCT/IDCT Roundtrip" begin
        for sz in [3, 4, 5, 8, 16]
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
        for (L, M, N) in [(4, 4, 8), (4, 8, 4), (8, 4, 4), (4, 8, 16), (3, 4, 5), (5, 5, 7)]
            x = rand(L, M, N)
            y = dct3d(x)
            x_rec = idct3d(y)
            @test x_rec ≈ x atol=1e-11
        end
    end
end
