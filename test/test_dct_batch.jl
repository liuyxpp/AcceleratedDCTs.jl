using Test
using AcceleratedDCTs
using LinearAlgebra

@testset "Batched Optimized DCT/IDCT (dct_fast)" begin
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
