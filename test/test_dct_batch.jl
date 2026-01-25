using Test
using AcceleratedDCTs
using AcceleratedDCTs: dct_batched, idct_batched
using AcceleratedDCTs: dct1d, dct2d, dct3d  # Needed for reference checking
using LinearAlgebra

@testset "Batched Optimized DCT/IDCT (dct_batched)" begin
    # 1D
    x = rand(16)
    y = dct_batched(x)
    y_ref = dct1d(x)
    @test y ≈ y_ref atol=1e-12
    @test idct_batched(y) ≈ x atol=1e-12
    
    # 2D
    x2 = rand(8, 8)
    y2 = dct_batched(x2)
    y2_ref = dct2d(x2)
    @test y2 ≈ y2_ref atol=1e-12
    @test idct_batched(y2) ≈ x2 atol=1e-12
    
    # 3D
    x3 = rand(4, 4, 4)
    y3 = dct_batched(x3)
    y3_ref = dct3d(x3)
    @test y3 ≈ y3_ref atol=1e-12
    @test idct_batched(y3) ≈ x3 atol=1e-12
end
