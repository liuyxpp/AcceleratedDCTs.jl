using Test
using AcceleratedDCTs
using AcceleratedDCTs: plan_dct_batched
using LinearAlgebra

@testset "DCTBatchedPlan API with cached buffers" begin
    # Test that plan_dct creates a plan and can be reused
    
    # 1D with plan
    x1 = rand(32)
    plan1 = AcceleratedDCTs.plan_dct_batched(x1)
    y1 = plan1 * x1
    x1_rec = plan1 \ y1
    @test x1_rec ≈ x1 atol=1e-12
    
    # Verify plan can be reused
    x1b = rand(32)
    y1b = plan1 * x1b
    @test (plan1 \ y1b) ≈ x1b atol=1e-12
    
    # 2D with plan
    x2 = rand(16, 16)
    plan2 = AcceleratedDCTs.plan_dct_batched(x2)
    y2 = plan2 * x2
    x2_rec = plan2 \ y2
    @test x2_rec ≈ x2 atol=1e-12
    
    # 2D non-square
    x2b = rand(8, 16)
    plan2b = AcceleratedDCTs.plan_dct_batched(x2b)
    y2b = plan2b * x2b
    @test (plan2b \ y2b) ≈ x2b atol=1e-12
    
    # 3D with plan
    x3 = rand(8, 8, 8)
    plan3 = AcceleratedDCTs.plan_dct_batched(x3)
    y3 = plan3 * x3
    x3_rec = plan3 \ y3
    @test x3_rec ≈ x3 atol=1e-11
    
    # 3D non-cube
    x3b = rand(4, 8, 16)
    plan3b = AcceleratedDCTs.plan_dct_batched(x3b)
    y3b = plan3b * x3b
    @test (plan3b \ y3b) ≈ x3b atol=1e-11
end

@testset "Zero-allocation mul!/ldiv!" begin
    using LinearAlgebra: mul!, ldiv!
    
    # 1D
    x1 = rand(32)
    y1 = similar(x1)
    x1_rec = similar(x1)
    plan1 = AcceleratedDCTs.plan_dct_batched(x1)
    mul!(y1, plan1, x1)
    ldiv!(x1_rec, plan1, y1)
    @test x1_rec ≈ x1 atol=1e-12
    
    # 2D
    x2 = rand(16, 16)
    y2 = similar(x2)
    x2_rec = similar(x2)
    plan2 = AcceleratedDCTs.plan_dct_batched(x2)
    mul!(y2, plan2, x2)
    ldiv!(x2_rec, plan2, y2)
    @test x2_rec ≈ x2 atol=1e-12
    
    # 2D non-square
    x2b = rand(8, 16)
    y2b = similar(x2b)
    x2b_rec = similar(x2b)
    plan2b = AcceleratedDCTs.plan_dct_batched(x2b)
    mul!(y2b, plan2b, x2b)
    ldiv!(x2b_rec, plan2b, y2b)
    @test x2b_rec ≈ x2b atol=1e-12
    
    # 3D
    x3 = rand(8, 8, 8)
    y3 = similar(x3)
    x3_rec = similar(x3)
    plan3 = AcceleratedDCTs.plan_dct_batched(x3)
    mul!(y3, plan3, x3)
    ldiv!(x3_rec, plan3, y3)
    @test x3_rec ≈ x3 atol=1e-11
    
    # 3D non-cube
    x3b = rand(4, 8, 16)
    y3b = similar(x3b)
    x3b_rec = similar(x3b)
    plan3b = AcceleratedDCTs.plan_dct_batched(x3b)
    mul!(y3b, plan3b, x3b)
    ldiv!(x3b_rec, plan3b, y3b)
    @test x3b_rec ≈ x3b atol=1e-11
    
    # Verify mul! matches plan * x
    x = rand(8, 8, 8)
    y_star = AcceleratedDCTs.plan_dct_batched(x) * x
    y_mul = similar(x)
    mul!(y_mul, AcceleratedDCTs.plan_dct_batched(x), x)
    @test y_mul ≈ y_star atol=1e-14
end
