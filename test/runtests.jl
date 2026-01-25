import FFTW
using AcceleratedDCTs
using Test
using LinearAlgebra

@testset "AcceleratedDCTs.jl" begin
    # Reference implementation tests (slow)
    include("test_reference.jl")
    
    # Existing batched implementation tests
    include("test_dct_batch.jl")
    
    # DCT Plan and cached buffer tests
    include("test_dct_plan.jl")
    
    # New Optimized Algorithm 2 tests
    include("test_dct_optimized.jl")
end
