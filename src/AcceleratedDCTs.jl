module AcceleratedDCTs

using LinearAlgebra
using KernelAbstractions
using AbstractFFTs

# Reference implementations (correct but not optimized)
include("dct_slow.jl")

# Batch implementations using R2C FFT and KernelAbstractions
include("dct_batch.jl")

# Optimized implementations using R2C FFT and KernelAbstractions
include("dct_optimized.jl")

public dct1d, idct1d, dct2d, idct2d, dct3d, idct3d  # reference implementations
public dct_batched, idct_batched  # batched implementations
public DCTBatchedPlan, plan_dct_batched  # planned batched implementations
public dct, idct, dct!, idct!, plan_dct, plan_idct, DCTPlan, IDCTPlan # optimized implementations

end # module AcceleratedDCTs

