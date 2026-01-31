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

# Optimized DCT-I implementations
include("dct1_optimized.jl")

# FFTW-based DCT-I for CPU (faster than generic implementation)
include("dct1_fftw.jl")

public dct1d, idct1d, dct2d, idct2d, dct3d, idct3d  # reference implementations
public dct_batched, idct_batched  # batched implementations
public DCTBatchedPlan, plan_dct_batched  # planned batched implementations
public dct, idct, dct!, idct!, plan_dct, plan_idct, DCTPlan, IDCTPlan # optimized implementations
public dct1, idct1, plan_dct1, plan_idct1, DCT1Plan, IDCT1Plan # optimized DCT-I implementations

end # module AcceleratedDCTs

