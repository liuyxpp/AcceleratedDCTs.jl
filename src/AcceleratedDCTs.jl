module AcceleratedDCTs

using KernelAbstractions
using AbstractFFTs

# Reference implementations (correct but not optimized)
include("dct_slow.jl")

# Batch implementations using R2C FFT and KernelAbstractions
include("dct_batch.jl")

# Optimized implementations using R2C FFT and KernelAbstractions
# include("dct_optimized.jl")

export dct1d, idct1d, dct2d, idct2d, dct3d, idct3d  # reference implementations
export dct_fast, idct_fast  # batched implementations
export DCTPlan, plan_dct, dct_fast!, idct_fast!  # planned implementations with buffer caching

end # module AcceleratedDCTs

