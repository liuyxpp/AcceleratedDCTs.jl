# Benchmark: Compare Algorithm 2 DCT (opt) vs cuFFT rfft on GPU
#
# This script compares the performance of:
# 1. dct_2d_opt (Algorithm 2) - new 2D implementation
# 2. cuFFT rfft - native CUDA R2C FFT
#
# Measured on GPU with N² grids.

using CUDA
using CUDA.CUFFT
using Statistics

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs

println("="^60)
println("Benchmark: 2D DCT (Algorithm 2) vs cuFFT rfft (GPU)")
println("="^60)
println()

# Setup
N = 2048
x_cpu = rand(Float64, N, N)
x_gpu = CuArray(x_cpu)

println("Grid size: $(N)x$(N)")
println("Input array size: $(size(x_gpu))")
println("Input array type: $(typeof(x_gpu))")
println("GPU: $(CUDA.name(CUDA.device()))")
println()

# Warmup
println("Warming up...")
# Warning: dct_2d_opt might fall back to CPU if not fully generic!
# We perform a try-catch to report if it fails on GPU types
try
    _ = dct_2d_opt(x_gpu)
catch e
    println("Error running dct_2d_opt on GPU: $e")
    println("Note: dct_2d_opt may need valid GPU array support.")
end
_ = rfft(x_gpu)
CUDA.synchronize()
println("Warmup complete.")
println()

# Benchmark function using manual timing
function benchmark_gpu(f, x; n_warmup=3, n_samples=20)
    # Warmup
    for _ in 1:n_warmup
        try
            f(x)
        catch
            return [NaN]
        end
        CUDA.synchronize()
    end
    
    # Measure
    times = Float64[]
    for _ in 1:n_samples
        CUDA.synchronize()
        t0 = time_ns()
        try
            f(x)
        catch
            return [NaN]
        end
        CUDA.synchronize()
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
    end
    return times
end

# ============================================================================
# Benchmark dct_2d_opt (Algorithm 2)
# ============================================================================
println("-"^60)
println("Benchmarking dct_2d_opt (Algorithm 2)...")
println("Note: If implementation allocates CPU Matrix, this includes transfer time.")
println("-"^60)

times_dct_opt = benchmark_gpu(dct_2d_opt, x_gpu)
if isnan(times_dct_opt[1])
    println("Benchmark failed (function error)")
else
    println("  Minimum time:  $(round(minimum(times_dct_opt), digits=3)) ms")
    println("  Median time:   $(round(median(times_dct_opt), digits=3)) ms")
    println("  Mean time:     $(round(mean(times_dct_opt), digits=3)) ms")
    println("  Maximum time:  $(round(maximum(times_dct_opt), digits=3)) ms")
end
println()

# ============================================================================
# Benchmark cuFFT rfft (2D)
# ============================================================================
println("-"^60)
println("Benchmarking cuFFT rfft (2D R2C FFT)...")
println("-"^60)

times_rfft = benchmark_gpu(rfft, x_gpu)
println("  Minimum time:  $(round(minimum(times_rfft), digits=3)) ms")
println("  Median time:   $(round(median(times_rfft), digits=3)) ms")
println("  Mean time:     $(round(mean(times_rfft), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_rfft), digits=3)) ms")
println()

# ============================================================================
# Performance Comparison Summary
# ============================================================================
println("="^60)
println("Performance Comparison (using median times)")
println("="^60)

if isnan(times_dct_opt[1])
    println("Skipping comparison due to dct_2d_opt failure")
else
    baseline = median(times_rfft)
    ratio_opt = median(times_dct_opt) / baseline

    println()
    println("FFT Methods:")
    println("  cuFFT rfft:              $(round(median(times_rfft), digits=3)) ms (baseline)")
    println()
    println("DCT Methods:")
    println("  dct_2d_opt (Alg 2):      $(round(median(times_dct_opt), digits=3)) ms ($(round(ratio_opt, digits=2))x slower vs FFT)")
    println()
end
