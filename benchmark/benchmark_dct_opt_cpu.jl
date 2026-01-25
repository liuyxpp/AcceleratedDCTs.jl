# Benchmark: Compare Algorithm 3 3D DCT (opt) vs dct_fast vs FFTW rfft on CPU
#
# This script compares the performance of:
# 1. dct_3d_opt (Algorithm 3) - new 3D implementation in pure Julia
# 2. dct_fast (Batched) - efficient separable 1D implementation
# 3. FFTW rfft - native FFTW 3D R2C FFT (baseline)
# 4. dct3d (Reference) - naive separable implementation
#
# Measured on CPU with N³ grids.

using FFTW
using Statistics
using LinearAlgebra

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs

println("="^60)
println("Benchmark: 3D DCT (Algorithm 3) vs dct_fast vs FFTW rfft (CPU)")
println("="^60)
println()

# Setup
N = 128
x_cpu = rand(Float64, N, N, N)

println("Grid size: $(N)x$(N)x$(N)")
println("Input array size: $(size(x_cpu))")
println("Input array type: $(typeof(x_cpu))")
println("CPU threads: $(Threads.nthreads())")
println("FFTW threads: $(FFTW.get_num_threads())")
println()

# Set FFTW to use multiple threads if available
FFTW.set_num_threads(Threads.nthreads())
println("Set FFTW threads to: $(FFTW.get_num_threads())")
println()

# Warmup
println("Warming up...")
_ = dct_3d_opt(x_cpu)
_ = dct_fast(x_cpu)
_ = rfft(x_cpu)
println("Warmup complete.")
println()

# Benchmark function for CPU
function benchmark_cpu(f, x; n_warmup=3, n_samples=10)
    # Warmup
    for _ in 1:n_warmup
        f(x)
    end
    
    # Measure
    times = Float64[]
    for _ in 1:n_samples
        GC.gc(false)  # Minor GC to reduce noise
        t0 = time_ns()
        f(x)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
    end
    return times
end

# ============================================================================
# Benchmark dct_3d_opt (Algorithm 3)
# ============================================================================
println("-"^60)
println("Benchmarking dct_3d_opt (Algorithm 3)...")
println("Description: Manual 3D RFFT wrapper with recursive post-processing")
println("-"^60)

times_dct_opt = benchmark_cpu(dct_3d_opt, x_cpu)
println("  Minimum time:  $(round(minimum(times_dct_opt), digits=3)) ms")
println("  Median time:   $(round(median(times_dct_opt), digits=3)) ms")
println("  Mean time:     $(round(mean(times_dct_opt), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_dct_opt), digits=3)) ms")
println()

# ============================================================================
# Benchmark dct_fast (Batched Separable)
# ============================================================================
println("-"^60)
println("Benchmarking dct_fast (Batched Separable, no plan reuse)...")
println("Description: 3x Separable 1D DCTs using batched kernels + transposes")
println("-"^60)

times_dct_fast = benchmark_cpu(dct_fast, x_cpu)
println("  Minimum time:  $(round(minimum(times_dct_fast), digits=3)) ms")
println("  Median time:   $(round(median(times_dct_fast), digits=3)) ms")
println("  Mean time:     $(round(mean(times_dct_fast), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_dct_fast), digits=3)) ms")
println()

# ============================================================================
# Benchmark FFTW rfft (3D)
# ============================================================================
println("-"^60)
println("Benchmarking FFTW rfft (3D R2C FFT)...")
println("Description: Native highly-optimized FFT library")
println("-"^60)

times_rfft = benchmark_cpu(rfft, x_cpu)
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

baseline = median(times_rfft)
time_opt = median(times_dct_opt)
time_fast = median(times_dct_fast)

ratio_opt = time_opt / baseline
ratio_fast = time_fast / baseline
speedup_opt_vs_fast = time_fast / time_opt

println()
println("Baselines:")
println("  FFTW rfft:               $(round(baseline, digits=3)) ms")
println()
println("DCT Variants:")
println("  dct_3d_opt (Algorithm 3): $(round(time_opt, digits=3)) ms ($(round(ratio_opt, digits=2))x slower vs FFT)")
println("  dct_fast (Batched):       $(round(time_fast, digits=3)) ms ($(round(ratio_fast, digits=2))x slower vs FFT)")
println()
println("Comparison:")
if speedup_opt_vs_fast > 1.0
    println("  • Algorithm 3 is $(round(speedup_opt_vs_fast, digits=2))x FASTER than dct_fast")
else
    println("  • Algorithm 3 is $(round(1/speedup_opt_vs_fast, digits=2))x SLOWER than dct_fast")
end
println()
