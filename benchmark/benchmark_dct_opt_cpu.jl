# Benchmark: Compare Algorithm 2 DCT (opt) vs FFTW rfft on CPU
#
# This script compares the performance of:
# 1. dct_2d_opt (Algorithm 2) - new 2D implementation
# 2. dct2d (reference) - separable 1D CPU implementation
# 3. FFTW rfft - native FFTW 2D R2C FFT
#
# Measured on CPU with N² grids.

using FFTW
using Statistics
using LinearAlgebra

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs

println("="^60)
println("Benchmark: 2D DCT (Algorithm 2) vs FFTW rfft (CPU)")
println("="^60)
println()

# Setup
N = 2048
x_cpu = rand(Float64, N, N)

println("Grid size: $(N)x$(N)")
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
_ = dct_2d_opt(x_cpu)
_ = dct2d(x_cpu)
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
# Benchmark dct_2d_opt (Algorithm 2)
# ============================================================================
println("-"^60)
println("Benchmarking dct_2d_opt (Algorithm 2)...")
println("-"^60)

times_dct_opt = benchmark_cpu(dct_2d_opt, x_cpu)
println("  Minimum time:  $(round(minimum(times_dct_opt), digits=3)) ms")
println("  Median time:   $(round(median(times_dct_opt), digits=3)) ms")
println("  Mean time:     $(round(mean(times_dct_opt), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_dct_opt), digits=3)) ms")
println()

# ============================================================================
# Benchmark dct2d (Reference Separable)
# ============================================================================
println("-"^60)
println("Benchmarking dct2d (Reference Separable)...")
println("-")

times_dct_ref = benchmark_cpu(dct2d, x_cpu)
println("  Minimum time:  $(round(minimum(times_dct_ref), digits=3)) ms")
println("  Median time:   $(round(median(times_dct_ref), digits=3)) ms")
println("  Mean time:     $(round(mean(times_dct_ref), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_dct_ref), digits=3)) ms")
println()


# ============================================================================
# Benchmark FFTW rfft (2D)
# ============================================================================
println("-"^60)
println("Benchmarking FFTW rfft (2D R2C FFT)...")
println("-")

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
ratio_opt = median(times_dct_opt) / baseline
ratio_ref = median(times_dct_ref) / baseline
speedup_vs_ref = median(times_dct_ref) / median(times_dct_opt)

println()
println("FFT Methods:")
println("  FFTW rfft:               $(round(median(times_rfft), digits=3)) ms (baseline)")
println()
println("DCT Methods:")
println("  dct_2d_opt (Alg 2):      $(round(median(times_dct_opt), digits=3)) ms ($(round(ratio_opt, digits=2))x slower vs FFT)")
println("  dct2d (Reference):       $(round(median(times_dct_ref), digits=3)) ms ($(round(ratio_ref, digits=2))x slower vs FFT)")
println()
println("Optimization Speedup:")
println("  • Algorithm 2 vs Reference: $(round(speedup_vs_ref, digits=2))x faster")
println()
