# Benchmark: Compare Algorithm 3 3D DCT (opt) vs dct_fast vs FFTW rfft on CPU
#
# This script compares the performance of:
# 1. dct_3d_opt (Algorithm 3)
# 2. dct_fast (Batched)
# 3. FFTW rfft - native FFTW 3D R2C FFT (baseline)
#
# Measured on CPU with N³ grids.

using FFTW
using Statistics
using LinearAlgebra

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs

println("="^60)
println("Benchmark: 3D DCT (Algorithm 3) w/ Plan vs dct_fast vs FFTW rfft (CPU)")
println("="^60)
println()

# Setup
N = 256
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
p = plan_dct_opt(x_cpu)
_ = p * x_cpu
_ = dct_3d_opt(x_cpu)
_ = dct_fast(x_cpu)
_ = FFTW.dct(x_cpu)
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

# 1a. One-shot
println("Type: One-shot (dct_3d_opt)")
times_dct_opt = benchmark_cpu(dct_3d_opt, x_cpu)
println("  Median time:   $(round(median(times_dct_opt), digits=3)) ms")
println()

# 1b. Plan-based
println("Type: Plan-based (p * x)")
p_cpu = plan_dct_opt(x_cpu)
times_dct_opt_plan = benchmark_cpu(x -> p_cpu * x, x_cpu)
println("  Median time:   $(round(median(times_dct_opt_plan), digits=3)) ms")
println()


# ============================================================================
# Benchmark dct_fast (Batched Separable)
# ============================================================================
println("-"^60)
println("Benchmarking dct_fast (Batched Separable, no plan reuse)...")
println("Description: 3x Separable 1D DCTs using batched kernels + transposes")
println("-"^60)

times_dct_fast = benchmark_cpu(dct_fast, x_cpu)
println("  Median time:   $(round(median(times_dct_fast), digits=3)) ms")
println()

# ============================================================================
# Benchmark FFTW rfft (3D)
# ============================================================================
println("-"^60)
println("Benchmarking FFTW rfft (3D R2C FFT)...")
println("Description: Native highly-optimized FFT library")
println("-"^60)

times_rfft = benchmark_cpu(rfft, x_cpu)
println("  Median time:   $(round(median(times_rfft), digits=3)) ms")
println()

# ============================================================================
# Benchmark FFTW dct (REDFT10)
# ============================================================================
println("-"^60)
println("Benchmarking FFTW dct (REDFT10)...")
println("Description: FFTW native DCT-II implementation")
println("-"^60)

# FFTW dct default is along all dims if not specified, but safe to verify
times_fftw_dct = benchmark_cpu(x -> FFTW.dct(x), x_cpu)
println("  Median time:   $(round(median(times_fftw_dct), digits=3)) ms")
println()

# ============================================================================
# Performance Comparison Summary
# ============================================================================
println("="^60)
println("Performance Comparison (using median times)")
println("="^60)

baseline = median(times_rfft)
time_opt = median(times_dct_opt)
time_opt_plan = median(times_dct_opt_plan)
time_fast = median(times_dct_fast)
time_fftw_dct = median(times_fftw_dct)

ratio_opt = time_opt / baseline
ratio_opt_plan = time_opt_plan / baseline
ratio_fast = time_fast / baseline
speedup_plan = time_opt / time_opt_plan

println()
println("Baselines:")
println("  FFTW rfft:               $(round(baseline, digits=3)) ms")
println("  FFTW dct (REDFT10):      $(round(time_fftw_dct, digits=3)) ms ($(round(time_fftw_dct/baseline, digits=2))x slower vs FFT)")
println()
println("DCT Variants:")
println("  dct_3d_opt (One-shot):   $(round(time_opt, digits=3)) ms ($(round(ratio_opt, digits=2))x slower vs FFT)")
println("  dct_3d_opt (Plan):       $(round(time_opt_plan, digits=3)) ms ($(round(ratio_opt_plan, digits=2))x slower vs FFT)")
println("     -> Plan Speedup:      $(round(speedup_plan, digits=2))x improvement")
println("  dct_fast (Batched):       $(round(time_fast, digits=3)) ms ($(round(ratio_fast, digits=2))x slower vs FFT)")
println()
println("Comparison:")
if time_opt_plan < time_fast
    speedup = time_fast / time_opt_plan
    println("  • Algorithm 3 (Plan) is $(round(speedup, digits=2))x FASTER than dct_fast")
else
    speedup = time_opt_plan / time_fast
    println("  • Algorithm 3 (Plan) is $(round(speedup, digits=2))x SLOWER than dct_fast")
end
println()
