# Benchmark: Compare dct_batch's 3D DCT vs FFTW rfft on CPU
#
# This script compares the performance of:
# 1. dct_fast (from dct_batch.jl) - our batched 1D DCT implementation
# 2. plan * x - our batched 1D DCT with cached plan
# 3. mul!(y, plan, x) - zero allocation DCT
# 4. FFTW rfft - native FFTW R2C FFT
#
# All measured on CPU with N³ grids.

using FFTW
using Statistics
using LinearAlgebra

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include the batch implementation directly
include(joinpath(@__DIR__, "..", "src", "dct_batch.jl"))

println("="^60)
println("Benchmark: 3D DCT (dct_batch) vs FFTW rfft (CPU)")
println("="^60)
println()

# Setup - use smaller grid for CPU (128³ instead of 256³)
N = 128
x_cpu = rand(Float64, N, N, N)

println("Grid size: $(N)³")
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
_ = dct_fast(x_cpu)
_ = dct_plan * x_cpu
_ = rfft(x_cpu)
_ = fftw_plan * x_cpu
println("Warmup complete.")
println()

# Create DCT plan (this is the key optimization - create once, reuse many times)
println("Creating DCT plan (one-time cost)...")
t_plan_start = time_ns()
dct_plan = plan_dct(x_cpu)
t_plan_end = time_ns()
plan_creation_time = (t_plan_end - t_plan_start) / 1e6
println("  Plan creation time: $(round(plan_creation_time, digits=3)) ms")
println()

# Create FFTW plan for comparison
println("Creating FFTW rfft plan...")
t_fftw_plan_start = time_ns()
fftw_plan = FFTW.plan_rfft(x_cpu)
t_fftw_plan_end = time_ns()
fftw_plan_creation_time = (t_fftw_plan_end - t_fftw_plan_start) / 1e6
println("  FFTW plan creation time: $(round(fftw_plan_creation_time, digits=3)) ms")
println()

# Benchmark function for CPU
function benchmark_cpu(f, x; n_warmup=3, n_samples=20)
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
# Benchmark dct_fast (without plan - creates plan each call)
# ============================================================================
println("-"^60)
println("Benchmarking dct_fast (creates plan each call)...")
println("-"^60)

times_dct = benchmark_cpu(dct_fast, x_cpu)
println("  Minimum time:  $(round(minimum(times_dct), digits=3)) ms")
println("  Median time:   $(round(median(times_dct), digits=3)) ms")
println("  Mean time:     $(round(mean(times_dct), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_dct), digits=3)) ms")
println()

# ============================================================================
# Benchmark plan * x (with cached plan - reuses buffers/twiddles)
# ============================================================================
println("-"^60)
println("Benchmarking plan * x (reuses cached plan)...")
println("-"^60)

times_dct_plan = benchmark_cpu(x -> dct_plan * x, x_cpu)
println("  Minimum time:  $(round(minimum(times_dct_plan), digits=3)) ms")
println("  Median time:   $(round(median(times_dct_plan), digits=3)) ms")
println("  Mean time:     $(round(mean(times_dct_plan), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_dct_plan), digits=3)) ms")
println()

# ============================================================================
# Benchmark mul!(y, plan, x) - zero allocation with ping-pong buffers
# ============================================================================
println("-"^60)
println("Benchmarking mul!(y, plan, x) (zero allocation)...")
println("-"^60)

y_preallocated = similar(x_cpu)

# Benchmark mul!
times_mul = Float64[]
for _ in 1:3  # Warmup
    mul!(y_preallocated, dct_plan, x_cpu)
end
for _ in 1:20
    GC.gc(false)
    t0 = time_ns()
    mul!(y_preallocated, dct_plan, x_cpu)
    t1 = time_ns()
    push!(times_mul, (t1 - t0) / 1e6)
end
println("  Minimum time:  $(round(minimum(times_mul), digits=3)) ms")
println("  Median time:   $(round(median(times_mul), digits=3)) ms")
println("  Mean time:     $(round(mean(times_mul), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_mul), digits=3)) ms")
println()

# ============================================================================
# Benchmark idct with plan (plan \ y)
# ============================================================================
println("-"^60)
println("Benchmarking plan \\ y (IDCT with cached plan)...")
println("-"^60)

y_cpu = dct_plan * x_cpu
times_idct_plan = benchmark_cpu(y -> dct_plan \ y, y_cpu)
println("  Minimum time:  $(round(minimum(times_idct_plan), digits=3)) ms")
println("  Median time:   $(round(median(times_idct_plan), digits=3)) ms")
println("  Mean time:     $(round(mean(times_idct_plan), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_idct_plan), digits=3)) ms")
println()

# ============================================================================
# Benchmark ldiv!(x, plan, y) - zero allocation IDCT
# ============================================================================
println("-"^60)
println("Benchmarking ldiv!(x, plan, y) (zero allocation IDCT)...")
println("-"^60)

x_preallocated = similar(x_cpu)
times_ldiv = Float64[]
for _ in 1:3  # Warmup
    ldiv!(x_preallocated, dct_plan, y_cpu)
end
for _ in 1:20
    GC.gc(false)
    t0 = time_ns()
    ldiv!(x_preallocated, dct_plan, y_cpu)
    t1 = time_ns()
    push!(times_ldiv, (t1 - t0) / 1e6)
end
println("  Minimum time:  $(round(minimum(times_ldiv), digits=3)) ms")
println("  Median time:   $(round(median(times_ldiv), digits=3)) ms")
println("  Mean time:     $(round(mean(times_ldiv), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_ldiv), digits=3)) ms")
println()

# ============================================================================
# Benchmark FFTW rfft (3D) - without plan
# ============================================================================
println("-"^60)
println("Benchmarking FFTW rfft (3D R2C FFT, no plan)...")
println("-"^60)

times_rfft = benchmark_cpu(rfft, x_cpu)
println("  Minimum time:  $(round(minimum(times_rfft), digits=3)) ms")
println("  Median time:   $(round(median(times_rfft), digits=3)) ms")
println("  Mean time:     $(round(mean(times_rfft), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_rfft), digits=3)) ms")
println()

# ============================================================================
# Benchmark FFTW rfft with plan (plan * x)
# ============================================================================
println("-"^60)
println("Benchmarking FFTW plan * x (3D R2C FFT, with plan)...")
println("-"^60)

times_rfft_plan = benchmark_cpu(x -> fftw_plan * x, x_cpu)
println("  Minimum time:  $(round(minimum(times_rfft_plan), digits=3)) ms")
println("  Median time:   $(round(median(times_rfft_plan), digits=3)) ms")
println("  Mean time:     $(round(mean(times_rfft_plan), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_rfft_plan), digits=3)) ms")
println()

# ============================================================================
# Benchmark FFTW mul! (zero allocation)
# ============================================================================
println("-"^60)
println("Benchmarking FFTW mul!(y, plan, x) (zero allocation)...")
println("-"^60)

y_fftw = similar(x_cpu, Complex{Float64}, (N÷2+1, N, N))
times_rfft_mul = Float64[]
for _ in 1:3  # Warmup
    mul!(y_fftw, fftw_plan, x_cpu)
end
for _ in 1:20
    GC.gc(false)
    t0 = time_ns()
    mul!(y_fftw, fftw_plan, x_cpu)
    t1 = time_ns()
    push!(times_rfft_mul, (t1 - t0) / 1e6)
end
println("  Minimum time:  $(round(minimum(times_rfft_mul), digits=3)) ms")
println("  Median time:   $(round(median(times_rfft_mul), digits=3)) ms")
println("  Mean time:     $(round(mean(times_rfft_mul), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_rfft_mul), digits=3)) ms")
println()

# ============================================================================
# Performance Comparison Summary
# ============================================================================
println("="^60)
println("Performance Comparison (using median times)")
println("="^60)

baseline = median(times_rfft_mul)  # Use FFTW mul! as baseline (fastest FFT)
ratio_dct = median(times_dct) / baseline
ratio_dct_plan = median(times_dct_plan) / baseline
ratio_mul = median(times_mul) / baseline
ratio_rfft = median(times_rfft) / baseline
ratio_rfft_plan = median(times_rfft_plan) / baseline

speedup_plan = median(times_dct) / median(times_dct_plan)
speedup_mul = median(times_dct) / median(times_mul)

println()
println("FFT Methods:")
println("  FFTW mul!(y,plan,x):       $(round(median(times_rfft_mul), digits=3)) ms (baseline)")
println("  FFTW plan * x:             $(round(median(times_rfft_plan), digits=3)) ms ($(round(ratio_rfft_plan, digits=2))x)")
println("  FFTW rfft (no plan):       $(round(median(times_rfft), digits=3)) ms ($(round(ratio_rfft, digits=2))x)")
println()
println("DCT Methods:")
println("  dct_fast (no plan):        $(round(median(times_dct), digits=3)) ms ($(round(ratio_dct, digits=2))x)")
println("  plan * x (cached):         $(round(median(times_dct_plan), digits=3)) ms ($(round(ratio_dct_plan, digits=2))x)")
println("  mul!(y,plan,x) (zero):     $(round(median(times_mul), digits=3)) ms ($(round(ratio_mul, digits=2))x)")
println()
println("DCT Optimization Benefits:")
println("  • Plan caching speedup:    $(round(speedup_plan, digits=2))x faster vs dct_fast")
println("  • Zero-alloc speedup:      $(round(speedup_mul, digits=2))x faster vs dct_fast")
println()

# ============================================================================
# Breakdown Analysis
# ============================================================================
println("="^60)
println("Breakdown Analysis: DCT operations")
println("="^60)

println()
println("3D DCT consists of:")
println("  - 3x dct_batch_dim1 (each with: preprocess kernel + rfft(dim=1) + postprocess kernel)")
println("  - 4x permutedims (to reorder dimensions)")
println()

# Benchmark single dct_batch_dim1
println("-"^60)
println("Benchmarking single dct_batch_dim1...")
println("-"^60)

times_single_dct = benchmark_cpu(dct_batch_dim1, x_cpu)
println("  Median time: $(round(median(times_single_dct), digits=3)) ms")
println()

# Benchmark permutedims
println("-"^60)
println("Benchmarking permutedims...")
println("-"^60)

times_perm = benchmark_cpu(x -> permutedims(x, (2, 1, 3)), x_cpu)
println("  Median time: $(round(median(times_perm), digits=3)) ms")
println()

# Theoretical breakdown
println("="^60)
println("Theoretical vs Actual Breakdown")
println("="^60)
theoretical_time = 3 * median(times_single_dct) + 4 * median(times_perm)
actual_time = median(times_dct_plan)
println()
println("  3x dct_batch_dim1:    $(round(3 * median(times_single_dct), digits=3)) ms")
println("  4x permutedims:       $(round(4 * median(times_perm), digits=3)) ms")
println("  ----------------------------------------")
println("  Theoretical total:    $(round(theoretical_time, digits=3)) ms")
println("  Actual (plan * x):    $(round(actual_time, digits=3)) ms")
println("  Difference:           $(round(actual_time - theoretical_time, digits=3)) ms")
println()

# Also compare 1D rfft batched vs 3D rfft
println("="^60)
println("Additional: Compare 1D batched rfft vs 3D rfft")
println("="^60)

println()
println("Benchmarking rfft(x, 1) (batched 1D along dim 1)...")
times_rfft_1d = benchmark_cpu(x -> rfft(x, 1), x_cpu)

println("  rfft(x, 1) median: $(round(median(times_rfft_1d), digits=3)) ms")
println("  rfft(x) 3D median: $(round(median(times_rfft), digits=3)) ms")
println("  Ratio: $(round(median(times_rfft) / median(times_rfft_1d), digits=2))x (3D is this many times of 1D)")
println()

# ============================================================================
# Final Summary
# ============================================================================
println("="^60)
println("Summary (CPU)")
println("="^60)
println()
println("Performance Results (relative to FFTW mul!):")
println("  • FFTW mul!(y,plan,x):    $(round(median(times_rfft_mul), digits=3)) ms (baseline)")
println("  • dct_fast (no plan):     $(round(median(times_dct), digits=3)) ms ($(round(ratio_dct, digits=2))x)")
println("  • plan * x (cached):      $(round(median(times_dct_plan), digits=3)) ms ($(round(ratio_dct_plan, digits=2))x)")
println("  • mul!(y,plan,x) (zero):  $(round(median(times_mul), digits=3)) ms ($(round(ratio_mul, digits=2))x)")
println()
println("Plan Creation Overhead:")
println("  • DCT plan creation:      $(round(plan_creation_time, digits=3)) ms")
println("  • FFTW plan creation:     $(round(fftw_plan_creation_time, digits=3)) ms")
println()

best_dct_time = min(median(times_dct_plan), median(times_mul))
best_dct_ratio = best_dct_time / baseline

if best_dct_ratio > 10
    println("⚠️  Significant performance gap vs FFTW. Consider:")
    println("   - Using multi-dimensional rfft to reduce overhead")
    println("   - Fusing kernels to reduce memory bandwidth")
elseif best_dct_ratio > 3
    println("⚠️  Moderate performance gap vs FFTW. This is expected due to:")
    println("   - Separable 1D approach requiring 3x rfft + 4x permutedims")
    println("   - Additional pre/post processing for DCT normalization")
else
    println("✓  Performance is within acceptable range.")
end
println()
