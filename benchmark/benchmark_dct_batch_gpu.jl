# Benchmark: Compare dct_batch's 3D DCT vs cuFFT rfft
#
# This script compares the performance of:
# 1. dct_fast (from dct_batch.jl) - our batched 1D DCT implementation (creates plan each call)
# 2. dct with plan_dct - our batched 1D DCT with cached plan (reuses buffers/twiddles)
# 3. cuFFT rfft - native CUDA R2C FFT
#
# Both are measured on 128³ or 256³ grids on GPU.

using CUDA
using CUDA.CUFFT
using Statistics

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# Include the batch implementation directly
include(joinpath(@__DIR__, "..", "src", "dct_batch.jl"))

println("="^60)
println("Benchmark: 3D DCT (dct_batch) vs cuFFT rfft")
println("="^60)
println()

# Setup
N = 256
x_cpu = rand(Float64, N, N, N)
x_gpu = CuArray(x_cpu)

println("Grid size: $(N)³")
println("Input array size: $(size(x_gpu))")
println("Input array type: $(typeof(x_gpu))")
println("GPU: $(CUDA.name(CUDA.device()))")
println()

# Create DCT plan (this is the key optimization - create once, reuse many times)
println("Creating DCT plan (one-time cost)...")
t_plan_start = time_ns()
dct_plan = plan_dct(x_gpu)
CUDA.synchronize()
t_plan_end = time_ns()
plan_creation_time = (t_plan_end - t_plan_start) / 1e6
println("  Plan creation time: $(round(plan_creation_time, digits=3)) ms")
println()

# Warmup
println("Warming up...")
_ = dct_fast(x_gpu)
_ = dct_plan * x_gpu
_ = rfft(x_gpu)
CUDA.synchronize()
println("Warmup complete.")
println()

# Benchmark function using manual timing
function benchmark_gpu(f, x; n_warmup=3, n_samples=20)
    # Warmup
    for _ in 1:n_warmup
        f(x)
        CUDA.synchronize()
    end
    
    # Measure
    times = Float64[]
    for _ in 1:n_samples
        CUDA.synchronize()
        t0 = time_ns()
        f(x)
        CUDA.synchronize()
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

times_dct = benchmark_gpu(dct_fast, x_gpu)
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

times_dct_plan = benchmark_gpu(x -> dct_plan * x, x_gpu)
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

using LinearAlgebra: mul!, ldiv!
y_preallocated = similar(x_gpu)

# Benchmark mul! - this should be the fastest since it reuses output buffer
times_mul = Float64[]
for _ in 1:3  # Warmup
    mul!(y_preallocated, dct_plan, x_gpu)
    CUDA.synchronize()
end
for _ in 1:20
    CUDA.synchronize()
    t0 = time_ns()
    mul!(y_preallocated, dct_plan, x_gpu)
    CUDA.synchronize()
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

y_gpu = dct_plan * x_gpu
times_idct_plan = benchmark_gpu(y -> dct_plan \ y, y_gpu)
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

x_preallocated = similar(x_gpu)
times_ldiv = Float64[]
for _ in 1:3  # Warmup
    ldiv!(x_preallocated, dct_plan, y_gpu)
    CUDA.synchronize()
end
for _ in 1:20
    CUDA.synchronize()
    t0 = time_ns()
    ldiv!(x_preallocated, dct_plan, y_gpu)
    CUDA.synchronize()
    t1 = time_ns()
    push!(times_ldiv, (t1 - t0) / 1e6)
end
println("  Minimum time:  $(round(minimum(times_ldiv), digits=3)) ms")
println("  Median time:   $(round(median(times_ldiv), digits=3)) ms")
println("  Mean time:     $(round(mean(times_ldiv), digits=3)) ms")
println("  Maximum time:  $(round(maximum(times_ldiv), digits=3)) ms")
println()

# ============================================================================
# Benchmark cuFFT rfft (3D)
# ============================================================================
println("-"^60)
println("Benchmarking cuFFT rfft (3D R2C FFT)...")
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

ratio_dct = median(times_dct) / median(times_rfft)
ratio_dct_plan = median(times_dct_plan) / median(times_rfft)
ratio_mul = median(times_mul) / median(times_rfft)
speedup_plan = median(times_dct) / median(times_dct_plan)
speedup_mul = median(times_dct) / median(times_mul)

println()
println("  cuFFT rfft median:           $(round(median(times_rfft), digits=3)) ms (baseline)")
println("  dct_fast median:             $(round(median(times_dct), digits=3)) ms ($(round(ratio_dct, digits=2))x slower)")
println("  plan * x median:             $(round(median(times_dct_plan), digits=3)) ms ($(round(ratio_dct_plan, digits=2))x slower)")
println("  mul!(y, plan, x) median:     $(round(median(times_mul), digits=3)) ms ($(round(ratio_mul, digits=2))x slower)")
println()
println("  Speedup from plan caching:   $(round(speedup_plan, digits=2))x faster")
println("  Speedup from mul! (zero-alloc): $(round(speedup_mul, digits=2))x faster")
println()

# ============================================================================
# Breakdown Analysis
# ============================================================================
println("="^60)
println("Breakdown Analysis: dct operations")
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

times_single_dct = benchmark_gpu(dct_batch_dim1, x_gpu)
println("  Median time: $(round(median(times_single_dct), digits=3)) ms")
println()

# Benchmark permutedims
println("-"^60)
println("Benchmarking permutedims...")
println("-"^60)

times_perm = benchmark_gpu(x -> permutedims(x, (2, 1, 3)), x_gpu)
println("  Median time: $(round(median(times_perm), digits=3)) ms")
println()

# Detailed breakdown of dct_batch_dim1 components
println("-"^60)
println("Detailed breakdown of dct_batch_dim1 components...")
println("-"^60)

# Setup for detailed benchmarking
dims = size(x_gpu)
Nx = dims[1]
halfN = Nx ÷ 2
Batch = prod(dims[2:end])

x_reshaped = reshape(x_gpu, (Nx, Batch))
v_buffer = similar(x_gpu, (Nx, Batch))
y_buffer = similar(x_gpu, (Nx, Batch))

be = get_backend(x_gpu)

# Preprocess kernel
println()
println("  Preprocess kernel:")
times_preprocess = Float64[]
for _ in 1:20
    CUDA.synchronize()
    t0 = time_ns()
    preprocess_batch_dim1_kernel!(be)(v_buffer, x_reshaped, Nx, halfN, Batch; ndrange=(Nx, Batch))
    KernelAbstractions.synchronize(be)
    t1 = time_ns()
    push!(times_preprocess, (t1 - t0) / 1e6)
end
println("    Median time: $(round(median(times_preprocess), digits=3)) ms")

# rfft along dim 1
println()
println("  rfft(v, 1):")
times_rfft_dim1 = benchmark_gpu(v -> rfft(v, 1), v_buffer)
println("    Median time: $(round(median(times_rfft_dim1), digits=3)) ms")

# Get V for postprocess benchmark
V = rfft(v_buffer, 1)

# Precompute twiddles
twiddles = get_twiddles(Nx, Float64, be)

# Postprocess kernel
println()
println("  Postprocess kernel:")
times_postprocess = Float64[]
for _ in 1:20
    CUDA.synchronize()
    t0 = time_ns()
    postprocess_batch_dim1_kernel!(be)(y_buffer, V, twiddles, Nx, halfN, Batch; ndrange=(Nx, Batch))
    KernelAbstractions.synchronize(be)
    t1 = time_ns()
    push!(times_postprocess, (t1 - t0) / 1e6)
end
println("    Median time: $(round(median(times_postprocess), digits=3)) ms")

# Memory allocation overhead
println()
println("  Memory allocation (similar):")
times_alloc = Float64[]
for _ in 1:20
    CUDA.synchronize()
    t0 = time_ns()
    _ = similar(x_gpu, (Nx, Batch))
    CUDA.synchronize()
    t1 = time_ns()
    push!(times_alloc, (t1 - t0) / 1e6)
end
println("    Median time: $(round(median(times_alloc), digits=3)) ms")

println()
println("  Summary of dct_batch_dim1 breakdown:")
component_total = median(times_preprocess) + median(times_rfft_dim1) + median(times_postprocess)
println("    preprocess + rfft + postprocess = $(round(component_total, digits=3)) ms")
println("    Actual dct_batch_dim1 = $(round(median(times_single_dct), digits=3)) ms")
println("    Overhead (alloc, sync, etc.) = $(round(median(times_single_dct) - component_total, digits=3)) ms")
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
times_rfft_1d = benchmark_gpu(x -> rfft(x, 1), x_gpu)

println("  rfft(x, 1) median: $(round(median(times_rfft_1d), digits=3)) ms")
println("  rfft(x) 3D median: $(round(median(times_rfft), digits=3)) ms")
println("  Ratio: $(round(median(times_rfft) / median(times_rfft_1d), digits=2))x (3D is this many times of 1D)")
println()

# ============================================================================
# Final Summary
# ============================================================================
println("="^60)
println("Summary")
println("="^60)
println()
println("Performance Results:")
println("  • cuFFT rfft (3D):        $(round(median(times_rfft), digits=3)) ms (baseline)")
println("  • dct_fast (no plan):     $(round(median(times_dct), digits=3)) ms ($(round(ratio_dct, digits=2))x slower)")
println("  • plan * x (cached):      $(round(median(times_dct_plan), digits=3)) ms ($(round(ratio_dct_plan, digits=2))x slower)")
println("  • mul!(y,plan,x) (zero):  $(round(median(times_mul), digits=3)) ms ($(round(ratio_mul, digits=2))x slower)")
println()
println("Optimization Benefits:")
println("  • Plan caching speedup:   $(round(speedup_plan, digits=2))x faster vs dct_fast")
println("  • Zero-alloc speedup:     $(round(speedup_mul, digits=2))x faster vs dct_fast")
println("  • Plan creation:          $(round(plan_creation_time, digits=3)) ms (one-time cost)")
println()

if ratio_mul > 10
    println("⚠️  Significant performance gap vs cuFFT. Consider:")
    println("   - Using multi-dimensional rfft to reduce overhead")
    println("   - Fusing kernels to reduce memory bandwidth")
elseif ratio_mul > 3
    println("⚠️  Moderate performance gap vs cuFFT. This is expected due to:")
    println("   - Separable 1D approach requiring 3x rfft + 4x permutedims")
    println("   - Additional pre/post processing for DCT normalization")
else
    println("✓  Performance is within acceptable range.")
end
println()

