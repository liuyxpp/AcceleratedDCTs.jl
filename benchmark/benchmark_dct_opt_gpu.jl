# Benchmark: Compare Algorithm 3 3D DCT (opt) vs dct_fast vs cuFFT on GPU
#
# This script compares the performance of:
# 1. dct_3d_opt (Algorithm 3)
# 2. dct_fast (Batched)
# 3. cuFFT rfft
#
# Measured on GPU. Includes robust OOM handling.

using CUDA
using CUDA.CUFFT
using Statistics

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs

println("="^60)
println("Benchmark: 3D DCT (Algorithm 3) vs dct_fast vs cuFFT (GPU)")
println("="^60)
println()

# Setup
N = 384
println("Grid size: $(N)x$(N)x$(N)")

# Check available memory
free_mem, total_mem = CUDA.Mem.info()
input_size_mb = (N^3 * 8) / 1024^2
println("Input array size: $(round(input_size_mb, digits=2)) MiB")
println("Free GPU Memory: $(round(free_mem/1024^2, digits=2)) MiB / $(round(total_mem/1024^2, digits=2)) MiB")
println("GPU: $(CUDA.name(CUDA.device()))")
println()

# Check if we assume we might OOM immediately
if input_size_mb * 6 > free_mem / 1024^2
    println("WARNING: Detailed benchmarks might OOM due to limited memory.")
    println("Estimated peak memory usage > $(round(input_size_mb*6, digits=2)) MiB")
end

println("Allocating input...")
x_cpu = rand(Float64, N, N, N)
x_gpu = CuArray(x_cpu)
println("Input allocated.")
println()

# Helper for robust benchmarking
function safe_benchmark(name, f, x; n_warmup=2, n_samples=10)
    println("-"^60)
    println("Benchmarking $name...")
    println("-"^60)
    
    # Aggressive Cleanup before start
    GC.gc()
    CUDA.reclaim()
    
    # Warmup
    try
        print("  Warming up... ")
        for _ in 1:n_warmup
            f(x)
            CUDA.synchronize()
        end
        println("Done.")
    catch e
        if isa(e, CUDA.OutOfMemoryError)
            println("\n  FAILED: Out Of Memory during warmup.")
            CUDA.reclaim()
            return [NaN]
        else
            println("\n  FAILED: Error during warmup: $e")
            return [NaN]
        end
    end
    
    # Measure
    times = Float64[]
    try
        for i in 1:n_samples
            # Aggressive cleanup between samples to survive tight memory conditions
            if i > 1
                GC.gc()
                CUDA.reclaim()
            end
            
            CUDA.synchronize()
            t0 = time_ns()
            f(x)
            CUDA.synchronize()
            t1 = time_ns()
            push!(times, (t1 - t0) / 1e6)  # ms
            print(".") 
        end
        println() # Newline after dots
        
        # Report
        min_t = minimum(times)
        med_t = median(times)
        max_t = maximum(times)
        
        println("  Minimum time:  $(round(min_t, digits=3)) ms")
        println("  Median time:   $(round(med_t, digits=3)) ms")
        println("  Maximum time:  $(round(max_t, digits=3)) ms")
        
        return times
    catch e
        if isa(e, CUDA.OutOfMemoryError)
            println("\n  FAILED: Out Of Memory during benchmark.")
            CUDA.reclaim()
            return [NaN]
        else
            println("\n  FAILED: Error during benchmark: $e")
            return [NaN]
        end
    end
end

# 1. Benchmark dct_3d_opt (No Plan Reuse)
println("Description: 3D DCT (Algorithm 3) - Plan creation + Execution")
times_dct_opt = safe_benchmark("dct_3d_opt (One-shot)", dct_3d_opt, x_gpu)
println()

# 1b. Benchmark dct_3d_opt (With Plan Reuse)
println("Description: 3D DCT (Algorithm 3) - Precomputed Plan (Execution Only)")
p = plan_dct_opt(x_gpu) # Precompute plan
times_dct_opt_plan = safe_benchmark("dct_3d_opt (Cached Plan)", x -> p * x, x_gpu)
println()

# 2. Benchmark dct_fast
println("Description: 3x Separable 1D DCTs (Batched)")
times_dct_fast = safe_benchmark("dct_fast", dct_fast, x_gpu)
println()

# 3. Benchmark cuFFT
println("Description: cuFFT R2C 3D FFT (Baseline)")
times_rfft = safe_benchmark("cuFFT rfft", rfft, x_gpu)
println()

# Comparison
println("="^60)
println("Performance Comparison (using median times)")
println("="^60)

if !isnan(times_rfft[1])
    baseline = median(times_rfft)
    println("Baselines:")
    println("  cuFFT rfft:              $(round(baseline, digits=3)) ms")
    println()
    
    println("DCT Variants:")
    if !isnan(times_dct_opt[1])
        t_opt = median(times_dct_opt)
        t_opt_p = median(times_dct_opt_plan)
        println("  dct_3d_opt (One-shot):   $(round(t_opt, digits=3)) ms ($(round(t_opt/baseline, digits=2))x slower vs FFT)")
        println("  dct_3d_opt (Plan):       $(round(t_opt_p, digits=3)) ms ($(round(t_opt_p/baseline, digits=2))x slower vs FFT)")
        println("     -> Plan Speedup:      $(round(t_opt/t_opt_p, digits=2))x improvement from caching")
    else
        println("  dct_3d_opt:              OOM / FAILED")
    end
    
    if !isnan(times_dct_fast[1])
        t_fast = median(times_dct_fast)
        println("  dct_fast (Batched):      $(round(t_fast, digits=3)) ms ($(round(t_fast/baseline, digits=2))x slower vs FFT)")
    else
        println("  dct_fast (Batched):      OOM / FAILED")
    end
else
    println("Baseline (cuFFT) failed. Cannot compare.")
end
println()
