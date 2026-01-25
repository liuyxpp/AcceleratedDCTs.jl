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
using Printf
using LinearAlgebra

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs
using AcceleratedDCTs: plan_dct, dct_batched

println("="^60)
println("Benchmark: 3D DCT (Algorithm 3) vs dct_fast vs cuFFT (GPU)")
println("="^60)
println()



# Helper for robust benchmarking
function safe_benchmark(name, f, x; n_warmup=2, n_samples=10)
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

# Sizes to benchmark
Ns = [16, 32, 64, 128, 256]
results = []

println("Benchmarking sizes: $Ns")
println()

# Check total GPU memory once
free_mem, total_mem = CUDA.Mem.info()
println("Free GPU Memory: $(round(free_mem/1024^2, digits=2)) MiB / $(round(total_mem/1024^2, digits=2)) MiB")
println("GPU: $(CUDA.name(CUDA.device()))")
println()

for N in Ns
    println("="^60)
    println("Running Benchmark for N = $N ($N x $N x $N)")
    println("="^60)
    
    # Alloc Input
    x_cpu = rand(Float64, N, N, N)
    x_gpu = CuArray(x_cpu)
    y_gpu = similar(x_gpu)
    
    # Alloc RFFT Output
    dims = size(x_gpu)
    cdims = ntuple(i -> ifelse(i == 1, dims[1] ÷ 2 + 1, dims[i]), 3)
    y_complex_gpu = CUDA.zeros(ComplexF64, cdims...)
    
    # Warmup / Pre-plan
    p = plan_dct(x_gpu)
    p_rfft = plan_rfft(x_gpu)
    
    # Measure cuFFT (Baseline - mul!)
    # Using small n_samples for stability across many sizes
    times_rfft = safe_benchmark("cuFFT rfft (mul!)", x -> mul!(y_complex_gpu, p_rfft, x), x_gpu; n_samples=5)
    t_rfft = isempty(times_rfft) || isnan(times_rfft[1]) ? NaN : median(times_rfft)
    
    # Measure Optimized DCT (Plan - mul!)
    times_opt = safe_benchmark("Opt DCT (Algorithm 3) w/ mul!", x -> mul!(y_gpu, p, x), x_gpu; n_samples=5)
    t_opt = isempty(times_opt) || isnan(times_opt[1]) ? NaN : median(times_opt)
    
    # Measure Batched DCT (Allocating)
    times_batched = safe_benchmark("Batched DCT", dct_batched, x_gpu; n_samples=5)
    t_batched = isempty(times_batched) || isnan(times_batched[1]) ? NaN : median(times_batched)

    push!(results, (N, t_rfft, t_opt, t_batched))
    
    # Cleanup between sizes
    x_gpu = nothing
    y_gpu = nothing
    y_complex_gpu = nothing
    p = nothing
    p_rfft = nothing
    GC.gc()
    CUDA.reclaim()
    println()
end

println("="^80)
println("GPU Performance Summary (Time in ms)")
println("="^80)
println("| Grid Size | cuFFT rfft | Opt 3D DCT | Batched DCT |")
println("|-----------|------------|------------|-------------|")
for (N, t_rfft, t_opt, t_batched) in results
    # Format: N^3 | rfft | opt | batched
    @printf("| %3d^3     | %10.3f | %10.3f | %11.3f |\n", 
            N, t_rfft, t_opt, t_batched)
end
println("="^80)

