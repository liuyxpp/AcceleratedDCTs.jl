# Benchmark: Compare DCT-I (dct1) performance on GPU
#
# This script compares the performance of:
# 1. AcceleratedDCTs dct1 (optimized)
# 2. cuFFT rfft - for reference (DCT-I uses 2M-2 size FFT internally)
#
# Note: CUDA does not have a native DCT-I, so we compare against raw FFT.
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
using AcceleratedDCTs: plan_dct1

println("="^60)
println("Benchmark: DCT-I Performance on GPU")
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
Ms = [16, 32, 64, 128, 256]
results = []

println("Benchmarking sizes: $Ms")
println()

# Check total GPU memory once
free_mem, total_mem = CUDA.Mem.info()
println("Free GPU Memory: $(round(free_mem/1024^2, digits=2)) MiB / $(round(total_mem/1024^2, digits=2)) MiB")
println("GPU: $(CUDA.name(CUDA.device()))")
println()

for M in Ms
    println("="^60)
    println("Running Benchmark for M = $M ($M x $M x $M)")
    println("="^60)
    
    # Alloc Input
    x_cpu = rand(Float64, M, M, M)
    x_gpu = CuArray(x_cpu)
    y_gpu = similar(x_gpu)
    
    # Alloc RFFT Output (DCT-I uses N=2M-2 internally)
    N = 2M - 2
    cdims = (N ÷ 2 + 1, N, N)
    y_complex_gpu = CUDA.zeros(ComplexF64, cdims...)
    x_extended_gpu = CUDA.zeros(Float64, N, N, N)
    
    # Create Plans
    p_dct1 = plan_dct1(x_gpu)
    p_rfft = plan_rfft(x_extended_gpu)
    
    # Measure cuFFT rfft (2M-2) - reference for internal FFT cost
    times_rfft = safe_benchmark("cuFFT rfft (2M-2)", x -> mul!(y_complex_gpu, p_rfft, x), x_extended_gpu; n_samples=5)
    t_rfft = isempty(times_rfft) || isnan(times_rfft[1]) ? NaN : median(times_rfft)
    
    # Measure AcceleratedDCTs DCT-I (mul!)
    times_opt = safe_benchmark("Opt DCT-I (mul!)", x -> mul!(y_gpu, p_dct1, x), x_gpu; n_samples=5)
    t_opt = isempty(times_opt) || isnan(times_opt[1]) ? NaN : median(times_opt)

    push!(results, (M, t_rfft, t_opt))
    
    # Cleanup between sizes
    x_gpu = nothing
    y_gpu = nothing
    y_complex_gpu = nothing
    x_extended_gpu = nothing
    p_dct1 = nothing
    p_rfft = nothing
    GC.gc()
    CUDA.reclaim()
    println()
end

println("="^80)
println("GPU DCT-I Performance Summary (Time in ms)")
println("="^80)
println("| Grid Size | cuFFT rfft (2M-2) | Opt DCT-I |")
println("|-----------|-------------------|-----------|")
for (M, t_rfft, t_opt) in results
    @printf("| %3d^3     | %17.3f | %9.3f |\n", 
            M, t_rfft, t_opt)
end
println("="^80)

