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
using AcceleratedDCTs: plan_dct1_mirror, plan_dct1

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
Ms = [33, 65, 129, 257]
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
    p_dct1 = plan_dct1_mirror(x_gpu)
    p_dct1_sep = plan_dct1(x_gpu)
    p_rfft = plan_rfft(x_extended_gpu)
    
    # Alloc Complex input for FFT (2M-2) comparisons
    # This is to compare against "Naive Complex FFT of roughly 2x size" 
    # (though typically Mirror uses RFFT 2M-2, or Separable uses FFT M-1).
    x_complex_2m_gpu = CUDA.zeros(ComplexF64, N, N, N)
    p_fft_2m = plan_fft!(x_complex_2m_gpu) # In-place complex plan for 2M-2
    
    # Measure cuFFT rfft (2M-2) - reference for internal FFT cost
    times_rfft_2m = safe_benchmark("cuFFT rfft (2M-2)", x -> mul!(y_complex_gpu, p_rfft, x), x_extended_gpu; n_samples=5)
    t_rfft_2m = isempty(times_rfft_2m) || isnan(times_rfft_2m[1]) ? NaN : median(times_rfft_2m)

    # Measure cuFFT fft (2M-2) - Complex-to-Complex
    # Note: 2M-2 size is large. Might OOM for 256^3 (512^3 complex ~ 2GB * buffers).
    # We use safe_benchmark to handle OOM.
    times_fft_2m = safe_benchmark("cuFFT fft (2M-2)", x -> mul!(x, p_fft_2m, x), x_complex_2m_gpu; n_samples=5)
    t_fft_2m = isempty(times_fft_2m) || isnan(times_fft_2m[1]) ? NaN : median(times_fft_2m)

    # Measure cuFFT rfft (M) - reference for same-size FFT
    y_complex_m_gpu = CUDA.zeros(ComplexF64, M ÷ 2 + 1, M, M)
    p_rfft_m = plan_rfft(x_gpu)
    times_rfft_m = safe_benchmark("cuFFT rfft (M)", x -> mul!(y_complex_m_gpu, p_rfft_m, x), x_gpu; n_samples=5)
    t_rfft_m = isempty(times_rfft_m) || isnan(times_rfft_m[1]) ? NaN : median(times_rfft_m)
    
    # Measure AcceleratedDCTs DCT-I (mul!)
    times_opt = safe_benchmark("Mirror DCT-I (mul!)", x -> mul!(y_gpu, p_dct1, x), x_gpu; n_samples=5)
    t_opt = isempty(times_opt) || isnan(times_opt[1]) ? NaN : median(times_opt)

    # Measure AcceleratedDCTs Separable DCT-I
    times_sep = safe_benchmark("Separable DCT-I", x -> mul!(y_gpu, p_dct1_sep, x), x_gpu; n_samples=5)
    t_sep = isempty(times_sep) || isnan(times_sep[1]) ? NaN : median(times_sep)

    push!(results, (M, t_rfft_2m, t_fft_2m, t_rfft_m, t_opt, t_sep))
    
    # Cleanup between sizes
    x_gpu = nothing
    y_gpu = nothing
    y_complex_gpu = nothing
    x_complex_2m_gpu = nothing # Cleanup new buffer
    y_complex_m_gpu = nothing
    x_extended_gpu = nothing
    p_dct1 = nothing
    p_dct1_sep = nothing
    p_rfft = nothing
    p_rfft_m = nothing
    GC.gc()
    CUDA.reclaim()
    println()
end

println("="^80)
println("GPU DCT-I Performance Summary (Time in ms)")
println("="^80)
println("| Grid Size | cuFFT rfft (2M-2) | cuFFT fft (2M-2)  | cuFFT rfft (M)    | Mirror DCT-I | Separable DCT-I |")
println("|-----------|-------------------|-------------------|-------------------|--------------|-----------------|")
for (M, t_rfft_2m, t_fft_2m, t_rfft_m, t_opt, t_sep) in results
    @printf("| %3d^3     | %17.3f | %17.3f | %17.3f | %12.3f | %15.3f |\n", 
            M, t_rfft_2m, t_fft_2m, t_rfft_m, t_opt, t_sep)
end
println("="^80)

