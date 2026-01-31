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
using Printf

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs
using AcceleratedDCTs: plan_dct, dct_batched

# Setup FFTW multi-threading
nthreads = Threads.nthreads()
FFTW.set_num_threads(nthreads)
println("FFTW using $nthreads threads")
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

# Sizes to benchmark
Ns = [16, 32, 64, 128, 256]
results = []

println("Benchmarking sizes: $Ns")
println()

for N in Ns
    println("="^60)
    println("Running Benchmark for N = $N ($N x $N x $N)")
    println("="^60)
    
    # Alloc Input/Output
    x_cpu = rand(Float64, N, N, N)
    y_cpu = similar(x_cpu)
    
    # Alloc RFFT Output
    dims = size(x_cpu)
    cdims = ntuple(i -> ifelse(i == 1, dims[1] ÷ 2 + 1, dims[i]), 3)
    y_complex = Array{ComplexF64}(undef, cdims)
    
    # Create Plans
    p_opt = plan_dct(x_cpu)
    p_rfft = plan_rfft(x_cpu)
    p_fftw_dct = FFTW.plan_dct(x_cpu)
    
    # Measure RFFT (mul!)
    print("  FFTW rfft (mul!)... ")
    times_rfft = benchmark_cpu(x -> mul!(y_complex, p_rfft, x), x_cpu; n_samples=5)
    t_rfft = median(times_rfft)
    println("Done ($t_rfft ms)")
    
    # Measure FFTW DCT (mul!)
    print("  FFTW dct (mul!)... ")
    times_fftw = benchmark_cpu(x -> mul!(y_cpu, p_fftw_dct, x), x_cpu; n_samples=5)
    t_fftw = median(times_fftw)
    println("Done ($t_fftw ms)")
    
    # Measure Optimized DCT (mul!)
    print("  Opt DCT (mul!)... ")
    times_opt = benchmark_cpu(x -> mul!(y_cpu, p_opt, x), x_cpu; n_samples=5)
    t_opt = median(times_opt)
    println("Done ($t_opt ms)")
    
    # Measure Batched DCT (Allocating - No mul! support)
    print("  Batched DCT... ")
    times_batched = benchmark_cpu(dct_batched, x_cpu; n_samples=5)
    t_batched = median(times_batched)
    println("Done ($t_batched ms)")
    
    padding = false # Placeholder
    push!(results, (N, t_rfft, t_opt, t_batched, t_fftw))
    
    # Cleanup
    p_opt = nothing
    p_rfft = nothing
    p_fftw_dct = nothing
    GC.gc()
    println()
end

println("="^80)
println("CPU Performance Summary (Time in ms)")
println("="^80)
println("| Grid Size | FFTW rfft | FFTW dct | Opt 3D DCT | Batched DCT |")
println("|-----------|-----------|----------|------------|-------------|")
for (N, t_rfft, t_opt, t_batched, t_fftw) in results
    # Format: N^3 | rfft | fftw_dct | opt | batched
    @printf("| %3d^3     | %9.3f | %8.3f | %10.3f | %11.3f |\n", 
            N, t_rfft, t_fftw, t_opt, t_batched)
end
println("="^80)

