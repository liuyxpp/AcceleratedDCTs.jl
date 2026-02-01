# Benchmark: Compare DCT-I (dct1) performance on CPU
#
# This script compares the performance of:
# 1. AcceleratedDCTs dct1 (optimized)
# 2. FFTW r2r (REDFT00) - native FFTW DCT-I (baseline)
# 3. FFTW rfft - for reference (DCT-I uses 2M-2 size FFT internally)
#
# Measured on CPU with M³ grids.

using FFTW
using Statistics
using LinearAlgebra
using Printf

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using AcceleratedDCTs
using AcceleratedDCTs: plan_dct1_mirror

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
Ms = [32, 33, 64, 65, 128, 129, 256, 257]
results = []

println("="^60)
println("Benchmark: DCT-I Performance on CPU")
println("="^60)
println()
println("Benchmarking sizes: $Ms")
println()

for M in Ms
    println("="^60)
    println("Running Benchmark for M = $M ($M x $M x $M)")
    println("="^60)
    
    # Alloc Input/Output
    x_cpu = rand(Float64, M, M, M)
    y_cpu = similar(x_cpu)
    
    # Alloc RFFT Output (for reference, DCT-I uses N=2M-2 internally)
    N = 2M - 2
    cdims = (N ÷ 2 + 1, N, N)
    y_complex = Array{ComplexF64}(undef, cdims)
    x_extended = zeros(Float64, N, N, N)
    
    # Create Plans
    p_dct1 = plan_dct1_mirror(x_cpu)
    p_fftw_dct1 = FFTW.plan_r2r(x_cpu, FFTW.REDFT00)
    p_rfft = plan_rfft(x_extended)
    
    # Measure FFTW DCT-I (r2r REDFT00)
    print("  FFTW DCT-I (mul!)... ")
    times_fftw = benchmark_cpu(x -> mul!(y_cpu, p_fftw_dct1, x), x_cpu; n_samples=5)
    t_fftw = median(times_fftw)
    println("Done ($t_fftw ms)")
    
    # Measure AcceleratedDCTs DCT-I (mul!)
    print("  Mirror DCT-I (mul!)... ")
    times_opt = benchmark_cpu(x -> mul!(y_cpu, p_dct1, x), x_cpu; n_samples=5)
    t_opt = median(times_opt)
    println("Done ($t_opt ms)")
    
    # Measure raw RFFT on 2M-2 size (reference for internal FFT cost)
    print("  FFTW rfft (2M-2)... ")
    times_rfft_2m = benchmark_cpu(x -> mul!(y_complex, p_rfft, x), x_extended; n_samples=5)
    t_rfft_2m = median(times_rfft_2m)
    println("Done ($t_rfft_2m ms)")

    # Measure raw RFFT on M size (reference for same-size FFT)
    # Re-use p_fftw_dct1 logic but make a new plan for M-size RFFT
    p_rfft_m = plan_rfft(x_cpu)
    y_complex_m = Array{ComplexF64}(undef, (M÷2+1, M, M))
    print("  FFTW rfft (M)...    ")
    times_rfft_m = benchmark_cpu(x -> mul!(y_complex_m, p_rfft_m, x), x_cpu; n_samples=5)
    t_rfft_m = median(times_rfft_m)
    println("Done ($t_rfft_m ms)")
    
    push!(results, (M, t_fftw, t_opt, t_rfft_2m, t_rfft_m))
    
    # Cleanup
    p_dct1 = nothing
    p_fftw_dct1 = nothing
    p_rfft = nothing
    GC.gc()
    println()
end

println("="^80)
println("CPU DCT-I Performance Summary (Time in ms)")
println("="^80)
println("| Grid Size | FFTW DCT-I | Mirror DCT-I | rfft (2M-2) | rfft (M)    |")
println("|-----------|------------|--------------|-------------|-------------|")
for (M, t_fftw, t_opt, t_rfft_2m, t_rfft_m) in results
    @printf("| %3d^3     | %10.3f | %12.3f | %11.3f | %11.3f |\n", 
            M, t_fftw, t_opt, t_rfft_2m, t_rfft_m)
end
println("="^80)

