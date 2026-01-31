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
using AcceleratedDCTs: plan_dct1


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
Ms = [16, 32, 64, 128, 256]
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
    p_dct1 = plan_dct1(x_cpu)
    p_fftw_dct1 = FFTW.plan_r2r(x_cpu, FFTW.REDFT00)
    p_rfft = plan_rfft(x_extended)
    
    # Measure FFTW DCT-I (r2r REDFT00)
    print("  FFTW DCT-I (mul!)... ")
    times_fftw = benchmark_cpu(x -> mul!(y_cpu, p_fftw_dct1, x), x_cpu; n_samples=5)
    t_fftw = median(times_fftw)
    println("Done ($t_fftw ms)")
    
    # Measure AcceleratedDCTs DCT-I (mul!)
    print("  Opt DCT-I (mul!)... ")
    times_opt = benchmark_cpu(x -> mul!(y_cpu, p_dct1, x), x_cpu; n_samples=5)
    t_opt = median(times_opt)
    println("Done ($t_opt ms)")
    
    # Measure raw RFFT on 2M-2 size (reference for internal FFT cost)
    print("  FFTW rfft (2M-2)... ")
    times_rfft = benchmark_cpu(x -> mul!(y_complex, p_rfft, x), x_extended; n_samples=5)
    t_rfft = median(times_rfft)
    println("Done ($t_rfft ms)")
    
    push!(results, (M, t_fftw, t_opt, t_rfft))
    
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
println("| Grid Size | FFTW DCT-I | Opt DCT-I | rfft (2M-2) |")
println("|-----------|------------|-----------|-------------|")
for (M, t_fftw, t_opt, t_rfft) in results
    @printf("| %3d^3     | %10.3f | %9.3f | %11.3f |\n", 
            M, t_fftw, t_opt, t_rfft)
end
println("="^80)

