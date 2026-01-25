# DCT Benchmark Script (1D Comparison) - Simplified using @time

using AcceleratedDCTs: dct, idct, dct_batched, idct_batched, plan_dct, plan_dct_batched, plan_idct
using LinearAlgebra
using Printf
using AcceleratedDCTs
using FFTW

# Check for GPU
args = [x for x in Base.loaded_modules if x.second == AcceleratedDCTs]
if isdefined(AcceleratedDCTs, :CUDA) && AcceleratedDCTs.CUDA.functional()
    using CUDA
    backend_name = "CUDA"
    T = Float32
    ArrayType = CuArray
    println("Running benchmarks on NVIDIA GPU (Float32)")
else
    backend_name = "CPU"
    T = Float64
    ArrayType = Array
    println("Running benchmarks on CPU (Float64)")
end

function run_benchmark()
    sizes = [1024, 4096, 65536, 1048576] # Reduced set
    
    println("\nBenchmarking 1D DCT (New Optimized vs Legacy Batched)")
    println("Backend: $backend_name")
    
    @printf("%-10s | %-15s | %-15s | %-10s\n", "Size (N)", "New Opt (ms)", "Batched (ms)", "Speedup")
    println("-"^60)
    
    for N in sizes
        x = rand(T, N) |> ArrayType
        
        # Dynamic sample count
        # Target ~0.1s total run time per measurement round
        samples = max(1, floor(Int, 10^7 / N)) 
        rounds = 5
        
        # --- Benchmark New Optimized ---
        p_opt = plan_dct(x)
        y_opt = similar(x)
        
        # Warmup specific to this size/plan
        mul!(y_opt, p_opt, x)
        
        times_opt = Float64[]
        for _ in 1:rounds
            GC.gc()
            t0 = time_ns()
            for _ in 1:samples
                mul!(y_opt, p_opt, x)
            end
            t_round = (time_ns() - t0) / samples / 1e6
            push!(times_opt, t_round)
        end
        t_opt = minimum(times_opt)
        
        # --- Benchmark Batched ---
        p_batch = plan_dct_batched(x)
        y_batch = similar(x)
        
        # Warmup
        mul!(y_batch, p_batch, x)
        
        times_batch = Float64[]
        for _ in 1:rounds
            GC.gc()
            t0 = time_ns()
            for _ in 1:samples
                mul!(y_batch, p_batch, x)
            end
            t_round = (time_ns() - t0) / samples / 1e6
            push!(times_batch, t_round)
        end
        t_batch = minimum(times_batch)
        
        speedup = t_batch / t_opt
        
        @printf("%-10d | %-15.4f | %-15.4f | %-10.2fx\n", N, t_opt, t_batch, speedup)
    end
    
    println("\nBenchmarking 1D IDCT")
    @printf("%-10s | %-15s | %-15s | %-10s\n", "Size (N)", "New Opt (ms)", "Batched (ms)", "Speedup")
    println("-"^60)
    
    for N in sizes
        x = rand(T, N) |> ArrayType
        y = dct(x)
        
        samples = max(1, floor(Int, 10^7 / N))
        rounds = 5
        
        # --- Benchmark New Optimized ---
        p_opt = plan_idct(x)
        x_out = similar(x)
        
        mul!(x_out, p_opt, y) # Warmup
        
        times_opt = Float64[]
        for _ in 1:rounds
            GC.gc()
            t0 = time_ns()
            for _ in 1:samples
                mul!(x_out, p_opt, y)
            end
            t_round = (time_ns() - t0) / samples / 1e6
            push!(times_opt, t_round)
        end
        t_opt = minimum(times_opt)
        
        # --- Benchmark Batched ---
        p_batch = plan_dct_batched(x)
        x_batch = similar(x)
        
        ldiv!(x_batch, p_batch, y) # Warmup
        
        times_batch = Float64[]
        for _ in 1:rounds
            GC.gc()
            t0 = time_ns()
            for _ in 1:samples
                ldiv!(x_batch, p_batch, y)
            end
            t_round = (time_ns() - t0) / samples / 1e6
            push!(times_batch, t_round)
        end
        t_batch = minimum(times_batch)
        
        speedup = t_batch / t_opt
        
        @printf("%-10d | %-15.4f | %-15.4f | %-10.2fx\n", N, t_opt, t_batch, speedup)
    end
end

run_benchmark()
