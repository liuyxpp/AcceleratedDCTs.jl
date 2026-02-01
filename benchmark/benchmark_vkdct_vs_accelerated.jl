
using BenchmarkTools
using CUDA
using CUDA.CUFFT
using LinearAlgebra
using Printf
using AcceleratedDCTs
using AbstractFFTs

# Include VkDCT module
include(joinpath(@__DIR__, "../lib/VkDCT/VkDCT.jl"))
using .VkDCT

function run_benchmark()
    if !CUDA.functional()
        @error "CUDA not available!"
        return
    end

    sizes = [
        (64, 64, 64),
        (65, 65, 65),
        (128, 128, 128),
        # (129, 129, 129) # Uncomment if VRAM allows and stability is checked
        (129, 129, 129)
    ]
    
    types = [Float32, Float64]

    @printf "%-20s %-10s %-15s %-15s %-15s %-10s %-10s\n" "Size" "Type" "AcceleratedDCTs" "VkDCT" "cuFFT(2M-2)" "Speedup(Acc)" "Speedup(FFT)"
    println("-"^105)

    for T in types
        for sz in sizes
            # Generate random data
            # Use a fresh array for each run to avoid cached plan issues if any
            h_x = rand(T, sz...)
            d_x = CuArray(h_x)
            
            # --- acceleratedDCTs ---
            # Create plan
            p_acc = try
                AcceleratedDCTs.plan_dct1(d_x)
            catch e
                nothing
            end

            t_acc = Inf
            if p_acc !== nothing
                # Warmup
                y_acc = similar(d_x)
                mul!(y_acc, p_acc, d_x)
                CUDA.synchronize()
                
                # Benchmark
                b_acc = @benchmark begin
                    mul!($y_acc, $p_acc, $d_x)
                    CUDA.synchronize()
                end samples=10 seconds=1
                t_acc = median(b_acc).time / 1e6 # ms
            else
                t_acc = Inf # Not supported or failed
            end

            # --- VkDCT ---
            p_vk = nothing
            t_vk = Inf
            try
                p_vk = VkDCT.plan_dct(d_x)
                
                 # Warmup
                y_vk = similar(d_x)
                mul!(y_vk, p_vk, d_x)
                CUDA.synchronize()

                # Benchmark
                b_vk = @benchmark begin 
                    mul!($y_vk, $p_vk, $d_x)
                    CUDA.synchronize()
                end samples=10 seconds=1
                t_vk = median(b_vk).time / 1e6 # ms
            catch e
                # print(e)
                t_vk = Inf
            end

            # Cleanup
            if p_vk !== nothing
                VkDCT.destroy_plan!(p_vk)
            end
            
            # --- cuFFT (rfft on (2M-2)^3) ---
            # DCT-I of size M corresponds to even extension of size 2M-2
            fft_sz = (2*sz[1]-2, 2*sz[2]-2, 2*sz[3]-2)
            t_fft = Inf
            
            # Check if size is reasonable (memory wise)
            # 129 -> 256^3 * 8 bytes ~ 134MB. Totally fine.
            try 
                d_fft_x = CUDA.rand(T, fft_sz...)
                p_fft = plan_rfft(d_fft_x)
                
                # Pre-allocate output for mul!
                # RFFT output size: (N1/2 + 1, N2, N3)
                out_fft_sz = (div(fft_sz[1], 2) + 1, fft_sz[2], fft_sz[3])
                y_fft = CUDA.zeros(Complex{T}, out_fft_sz)

                # Warmup
                mul!(y_fft, p_fft, d_fft_x)
                CUDA.synchronize()
                
                # Benchmark (using mul! for fairness)
                b_fft = @benchmark begin
                    mul!($y_fft, $p_fft, $d_fft_x)
                    CUDA.synchronize()
                end samples=10 seconds=1
                t_fft = median(b_fft).time / 1e6 # ms
            catch e
                println("Error in cuFFT: ", e)
                t_fft = Inf
            end

            # --- Report ---
            speedup_acc = t_acc / t_vk
            speedup_fft = t_fft / t_vk
            
            sz_str = string(sz)
            acc_str = t_acc == Inf ? "Fail" : @sprintf("%.3f ms", t_acc)
            vk_str = t_vk == Inf ? "Fail" : @sprintf("%.3f ms", t_vk)
            fft_str = t_fft == Inf ? "Fail" : @sprintf("%.3f ms", t_fft)
            
            spd_acc_str = t_acc == Inf || t_vk == Inf ? "-" : @sprintf("%.2fx", speedup_acc)
            spd_fft_str = t_fft == Inf || t_vk == Inf ? "-" : @sprintf("%.2fx", speedup_fft)
            
            @printf "%-20s %-10s %-15s %-15s %-15s %-10s %-10s\n" sz_str string(T) acc_str vk_str fft_str spd_acc_str spd_fft_str
            
            GC.gc()
            CUDA.reclaim()
        end
    end
end

run_benchmark()
