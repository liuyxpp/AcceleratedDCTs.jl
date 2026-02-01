
using Test
using CUDA
using LinearAlgebra
using FFTW

# Include VkDCT module
include(joinpath(@__DIR__, "../lib/VkDCT/VkDCT.jl"))
using .VkDCT

# Helper function to compute FFTW's DCT-I (REDFT00) on CPU
# This serves as the ground truth
function fftw_dct1(x::AbstractArray{T}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

@testset "VkDCT Correctness Verification" begin
    if !CUDA.functional()
        @warn "CUDA not available, skipping VkDCT tests"
        return
    end

    @testset "Basic 3D DCT-I Consistency" begin
        # Test various sizes, including non-power-of-2 and non-cubic
        sizes = [
            (32, 32, 32),
            (33, 33, 33),
            (64, 64, 64),
            (32, 64, 32),
            (16, 32, 16),
            (17, 17, 17), # Prime sizes if supported, or at least odd
             # Small sizes
             (8, 8, 8),
             (4, 5, 6)
        ]

        for sz in sizes
            @testset "Size $sz" begin
                # 1. Generate random data on CPU
                h_x = rand(Float32, sz...)

                # 2. Compute Ground Truth (FFTW on CPU)
                h_y_ref = fftw_dct1(h_x)

                # 3. Compute VkDCT (on GPU)
                d_x = CuArray(h_x)

                # Create Plan
                p = VkDCT.plan_dct(d_x)

                # Forward Transform
                d_y = p * d_x

                # Download result to CPU
                h_y_vk = Array(d_y)

                # 4. Compare
                # Note: VkDCT is unnormalized, and FFTW REDFT00 is also unnormalized.
                # They should match directly.
                # Tolerance: Float32 precision might require reasonably loose tolerance
                @test h_y_vk ≈ h_y_ref rtol=1e-4

                # 5. Roundtrip (Inverse)
                # VkDCT inverse is unnormalized (same as forward).
                # Mathematically: DCT1(DCT1(x)) = (2(N-1))^d * x  for each dimension d
                # For 3D: scale_factor = (2(Nx-1)) * (2(Ny-1)) * (2(Nz-1))
                d_rec = p \ d_y
                h_rec = Array(d_rec)

                @test h_rec ≈ h_x rtol=1e-4

                # Clean up plan explicitly (optional, but good for testing)
                VkDCT.destroy_plan!(p)
            end
        end
    end
end
