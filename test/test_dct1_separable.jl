import FFTW
import AbstractFFTs
using Test
using AcceleratedDCTs
using AcceleratedDCTs: dct1, idct1, plan_dct1, plan_idct1, DCT1Plan, IDCT1Plan
using LinearAlgebra: mul!, ldiv!
import KernelAbstractions

# Helper function to compute FFTW's DCT-I (REDFT00)
function fftw_dct1(x::AbstractVector{T}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

function fftw_dct1(x::AbstractMatrix{T}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

function fftw_dct1(x::AbstractArray{T, 3}) where T
    return FFTW.r2r(x, FFTW.REDFT00)
end

# Wrapper to force generic fallback (testing dct1_optimized.jl on CPU)
struct GenericWrapper{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
Base.size(w::GenericWrapper) = size(w.data)
Base.getindex(w::GenericWrapper, i...) = getindex(w.data, i...)
Base.setindex!(w::GenericWrapper, v, i...) = setindex!(w.data, v, i...)
Base.IndexStyle(::Type{<:GenericWrapper}) = IndexLinear()
Base.similar(w::GenericWrapper, ::Type{T}, dims::Dims) where T = GenericWrapper(similar(w.data, T, dims))
# KA backend forwarding
KernelAbstractions.get_backend(w::GenericWrapper) = KernelAbstractions.get_backend(w.data)

@testset "Optimized DCT-I (1D)" begin
    @testset "Consistency with FFTW DCT-I" begin
        for M in [4, 5, 8, 9, 16, 17, 32, 33, 64]
            x = rand(Float64, M)
            y_fftw = fftw_dct1(x)
            y_opt = dct1(x)
            @test y_opt ≈ y_fftw atol=1e-12
        end
    end

    @testset "Roundtrip Accuracy" begin
        for M in [4, 5, 8, 9, 16, 17, 32, 33, 64]
            x = rand(M)
            y = dct1(x)
            x_rec = idct1(y)
            @test x_rec ≈ x atol=1e-12
        end
    end

    @testset "Generic Fallback (New Optimization)" begin
        M = 16
        x_data = rand(Float64, M)
        x = GenericWrapper(x_data)
        
        # Plan verification
        p = plan_dct1(x)
        @test p isa DCT1Plan # Must be the new generic plan, not FFTWBased
        
        y = p * x
        @test y isa GenericWrapper
        @test y.data ≈ fftw_dct1(x_data)
        
        y_direct = dct1(x)
        @test y_direct.data ≈ y.data
        
        x_rec = idct1(y)
        @test x_rec.data ≈ x.data atol=1e-12
        
        # Check that it actually uses the right size buffer
        @test size(p.tmp_comp) == size(x)
    end
end

@testset "Optimized DCT-I (2D)" begin
    @testset "Consistency with FFTW DCT-I" begin
        sizes = [(4, 4), (5, 5), (8, 8), (8, 9), (9, 8), (16, 16), (4, 8)]
        for (M1, M2) in sizes
            x = rand(Float64, M1, M2)
            y_fftw = fftw_dct1(x)
            @test dct1(x) ≈ y_fftw atol=1e-11
        end
    end

    @testset "Generic Fallback (2D)" begin
        M1, M2 = 8, 9
        x_data = rand(M1, M2)
        x = GenericWrapper(x_data)
        
        y = dct1(x)
        @test y.data ≈ fftw_dct1(x_data)
        
        x_rec = idct1(y)
        @test x_rec.data ≈ x_data atol=1e-12
    end
end

@testset "Optimized DCT-I (3D)" begin
    @testset "Consistency with FFTW DCT-I" begin
        sizes = [(4, 4, 4), (5, 5, 5), (4, 5, 6), (8, 8, 8), (4, 8, 4)]
        for sz in sizes
            x = rand(Float64, sz...)
            y_fftw = fftw_dct1(x)
            @test dct1(x) ≈ y_fftw atol=1e-10
        end
    end

    @testset "Generic Fallback (3D)" begin
        sz = (4, 5, 6)
        x_data = rand(sz...)
        x = GenericWrapper(x_data)
        
        y = dct1(x)
        @test y.data ≈ fftw_dct1(x_data)
        
        x_rec = idct1(y)
        @test x_rec.data ≈ x_data atol=1e-11
    end
end
