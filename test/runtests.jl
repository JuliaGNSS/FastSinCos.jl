using FastSinCos
using SIMD
using Test

@testset "FastSinCos.jl" begin
    @testset "u3500" begin
        v = SIMD.Vec{8,Float32}(ntuple(i -> Float32(i * 0.7), 8))
        s, c = fast_sincos_u3500(v)
        s_ref = sin.(Float64.(NTuple{8,Float32}(v)))
        c_ref = cos.(Float64.(NTuple{8,Float32}(v)))
        s_tup = NTuple{8,Float32}(s)
        c_tup = NTuple{8,Float32}(c)
        for i in 1:8
            @test abs(s_tup[i] - Float32(s_ref[i])) < 1e-4
            @test abs(c_tup[i] - Float32(c_ref[i])) < 1e-4
        end
    end

    @testset "u35" begin
        v = SIMD.Vec{8,Float32}(ntuple(i -> Float32(i * 0.7), 8))
        s, c = fast_sincos_u35(v)
        s_ref = sin.(Float64.(NTuple{8,Float32}(v)))
        c_ref = cos.(Float64.(NTuple{8,Float32}(v)))
        s_tup = NTuple{8,Float32}(s)
        c_tup = NTuple{8,Float32}(c)
        for i in 1:8
            @test abs(s_tup[i] - Float32(s_ref[i])) < 1e-6
            @test abs(c_tup[i] - Float32(c_ref[i])) < 1e-6
        end
    end

    @testset "u35 quadrant coverage" begin
        # Test values spanning all four quadrants
        vals = Float32[0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]
        v = SIMD.Vec{8,Float32}(ntuple(i -> vals[i], 8))
        s, c = fast_sincos_u35(v)
        s_tup = NTuple{8,Float32}(s)
        c_tup = NTuple{8,Float32}(c)
        for i in 1:8
            ref_s = sin(Float64(vals[i]))
            ref_c = cos(Float64(vals[i]))
            @test abs(s_tup[i] - Float32(ref_s)) < 1e-6
            @test abs(c_tup[i] - Float32(ref_c)) < 1e-6
        end
    end

    @testset "u3500 quadrant coverage" begin
        vals = Float32[0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]
        v = SIMD.Vec{8,Float32}(ntuple(i -> vals[i], 8))
        s, c = fast_sincos_u3500(v)
        s_tup = NTuple{8,Float32}(s)
        c_tup = NTuple{8,Float32}(c)
        for i in 1:8
            ref_s = sin(Float64(vals[i]))
            ref_c = cos(Float64(vals[i]))
            @test abs(s_tup[i] - Float32(ref_s)) < 1e-4
            @test abs(c_tup[i] - Float32(ref_c)) < 1e-4
        end
    end

    @testset "different Vec widths" begin
        for N in (4, 8, 16)
            v = SIMD.Vec{N,Float32}(ntuple(i -> Float32(i * 0.3), N))
            s35, c35 = fast_sincos_u35(v)
            s3500, c3500 = fast_sincos_u3500(v)
            @test s35 isa SIMD.Vec{N,Float32}
            @test c35 isa SIMD.Vec{N,Float32}
            @test s3500 isa SIMD.Vec{N,Float32}
            @test c3500 isa SIMD.Vec{N,Float32}
        end
    end
end
