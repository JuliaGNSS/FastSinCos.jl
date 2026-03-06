using BenchmarkTools
using FastSinCos
using SIMD

const SUITE = BenchmarkGroup()

SUITE["sincos"] = BenchmarkGroup()

# Benchmark the raw sincos functions at different Vec widths
for W in (4, 8, 16)
    v = SIMD.Vec{W,Float32}(ntuple(i -> Float32(i * 0.7), W))
    SUITE["sincos"]["u100k_Vec$W"] = @benchmarkable fast_sincos_u100k($v)
    SUITE["sincos"]["u3500_Vec$W"] = @benchmarkable fast_sincos_u3500($v)
    SUITE["sincos"]["u35_Vec$W"] = @benchmarkable fast_sincos_u35($v)
end

# Benchmark a realistic downconvert workload (N=25000 samples)
SUITE["downconvert"] = BenchmarkGroup()

const N_SAMP = 25000
const VECW = 8

# Basic 1x loop
function downconvert_u100k!(ds_re, ds_im, s_re, s_im,
        carrier_freq, sampling_freq, start_phase, num_samples)
    two_pi = Float32(2π)
    freq_ratio = carrier_freq / sampling_freq
    offsets = SIMD.Vec{VECW,Float32}(ntuple(k -> Float32(k - 1), VECW))
    i = 1
    @inbounds while i + VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx = base_idx + offsets
        phase = two_pi * (idx * freq_ratio + start_phase)
        s, c = fast_sincos_u100k(phase)
        sr = vload(SIMD.Vec{VECW,Float32}, pointer(s_re, i))
        si = vload(SIMD.Vec{VECW,Float32}, pointer(s_im, i))
        vstore(sr * c + si * s, pointer(ds_re, i))
        vstore(si * c - sr * s, pointer(ds_im, i))
        i += VECW
    end
end

function downconvert_u100k_unroll4!(ds_re, ds_im, s_re, s_im,
        carrier_freq, sampling_freq, start_phase, num_samples)
    two_pi = Float32(2π)
    freq_ratio = carrier_freq / sampling_freq
    offsets = SIMD.Vec{VECW,Float32}(ntuple(k -> Float32(k - 1), VECW))
    stride = SIMD.Vec{VECW,Float32}(ntuple(_ -> Float32(VECW), VECW))
    stride2 = stride + stride
    stride3 = stride2 + stride

    i = 1
    @inbounds while i + 4VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx0 = base_idx + offsets
        idx1 = idx0 + stride
        idx2 = idx0 + stride2
        idx3 = idx0 + stride3

        phase0 = two_pi * (idx0 * freq_ratio + start_phase)
        phase1 = two_pi * (idx1 * freq_ratio + start_phase)
        phase2 = two_pi * (idx2 * freq_ratio + start_phase)
        phase3 = two_pi * (idx3 * freq_ratio + start_phase)

        s0, c0 = fast_sincos_u100k(phase0)
        s1, c1 = fast_sincos_u100k(phase1)
        s2, c2 = fast_sincos_u100k(phase2)
        s3, c3 = fast_sincos_u100k(phase3)

        p_sre = pointer(s_re, i)
        p_sim = pointer(s_im, i)
        p_dre = pointer(ds_re, i)
        p_dim = pointer(ds_im, i)

        sr0 = vload(SIMD.Vec{VECW,Float32}, p_sre)
        si0 = vload(SIMD.Vec{VECW,Float32}, p_sim)
        vstore(sr0 * c0 + si0 * s0, p_dre)
        vstore(si0 * c0 - sr0 * s0, p_dim)

        sr1 = vload(SIMD.Vec{VECW,Float32}, p_sre + VECW*4)
        si1 = vload(SIMD.Vec{VECW,Float32}, p_sim + VECW*4)
        vstore(sr1 * c1 + si1 * s1, p_dre + VECW*4)
        vstore(si1 * c1 - sr1 * s1, p_dim + VECW*4)

        sr2 = vload(SIMD.Vec{VECW,Float32}, p_sre + 2VECW*4)
        si2 = vload(SIMD.Vec{VECW,Float32}, p_sim + 2VECW*4)
        vstore(sr2 * c2 + si2 * s2, p_dre + 2VECW*4)
        vstore(si2 * c2 - sr2 * s2, p_dim + 2VECW*4)

        sr3 = vload(SIMD.Vec{VECW,Float32}, p_sre + 3VECW*4)
        si3 = vload(SIMD.Vec{VECW,Float32}, p_sim + 3VECW*4)
        vstore(sr3 * c3 + si3 * s3, p_dre + 3VECW*4)
        vstore(si3 * c3 - sr3 * s3, p_dim + 3VECW*4)

        i += 4VECW
    end
    @inbounds while i + VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx = base_idx + offsets
        phase = two_pi * (idx * freq_ratio + start_phase)
        s, c = fast_sincos_u100k(phase)
        sr = vload(SIMD.Vec{VECW,Float32}, pointer(s_re, i))
        si = vload(SIMD.Vec{VECW,Float32}, pointer(s_im, i))
        vstore(sr * c + si * s, pointer(ds_re, i))
        vstore(si * c - sr * s, pointer(ds_im, i))
        i += VECW
    end
end

function downconvert_u3500!(ds_re, ds_im, s_re, s_im,
        carrier_freq, sampling_freq, start_phase, num_samples)
    two_pi = Float32(2π)
    freq_ratio = carrier_freq / sampling_freq
    offsets = SIMD.Vec{VECW,Float32}(ntuple(k -> Float32(k - 1), VECW))
    i = 1
    @inbounds while i + VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx = base_idx + offsets
        phase = two_pi * (idx * freq_ratio + start_phase)
        s, c = fast_sincos_u3500(phase)
        sr = vload(SIMD.Vec{VECW,Float32}, pointer(s_re, i))
        si = vload(SIMD.Vec{VECW,Float32}, pointer(s_im, i))
        vstore(sr * c + si * s, pointer(ds_re, i))
        vstore(si * c - sr * s, pointer(ds_im, i))
        i += VECW
    end
end

function downconvert_u35!(ds_re, ds_im, s_re, s_im,
        carrier_freq, sampling_freq, start_phase, num_samples)
    two_pi = Float32(2π)
    freq_ratio = carrier_freq / sampling_freq
    offsets = SIMD.Vec{VECW,Float32}(ntuple(k -> Float32(k - 1), VECW))
    i = 1
    @inbounds while i + VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx = base_idx + offsets
        phase = two_pi * (idx * freq_ratio + start_phase)
        s, c = fast_sincos_u35(phase)
        sr = vload(SIMD.Vec{VECW,Float32}, pointer(s_re, i))
        si = vload(SIMD.Vec{VECW,Float32}, pointer(s_im, i))
        vstore(sr * c + si * s, pointer(ds_re, i))
        vstore(si * c - sr * s, pointer(ds_im, i))
        i += VECW
    end
end

# 4x unrolled loop (reduces branch overhead, better ILP)
function downconvert_u3500_unroll4!(ds_re, ds_im, s_re, s_im,
        carrier_freq, sampling_freq, start_phase, num_samples)
    two_pi = Float32(2π)
    freq_ratio = carrier_freq / sampling_freq
    offsets = SIMD.Vec{VECW,Float32}(ntuple(k -> Float32(k - 1), VECW))
    stride = SIMD.Vec{VECW,Float32}(ntuple(_ -> Float32(VECW), VECW))
    stride2 = stride + stride
    stride3 = stride2 + stride

    i = 1
    @inbounds while i + 4VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx0 = base_idx + offsets
        idx1 = idx0 + stride
        idx2 = idx0 + stride2
        idx3 = idx0 + stride3

        phase0 = two_pi * (idx0 * freq_ratio + start_phase)
        phase1 = two_pi * (idx1 * freq_ratio + start_phase)
        phase2 = two_pi * (idx2 * freq_ratio + start_phase)
        phase3 = two_pi * (idx3 * freq_ratio + start_phase)

        s0, c0 = fast_sincos_u3500(phase0)
        s1, c1 = fast_sincos_u3500(phase1)
        s2, c2 = fast_sincos_u3500(phase2)
        s3, c3 = fast_sincos_u3500(phase3)

        p_sre = pointer(s_re, i)
        p_sim = pointer(s_im, i)
        p_dre = pointer(ds_re, i)
        p_dim = pointer(ds_im, i)

        sr0 = vload(SIMD.Vec{VECW,Float32}, p_sre)
        si0 = vload(SIMD.Vec{VECW,Float32}, p_sim)
        vstore(sr0 * c0 + si0 * s0, p_dre)
        vstore(si0 * c0 - sr0 * s0, p_dim)

        sr1 = vload(SIMD.Vec{VECW,Float32}, p_sre + VECW*4)
        si1 = vload(SIMD.Vec{VECW,Float32}, p_sim + VECW*4)
        vstore(sr1 * c1 + si1 * s1, p_dre + VECW*4)
        vstore(si1 * c1 - sr1 * s1, p_dim + VECW*4)

        sr2 = vload(SIMD.Vec{VECW,Float32}, p_sre + 2VECW*4)
        si2 = vload(SIMD.Vec{VECW,Float32}, p_sim + 2VECW*4)
        vstore(sr2 * c2 + si2 * s2, p_dre + 2VECW*4)
        vstore(si2 * c2 - sr2 * s2, p_dim + 2VECW*4)

        sr3 = vload(SIMD.Vec{VECW,Float32}, p_sre + 3VECW*4)
        si3 = vload(SIMD.Vec{VECW,Float32}, p_sim + 3VECW*4)
        vstore(sr3 * c3 + si3 * s3, p_dre + 3VECW*4)
        vstore(si3 * c3 - sr3 * s3, p_dim + 3VECW*4)

        i += 4VECW
    end
    @inbounds while i + VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx = base_idx + offsets
        phase = two_pi * (idx * freq_ratio + start_phase)
        s, c = fast_sincos_u3500(phase)
        sr = vload(SIMD.Vec{VECW,Float32}, pointer(s_re, i))
        si = vload(SIMD.Vec{VECW,Float32}, pointer(s_im, i))
        vstore(sr * c + si * s, pointer(ds_re, i))
        vstore(si * c - sr * s, pointer(ds_im, i))
        i += VECW
    end
end

# 4x unrolled loop for u35
function downconvert_u35_unroll4!(ds_re, ds_im, s_re, s_im,
        carrier_freq, sampling_freq, start_phase, num_samples)
    two_pi = Float32(2π)
    freq_ratio = carrier_freq / sampling_freq
    offsets = SIMD.Vec{VECW,Float32}(ntuple(k -> Float32(k - 1), VECW))
    stride = SIMD.Vec{VECW,Float32}(ntuple(_ -> Float32(VECW), VECW))
    stride2 = stride + stride
    stride3 = stride2 + stride

    i = 1
    @inbounds while i + 4VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx0 = base_idx + offsets
        idx1 = idx0 + stride
        idx2 = idx0 + stride2
        idx3 = idx0 + stride3

        phase0 = two_pi * (idx0 * freq_ratio + start_phase)
        phase1 = two_pi * (idx1 * freq_ratio + start_phase)
        phase2 = two_pi * (idx2 * freq_ratio + start_phase)
        phase3 = two_pi * (idx3 * freq_ratio + start_phase)

        s0, c0 = fast_sincos_u35(phase0)
        s1, c1 = fast_sincos_u35(phase1)
        s2, c2 = fast_sincos_u35(phase2)
        s3, c3 = fast_sincos_u35(phase3)

        p_sre = pointer(s_re, i)
        p_sim = pointer(s_im, i)
        p_dre = pointer(ds_re, i)
        p_dim = pointer(ds_im, i)

        sr0 = vload(SIMD.Vec{VECW,Float32}, p_sre)
        si0 = vload(SIMD.Vec{VECW,Float32}, p_sim)
        vstore(sr0 * c0 + si0 * s0, p_dre)
        vstore(si0 * c0 - sr0 * s0, p_dim)

        sr1 = vload(SIMD.Vec{VECW,Float32}, p_sre + VECW*4)
        si1 = vload(SIMD.Vec{VECW,Float32}, p_sim + VECW*4)
        vstore(sr1 * c1 + si1 * s1, p_dre + VECW*4)
        vstore(si1 * c1 - sr1 * s1, p_dim + VECW*4)

        sr2 = vload(SIMD.Vec{VECW,Float32}, p_sre + 2VECW*4)
        si2 = vload(SIMD.Vec{VECW,Float32}, p_sim + 2VECW*4)
        vstore(sr2 * c2 + si2 * s2, p_dre + 2VECW*4)
        vstore(si2 * c2 - sr2 * s2, p_dim + 2VECW*4)

        sr3 = vload(SIMD.Vec{VECW,Float32}, p_sre + 3VECW*4)
        si3 = vload(SIMD.Vec{VECW,Float32}, p_sim + 3VECW*4)
        vstore(sr3 * c3 + si3 * s3, p_dre + 3VECW*4)
        vstore(si3 * c3 - sr3 * s3, p_dim + 3VECW*4)

        i += 4VECW
    end
    @inbounds while i + VECW - 1 <= num_samples
        base_idx = Float32(i - 1)
        idx = base_idx + offsets
        phase = two_pi * (idx * freq_ratio + start_phase)
        s, c = fast_sincos_u35(phase)
        sr = vload(SIMD.Vec{VECW,Float32}, pointer(s_re, i))
        si = vload(SIMD.Vec{VECW,Float32}, pointer(s_im, i))
        vstore(sr * c + si * s, pointer(ds_re, i))
        vstore(si * c - sr * s, pointer(ds_im, i))
        i += VECW
    end
end

let ds_re = zeros(Float32, N_SAMP),
    ds_im = zeros(Float32, N_SAMP),
    s_re = rand(Float32, N_SAMP),
    s_im = rand(Float32, N_SAMP),
    carrier_freq = Float32(1000.0),
    sampling_freq = Float32(5e6),
    start_phase = Float32(0.3)

    SUITE["downconvert"]["u100k"] = @benchmarkable downconvert_u100k!(
        $ds_re, $ds_im, $s_re, $s_im, $carrier_freq, $sampling_freq, $start_phase, $N_SAMP)
    SUITE["downconvert"]["u100k_unroll4"] = @benchmarkable downconvert_u100k_unroll4!(
        $ds_re, $ds_im, $s_re, $s_im, $carrier_freq, $sampling_freq, $start_phase, $N_SAMP)
    SUITE["downconvert"]["u3500"] = @benchmarkable downconvert_u3500!(
        $ds_re, $ds_im, $s_re, $s_im, $carrier_freq, $sampling_freq, $start_phase, $N_SAMP)
    SUITE["downconvert"]["u35"] = @benchmarkable downconvert_u35!(
        $ds_re, $ds_im, $s_re, $s_im, $carrier_freq, $sampling_freq, $start_phase, $N_SAMP)
    SUITE["downconvert"]["u3500_unroll4"] = @benchmarkable downconvert_u3500_unroll4!(
        $ds_re, $ds_im, $s_re, $s_im, $carrier_freq, $sampling_freq, $start_phase, $N_SAMP)
    SUITE["downconvert"]["u35_unroll4"] = @benchmarkable downconvert_u35_unroll4!(
        $ds_re, $ds_im, $s_re, $s_im, $carrier_freq, $sampling_freq, $start_phase, $N_SAMP)
end
