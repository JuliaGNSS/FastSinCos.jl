# FastSinCos.jl

Fast SIMD sincos approximations for Float32, using [SIMD.jl](https://github.com/eschnett/SIMD.jl).

Ported from [SLEEF](https://github.com/shibatch/sleef) polynomial approximations, with no deprecated dependencies (no LoopVectorization, VectorizationBase, or SLEEFPirates).

## Functions

Two accuracy variants are provided:

| Function | Accuracy | Speed | Best for |
|----------|----------|-------|----------|
| [`fast_sincos_u3500`](@ref) | ~3500 ULP | Fastest | Small inputs (`|d| ≲ 100`), maximum throughput |
| [`fast_sincos_u35`](@ref) | ~35 ULP | Fast | Any input range, when accuracy matters |

Both operate on `SIMD.Vec{N,Float32}` and return `(sin, cos)` as a tuple of vectors.

## Usage guide

### What is SIMD?

SIMD (Single Instruction, Multiple Data) allows a CPU to perform the same operation
on multiple values simultaneously. Instead of computing `sin` one float at a time,
SIMD processes 4, 8, or 16 floats in a single instruction.

[SIMD.jl](https://github.com/eschnett/SIMD.jl) provides `Vec{N,T}` types that
represent N values of type T packed into a SIMD register. For example,
`Vec{8,Float32}` holds 8 Float32 values that are processed in parallel.

### Basic usage

```julia
using FastSinCos
using SIMD

# Create a SIMD vector of 8 Float32 phases
phases = SIMD.Vec{8,Float32}((0.1f0, 0.5f0, 1.0f0, 1.5f0, 2.0f0, 2.5f0, 3.0f0, 3.5f0))

# Compute sin and cos of all 8 values at once
s, c = fast_sincos_u35(phases)
# s and c are each SIMD.Vec{8,Float32}
```

### Processing arrays in a loop

To process a Float32 array, load chunks into SIMD vectors using `vload`, process
them, and write results back with `vstore`:

```julia
using FastSinCos
using SIMD

const W = 8  # SIMD width (8 for AVX2 Float32)

function compute_sincos!(sin_out, cos_out, input, n)
    i = 1
    @inbounds while i + W - 1 <= n
        # Load W floats from the array into a SIMD vector
        v = vload(SIMD.Vec{W,Float32}, pointer(input, i))

        # Compute sin and cos of all W values simultaneously
        s, c = fast_sincos_u35(v)

        # Store results back to arrays
        vstore(s, pointer(sin_out, i))
        vstore(c, pointer(cos_out, i))

        i += W
    end
    # Handle remaining elements with scalar fallback
    @inbounds for i in i:n
        sin_out[i], cos_out[i] = sincos(input[i])
    end
end
```

### Choosing a variant

- Use [`fast_sincos_u3500`](@ref) when inputs are small (`|d| ≲ 100`) and you need
  maximum speed. Typical use case: GNSS carrier generation where phases are
  wrapped to `[-π, π]`.
- Use [`fast_sincos_u35`](@ref) when inputs can be large or accuracy matters.
  The Cody-Waite range reduction keeps error flat at ~6e-8 even for `|d| > 10⁶`.

### Performance tips

- **4x loop unrolling** can give an additional ~25% speedup by reducing loop
  overhead and improving instruction-level parallelism. Compute 4 SIMD chunks
  of phases before doing the loads/stores.
- **Phase accumulation** (`phase += delta` instead of recomputing from index)
  avoids an int→float conversion per iteration. This is faster but accumulates
  floating-point drift over many iterations — only use for short blocks.

## Accuracy vs input range

The `u3500` variant uses single-precision range reduction, so its error grows
linearly with input magnitude. The `u35` variant uses Cody-Waite range reduction
and maintains ~6e-8 absolute error across the entire valid range:

```@example accuracy
using FastSinCos, SIMD, CairoMakie

function max_abs_error(sincos_fn, range_max; n=10000)
    max_sin_err = 0.0
    max_cos_err = 0.0
    for _ in 1:n÷8
        vals = ntuple(_ -> Float32(rand() * 2 * range_max - range_max), 8)
        v = SIMD.Vec{8,Float32}(vals)
        s, c = sincos_fn(v)
        s_tup = NTuple{8,Float32}(s)
        c_tup = NTuple{8,Float32}(c)
        for i in 1:8
            ref_s = sin(Float64(vals[i]))
            ref_c = cos(Float64(vals[i]))
            max_sin_err = max(max_sin_err, abs(Float64(s_tup[i]) - ref_s))
            max_cos_err = max(max_cos_err, abs(Float64(c_tup[i]) - ref_c))
        end
    end
    return max(max_sin_err, max_cos_err)
end

ranges = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 1_000_000]
err_u3500 = [max_abs_error(fast_sincos_u3500, r) for r in ranges]
err_u35 = [max_abs_error(fast_sincos_u35, r) for r in ranges]

fig = Figure(size=(700, 400))
ax = Axis(fig[1, 1];
    xlabel="|d| max input range",
    ylabel="Max absolute error",
    xscale=log10, yscale=log10,
    title="Accuracy degradation vs input range")
scatterlines!(ax, Float64.(ranges), err_u3500; label="u3500", marker=:circle)
scatterlines!(ax, Float64.(ranges), err_u35; label="u35", marker=:diamond)
axislegend(ax; position=:lt)
fig
```

| `|d|` range | u3500 max error | u35 max error |
|-------------|-----------------|---------------|
| ±10 | 3.5e-6 | 6e-8 |
| ±100 | 6e-6 | 6e-8 |
| ±1,000 | 4e-5 | 6e-8 |
| ±10,000 | 8e-4 | 7e-8 |
| ±100,000 | 3e-3 | 6e-8 |
| ±1,000,000 | 4e-2 | 6e-8 |

## Performance

On a typical x86-64 CPU with AVX2 (Vec width 8), processing 25K Float32 samples
in a downconvert workload:

| Approach | Time |
|----------|------|
| FastSinCos u3500 (4x unrolled) | ~15 μs |
| LoopVectorization @avx + SLEEF | ~17 μs |
| FastSinCos u3500 (basic loop) | ~20 μs |
| FastSinCos u35 (basic loop) | ~22 μs |
| SIMD.jl built-in sin/cos | ~127 μs |
| Base.sincos | ~370 μs |
