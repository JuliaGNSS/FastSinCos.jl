# Ultra-fast ~100,000 ULP sincos for Float32
# Single Cody-Waite range reduction to [-π/4, π/4], then 2-coeff sin + 2-coeff cos
# polynomials with quadrant-based swap. Fastest variant, suitable when ~3e-4 error
# is acceptable.

"""
    fast_sincos_u100k(d::SIMD.Vec{N,Float32}) -> (sin, cos)

Compute sin and cos simultaneously using an ultra-fast ~100,000 ULP approximation.

Uses a single Cody-Waite range reduction (π/2 split into high + low parts)
to `[-π/4, π/4]`, then evaluates 2-coefficient sin and 2-coefficient cos
polynomials, swapping based on quadrant.

# Input range
Cody-Waite range reduction keeps error flat across the full valid range.
The hard limit is `|d| < (π/2) × 2³¹ ≈ 3.4 × 10⁹` (Int32 overflow in the
quadrant index).

# Accuracy
~100,000 ULP (~3e-4 max absolute error) across all input ranges. ~27% faster than
[`fast_sincos_u35`](@ref) due to minimal polynomial terms (2+2 vs 3+5 coefficients).
"""
@inline function fast_sincos_u100k(d::SIMD.Vec{N,Float32}) where N
    ONE_I = SIMD.Vec{N,Int32}(Int32(1))
    TWO_I = SIMD.Vec{N,Int32}(Int32(2))
    ZERO_I = SIMD.Vec{N,Int32}(Int32(0))
    ONE_F = SIMD.Vec{N,Float32}(1.0f0)

    # Single range reduction: q = round(d * 2/π), s = d - q * π/2
    q_f = round(d * Float32(2/π))
    q = convert(SIMD.Vec{N,Int32}, q_f)

    # Cody-Waite range reduction (split π/2 into high + low parts)
    s = muladd(q_f, SIMD.Vec{N,Float32}(-1.5707963705062866f0), d)
    s = muladd(q_f, SIMD.Vec{N,Float32}(Float32(4.371139000186241e-8)), s)

    t = s
    s2 = s * s

    # Sin polynomial (2 coefficients): sin(s) ≈ s + s³·(C1·s² + C2)
    u_sin = muladd(SIMD.Vec{N,Float32}(Float32(0.8323502727e-2)), s2,
                   SIMD.Vec{N,Float32}(Float32(-0.1666651368e+0)))
    rx = muladd(u_sin * s2, t, t)

    # Cos polynomial (2 coefficients): cos(s) ≈ 1 + s²·(D1·s² + D2)
    u_cos = muladd(SIMD.Vec{N,Float32}(0.0416666641831398010253906f0), s2,
                   SIMD.Vec{N,Float32}(-0.5f0))
    ry = muladd(u_cos, s2, ONE_F)

    # Swap sin/cos based on quadrant: if q is odd, swap
    q_odd = (q & ONE_I) != ZERO_I
    sin_result = vifelse(q_odd, ry, rx)
    cos_result = vifelse(q_odd, rx, ry)

    # Sign correction
    sin_flip = (q & TWO_I) != ZERO_I
    sin_result = vifelse(sin_flip, -sin_result, sin_result)
    cos_flip = ((q + ONE_I) & TWO_I) != ZERO_I
    cos_result = vifelse(cos_flip, -cos_result, cos_result)

    return sin_result, cos_result
end
