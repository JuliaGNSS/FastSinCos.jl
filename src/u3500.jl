# Fast ~3500 ULP sincos for Float32
# Single Cody-Waite range reduction to [-π/4, π/4], then 3-coeff sin + 3-coeff cos
# polynomials with quadrant-based swap. Inspired by SLEEF u3500 coefficients.

"""
    fast_sincos_u3500(d::SIMD.Vec{N,Float32}) -> (sin, cos)

Compute sin and cos simultaneously using a fast ~3500 ULP approximation.

Uses a single Cody-Waite range reduction (π/2 split into high + low parts)
to `[-π/4, π/4]`, then evaluates 3-coefficient sin and 3-coefficient cos
polynomials, swapping based on quadrant.

# Input range
Cody-Waite range reduction keeps error flat at ~3.6e-6 across the full valid range.
The hard limit is `|d| < (π/2) × 2³¹ ≈ 3.4 × 10⁹` (Int32 overflow in the
quadrant index).

# Accuracy
~3500 ULP (~3.6e-6 max absolute error) across all input ranges. ~15% faster than
[`fast_sincos_u35`](@ref) due to fewer polynomial terms (3+3 vs 3+5 coefficients).
"""
@inline function fast_sincos_u3500(d::SIMD.Vec{N,Float32}) where N
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

    # Sin polynomial (3 coefficients): sin(s) ≈ s + s³·(C1·s⁴ + C2·s² + C3)
    u_sin = muladd(SIMD.Vec{N,Float32}(Float32(-0.1881748176e-3)), s2,
                   SIMD.Vec{N,Float32}(Float32(0.8323502727e-2)))
    u_sin = muladd(u_sin, s2, SIMD.Vec{N,Float32}(Float32(-0.1666651368e+0)))
    rx = muladd(u_sin * s2, t, t)

    # Cos polynomial (3 coefficients): cos(s) ≈ 1 + s²·(D1·s⁴ + D2·s² + D3)
    u_cos = muladd(SIMD.Vec{N,Float32}(-0.00138888787478208541870117f0), s2,
                   SIMD.Vec{N,Float32}(0.0416666641831398010253906f0))
    u_cos = muladd(u_cos, s2, SIMD.Vec{N,Float32}(-0.5f0))
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
