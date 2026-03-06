# Port of SLEEFPirates sincos_fast (~35 ULP accuracy for Float32)
# Single range reduction q = round(d * 2/π), separate sin/cos polynomials,
# swap based on quadrant.

const TWO_OVER_PI_F = Float32(2/π)

"""
    fast_sincos_u35(d::SIMD.Vec{N,Float32}) -> (sin, cos)

Compute sin and cos simultaneously using a fast ~35 ULP approximation.
Port of SLEEFPirates' `sincos_fast` for SIMD.jl Vec types.

Uses a single Cody-Waite range reduction (π/2 split into high + low parts)
to `[-π/4, π/4]`, then evaluates separate sin (3-coeff) and cos (5-coeff)
polynomials, swapping based on quadrant.

# Input range
The Cody-Waite range reduction preserves precision for larger inputs than
[`fast_sincos_u3500`](@ref). Accurate for `|d| ≲ 1e5`. The hard limit is
`|d| < (π/2) × 2³¹ ≈ 3.4 × 10⁹` (Int32 overflow in the quadrant index).

# Accuracy
~35 ULP across the valid input range. More accurate than
[`fast_sincos_u3500`](@ref) (~3500 ULP) but slightly slower due to more
polynomial terms (3+5 vs 3 coefficients).
"""
@inline function fast_sincos_u35(d::SIMD.Vec{N,Float32}) where N
    ONE_I = SIMD.Vec{N,Int32}(Int32(1))
    TWO_I = SIMD.Vec{N,Int32}(Int32(2))
    ZERO_I = SIMD.Vec{N,Int32}(Int32(0))
    ONE_F = SIMD.Vec{N,Float32}(1.0f0)

    # Range reduction: q = round(d * 2/π), s = d - q * π/2
    q_f = round(d * TWO_OVER_PI_F)
    q = convert(SIMD.Vec{N,Int32}, q_f)

    # Cody-Waite range reduction (split π/2 into high + low parts)
    s = muladd(q_f, SIMD.Vec{N,Float32}(-1.5707963705062866f0), d)
    s = muladd(q_f, SIMD.Vec{N,Float32}(Float32(4.371139000186241e-8)), s)

    t = s
    s2 = s * s

    # Sin polynomial: sin(s) ≈ s + s³·P(s²)
    u_sin = SIMD.Vec{N,Float32}(-0.000195169282960705459117889f0)
    u_sin = muladd(u_sin, s2, SIMD.Vec{N,Float32}(0.00833215750753879547119141f0))
    u_sin = muladd(u_sin, s2, SIMD.Vec{N,Float32}(-0.166666537523269653320312f0))
    rx = muladd(u_sin * s2, t, t)

    # Cos polynomial: cos(s) ≈ 1 + s²·Q(s²)
    u_cos = SIMD.Vec{N,Float32}(Float32(-2.71811842367242206819355e-07))
    u_cos = muladd(u_cos, s2, SIMD.Vec{N,Float32}(Float32(2.47990446951007470488548e-05)))
    u_cos = muladd(u_cos, s2, SIMD.Vec{N,Float32}(-0.00138888787478208541870117f0))
    u_cos = muladd(u_cos, s2, SIMD.Vec{N,Float32}(0.0416666641831398010253906f0))
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
