# Port of SLEEF Sleef_fastsinf_u3500 / Sleef_fastcosf_u3500
# Simple 3-coefficient polynomial, ~3500 ULP accuracy for Float32
# Uses cos(x) = sin(x + π/2) to share the same polynomial for both

const M_1_PI_F = Float32(1/π)
const NEG_PI_F = Float32(-π)
const HALF_PI_F = Float32(π/2)

"""
    fast_sincos_u3500(d::SIMD.Vec{N,Float32}) -> (sin, cos)

Compute sin and cos simultaneously using a fast ~3500 ULP approximation.
Port of SLEEF's `Sleef_fastsinf_u3500` / `Sleef_fastcosf_u3500`.

Both sin and cos are evaluated using the same 3-coefficient polynomial
with separate range reductions (`cos(x) = sin(x + π/2)`).

# Input range
The range reduction uses a single Float32 constant for π, so precision degrades
for large inputs due to catastrophic cancellation. Best accuracy for `|d| ≲ 100`.
For larger inputs, use [`fast_sincos_u35`](@ref) which uses Cody-Waite range
reduction for better precision.

The hard limit is `|d| < π × 2³¹ ≈ 6.7 × 10⁹` (Int32 overflow in the
quadrant index).

# Accuracy
~3500 ULP for small inputs. Faster than [`fast_sincos_u35`](@ref) due to fewer
polynomial terms (3 vs 3+5 coefficients).
"""
@inline function fast_sincos_u3500(d::SIMD.Vec{N,Float32}) where N
    ONE_I = SIMD.Vec{N,Int32}(Int32(1))
    ZERO_I = SIMD.Vec{N,Int32}(Int32(0))

    # sin: q_s = round(d/π), d_s = d - q_s*π
    s_s = d * M_1_PI_F
    u_s = round(s_s)
    q_s = convert(SIMD.Vec{N,Int32}, u_s)
    d_s = muladd(u_s, NEG_PI_F, d)
    s2_s = d_s * d_s

    # cos: q_c = round((d - π/2)/π), d_c = d - π/2 - q_c*π
    s_c = muladd(d, M_1_PI_F, Float32(-0.5))
    u_c = round(s_c)
    q_c = convert(SIMD.Vec{N,Int32}, u_c)
    d_c = muladd(u_c, NEG_PI_F, d - HALF_PI_F)
    s2_c = d_c * d_c

    # Shared polynomial coefficients: sin(x) ≈ x + x³·(C1·x⁴ + C2·x² + C3)
    C1 = SIMD.Vec{N,Float32}(Float32(-0.1881748176e-3))
    C2 = SIMD.Vec{N,Float32}(Float32(0.8323502727e-2))
    C3 = SIMD.Vec{N,Float32}(Float32(-0.1666651368e+0))

    u_s2 = muladd(C1, s2_s, C2)
    u_s2 = muladd(u_s2, s2_s, C3)
    sin_result = muladd(s2_s * d_s, u_s2, d_s)

    u_c2 = muladd(C1, s2_c, C2)
    u_c2 = muladd(u_c2, s2_c, C3)
    cos_result = muladd(s2_c * d_c, u_c2, d_c)

    # Sign flips based on quadrant
    sin_result = vifelse((q_s & ONE_I) == ONE_I, -sin_result, sin_result)
    cos_result = vifelse((q_c & ONE_I) == ZERO_I, -cos_result, cos_result)

    return sin_result, cos_result
end
