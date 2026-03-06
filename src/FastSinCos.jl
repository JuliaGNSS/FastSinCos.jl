module FastSinCos

using SIMD

export fast_sincos_u3500, fast_sincos_u35

include("u3500.jl")
include("u35.jl")

end
