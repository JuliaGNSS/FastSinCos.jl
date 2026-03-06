# FastSinCos.jl

[![CI](https://github.com/JuliaGNSS/FastSinCos.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGNSS/FastSinCos.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JuliaGNSS/FastSinCos.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNSS/FastSinCos.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNSS.github.io/FastSinCos.jl/dev)

Fast SIMD sincos approximations for Float32 using [SIMD.jl](https://github.com/eschnett/SIMD.jl). No deprecated dependencies.

```julia
using FastSinCos, SIMD

v = SIMD.Vec{8,Float32}((0.1f0, 0.5f0, 1.0f0, 1.5f0, 2.0f0, 2.5f0, 3.0f0, 3.5f0))
s, c = fast_sincos_u35(v)
```

See the [documentation](https://JuliaGNSS.github.io/FastSinCos.jl/dev) for details on accuracy, input ranges, and performance.
