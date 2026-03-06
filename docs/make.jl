using Documenter
using Documenter: Remotes
using FastSinCos

DocMeta.setdocmeta!(FastSinCos, :DocTestSetup, :(using FastSinCos); recursive=true)

makedocs(;
    modules=[FastSinCos],
    authors="Soeren Schoenbrod",
    repo=Remotes.GitHub("JuliaGNSS", "FastSinCos.jl"),
    sitename="FastSinCos.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNSS.github.io/FastSinCos.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNSS/FastSinCos.jl",
    devbranch="main",
)
