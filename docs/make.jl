using Documenter, GMMInference

makedocs(;
    modules=[GMMInference],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/schrimpf/GMMInference.jl/blob/{commit}{path}#L{line}",
    sitename="GMMInference.jl",
    authors="Paul Schrimpf <paul.schrimpf@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/schrimpf/GMMInference.jl",
)
