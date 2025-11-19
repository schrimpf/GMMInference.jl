using Documenter, GMMInference, DocumenterCitations
bib = CitationBibliography(joinpath(@__DIR__,"ee.bib"), style=:authoryear)

makedocs(
    modules=[GMMInference],
    clean=true,
    pages=[
        "Extremum Estimation" => "extremumEstimation.md",
        "Identification Robust Inference" => "identificationRobustInference.md",
        "Bootstrap" => "bootstrap.md",
        "Empirical Likelihood" => "empiricalLikelihood.md",
        "Autodocs" => "index.md",
        "References" => "references.md",
        "License" => "license.md"
    ],
    repo=Remotes.GitHub("schrimpf","GMMInference.jl"), #"https://github.com/schrimpf/GMMInference.jl/blob/{commit}{path}#L{line}",
    sitename="GMMInference.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="master",
        assets=String[],
    ),

    authors="Paul Schrimpf <paul.schrimpf@ubc.ca>",
    draft=false,
    plugins=[bib]
)


include("faketravis.jl")

deploydocs(deps = nothing, make = nothing,
  repo="github.com/schrimpf/GMMInference.jl.git",
  target = "build",
  branch = "gh-pages",
  devbranch = "master",
)
