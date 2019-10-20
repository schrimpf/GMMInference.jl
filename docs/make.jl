using Documenter, GMMInference, DocumenterMarkdown

runweave=true
runnotebook=false
if runweave
  using Weave
  wd = pwd()
  try
    builddir=joinpath(dirname(Base.pathof(GMMInference)),"..","docs","build")
    mkpath(builddir)
    cd(builddir)
    jmdfiles = filter(x->occursin(".jmd",x), readdir(joinpath("..","jmd")))
    for f in jmdfiles 
      weave(joinpath("..","jmd",f),out_path=joinpath("..","build"),
            cache=:user,
            cache_path=joinpath("..","weavecache"),
            doctype="github", mod=Main,
            args=Dict("md" => true))
      if (runnotebook)
        notebook(joinpath("..","jmd",f),out_path=joinpath("..","build"),
                 nbconvert_options="--allow-errors")
      end
    end
  finally 
    cd(wd)
  end
  if (isfile("build/temp.md"))
    rm("build/temp.md")
  end
end

makedocs(
  modules=[GMMInference],
  format=Markdown(),
  clean=false,
  pages=[
    "Home" => "index.md", # this won't get used anyway; we use mkdocs instead for interoperability with weave's markdown output.
  ],
  repo="https://github.com/schrimpf/GMMInference.jl/blob/{commit}{path}#L{line}",
  sitename="GMMInference.jl",
  authors="Paul Schrimpf <paul.schrimpf@gmail.com>",
)

run(`mkdocs build`)

#deploydocs(;
#    repo="github.com/schrimpf/GMMInference.jl",
#)

deploy=true
if deploy || "deploy" in ARGS
  run(`mkdocs gh-deploy`)
end
