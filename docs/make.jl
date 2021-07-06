using FairnessInprocessing
using Documenter

DocMeta.setdocmeta!(FairnessInprocessing, :DocTestSetup, :(using FairnessInprocessing); recursive=true)

makedocs(;
    modules=[FairnessInprocessing],
    authors="Sumantrak Mukherjee <sumantrak.mukherjee@gmail.com> and contributors",
    repo="https://github.com/samurai-kant/FairnessInprocessing.jl/blob/{commit}{path}#{line}",
    sitename="FairnessInprocessing.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://samurai-kant.github.io/FairnessInprocessing.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/samurai-kant/FairnessInprocessing.jl",
)
