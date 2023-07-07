using FastMatrixMultiplication
using Documenter

DocMeta.setdocmeta!(FastMatrixMultiplication, :DocTestSetup, :(using FastMatrixMultiplication); recursive=true)

makedocs(;
    modules=[FastMatrixMultiplication],
    authors="Ian McInerney <i.mcinerney17@imperial.ac.uk> and contributors",
    repo="https://github.com/imciner2/FastMatrixMultiplication.jl/blob/{commit}{path}#{line}",
    sitename="FastMatrixMultiplication.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://imciner2.github.io/FastMatrixMultiplication.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/imciner2/FastMatrixMultiplication.jl",
    devbranch="main",
)
