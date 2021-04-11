using BipolarSphericalHarmonics
using Documenter

DocMeta.setdocmeta!(BipolarSphericalHarmonics, :DocTestSetup, :(using BipolarSphericalHarmonics); recursive=true)

makedocs(;
    modules=[BipolarSphericalHarmonics],
    authors="jishnub <jishnub@users.noreply.github.com> and contributors",
    repo="https://github.com/jishnub/BipolarSphericalHarmonics.jl/blob/{commit}{path}#{line}",
    sitename="BipolarSphericalHarmonics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jishnub.github.io/BipolarSphericalHarmonics.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jishnub/BipolarSphericalHarmonics.jl",
)
