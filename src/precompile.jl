for SHT in [SH, GSH]
    precompile(cache, (SHT, Type{Float64}, Int))
    precompile(biposh_flippoints, (SHT, Float64, Float64, Float64, Float64, Int, Int, Int, Int))
    precompile(biposh_flippoints, (SHT, Float64, Float64, Float64, Float64, Colon, Colon, Int, Int))
    precompile(biposh_flippoints, (SHT, Float64, Float64, Float64, Float64, Int, Int, L2L1Triangle))
    precompile(biposh_flippoints, (SHT, Float64, Float64, Float64, Float64, Colon, Colon, L2L1Triangle))
end
for YT in [PB, Hansen, Irreducible], B in [Cartesian, Polar, SphericalCovariant, HelicityCovariant]
    precompile(cache, (VSH{YT,B}, Type{Float64}, Int))
    precompile(biposh_flippoints, (VSH{YT,B}, Float64, Float64, Float64, Float64, Int, Int, Int, Int))
    precompile(biposh_flippoints, (VSH{YT,B}, Float64, Float64, Float64, Float64, Colon, Colon, Int, Int))
    precompile(biposh_flippoints, (VSH{YT,B}, Float64, Float64, Float64, Float64, Int, Int, L2L1Triangle))
    precompile(biposh_flippoints, (VSH{YT,B}, Float64, Float64, Float64, Float64, Colon, Colon, L2L1Triangle))
end


