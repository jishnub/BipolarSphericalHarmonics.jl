using BipolarSphericalHarmonics
using Test
using BipolarSphericalHarmonics: kronindex, cache, monopolarharmonics, GSHPerm, VSHPerm

using LegendrePolynomials
using SphericalHarmonics
using VectorSphericalHarmonics
using VectorSphericalHarmonics: basisconversionmatrix
using SphericalHarmonicModes

using SphericalHarmonicArrays
using SphericalHarmonics: NorthPole, SouthPole
import WignerSymbols
using OffsetArrays
using LinearAlgebra
using StaticArrays
using Rotations
using WignerD

using Aqua
@testset "project quality" begin
    Aqua.test_all(BipolarSphericalHarmonics, ambiguities=(recursive=false))
end

isapproxdefault(x, y) = isapprox(x, y, atol = 1e-14, rtol = 1e-8)
isapproxdefault(x::Nothing, y::Nothing) = true

cosχ(θ1, ϕ1, θ2, ϕ2) = cos(θ1)cos(θ2) + sin(θ1)sin(θ2)cos(ϕ1-ϕ2)
∂ϕ₂cosχ(θ1, ϕ1, θ2, ϕ2) = sin(θ1)sin(θ2)sin(ϕ1-ϕ2)

@testset "Clebsch-Gordan" begin

    # Computes the Clebsch-Gordan coefficient C_{l₁m₁l₂m₂}^{lm} for all valid l
    function clebschgordan(l1::Integer, m1::Integer, l2::Integer, m::Integer)
        m₂ = m - m1
        w3j = zeros(Cdouble, l1 + l2 + 1)
        jmin, jmax = extrema(BipolarSphericalHarmonics.degreerange(l1, l2, m))
        C = zeros(jmin:jmax)
        BipolarSphericalHarmonics.clebschgordan!(C, w3j, l1, m1,l2, m)
        return C
    end

    @testset "allocating" begin
        CG = clebschgordan(1,1,1,0)

        @test CG[0] ≈ WignerSymbols.clebschgordan(1,1,1,-1,0,0) ≈ 1/√3
        @test CG[1] ≈ WignerSymbols.clebschgordan(1,1,1,-1,1,0) ≈ 1/√2
        @test CG[2] ≈ WignerSymbols.clebschgordan(1,1,1,-1,2,0) ≈ 1/√6

        CG = clebschgordan(1,-1,1,0)

        @test CG[0] ≈ WignerSymbols.clebschgordan(1,-1,1,1,0,0) ≈ 1/√3
        @test CG[1] ≈ WignerSymbols.clebschgordan(1,-1,1,1,1,0) ≈ -1/√2
        @test CG[2] ≈ WignerSymbols.clebschgordan(1,-1,1,1,2,0) ≈ 1/√6

        CG = clebschgordan(1,0,1,0)

        @test CG[0] ≈ WignerSymbols.clebschgordan(1,0,1,0,0,0) ≈ -1/√3
        @test CG[1] ≈ WignerSymbols.clebschgordan(1,0,1,0,1,0) ≈ 0
        @test CG[2] ≈ WignerSymbols.clebschgordan(1,0,1,0,2,0) ≈ √(2/3)

        for l1 in 0:10, l2 in 0:10
            j = l1 + l2
            m = j
            CG = clebschgordan(l1,l1,l2,m)
            @test CG[j] ≈ 1
        end
    end

    @testset "prime factorization" begin
        C1 = zeros(200)
        C2 = zero(C1)
        C3 = zero(C1)
        W1 = zero(C1)
        W2 = zero(C2)
        W3 = zero(C3)
        BipolarSphericalHarmonics.clebschgordan!(C1, W1, 50, 0, 50, 0, primefactorization_cutoff = 0)
        BipolarSphericalHarmonics.clebschgordan!(C2, W2, 50, 0, 50, 0, primefactorization_cutoff = 200)
        BipolarSphericalHarmonics.clebschgordan!(C3, W3, 50, 0, 50, 0, primefactorization_cutoff = 50)
        @test C1 ≈ C2 ≈ C3
        C1 .= 0
        C2 .= 0
        C3 .= 0
        BipolarSphericalHarmonics.clebschgordan!(C1, W1, 50, 50, 50, 50, primefactorization_cutoff = 0)
        BipolarSphericalHarmonics.clebschgordan!(C2, W2, 50, 50, 50, 50, primefactorization_cutoff = 200)
        BipolarSphericalHarmonics.clebschgordan!(C3, W3, 50, 50, 50, 50, primefactorization_cutoff = 50)
        @test C1 ≈ C2 ≈ C3
    end
end

function _biposh(m, l1, l2, θ1, ϕ1, θ2, ϕ2, B = cache(SH(), max(l1, l2)))
    jrange = BipolarSphericalHarmonics.degreerange(l1, l2, m)
    Y = zeros(ComplexF64, LM(jrange, m))
    computePlmcostheta!(B.S1, θ1, l1)
    Y1 = computeYlm!(B.S1, θ1, ϕ1, l1)
    computePlmcostheta!(B.S2, θ2, l2)
    Y2 = computeYlm!(B.S2, θ2, ϕ2, l2)
    for m1 in -l1:l1
        m2 = m - m1
        abs(m2) <= l2 || continue
        BipolarSphericalHarmonics.clebschgordan!(B.C, B.W3j, l1, m1, l2, m)
        C = BipolarSphericalHarmonics.C_view_j1j2m_j(B.C, l1, l2, m, jrange)
        Y1_l1m1 = Y1[(l1, m1)]
        Y2_l2m2 = Y2[(l2, m2)]
        for ind in eachindex(Y, C)
            Y[ind] += C[ind] * Y1_l1m1 * Y2_l2m2
        end
    end
    return Y
end

@testset "kronindex" begin
    @testset "GSH" begin
        r = -1:1
        for (ind, (i, j)) in enumerate(Base.product(r,r))
            @test kronindex(GSH(), j, i) == ind
        end

        @testset "values" begin
            θ1, ϕ1, θ2, ϕ2 = pi/2, 0, pi/2, pi/3
            j, m, j1, j2 = 0, 0, 2, 2
            B1 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
            BG = biposh(GSH(), θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
            @test BG[kronindex(GSH(), 0, 0)] ≈ B1 atol=1e-14 rtol=1e-8
        end
    end

    @testset "VSH" begin
        struct TupMerge{N}
           t :: NTuple{N,Int}
        end
        Base.Tuple(t::TupMerge) = t.t
        Base.:(*)(a::TupMerge, b::TupMerge) = TupMerge((a.t..., b.t...))
        S = [TupMerge((i,j)) for i in -1:1, j in -1:1];
        SS = kron(S, S);
        for (ind,t) in zip(CartesianIndices(SS), SS)
            ttup = Tuple(t)
            @test kronindex(VSH(PB(), HelicityCovariant()), ttup...) == ind
            @test kronindex(VSH(PB(), SphericalCovariant()), ttup...) == ind
            indt11, indt12, indt21, indt22 = ttup
            indt11s = indt11 + 2
            indt21s = indt21 + 2
            ttups = (indt11s, indt12, indt21s, indt22)
            @test kronindex(VSH(PB(), Polar()), ttups...) == ind
            @test kronindex(VSH(PB(), Cartesian()), ttups...) == ind
        end

        # For Hansen, H⁻¹ⱼₘ(̂n) = Yⱼₘ(̂n)̂n, therefore its radial component is Yⱼₘ(̂n)
        # The radial basis vector ̂n is the same as the HelicityCovariant basis vector χ₀
        # The Hansen vector H⁻¹ⱼₘ(̂n) is equivalent to the PB vector P⁰ⱼₘ(̂n)
        @testset "values" begin
            θ1, ϕ1, θ2, ϕ2 = pi/2, 0, pi/2, pi/3
            j, m, j1, j2 = 0, 0, 2, 2
            B1 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)

            BV = biposh(VSH(Hansen(), Polar()), θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
            @test BV[kronindex(VSH(Hansen(), Polar()), 1, -1, 1, -1)] ≈ B1 atol=1e-14 rtol=1e-8
            BV = biposh(VSH(Hansen(), HelicityCovariant()), θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
            @test BV[kronindex(VSH(Hansen(), HelicityCovariant()), 0, -1, 0, -1)] ≈ B1 atol=1e-14 rtol=1e-8

            BV = biposh(VSH(PB(), Polar()), θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
            @test BV[kronindex(VSH(PB(), Polar()), 1, 0, 1, 0)] ≈ B1 atol=1e-14 rtol=1e-8
            BV = biposh(VSH(PB(), HelicityCovariant()), θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
            @test BV[kronindex(VSH(PB(), HelicityCovariant()), 0, 0, 0, 0)] ≈ B1 atol=1e-14 rtol=1e-8
        end
    end

    @testset "broadcasting" begin
        @testset "GSH" begin
            for r1_st in -1:1, r1_end in -1:1
                r1 = r1_st:r1_end
                for i2 in -1:1
                    ki = kronindex.(GSH(), r1, i2)
                    @test ki == kronindex.(GSH(), collect(r1), collect(i2))
                    @test length(ki) == length(r1)
                    for (ind, i1) in enumerate(r1)
                        @test ki[ind] == kronindex(GSH(), i1, i2)
                    end
                end
            end
            for r2_st in -1:1, r2_end in -1:1
                r2 = r2_st:r2_end
                for i1 in -1:1
                    ki = kronindex.(GSH(), i1, r2)
                    @test ki == kronindex.(GSH(), collect(i1), collect(r2))
                    @test length(ki) == length(r2)
                    for (ind, i2) in enumerate(r2)
                        @test ki[ind] == kronindex(GSH(), i1, i2)
                    end
                end
            end
            for r2_st in -1:1, r2_end in -1:1
                r2 = r2_st:r2_end
                for r1_st in -1:1, r1_end in -1:1
                    r1 = r1_st:r1_end
                    if length(r1) != 1 && length(r2) != 1 && length(r1) != length(r2)
                        continue
                    end
                    @inferred Broadcast.broadcast(kronindex, Ref(GSH()), r1, r2)
                    ki = kronindex.(GSH(), r1, r2)
                    @test ki == kronindex.(GSH(), collect(r1), collect(r2))
                    els = ((x,y) -> (x,y)).(r1,r2)
                    for (ind, (i1, i2)) in enumerate(els)
                        @test ki[ind] == kronindex(GSH(), i1, i2)
                    end
                    if length(r2) == 1
                        @test length(ki) == length(r1)
                        for (ind, i1) in enumerate(r1)
                            @test ki[ind] == kronindex.(GSH(), i1, r2)[1]
                        end
                    end
                    if length(r1) == 1
                        @test length(ki) == length(r2)
                        for (ind, i2) in enumerate(r2)
                            @test ki[ind] == kronindex.(GSH(), r1, i2)[1]
                        end
                    end
                    if length(r1) == 1 && length(r2) == 1
                        @test length(ki) == 1
                        @test ki[1] == kronindex(GSH(), first(r1), first(r2))
                    end
                    if length(r1) == 0 || length(r2) == 0
                        @test length(ki) == 0
                    end
                end
            end
        end
    end
end

@testset "monopolarharmonics" begin
    lmax = 3
    @testset "SH" begin
        B = cache(SH(), lmax);
        YSH1 = computeYlm(pi/2, 0, lmax = lmax)
        YSH2 = computeYlm(pi/3, pi/4, lmax = lmax)
        BipolarSphericalHarmonics.monopolarharmonics!(B, pi/2, 0, pi/3, pi/4);
        Y1, Y2 = monopolarharmonics(B);
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y1)), Y1)
            @test isapproxdefault(Y, YSH1[(j,m)])
        end
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y2)), Y2)
            @test isapproxdefault(Y, YSH2[(j,m)])
        end
        BipolarSphericalHarmonics.monopolarharmonics!(B, pi/2, 0, pi/3, pi/4, ((1,2),(2,1)));
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y1)), Y1)
            @test isapproxdefault(Y, YSH1[(j,m)])
        end
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y2)), Y2)
            @test isapproxdefault(Y, YSH2[(j,m)])
        end
    end
    @testset "GSH" begin
        B = cache(GSH(), lmax);
        BipolarSphericalHarmonics.monopolarharmonics!(B, pi/2, 0, pi/3, pi/4);
        Y1, Y2 = monopolarharmonics(B);
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y1)), Y1)
            @test isapproxdefault(Y, genspharm(j, m, pi/2, 0))
        end
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y2)), Y2)
            @test isapproxdefault(Y, genspharm(j, m, pi/3, pi/4))
        end
        BipolarSphericalHarmonics.monopolarharmonics!(B, pi/2, 0, pi/3, pi/4, ((1,2),(2,1)));
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y1)), Y1)
            @test isapproxdefault(Y, genspharm(j, m, pi/2, 0))
        end
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y2)), Y2)
            @test isapproxdefault(Y, genspharm(j, m, pi/3, pi/4))
        end
    end
    @testset "VSH" begin
        B = cache(VSH(PB(), Polar()), lmax);
        BipolarSphericalHarmonics.monopolarharmonics!(B, pi/2, 0, pi/3, pi/4);
        Y1, Y2 = monopolarharmonics(B);
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y1)), Y1)
            @test isapproxdefault(Y, vshbasis(PB(), Polar(), j, m, pi/2, 0))
        end
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y2)), Y2)
            @test isapproxdefault(Y, vshbasis(PB(), Polar(), j, m, pi/3, pi/4))
        end
        BipolarSphericalHarmonics.monopolarharmonics!(B, pi/2, 0, pi/3, pi/4, ((1,2),(2,1)));
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y1)), Y1)
            @test isapproxdefault(Y, vshbasis(PB(), Polar(), j, m, pi/2, 0))
        end
        for ((j,m), Y) in zip(first(SphericalHarmonicArrays.modes(Y2)), Y2)
            @test isapproxdefault(Y, vshbasis(PB(), Polar(), j, m, pi/3, pi/4))
        end
    end
end

@testset "NorthPole" begin
    θ1 = NorthPole(); ϕ1 = 0;
    l1 = 1; l2 = 1;
    B = BipolarSphericalHarmonics.cache(BipolarSphericalHarmonics.SH(), max(l1,l2))
    Y12 = zeros(ComplexF64, l1 + l2 + 1);

    m = -2;
    jrange = BipolarSphericalHarmonics.degreerange(l1, l2, m)
    modes = LM(jrange, m)
    Y = SHArray((@view Y12[1:length(modes)]), modes)
    for θ2 in LinRange(0, pi, 10), ϕ2 in LinRange(0, 2pi, 10)
        BipolarSphericalHarmonics.monopolarharmonics!(B, θ1, ϕ1, θ2, ϕ2)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ1, ϕ1, θ2, ϕ2, :, m, l1, l2)
        @test all(iszero, Y)
    end

    m = -1;
    jrange = BipolarSphericalHarmonics.degreerange(l1, l2, m)
    modes = LM(jrange, m)
    Y = SHArray((@view Y12[1:length(modes)]), modes)
    for θ2 in LinRange(0, pi, 10), ϕ2 in LinRange(0, 2pi, 10)
        BipolarSphericalHarmonics.monopolarharmonics!(B, θ1, ϕ1, θ2, ϕ2)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ1, ϕ1, θ2, ϕ2, :, m, l1, l2)
        Y2 = _biposh(m, l1, l2, θ1, ϕ1, θ2, ϕ2, B)
        @test isapproxdefault(Y[(1,-1)], 3/8pi*sin(θ2)*cis(-ϕ2))
        @test isapproxdefault(Y[(2,-1)], 3/8pi*sin(θ2)*cis(-ϕ2))
        @test isapproxdefault(Y, Y2)

        BipolarSphericalHarmonics.monopolarharmonics!(B, θ2, ϕ2, θ1, ϕ1)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ2, ϕ2, θ1, ϕ1, :, m, l1, l2)
        Y2 = _biposh(m, l1, l2, θ2, ϕ2, θ1, ϕ1, B)
        @test isapproxdefault(Y, Y2)
    end

    m = 0;
    jrange = BipolarSphericalHarmonics.degreerange(l1, l2, m)
    modes = LM(jrange, m)
    Y = SHArray((@view Y12[1:length(modes)]), modes)
    for θ2 in LinRange(0, pi, 10), ϕ2 in LinRange(0, 2pi, 10)
        BipolarSphericalHarmonics.monopolarharmonics!(B, θ1, ϕ1, θ2, ϕ2)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ1, ϕ1, θ2, ϕ2, :, m, l1, l2)
        Y2 = _biposh(m, l1, l2, θ1, ϕ1, θ2, ϕ2, B)
        @test isapproxdefault(Y[(0,0)], -√3/4pi * cos(θ2))
        @test isapproxdefault(Y[(1,0)], 0)
        @test isapproxdefault(Y[(2,0)], √(3/2)/2pi * cos(θ2))
        @test isapproxdefault(Y, Y2)

        BipolarSphericalHarmonics.monopolarharmonics!(B, θ2, ϕ2, θ1, ϕ1)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ2, ϕ2, θ1, ϕ1, :, m, l1, l2)
        Y2 = _biposh(m, l1, l2, θ2, ϕ2, θ1, ϕ1, B)
        @test isapproxdefault(Y, Y2)
    end

    m = 1;
    jrange = BipolarSphericalHarmonics.degreerange(l1, l2, m)
    modes = LM(jrange, m)
    Y = SHArray((@view Y12[1:length(modes)]), modes)
    for θ2 in LinRange(0, pi, 10), ϕ2 in LinRange(0, 2pi, 10)
        BipolarSphericalHarmonics.monopolarharmonics!(B, θ1, ϕ1, θ2, ϕ2)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ1, ϕ1, θ2, ϕ2, :, m, l1, l2)
        Y2 = _biposh(m, l1, l2, θ1, ϕ1, θ2, ϕ2, B)
        @test isapproxdefault(Y[(1,1)], 3/8pi*sin(θ2)*cis(ϕ2))
        @test isapproxdefault(Y[(2,1)], -3/8pi*sin(θ2)*cis(ϕ2))
        @test isapproxdefault(Y, Y2)

        BipolarSphericalHarmonics.monopolarharmonics!(B, θ2, ϕ2, θ1, ϕ1)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ2, ϕ2, θ1, ϕ1, :, m, l1, l2)
        Y2 = _biposh(m, l1, l2, θ2, ϕ2, θ1, ϕ1, B)
        @test isapproxdefault(Y, Y2)
    end

    m = 2;
    jrange = BipolarSphericalHarmonics.degreerange(l1, l2, m)
    modes = LM(jrange, m)
    Y = SHArray((@view Y12[1:length(modes)]), modes)
    for θ2 in LinRange(0, pi, 10), ϕ2 in LinRange(0, 2pi, 10)
        BipolarSphericalHarmonics.monopolarharmonics!(B, θ1, ϕ1, θ2, ϕ2)
        BipolarSphericalHarmonics.biposh!(Y12, B, θ1, ϕ1, θ2, ϕ2, :, m, l1, l2)
        @test all(iszero, Y)
    end

    # check NorthPole and 0
    lmax = 5
    B = BipolarSphericalHarmonics.cache(BipolarSphericalHarmonics.SH(), lmax)
    for l1 in 0:lmax, l2 in 0:lmax
        BipolarSphericalHarmonics.monopolarharmonics!(B, 0, 0, 0, 0)
        B1 = biposh(B, 0, 0, 0, 0, :, :, l1, l2)
        BipolarSphericalHarmonics.monopolarharmonics!(B, NorthPole(), 0, 0, 0)
        B2 = biposh(B, NorthPole(), 0, 0, 0, :, :, l1, l2)
        BipolarSphericalHarmonics.monopolarharmonics!(B, 0, 0, NorthPole(), 0)
        B3 = biposh(B, 0, 0, NorthPole(), 0, :, :, l1, l2)
        BipolarSphericalHarmonics.monopolarharmonics!(B, NorthPole(), 0, NorthPole(), 0)
        B4 = biposh(B, NorthPole(), 0, NorthPole(), 0, :, :, l1, l2)
        @test isapproxdefault(B1, B2)
        @test isapproxdefault(B1, B3)
        @test isapproxdefault(B1, B4)
    end
end

@testset "l1l2modes" begin
    l1range = 1:2; l2range = 1:2
    θ1, ϕ1 = pi/3, pi/4
    θ2, ϕ2 = pi/2, pi/3
    for l1l2modes = Any[Iterators.product(l1range, l2range),
            L2L1Triangle(0:1, 1, 0:1), L1L2Triangle(0:1, 1, 0:1)]
        Y = biposh(SH(), θ1, ϕ1, θ2, ϕ2, :, :, l1l2modes)
        for (ind, (l1, l2)) in enumerate(l1l2modes)
            Y2 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, :, :, l1, l2)
            @test isapproxdefault(Y[ind], Y2)
        end
        Y = biposh(SH(), θ1, ϕ1, θ2, ϕ2, :, 0, l1l2modes)
        for (ind, (l1, l2)) in enumerate(l1l2modes)
            Y2 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, :, 0, l1, l2)
            @test isapproxdefault(Y[ind], Y2)
        end
    end

    l1range = 2:2; l2range = 2:2
    jrange = 0:4
    θ1, ϕ1 = pi/3, pi/4
    θ2, ϕ2 = pi/2, pi/3
    l1l2modes = Iterators.product(l1range, l2range)
    Y = biposh(SH(), θ1, ϕ1, θ2, ϕ2, jrange, :, l1l2modes)
    for (ind, (l1, l2)) in enumerate(l1l2modes)
        Y2 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, jrange, :, l1, l2)
        @test isapproxdefault(Y[ind], Y2)
    end
    Y = biposh(SH(), θ1, ϕ1, θ2, ϕ2, jrange, 0, l1l2modes)
    for (ind, (l1, l2)) in enumerate(l1l2modes)
        Y2 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, jrange, 0, l1, l2)
        @test isapproxdefault(Y[ind], Y2)
    end

    jₒjₛ_allmodes = L2L1Triangle(1:50, 2, 1:52)
    Y12 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, 2, 0, jₒjₛ_allmodes)
    for (ind,(j2,j1)) in enumerate(jₒjₛ_allmodes)
        @test Y12[ind] ≈ biposh(SH(), θ1, ϕ1, θ2, ϕ2, 2, 0, j2, j1)
    end

    jₒjₛ_allmodes = L1L2Triangle(1:50, 2, 1:52)
    Y12 = biposh(SH(), θ1, ϕ1, θ2, ϕ2, 2, 0, jₒjₛ_allmodes)
    for (ind,(j1,j2)) in enumerate(jₒjₛ_allmodes)
        @test Y12[ind] ≈ biposh(SH(), θ1, ϕ1, θ2, ϕ2, 2, 0, j1, j2)
    end
end

@testset "Y^{11}_{00}" begin
    θ1, ϕ1 = pi*rand(), 2pi*rand()
    θ2, ϕ2 = pi*rand(), 2pi*rand()

    @testset "SH" begin
        B = biposh(SH(), θ1, ϕ1, θ2, ϕ2, 0, 0, 1, 1)
        @test B ≈ -√3/4π * cosχ(θ1, ϕ1, θ2, ϕ2)
    end

    @testset "GSH" begin
        B = biposh(GSH(), θ1, ϕ1, θ2, ϕ2, 0, 0, 1, 1)
        @test B[kronindex(GSH(), 0, 0)] ≈ -√3/4π * cosχ(θ1, ϕ1, θ2, ϕ2)
        for α1 = -1:1, α2 = -1:1
            @test isapprox(B[kronindex(GSH(), -α1, -α2)], (-1)^(α1 + α2) * conj(B[kronindex(GSH(), α1, α2)]),
                atol = 1e-14, rtol = 1e-8)
        end

        # We check the values at the Equator
        B = biposh(GSH(), pi/2, pi/4, pi/2, pi/2, 0, 0, 1, 1)
        B_exp = vec(transpose(
                [√3/8π*(-1 + cosh(im*(-pi/4)))    √(3/2)/4π*sinh(im*(-pi/4))     √3/8π*(1 + cosh(im*(-pi/4)))
                -√(3/2)/4π*sinh(im*(-pi/4))       -√3/4π*cosh(im*(-pi/4))       -√(3/2)/4π*sinh(im*(-pi/4))
                √3/8π*(1 + cosh(im*(-pi/4)))     √(3/2)/4π*sinh(im*(-pi/4))      √3/8π*(-1 + cosh(im*(-pi/4)))]))

        @test isapproxdefault(B, B_exp)
    end
end

@testset "Y^{ll}_{00}" begin
    θ1, ϕ1 = pi*rand(), 2pi*rand()
    θ2, ϕ2 = pi*rand(), 2pi*rand()

    lmax = 10
    P = collectPl(cosχ(θ1, ϕ1, θ2, ϕ2), lmax = lmax)
    Yll_00 = [P[l]*(-1)^l * √(2l+1)/4π for l in 1:lmax]
    @testset "SH" begin
        YB_00 = [biposh(SH(), θ1, ϕ1, θ2, ϕ2, 0, 0, l, l) for l in 1:lmax]
        @test Yll_00 ≈ YB_00
    end

    @testset "GSH" begin
        for l in 1:lmax
            B = biposh(GSH(), θ1, ϕ1, θ2, ϕ2, 0, 0, l, l)
            for α1 = -1:1, α2 = -1:1
                @test isapprox(B[kronindex(GSH(), -α1, -α2)], (-1)^(α1 + α2) * conj(B[kronindex(GSH(), α1, α2)]),
                    atol = 1e-14, rtol = 1e-8)
            end
            @test B[kronindex(GSH(), 0, 0)] ≈ Yll_00[l]
        end
    end
end

@testset "Y^{j1j1}_{10}" begin
    θ1, ϕ1 = pi*rand(), 2pi*rand()
    θ2, ϕ2 = pi*rand(), 2pi*rand()

    lmax = 10
    dP = collectdnPl(cosχ(θ1, ϕ1, θ2, ϕ2), lmax = lmax, n = 1)
    Yll_10 = [dP[l]*im*(-1)^l * √(3*(2l+1)/(l*(l+1)))/4π * ∂ϕ₂cosχ(θ1, ϕ1, θ2, ϕ2) for l in 1:lmax]

    @testset "SH" begin
        YB_10_n1n2 = zeros(ComplexF64, 1:lmax)
        YB_10_n2n1 = similar(YB_10_n1n2)
        for l in 1:lmax
            YB_10_n1n2[l] = biposh(SH(), θ1, ϕ1, θ2, ϕ2, 1, 0, l, l)
            YB_10_n2n1[l] = biposh(SH(), θ2, ϕ2, θ1, ϕ1, 1, 0, l, l)
        end

        @test Yll_10 ≈ YB_10_n1n2
        @test YB_10_n1n2 ≈ -YB_10_n2n1

        @test biposh(SH(), θ1, ϕ1, θ2, ϕ2, LM(1, 0), 1, 1)[1] == biposh(SH(), θ1, ϕ1, θ2, ϕ2, 1, 0, 1, 1)
    end

    @testset "GSH" begin
        for l in 1:lmax
            B = biposh(GSH(), θ1, ϕ1, θ2, ϕ2, 1, 0, l, l)
            for α1 = -1:1, α2 = -1:1
                @test isapprox(B[kronindex(GSH(), -α1, -α2)], (-1)^(α1 + α2 + 1) * conj(B[kronindex(GSH(), α1, α2)]),
                    atol = 1e-14, rtol = 1e-8)
            end
            @test B[kronindex(GSH(), 0, 0)] ≈ Yll_10[l]
        end

        # We check the values for specific j1
        B = biposh(GSH(), pi/2, pi/4, pi/2, pi/2, 1, 0, 1, 1)
        B_exp = vec(transpose(
                [3/(8*√2π)*sinh(im*(-pi/4))    3/8π*cosh(im*(-pi/4))     3/(8*√2π)*sinh(im*(-pi/4))
                -3/8π*cosh(im*(-pi/4))       -3/(4*√2π)*sinh(im*(-pi/4))      -3/8π*cosh(im*(-pi/4))
                3/(8*√2π)*sinh(im*(-pi/4))     3/8π*cosh(im*(-pi/4))      3/(8*√2π)*sinh(im*(-pi/4))]))

        @test isapproxdefault(B, B_exp)
    end
end

# test symmetries
@testset "Y^{j1j1}_{j0}" begin
    θ1, ϕ1 = pi*rand(), 2pi*rand()
    θ2, ϕ2 = pi*rand(), 2pi*rand()

    lmax = 5; jmax = 5
    @testset "SH" begin
        for j in 2:jmax, l in div(j,2)+1:lmax
            B = biposh(SH(), θ1, ϕ1, θ2, ϕ2, j, 0, l, l)
            if iseven(j)
                @test isapprox(imag(B), 0, atol = 1e-14)
            else
                @test isapprox(real(B), 0, atol = 1e-14)
            end
        end
    end
    @testset "GSH" begin
        for j in 2:jmax, l in div(j,2)+1:lmax
            B = biposh(GSH(), θ1, ϕ1, θ2, ϕ2, j, 0, l, l)
            for α1 = -1:1, α2 = -1:1
                @test isapprox(B[kronindex(GSH(), -α1, -α2)], (-1)^(α1 + α2 + j) * conj(B[kronindex(GSH(), α1, α2)]),
                    atol = 1e-14, rtol = 1e-8)
            end
            if iseven(j)
                @test isapprox(imag(B[kronindex(GSH(), 0, 0)]), 0, atol = 1e-14)
            else
                @test isapprox(real(B[kronindex(GSH(), 0, 0)]), 0, atol = 1e-14)
            end
        end

        # test values
        B = biposh(GSH(), pi/2, pi/4, pi/2, pi/2, 2, 0, 1, 1)
        B_exp = vec(transpose(
                [√(3/2)/8π*(2 + cosh(im*(-pi/4)))    √3/8π*sinh(im*(-pi/4))       √(3/2)/8π*(-2 + cosh(im*(-pi/4)))
                 -√3/8π*sinh(im*(-pi/4))            -√(3/2)/4π*cosh(im*(-pi/4))   -√3/8π*sinh(im*(-pi/4))
                 √(3/2)/8π*(-2 + cosh(im*(-pi/4)))   √3/8π*sinh(im*(-pi/4))       √(3/2)/8π*(2 + cosh(im*(-pi/4))) ]))

        @test isapproxdefault(B, B_exp)
    end
end

@testset "Y^{j1j2}_{jm}" begin
    θ1, ϕ1 = pi*rand(), 2pi*rand()
    θ2, ϕ2 = pi*rand(), 2pi*rand()

    lmax = 2;

    function testmultiplemodes(C, θ1, ϕ1, θ2, ϕ2, lmax)
        for l1 in 0:lmax, l2 in 0:lmax
            B = biposh(C, θ1, ϕ1, θ2, ϕ2, :, :, l1, l2)
            if l1 + l2 > 0
                Bm = biposh(C, θ1, ϕ1, θ2, ϕ2, :, 1:(l1+l2), l1, l2)
            end
            for j in abs(l1 - l2):l1 + l2
                Bj = biposh(C, θ1, ϕ1, θ2, ϕ2, j, :, l1, l2)
                Bjneg = biposh(C, θ1, ϕ1, θ2, ϕ2, j, -j:-min(1,j), l1, l2)
                Bjpos = biposh(C, θ1, ϕ1, θ2, ϕ2, j, min(1,j):j, l1, l2)
                for m in -j:j
                    Bjm = biposh(C, θ1, ϕ1, θ2, ϕ2, j, m, l1, l2)
                    @test B[(j,m)] ≈ Bjm
                    @test Bj[(j,m)] ≈ Bjm
                    B2jm = biposh(C, θ1, ϕ1, θ2, ϕ2, j, m, ((l1, l2),))
                    @test B2jm[1] ≈ Bjm

                    B3jm = biposh(C, θ1, ϕ1, θ2, ϕ2, LM(j, m), l1, l2)
                    @test B3jm[(j,m)] ≈ Bjm
                    B4jm = biposh(C, θ1, ϕ1, θ2, ϕ2, LM(j, m), ((l1, l2),))
                    @test B4jm[1][(j,m)] ≈ Bjm

                    if m < 0
                        @test isapproxdefault(Bjneg[(j,m)], Bjm)
                    elseif m > 0
                        @test isapproxdefault(Bjpos[(j,m)], Bjm)
                        @test isapproxdefault(Bm[(j,m)], Bjm)
                    end
                end
            end
        end
    end

    @testset "SH" begin
        C = monopolarharmonics(SH(), θ1, ϕ1, θ2, ϕ2, lmax);

        @test biposh(C, θ1, ϕ1, θ2, ϕ2, 3, 3, 1, 1) === nothing

        @testset "multiple vs one" begin
            testmultiplemodes(C, θ1, ϕ1, θ2, ϕ2, lmax)
        end
    end

    @testset "GSH" begin
        C = monopolarharmonics(GSH(), θ1, ϕ1, θ2, ϕ2, lmax);
        for l1 in 0:lmax, l2 in 0:lmax, j in abs(l1 - l2):l1 + l2, m in 0:j
            Bm = biposh(C, θ1, ϕ1, θ2, ϕ2, j, m, l1, l2)
            Bminm = biposh(C, θ1, ϕ1, θ2, ϕ2, j, -m, l1, l2)
            for α1 = -1:1, α2 = -1:1
                @test isapprox(Bm[kronindex(GSH(), -α1, -α2)], (-1)^(α1 + α2 + j + m + l1 + l2) * conj(Bminm[kronindex(GSH(), α1, α2)]),
                    atol = 1e-14, rtol = 1e-8)
            end
        end

        @test biposh(C, θ1, ϕ1, θ2, ϕ2, 3, 3, 1, 1) === nothing

        @testset "multiple vs one" begin
            testmultiplemodes(C, θ1, ϕ1, θ2, ϕ2, lmax)
        end
    end

    @testset "VSH" begin
        @testset "PB" begin
            CGSH = monopolarharmonics(GSH(), θ1, ϕ1, θ2, ϕ2, lmax);
            @testset "PB HelicityCovariant" begin
                CVSH = monopolarharmonics(VSH(PB(), HelicityCovariant()), θ1, ϕ1, θ2, ϕ2, lmax);
                for l1 in 0:lmax, l2 in 0:lmax
                    G = biposh(CGSH, θ1, ϕ1, θ2, ϕ2, :, :, l1, l2)
                    V = biposh(CVSH, θ1, ϕ1, θ2, ϕ2, :, :, l1, l2)
                    for j in abs(l1 - l2):l1 + l2, m in -j:j
                        @test diag(V[(j,m)]) ≈ G[(j,m)]
                    end
                end
            end
        end
        for YT in [PB(), Hansen(), Irreducible()], B in [SphericalCovariant(), HelicityCovariant(), Polar(), Cartesian()]
            @testset "$(typeof(YT)) $(typeof(B))" begin
                CVSH = monopolarharmonics(VSH(YT, B), θ1, ϕ1, θ2, ϕ2, lmax);
                testmultiplemodes(CVSH, θ1, ϕ1, θ2, ϕ2, lmax)
                @test biposh(CVSH, θ1, ϕ1, θ2, ϕ2, 3, 3, 1, 1) === nothing
            end
        end
    end
end

@testset "swap points" begin
    θ1, ϕ1 = pi*rand(), 2pi*rand()
    θ2, ϕ2 = pi*rand(), 2pi*rand()

    jmax = 2;

    @testset "biposh" begin
        function testswap(SHT, θ1, ϕ1, θ2, ϕ2, jmax, perm)
            C12 = monopolarharmonics(SHT, θ1, ϕ1, θ2, ϕ2, jmax);
            C21 = monopolarharmonics(SHT, θ2, ϕ2, θ1, ϕ1, jmax);
            testswap(C12, C21, θ1, ϕ1, θ2, ϕ2, jmax, perm)
        end

        function testswap(C12, C21, θ1, ϕ1, θ2, ϕ2, jmax, perm)
            for j1 in 0:jmax, j2 in 0:jmax
                for j in abs(j1 - j2):j1+j2, m in -j:j
                    B12 = biposh(C12, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2);
                    B21 = biposh(C21, θ2, ϕ2, θ1, ϕ1, j, m, j2, j1);
                    @test begin
                        res = isapproxdefault(B21, (-1)^(j1 + j2 - j) * B12[perm])
                        if !res
                            @show j1, j2, j, m
                        end
                        res
                    end
                end
            end
        end

        @testset "SH" begin
            tstart = time()
            testswap(SH(), θ1, ϕ1, θ2, ϕ2, jmax, 1)
            tend = time()
            @info "Finished testing SH() in $(round(tend - tstart, sigdigits = 1)) seconds"
        end

        @testset "GSH" begin
            tstart = time()
            testswap(GSH(), θ1, ϕ1, θ2, ϕ2, jmax, GSHPerm)
            tend = time()
            @info "Finished testing GSH() in $(round(tend - tstart, sigdigits = 1)) seconds"
        end

        @testset "VSH" begin
            for YT in [PB(), Hansen(), Irreducible()], B in [Polar(), Cartesian(), HelicityCovariant(), SphericalCovariant()]
                @testset "$(typeof(YT)) $(typeof(B))" begin
                    tstart = time()
                    testswap(VSH(YT, B), θ1, ϕ1, θ2, ϕ2, jmax, VSHPerm)
                    tend = time()
                    @info "Finished testing $YT $B in $(round(tend - tstart, sigdigits = 1)) seconds"
                end
            end
        end
    end

    @testset "biposh_flippoints" begin
        j1j2modes = L2L1Triangle(1:jmax, jmax, 1:jmax)

        function testswap(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes)
            B12 = monopolarharmonics(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes);
            B21 = monopolarharmonics(SHT, θ2, ϕ2, θ1, ϕ1, j1j2modes);
            Y12_alljm, Y21_alljm = biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, :, :, j1j2modes);
            for (indj1j2, (j1, j2)) in enumerate(j1j2modes)
                Y12_j1j2_alljm = biposh(B12, θ1, ϕ1, θ2, ϕ2, :, :, j1, j2)
                @test isapproxdefault(Y12_alljm[indj1j2], Y12_j1j2_alljm)
                Y21_j1j2_alljm = biposh(B21, θ2, ϕ2, θ1, ϕ1, :, :, j1, j2)
                @test isapproxdefault(Y21_alljm[indj1j2], Y21_j1j2_alljm)
            end
            j, m = 1, 1
            Y12_jm, Y21_jm = biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, j, m, j1j2modes);
            for (indj1j2, (j1, j2)) in enumerate(j1j2modes)
                Y12_j1j2_jm = biposh(B12, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
                if Y12_j1j2_jm === nothing
                    @test Y12_jm[indj1j2] === nothing
                else
                    @test isapproxdefault(Y12_jm[indj1j2], Y12_j1j2_jm)
                end
                Y21_j1j2_jm = biposh(B21, θ2, ϕ2, θ1, ϕ1, j, m, j1, j2)
                if Y21_j1j2_jm === nothing
                    @test Y21_jm[indj1j2] === nothing
                else
                    @test isapproxdefault(Y21_jm[indj1j2], Y21_j1j2_jm)
                end
            end

            j1, j2 = 2, 2
            Y12, Y21 = biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2);
            Y12_j1j2 = biposh(B12, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)
            Y21_j1j2 = biposh(B21, θ2, ϕ2, θ1, ϕ1, j, m, j1, j2)
            @test isapproxdefault(Y12, Y12_j1j2)
            @test isapproxdefault(Y21, Y21_j1j2)

            @test biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, 3, 3, 1, 1) === nothing
            @test biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, LM(3, 3), 1, 1) === nothing
            @test biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, 3, 3, L2L1Triangle(1:1, 0)) == ([nothing], [nothing])
            @test biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, LM(3, 3), L2L1Triangle(1:1, 0)) == ([nothing], [nothing])
            return nothing
        end

        for SHT in [SH(), GSH()]
            @testset "$(typeof(SHT))" begin
                tstart = time()
                testswap(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes)
                Y12_jmj1j2 = biposh_flippoints(SHT, θ1, ϕ1, θ2, ϕ2, 1, 1, 1, 1)[1]
                Y12_jmj1j2modes = biposh_flippoints(SHT, θ1, ϕ1, θ2, ϕ2, LM(1, 1), 1, 1)[1][1]
                @test isapproxdefault(Y12_jmj1j2, Y12_jmj1j2)
                tend = time()
                @info "Finished testing $SHT in $(round(tend - tstart, sigdigits = 1)) seconds"
            end
        end
        @testset "VSH" begin
            for YT in [PB(), Hansen(), Irreducible()], B in [Polar(), Cartesian(), HelicityCovariant(), SphericalCovariant()]
                @testset "$(typeof(YT)) $(typeof(B))" begin
                    tstart = time()
                    testswap(VSH(YT, B), θ1, ϕ1, θ2, ϕ2, j1j2modes)
                    tend = time()
                    @info "Finished testing $YT $B in $(round(tend - tstart, sigdigits = 1)) seconds"
                end
            end
        end
    end
end

@testset "Rotation of coordinates" begin
    # define some helper functions to rotate the points
    cartvec(θ, ϕ) = SVector{3}(sin(θ)cos(ϕ), sin(θ)sin(ϕ), cos(θ))
    function polcoords(v)
        # This is not unique at the poles, as (θ=0, ϕ) map to the north pole for all ϕ
        vunit = normalize(v)
        if vunit[3] == 1
            # north pole
            return SVector{2}(promote(oftype(float(vunit[3]), 0), 0))
        elseif vunit[3] == -1
            # south pole
            return SVector{2}(promote(oftype(float(vunit[3]), pi), 0))
        end
        θ = acos(vunit[3])
        ϕ = mod2pi(atan(vunit[2], vunit[1]))
        SVector{2}(promote(θ, ϕ))
    end
    for θ1 in LinRange(0, pi, 10), ϕ1 in LinRange(0, 2pi, 10)
        @test polcoords(cartvec(θ1, ϕ1)) ≈ SVector{2}(θ1, abs(cos(θ1)) == 1 ? 0 : ϕ1)
    end
    # A rotation that takes (θ1, ϕ1) to (θ1′, ϕ1′). The rotation is not unique
    rotn1n1′(θ1, ϕ1, θ1′, ϕ1′) = RotZYZ(ϕ1′, θ1′ - θ1, -ϕ1)
    for θ1 in LinRange(0, pi, 4), ϕ1 in LinRange(0, 2pi, 4), θ1′ in LinRange(0, pi, 4), ϕ1′ in LinRange(0, 2pi, 4)
        R = rotn1n1′(θ1, ϕ1, θ1′, ϕ1′)
        @test det(R) ≈ 1
        @test R'R ≈ I
        @test R * cartvec(θ1, ϕ1) ≈ cartvec(θ1′, ϕ1′)
    end
    # A rotation about the axis n1′ that takes n2′′ to n2′
    function rotn1′n2n2′(n1′, n2′′, n2′)
        n1′x, n1′y, n1′z = n1′
        a = n2′′ - n1′
        b = n2′ - n1′
        if (isapproxdefault(cross(n1′, a), zero(SVector{3})) || isapproxdefault(cross(n1′, b), zero(SVector{3})))
            return one(SMatrix{3,3,eltype(n1′)})
        end
        sinω = b ⋅ cross(n1′, a) / norm(cross(n1′, a))^2
        cosω = b ⋅ (a - (n1′⋅a)n1′) / norm(a - (n1′⋅a)n1′)^2
        n1′C = SMatrix{3,3}(0, n1′z, -n1′y, -n1′z, 0, n1′x, n1′y, -n1′x, 0)
        n1′n1′ = n1′ * n1′'
        R = cosω*I + sinω * n1′C + (1 - cosω)*n1′n1′
    end
    for Δθ in LinRange(0, pi, 4), ϕ1 in LinRange(0, 2pi, 4), ϕ2 in LinRange(0, 2pi, 4)
        n1′ = cartvec(0, 0)
        n2′′ = cartvec(Δθ, ϕ1)
        n2′ = cartvec(Δθ, ϕ2)
        R = rotn1′n2n2′(n1′, n2′′, n2′)
        @test det(R) ≈ 1
        @test R'R ≈ I
        @test R * n1′ ≈ n1′
        @test R * n2′′ ≈ n2′
    end
    # a rotation that takes (θ1, ϕ1) and (θ2, ϕ2) to (θ1′, ϕ1′) and (θ2′, ϕ2′) for pairs of points that are equispaced in angle
    function rotn1n2n1′n2′(θ1, ϕ1, θ2, ϕ2, θ1′, ϕ1′, θ2′, ϕ2′)
        local n1, n2
        n1 = cartvec(θ1, ϕ1)
        n2 = cartvec(θ2, ϕ2)
        n1′ = cartvec(θ1′, ϕ1′)
        n2′ = cartvec(θ2′, ϕ2′)
        n1 ⋅ n2 ≈ n1′ ⋅ n2′ || throw(ArgumentError("Points are not related by a rotation"))
        R11′ = rotn1n1′(θ1, ϕ1, θ1′, ϕ1′)
        n2′′ = R11′ * n2
        R2′′2′ = rotn1′n2n2′(n1′, n2′′, n2′)
        R = RotZYZ(R2′′2′ * R11′)
    end
    for θ1 in LinRange(0, pi, 4), θ2 in LinRange(0, pi, 4), ϕ1′ in LinRange(0, 2pi, 4)
        Δθ21 = θ2 - θ1
        ϕ1 = ϕ2 = 0.0
        n1 = cartvec(θ1, ϕ1)
        n2 = cartvec(θ2, ϕ2)

        θ1′ = θ2′ = pi/2
        Δϕ2′1′ = Δθ21
        ϕ2′ = ϕ1′ + Δϕ2′1′
        n1′ = cartvec(θ1′, ϕ1′)
        n2′ = cartvec(θ2′, ϕ2′)
        @test dot(n1, n2) ≈ dot(n1′, n2′)

        R = rotn1n2n1′n2′(θ1, ϕ1, θ2, ϕ2, θ1′, ϕ1′, θ2′, ϕ2′)
        @test det(R) ≈ 1
        @test R'R ≈ I
        @test R * n1 ≈ n1′
        @test R * n2 ≈ n2′

        # Test by rotating the points n2 and n2′
        # We rotate n2 about n1 and n2′ about n1′
        # This preserves the angular distance between the pairs
        for ω1 in LinRange(0, 2pi, 6), ω2 in LinRange(0, 2pi, 6)
            n2_rot = AngleAxis(ω1, n1...) * n2
            θ2_rot, ϕ2_rot = polcoords(n2_rot)
            @test dot(n1, n2_rot) ≈ dot(n1, n2)

            n2′_rot = AngleAxis(ω2, n1′...) * n2′
            @test dot(n1′, n2′_rot) ≈ dot(n1′, n2′)
            θ2′_rot, ϕ2′_rot = polcoords(n2′_rot)

            R = rotn1n2n1′n2′(θ1, ϕ1, θ2_rot, ϕ2_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot)
            @test det(R) ≈ 1
            @test R'R ≈ I
            @test R * n1 ≈ n1′
            @test R * n2_rot ≈ n2′_rot
        end
    end

    lmax = 5
    modes = ML(0:lmax)
    Dvec = OffsetArray([zeros(ComplexF64, 2l+1, 2l+1) for l in 0:2lmax], 0:2lmax);

    θ1, ϕ1 = pi/4, 0; n1 = cartvec(θ1, ϕ1);
    θ2, ϕ2 = pi/3, pi/3; n2 = cartvec(θ2, ϕ2);

    @testset "SH" begin
        B12 = cache(SH(), lmax);
        B1′2′ = cache(SH(), lmax);
        BipolarSphericalHarmonics.monopolarharmonics!(B12, θ1, ϕ1, θ2, ϕ2);
        for θ1′ in LinRange(0, pi, 6), ϕ1′ in LinRange(0, 2pi, 6)
            # start by choosing the rotation that maps n1 to n1′
            R11′ = rotn1n1′(θ1, ϕ1, θ1′, ϕ1′)
            n2′ = R11′ * cartvec(θ2, ϕ2)
            θ2′, ϕ2′ = polcoords(n2′)
            n1′ = cartvec(θ1′, ϕ1′)
            BipolarSphericalHarmonics.monopolarharmonics!(B1′2′, θ1′, ϕ1′, θ2′, ϕ2′);
            @test R11′ * n1 ≈ n1′
            @test R11′ * n2 ≈ n2′
            for j in 0:2lmax
                Dp = wignerD!(Dvec[j], j, ϕ1, θ1 - θ1′, -ϕ1′);
                D = OffsetArray(Dp, -j:j, -j:j)
                for l1 in 0:lmax, l2 in abs(j - l1):min(lmax, l1 + j)
                    Y  = biposh(B12, θ1, ϕ1, θ2, ϕ2, j, :, l1, l2)
                    Y′ = biposh(B1′2′, θ1′, ϕ1′, θ2′, ϕ2′, j, :, l1, l2)
                    Y_rot = parent([sum(D[m′, m] * Y[(j, m′)] for m′ in -j:j) for m in axes(D, 2)])
                    @test isapproxdefault(Y′, Y_rot)
                end
            end

            # next rotate (θ2′,ϕ2′) to a new position
            for ω2 in LinRange(0, 2pi, 4)[2:end]
                n2_rot = AngleAxis(ω2, n1...) * n2
                θ2_rot, ϕ2_rot = polcoords(n2_rot)
                @test dot(n1, n2_rot) ≈ dot(n1, n2)
                BipolarSphericalHarmonics.monopolarharmonics!(B12, θ1, ϕ1, θ2_rot, ϕ2_rot);

                for ω2′ in LinRange(0, 2pi, 4)[2:end]
                    n2′_rot = AngleAxis(ω2′, n1′...) * n2′
                    θ2′_rot, ϕ2′_rot = polcoords(n2′_rot)
                    @test dot(n1′, n2′_rot) ≈ dot(n1′, n2′) ≈ dot(n1, n2)

                    BipolarSphericalHarmonics.monopolarharmonics!(B1′2′, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot);
                    R = rotn1n2n1′n2′(θ1, ϕ1, θ2_rot, ϕ2_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot)

                    @test det(R) ≈ 1
                    @test R'R ≈ I
                    @test R * n1 ≈ n1′
                    @test R * n2_rot ≈ n2′_rot
                    # The passive rotation for the frame is the inverse of R11′
                    RSS′ = inv(R)
                    α, β, γ = RSS′.theta1, RSS′.theta2, RSS′.theta3
                    for j in 0:lmax
                        # The rotation that transforms n1 to n1′ is the inverse of the one that transforms the harmoncics
                        Dp = wignerD!(Dvec[j], j, α, β, γ);
                        D = OffsetArray(Dp, -j:j, -j:j)
                        for l1 in 0:lmax, l2 in abs(j - l1):min(lmax, l1 + j)
                            Y  = biposh(B12, θ1, ϕ1, θ2_rot, ϕ2_rot, j, :, l1, l2)
                            Y′ = biposh(B1′2′, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot, j, :, l1, l2)
                            Y_rot = parent([sum(D[m′, m] * Y[(j, m′)] for m′ in -j:j) for m in axes(D, 2)])
                            @test isapproxdefault(Y′, Y_rot)
                        end
                    end
                end
            end
        end
    end

    @testset "GSH" begin
        B12 = cache(GSH(), lmax);
        B1′2′ = cache(GSH(), lmax);
        B12_rot = cache(GSH(), lmax);
        B1′2′_rot = cache(GSH(), lmax);
        BipolarSphericalHarmonics.monopolarharmonics!(B12, θ1, ϕ1, θ2, ϕ2);
        Un1 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ1, ϕ1);
        Un2 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ2, ϕ2);
        x̂_n1 = Un1 * SVector{3}(1,0,0)
        x̂_n2 = Un2 * SVector{3}(1,0,0)
        x̂x̂_n1n2 = kron(x̂_n1, x̂_n2)

        for θ1′ in LinRange(0, pi, 6), ϕ1′ in LinRange(0, 2pi, 6)
            # start by choosing the rotation that maps n1 to n1′
            n1′ = cartvec(θ1′, ϕ1′)
            R11′ = rotn1n1′(θ1, ϕ1, θ1′, ϕ1′)
            n2′ = R11′ * n2
            θ2′, ϕ2′ = polcoords(n2′)
            @test dot(n1, n2) ≈ dot(n1′, n2′)
            @test R11′ * n1 ≈ n1′
            @test R11′ * n2 ≈ n2′
            BipolarSphericalHarmonics.monopolarharmonics!(B1′2′, θ1′, ϕ1′, θ2′, ϕ2′);
            # The passive rotation for the frame is the inverse of R11′
            RSS′ = inv(R11′)
            α, β, γ = RSS′.theta1, RSS′.theta2, RSS′.theta3
            U′n1 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ1′, ϕ1′);
            U′n2 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ2′, ϕ2′);
            M′1 = U′n1 * R11′ * Un1'
            M′2 = U′n2 * R11′ * Un2'
            M′ = kron(M′1, M′2)
            # For HelicityCovariant both M′1 and M′2, and consequently their kronecker product, are diagonal
            @test M′ ≈ Diagonal(M′)

            x̂x̂_n1′n2′ = kron(M′1 * x̂_n1, M′2 * x̂_n2)
            R⁻¹x̂R⁻¹x̂_n1n2 = kron(inv(M′1) * x̂_n1, inv(M′2) * x̂_n2)

            for j in 0:2lmax
                Dp = wignerD!(Dvec[j], j, α, β, γ);
                D = OffsetArray(Dp, -j:j, -j:j);
                for l1 in 0:lmax, l2 in abs(j - l1):min(lmax, l1 + j)
                    for m in -j:j
                        DY = sum(D[m′, m] * biposh(B12, θ1, ϕ1, θ2, ϕ2, j, m′, l1, l2) for m′ in -j:j)
                        Y_rot = M′ * DY
                        Y′ = biposh(B1′2′, θ1′, ϕ1′, θ2′, ϕ2′, j, m, l1, l2)
                        @test isapproxdefault(Y′, Y_rot)
                        # x′x′ : Bjm′(n1′,n2′) = M1xM2x : M1M2 ∑Dlmm′ Bjm′(n1,n2) = xx : ∑Dlmm′ Bjm′(n1,n2)
                        xxY12_rot = x̂x̂_n1n2' * DY
                        xxY1′2′ = x̂x̂_n1′n2′' * Y′
                        @test isapproxdefault(xxY1′2′, xxY12_rot)
                        # xx : Bjm′(n1′,n2′) = xx : M1M2 ∑Dlmm′ Bjm′(n1,n2) = M1⁻¹xM2⁻¹x : ∑Dlmm′ Bjm′(n1,n2)
                        xxY12_rot = R⁻¹x̂R⁻¹x̂_n1n2' * DY
                        xxY1′2′ = x̂x̂_n1n2' * Y′
                        @test isapproxdefault(xxY1′2′, xxY12_rot)
                    end
                end
            end

            # next rotate (θ2,ϕ2) and (θ2′,ϕ2′) to new positions
            for ω2 in LinRange(0, 2pi, 4)[2:end]
                R = AngleAxis(ω2, n1...)
                n2_rot = R * n2
                θ2_rot, ϕ2_rot = polcoords(n2_rot)
                @test dot(n1, n2_rot) ≈ dot(n1, n2)
                BipolarSphericalHarmonics.monopolarharmonics!(B12_rot, θ1, ϕ1, θ2_rot, ϕ2_rot);

                RSS′ = inv(R) # passive rotation that maps n2 to n2_rot
                Un2_rot = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ2_rot, ϕ2_rot);
                M22_rot = Un2_rot * RSS′' * Un2'

                x̂_n2_rot = M22_rot * SVector{3}(1,0,0)
                x̂x̂_n1n2_rot = kron(x̂_n1, x̂_n2_rot)

                for ω2′ in LinRange(0, 2pi, 4)[2:end]
                    n2′_rot = AngleAxis(ω2′, n1′...) * n2′
                    θ2′_rot, ϕ2′_rot = polcoords(n2′_rot)
                    @test dot(n1′, n2′_rot) ≈ dot(n1′, n2′) ≈ dot(n1, n2)
                    R = rotn1n2n1′n2′(θ1, ϕ1, θ2_rot, ϕ2_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot)

                    @test det(R) ≈ 1
                    @test R'R ≈ I
                    @test R * n1 ≈ n1′
                    @test R * n2_rot ≈ n2′_rot
                    RSS′ = inv(R)
                    α, β, γ = RSS′.theta1, RSS′.theta2, RSS′.theta3
                    U′n2_rot = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ2′_rot, ϕ2′_rot);
                    M′1 = U′n1 * R * Un1'
                    M′2 = U′n2_rot * R * Un2_rot'
                    M′ = kron(M′1, M′2)
                    @test M′ ≈ Diagonal(M′)

                    x̂_n1′ = M′1 * x̂_n1
                    x̂_n2′_rot = M′2 * x̂_n2_rot
                    x̂x̂_n1′n2′_rot = kron(x̂_n1′, x̂_n2′_rot)

                    BipolarSphericalHarmonics.monopolarharmonics!(B1′2′_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot);

                    for j in 0:lmax
                        Dp = wignerD!(Dvec[j], j, α, β, γ);
                        D = OffsetArray(Dp, -j:j, -j:j)
                        for l1 in 0:lmax, l2 in abs(j - l1):min(lmax, l1 + j), m in -j:j
                            DY = sum(D[m′, m] * biposh(B12_rot, θ1, ϕ1, θ2_rot, ϕ2_rot, j, m′, l1, l2) for m′ in -j:j)
                            RY12_rot = M′ * DY
                            Y1′2′_rot = biposh(B1′2′_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot, j, m, l1, l2)
                            @test isapproxdefault(Y1′2′_rot, RY12_rot)
                            # x′x′ : Bjm′(n1′,n2′) = M1xM2x : M1M2 ∑Dlmm′ Bjm′(n1,n2) = xx : ∑Dlmm′ Bjm′(n1,n2)
                            RxxY12_rot = x̂x̂_n1n2_rot' * DY
                            xxY1′2′_rot = x̂x̂_n1′n2′_rot' * Y1′2′_rot
                            @test isapproxdefault(xxY1′2′_rot, RxxY12_rot)
                        end
                    end
                end
            end
        end

        # Compare Equator with the prime meridian (as both are great circles)
        ϕ1_Mer = ϕ2_Mer = 0
        θ1_Equator = θ2_Equator = pi/2
        for θ1_Mer in LinRange(0, pi, 4), θ2_Mer in LinRange(0, pi, 4),
            Δθ = θ2_Mer - θ1_Mer
            BipolarSphericalHarmonics.monopolarharmonics!(B12, θ1_Mer, ϕ1_Mer, θ2_Mer, ϕ2_Mer);

            Un1 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ1_Mer, ϕ1_Mer);
            Un2 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ2_Mer, ϕ2_Mer);

            for ϕ1_Equator in LinRange(0, pi, 4)
                Δϕ = Δθ
                ϕ2_Equator = ϕ1_Equator + Δϕ
                U′n1 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ1_Equator, ϕ1_Equator);
                U′n2 = basisconversionmatrix(Cartesian(), HelicityCovariant(), θ2_Equator, ϕ2_Equator);
                BipolarSphericalHarmonics.monopolarharmonics!(B1′2′, θ1_Equator, ϕ1_Equator, θ2_Equator, ϕ2_Equator);

                R = rotn1n2n1′n2′(θ1_Mer, ϕ1_Mer, θ2_Mer, ϕ2_Mer, θ1_Equator, ϕ1_Equator, θ2_Equator, ϕ2_Equator)
                RSS′ = inv(R)
                α, β, γ = RSS′.theta1, RSS′.theta2, RSS′.theta3
                M′1 = U′n1 * RSS′' * Un1'
                M′2 = U′n2 * RSS′' * Un2'
                M′ = kron(M′1, M′2)
                @test M′ ≈ Diagonal(M′)

                for j in 0:2lmax
                    Dp = wignerD!(Dvec[j], j, α, β, γ);
                    D = OffsetArray(Dp, -j:j, -j:j);
                    for l1 in 0:lmax, l2 in abs(j - l1):min(lmax, l1 + j), m in -j:j
                        Y_rot = M′ * sum(D[m′, m] * biposh(B12, θ1_Mer, ϕ1_Mer, θ2_Mer, ϕ2_Mer, j, m′, l1, l2) for m′ in -j:j)
                        Y′ = biposh(B1′2′, θ1_Equator, ϕ1_Equator, θ2_Equator, ϕ2_Equator, j, m, l1, l2)
                        @test isapproxdefault(Y′, Y_rot)
                    end
                end
            end
        end
    end

    @testset "VSH" begin
        for YT in [Irreducible(), PB(), Hansen()], B in [Cartesian(), Polar(), SphericalCovariant(), HelicityCovariant()]
            tstart = time()
            B12 = cache(VSH(YT, B), lmax);
            B12_rot = cache(VSH(YT, B), lmax);
            B1′2′ = cache(VSH(YT, B), lmax);
            B1′2′_rot = cache(VSH(YT, B), lmax);
            Un1 = basisconversionmatrix(Cartesian(), B, θ1, ϕ1);
            Un2 = basisconversionmatrix(Cartesian(), B, θ2, ϕ2);
            BipolarSphericalHarmonics.monopolarharmonics!(B12, θ1, ϕ1, θ2, ϕ2);
            x̂_n1 = Un1 * SVector{3}(1,0,0)
            x̂_n2 = Un2 * SVector{3}(1,0,0)
            x̂x̂_n1n2 = kron(x̂_n1, x̂_n2)
            for θ1′ in LinRange(0, pi, 4), ϕ1′ in LinRange(0, 2pi, 5)
                # start by choosing the rotation that maps n1 to n1′
                n1′ = cartvec(θ1′, ϕ1′)
                R11′ = rotn1n1′(θ1, ϕ1, θ1′, ϕ1′)
                n2′ = R11′ * n2
                θ2′, ϕ2′ = polcoords(n2′)
                @test dot(n1, n2) ≈ dot(n1′, n2′)
                @test R11′ * n1 ≈ n1′
                @test R11′ * n2 ≈ n2′

                BipolarSphericalHarmonics.monopolarharmonics!(B1′2′, θ1′, ϕ1′, θ2′, ϕ2′);
                # The passive rotation for the frame is the inverse of R11′
                RSS′ = inv(R11′)
                α, β, γ = RSS′.theta1, RSS′.theta2, RSS′.theta3
                U′n1 = basisconversionmatrix(Cartesian(), B, θ1′, ϕ1′);
                U′n2 = basisconversionmatrix(Cartesian(), B, θ2′, ϕ2′);
                M′1 = U′n1 * R11′ * Un1'
                M′2 = U′n2 * R11′ * Un2'
                M′ = kron(M′1, M′2)

                x̂_n1′ = M′1 * x̂_n1
                x̂_n2′ = M′2 * x̂_n2
                x̂x̂_n1′n2′ = kron(x̂_n1′, x̂_n2′)
                R⁻¹x̂R⁻¹x̂_n1n2 = kron(inv(M′1) * x̂_n1, inv(M′2) * x̂_n2)

                for j in 0:lmax
                    Dp = wignerD!(Dvec[j], j, α, β, γ);
                    D = OffsetArray(Dp, -j:j, -j:j);
                    for l1 in 0:lmax, l2 in abs(j - l1):min(lmax, l1 + j)
                        for m in -j:j
                            DY = sum(D[m′, m] * biposh(B12, θ1, ϕ1, θ2, ϕ2, j, m′, l1, l2) for m′ in -j:j)
                            RY = M′ * DY
                            Y′ = biposh(B1′2′, θ1′, ϕ1′, θ2′, ϕ2′, j, m, l1, l2)
                            @test isapproxdefault(Y′, RY)

                            # x′x′ : Bjm′(n1′,n2′) = M1xM2x : M1M2 ∑Dlmm′ Bjm′(n1,n2) = xx : ∑Dlmm′ Bjm′(n1,n2)
                            RxxY12 = x̂x̂_n1n2' * DY
                            xxY1′2′ = x̂x̂_n1′n2′' * Y′
                            @test isapproxdefault(xxY1′2′, RxxY12)

                            # xx : Bjm′(n1′,n2′) = xx : M1M2 ∑Dlmm′ Bjm′(n1,n2) = M1⁻¹xM2⁻¹x : ∑Dlmm′ Bjm′(n1,n2)
                            xxY12_rot = R⁻¹x̂R⁻¹x̂_n1n2' * DY
                            xxY1′2′ = x̂x̂_n1n2' * Y′
                            @test isapproxdefault(xxY1′2′, xxY12_rot)
                        end
                    end
                end

                # next rotate (θ2,ϕ2) and (θ2′,ϕ2′) to new positions
                for ω2 in LinRange(0, pi, 4)[2:end]
                    R = AngleAxis(ω2, n1...)
                    n2_rot = R * n2
                    θ2_rot, ϕ2_rot = polcoords(n2_rot)
                    @test dot(n1, n2_rot) ≈ dot(n1, n2)
                    BipolarSphericalHarmonics.monopolarharmonics!(B12_rot, θ1, ϕ1, θ2_rot, ϕ2_rot);
                    Un2_rot = basisconversionmatrix(Cartesian(), B, θ2_rot, ϕ2_rot);

                    RSS′ = inv(R) # passive rotation that maps n2 to n2_rot
                    M22_rot = Un2_rot * R * Un2'

                    x̂_n2_rot = M22_rot * SVector{3}(1,0,0)
                    x̂x̂_n1n2_rot = kron(x̂_n1, x̂_n2_rot)

                    for ω2′ in LinRange(0, pi, 4)[2:end]
                        n2′_rot = AngleAxis(ω2′, n1′...) * n2′
                        θ2′_rot, ϕ2′_rot = polcoords(n2′_rot)
                        @test dot(n1′, n2′_rot) ≈ dot(n1′, n2′) ≈ dot(n1, n2)
                        R = rotn1n2n1′n2′(θ1, ϕ1, θ2_rot, ϕ2_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot)
                        @test det(R) ≈ 1
                        @test R'R ≈ I
                        @test R * n1 ≈ n1′
                        @test R * n2_rot ≈ n2′_rot

                        RSS′ = inv(R)
                        α, β, γ = RSS′.theta1, RSS′.theta2, RSS′.theta3
                        U′n2_rot = basisconversionmatrix(Cartesian(), B, θ2′_rot, ϕ2′_rot);
                        M′1 = U′n1 * R * Un1'
                        M′2 = U′n2_rot * R * Un2_rot'
                        M′ = kron(M′1, M′2)

                        x̂x̂_n1′n2′_rot = kron(M′1 * x̂_n1, M′2 * x̂_n2_rot)
                        R⁻¹x̂R⁻¹x̂_n1n2_rot = kron(inv(M′1) * x̂_n1, inv(M′2) * x̂_n2_rot)

                        BipolarSphericalHarmonics.monopolarharmonics!(B1′2′_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot);

                        for j in 0:lmax
                            Dp = wignerD!(Dvec[j], j, α, β, γ);
                            D = OffsetArray(Dp, -j:j, -j:j)
                            for l1 in 0:2:lmax, l2 in abs(j - l1):2:min(lmax, l1 + j), m in -j:2:j
                                DY = sum(D[m′, m] * biposh(B12_rot, θ1, ϕ1, θ2_rot, ϕ2_rot, j, m′, l1, l2) for m′ in -j:j)
                                RY12_rot = M′ * DY
                                Y1′2′_rot = biposh(B1′2′_rot, θ1′, ϕ1′, θ2′_rot, ϕ2′_rot, j, m, l1, l2)
                                @test isapproxdefault(Y1′2′_rot, RY12_rot)

                                # x′x′ : Bjm′(n1′,n2′) = M1xM2x : M1M2 ∑Dlmm′ Bjm′(n1,n2) = xx : ∑Dlmm′ Bjm′(n1,n2)
                                xxY1′2′_rot = x̂x̂_n1′n2′_rot' * Y1′2′_rot
                                RxxY12_rot = x̂x̂_n1n2_rot' * DY
                                @test isapproxdefault(xxY1′2′_rot, RxxY12_rot)

                                # xx : Bjm′(n1′,n2′) = xx : M1M2 ∑Dlmm′ Bjm′(n1,n2) = M1⁻¹xM2⁻¹x : ∑Dlmm′ Bjm′(n1,n2)
                                xxY1′2′_rot = x̂x̂_n1n2_rot' * Y1′2′_rot
                                R⁻¹xR⁻¹xY12_rot = R⁻¹x̂R⁻¹x̂_n1n2_rot' * DY
                                @test isapproxdefault(xxY1′2′_rot, R⁻¹xR⁻¹xY12_rot)
                            end
                        end
                    end
                end
            end

            # Compare Equator with the prime meridian (as both are great circles)
            ϕ1_Mer = ϕ2_Mer = 0
            θ1_Equator = θ2_Equator = pi/2
            for θ1_Mer in LinRange(0, pi, 4), θ2_Mer in LinRange(0, pi, 4),
                Δθ = θ2_Mer - θ1_Mer
                BipolarSphericalHarmonics.monopolarharmonics!(B12, θ1_Mer, ϕ1_Mer, θ2_Mer, ϕ2_Mer);

                Un1 = basisconversionmatrix(Cartesian(), B, θ1_Mer, ϕ1_Mer);
                Un2 = basisconversionmatrix(Cartesian(), B, θ2_Mer, ϕ2_Mer);

                for ϕ1_Equator in LinRange(0, pi, 4)
                    Δϕ = Δθ
                    ϕ2_Equator = ϕ1_Equator + Δϕ
                    U′n1 = basisconversionmatrix(Cartesian(), B, θ1_Equator, ϕ1_Equator);
                    U′n2 = basisconversionmatrix(Cartesian(), B, θ2_Equator, ϕ2_Equator);
                    BipolarSphericalHarmonics.monopolarharmonics!(B1′2′, θ1_Equator, ϕ1_Equator, θ2_Equator, ϕ2_Equator);

                    R = rotn1n2n1′n2′(θ1_Mer, ϕ1_Mer, θ2_Mer, ϕ2_Mer, θ1_Equator, ϕ1_Equator, θ2_Equator, ϕ2_Equator)
                    RSS′ = inv(R)
                    α, β, γ = RSS′.theta1, RSS′.theta2, RSS′.theta3
                    M′1 = U′n1 * R * Un1'
                    M′2 = U′n2 * R * Un2'
                    M′ = kron(M′1, M′2)

                    for j in 0:lmax
                        Dp = wignerD!(Dvec[j], j, α, β, γ);
                        D = OffsetArray(Dp, -j:j, -j:j);
                        for l1 in 0:2:lmax, l2 in abs(j - l1):2:min(lmax, l1 + j), m in -j:2:j
                            RY_mer = M′ * sum(D[m′, m] * biposh(B12, θ1_Mer, ϕ1_Mer, θ2_Mer, ϕ2_Mer, j, m′, l1, l2) for m′ in -j:j)
                            Y_Equator = biposh(B1′2′, θ1_Equator, ϕ1_Equator, θ2_Equator, ϕ2_Equator, j, m, l1, l2)
                            @test isapproxdefault(Y_Equator, RY_mer)
                        end
                    end
                end
            end

            tend = time()
            @info "Finished testing $YT $B in $(round(tend - tstart, sigdigits = 1)) seconds"
        end
    end
end

@testset "allocation" begin
    θ1, ϕ1, θ2, ϕ2 = pi*rand(), 2pi*rand(), pi*rand(), 2pi*rand()

    function test_alloc(SHT, θ1, ϕ1, θ2, ϕ2)
        B12 = monopolarharmonics(SHT, θ1, ϕ1, θ2, ϕ2, 1, 1)
        Y12 = zeros(BipolarSphericalHarmonics.eltypeY(B12), 1)
        biposh!(Y12, B12, θ1, ϕ1, θ2, ϕ2, 1, 1, 1, 1)
        @test (@allocated biposh!(Y12, B12, θ1, ϕ1, θ2, ϕ2, 1, 1, 1, 1)) == 0
    end

    test_alloc(SH(), θ1, ϕ1, θ2, ϕ2)
    test_alloc(GSH(), θ1, ϕ1, θ2, ϕ2)
    for YT in [PB(), Hansen(), Irreducible()], B in [Polar(), Cartesian(), HelicityCovariant(), SphericalCovariant()]
        test_alloc(VSH(YT, B), θ1, ϕ1, θ2, ϕ2)
    end
end
