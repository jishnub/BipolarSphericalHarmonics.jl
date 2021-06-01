module BipolarSphericalHarmonics

using SphericalHarmonicModes
using SphericalHarmonicArrays
using SphericalHarmonics
using SphericalHarmonics: NorthPole, SouthPole, getY
import SphericalHarmonics: eltypeY
using VectorSphericalHarmonics
using SHTOOLS
using LinearAlgebra
using StaticArrays
using Base: @propagate_inbounds

abstract type SHType end
struct SH <:SHType end
struct VSH{VSHType, Basis} <:SHType
    YT :: VSHType
    B :: Basis
end
struct GSH <:SHType end
Base.broadcastable(S::SHType) = Ref(S)

export monopolarharmonics
export monopolarharmonics!
export biposh
export biposh!
export biposh_flippoints
export SH
export VSH
export GSH

"""
    BSHCache

Wrapper around monopolar harmonics that are coupled to evaluate the bipolar harmonics. The cache may be constructed using
[`cache`](@ref).
"""
struct BSHCache{T, ST<:SHType, S, W}
    SHT :: ST
    S1 :: S
    S2 :: S
    W3j :: W
    C :: W
end

BSHCache{T}(SHT::SHType, S1, S2, W, C) where {T} = BSHCache{T, typeof(SHT), typeof(S1), typeof(W)}(SHT, S1, S2, W, C)
flip(B::BSHCache) = typeof(B)(B.SHT, B.S2, B.S1, B.W3j, B.C)

_cache(::SH, T, lmax) = SphericalHarmonics.cache(T, lmax)
_cache(V::VSH, T, lmax) = VectorSphericalHarmonics.VSHCache(T, V.YT, V.B, ML(ZeroTo(lmax)))
_cache(::GSH, T, lmax) = VectorSphericalHarmonics.VSHCache(T, ML(ZeroTo(lmax)))

_bshtype(SHT, C, Y1, Y2) = typeof(C * _kron(Y1, Y2))

"""
    cache(SHT, T::Type, j12max)

Allocate the arrays required to evaluate the monopolar harmonics for all modes `0 ≤ j1, j2 ≤ j12max`.
`SHT` may be one of `SH()`, `GSH()`, or `VSH(YT(), B())` where `YT` and `B` are vector harmonic and basis
types offered by [`VectorSphericalHarmonics.jl`](https://github.com/jishnub/VectorSphericalHarmonics.jl).
`T` is a real type that determines the precision of the monopolar harmonics.

!!! note
    The monopolar harmonics must be initialized by calling [`monopolarharmonics!`](@ref) before the cache is used.
"""
function cache(SHT::SHType, T::Type, lmax)
    S1 = _cache(SHT, T, lmax)
    S2 = _cache(SHT, T, lmax)
    W = zeros(Cdouble, 2lmax+1)
    C = copy(W)
    TB = _bshtype(SHT, first(C), _parent(first(getY(S1))), _parent(first(getY(S2))))
    BSHCache{TB}(SHT, S1, S2, C, W)
end
cache(SHT, args...) = cache(SHT, Float64, args...)

eltypeY(::BSHCache{T}) where {T} = T

SHType(B::BSHCache) = B.SHT

_parent(A::AbstractArray) = parent(A)
_parent(A::Number) = A

_kron(Y1, Y2) = kron(Y1, Y2)
if VERSION <= v"1.7.0-DEV.1046"
    # preserve sizes in kron
    function _kron(Y1::Diagonal{<:Any, <:SVector}, Y2::Diagonal{<:Any, <:SVector})
        YD1 = diag(Y1)
        YD2 = diag(Y2)
        Diagonal(kron(YD1, YD2))
    end
end

_kronindex(ind1, ind2) = 3(ind1 + 1) + ind2 + 2
"""
    kronindex(::GSH, ind1, ind2)

Given the indices of each `GSH` vector, return the corresponding index of the bipolar spherical harmonic.
The indices `ind1` and `ind2` must lie within `-1` and `1`.
"""
function kronindex(::GSH, ind1, ind2)
    @assert -1 <= ind1 <= 1 "ind1 must satisfy -1 ≤ ind1 ≤ 1"
    @assert -1 <= ind2 <= 1 "ind2 must satisfy -1 ≤ ind2 ≤ 1"
    _kronindex(ind1, ind2)
end
"""
    kronindex(::VSH, ind11, ind12, ind21, ind22)

Given the indices of each `VSH` matrix, return the corresponding index of the bipolar spherical harmonic.
Note that the indices correspond to `ℓ - j` for the `Irreducible` vector spherical harmonic ``\\mathbf{Y}^{\\ell}_{jm}(\\hat{n})``.
"""
function kronindex(::VSH, ind11, ind12, ind21, ind22)
    @assert -1 <= ind11 <= 1 "ind11 must satisfy -1 ≤ ind11 ≤ 1"
    @assert -1 <= ind12 <= 1 "ind12 must satisfy -1 ≤ ind12 ≤ 1"
    @assert -1 <= ind21 <= 1 "ind21 must satisfy -1 ≤ ind21 ≤ 1"
    @assert -1 <= ind22 <= 1 "ind22 must satisfy -1 ≤ ind22 ≤ 1"
    CartesianIndex(_kronindex(ind11, ind21), _kronindex(ind12, ind22))
end

function Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, kronindex, ::Base.Ref{GSH}, i1::Integer, r2::AbstractUnitRange{<:Integer})
    ind1 = kronindex(GSH(), i1, first(r2))
    ind2 = kronindex(GSH(), i1, last(r2))
    ind1:ind2
end
function Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, kronindex, ::Base.Ref{GSH}, r1::AbstractUnitRange{<:Integer}, r2::Union{Integer, AbstractUnitRange{<:Integer}})
    ind1 = kronindex(GSH(), first(r1), first(r2))
    ind2 = kronindex(GSH(), last(r1), last(r2))
    s = length(r1) == 1 ? 1 : length(r2) == 1 ? 3 : 4
    ind1:s:ind2
end

_containertype(::Type{T}, ::Any) where {T} = Vector{T}
_containertype(::Type{T}, ::LM{SingleValuedRange, SingleValuedRange}) where {T} = MVector{1,T}
_container(el, modes) = [el for i = 1:length(modes)]
_container(el, modes::LM{SingleValuedRange, SingleValuedRange}) = MVector{1,typeof(el)}((el,))

_zerostype(::Type{T}, modes) where {T} = SHArray{T, 1, _containertype(T, modes), Tuple{typeof(modes)}}
_zeros(::Type{T}, modes) where {T} = SHArray(_container(zero(T), modes), modes)

_maybestripwrapper(T, ZT, ::Integer, ::Integer) = T
_maybestripwrapper(T, ZT, jm...) = ZT

_j1j2(::BSHCache{<:Any, SH}, j1, j2) = (j1,), (j2,)
_j1j2(::BSHCache, j1, j2) = (ML(ZeroTo(j1)),), (ML(ZeroTo(j2)),)
_j1j2(::BSHCache) = (), ()
_j1j2(B::BSHCache, j1j2modes) = _j1j2(B, _j12max(j1j2modes)...)

_j12max(j1j2modes::L2L1Triangle) = maximum(l2_range(j1j2modes)), maximum(l1_range(j1j2modes))
_j12max(j1j2modes) = maximum(first, j1j2modes), maximum(last, j1j2modes)

"""
    monopolarharmonics!(B::BSHCache, θ1, ϕ1, θ2, ϕ2, j1, j2)

Update `B` with the monopolar harmonics ``Y_{j_1,m_1}(\\theta_1, \\phi_1)`` and ``Y_{j_2,m_2}(\\theta_2, \\phi_2)``,
where ``Y`` may either be a scalar or a vector harmonic depending on `B`.
"""
function monopolarharmonics!(B::BSHCache{<:Any, SH}, θ1, ϕ1, θ2, ϕ2, j1j2...)
    j1tup, j2tup = _j1j2(B, j1j2...)
    computePlmcostheta!(B.S1, θ1, j1tup...)
    computeYlm!(B.S1, θ1, ϕ1, j1tup...)
    computePlmcostheta!(B.S2, θ2, j2tup...)
    computeYlm!(B.S2, θ2, ϕ2, j2tup...)
    return B
end

function monopolarharmonics!(B::BSHCache{<:Any, GSH}, θ1, ϕ1, θ2, ϕ2, j1j2...)
    j1tup, j2tup = _j1j2(B, j1j2...)
    VectorSphericalHarmonics.genspharm!(B.S1, j1tup..., θ1, ϕ1)
    VectorSphericalHarmonics.genspharm!(B.S2, j2tup..., θ2, ϕ2)
    return B
end

function monopolarharmonics!(B::BSHCache{<:Any, <:VSH}, θ1, ϕ1, θ2, ϕ2, j1j2...)
    j1tup, j2tup = _j1j2(B, j1j2...)
    VectorSphericalHarmonics.vshbasis!(B.S1, B.SHT.YT, B.SHT.B, j1tup..., θ1, ϕ1)
    VectorSphericalHarmonics.vshbasis!(B.S2, B.SHT.YT, B.SHT.B, j2tup..., θ2, ϕ2)
    return B
end

"""
    monopolarharmonics(SHT, θ1, ϕ1, θ2, ϕ2, j1, j2)

Evaluate the monopolar harmonics ``Y_{j_1,m_1}(\\theta_1, \\phi_1)`` and ``Y_{j_2,m_2}(\\theta_2, \\phi_2)``,
where ``Y`` may either be a scalar or a vector harmonic depending on `SHT`, which may be one of `SH()`, `GSH()`,
or `VSH(YT(), B())` where `YT` and `B` are vector harmonic and basis types offered by
[`VectorSphericalHarmonics.jl`](https://github.com/jishnub/VectorSphericalHarmonics.jl).
"""
function monopolarharmonics(SHT::SHType, θ1, ϕ1, θ2, ϕ2, j1, j2)
    T = float(mapreduce(typeof, promote_type, (θ1, ϕ1, θ2, ϕ2)))
    B = cache(SHT, T, max(j1, j2))
    monopolarharmonics!(B, θ1, ϕ1, θ2, ϕ2, j1, j2)
    return B
end

function monopolarharmonics(SHT::SHType, θ1, ϕ1, θ2, ϕ2, j1j2modes...)
    j1j2 = _j12max(j1j2modes...)
    monopolarharmonics(SHT, θ1, ϕ1, θ2, ϕ2, j1j2...)
end

monopolarharmonics(B::BSHCache) = getY(B.S1), getY(B.S2)

function degreerange(j1, j2, m::Integer)
    jmin = max(abs(j1 - j2), abs(m))
    jmax = j1 + j2
    jmin:jmax
end
degreerange(j1, j2, m::AbstractUnitRange) = degreerange(j1, j2, minimum(abs, m))
degreerange(j1, j2) = degreerange(j1, j2, 0)

neg1pow(x::Integer) = isodd(x) ? -1 : 1

_maybewrapj2j1(M, modes::L2L1Triangle) = SHArray(M, modes)
_maybewrapj2j1(M, modes) = M

_modes(::Colon, ::Colon, j1, j2) = LM(degreerange(j1, j2))
_modes(::Colon, m::Union{Integer, AbstractUnitRange{<:Integer}}, j1, j2) = LM(degreerange(j1, j2, m), m)
_modes(j::Union{Integer, AbstractUnitRange{<:Integer}}, ::Colon, j1, j2) = LM(j)
_modes(j::Union{Integer, AbstractUnitRange{<:Integer}}, m::Union{Integer, AbstractUnitRange{<:Integer}}, j1, j2) = LM(j, m)
function _modes(j, m, j1j2modes)
    lmax = max(maximum(first, j1j2modes), maximum(last, j1j2modes))
    _modes(j, m, lmax, lmax)
end
_modes(jm::LM, j1j2modes) = jm

_nonnegative(r::AbstractUnitRange) = max(minimum(r), 0):maximum(r)
_negative(r::AbstractUnitRange) = minimum(r):min(-1, maximum(r))

_maybeonly(M, jm...) = M
_maybeonly(M::SHArray{<:Any,1,<:Any,Tuple{LM{SingleValuedRange, SingleValuedRange}}}, j::Integer, m::Integer) = M[1]

_modes_intersect(jm, j1, j2) = intersect(jm, LM(degreerange(j1, j2)))
_modes_intersect(j, m, j1, j2) = intersect(_modes(j, m, j1, j2), LM(degreerange(j1, j2)))

const perminds = SVector{9}((1,4,7,2,5,8,3,6,9))
const GSHPerm = CartesianIndex.(perminds)
const VSHPerm = CartesianIndex.(perminds, perminds')

_permutation(::VSH) = VSHPerm
_permutation(::GSH) = GSHPerm

function _permuteterms!(Y21, indj1j2, ::Nothing, SHT, j1, j2, jm)
    @inbounds Y21[indj1j2] = nothing
    return nothing
end
function _permuteterms!(Y21, indj1j2, Y12_j2j1::SHArray, SHT, j1, j2, jm)
    _Y21 = similar(Y12_j2j1)
    @inbounds begin
        for (indjm, (j,m)) in enumerate(first(SphericalHarmonicArrays.modes(Y12_j2j1)))
            phase = (-1)^(j1 + j2 - j)
            _Y21[indjm] = _permuteterms(Y12_j2j1[indjm], SHT) * phase
        end
        Y21[indj1j2] = _Y21
    end
    return nothing
end
function _permuteterms!(Y21, indj1j2, Y12_j2j1::Union{SArray,Diagonal{<:Any,<:SVector},Number}, SHT, j1, j2, jm::LM{SingleValuedRange, SingleValuedRange})
    j, _ = first(jm)
    phase = (-1)^(j1 + j2 - j)
    @inbounds Y21[indj1j2] = _permuteterms(Y12_j2j1, SHT) * phase
    return nothing
end
_permuteterms(Y12_j2j1_jm, SHT) = @inbounds Y12_j2j1_jm[_permutation(SHT)]
function _permuteterms(Y12_j2j1_jm::Diagonal{<:Any, <:SVector}, SHT)
    d = diag(Y12_j2j1_jm)
    v = typeof(d)(d[GSHPerm])
    Diagonal(v)
end
_permuteterms(Y12_j2j1_jm::Number, SHT) = Y12_j2j1_jm

"""
    biposh_flippoints(SHT, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)

Evaluate the bipolar harmonics ``Y_{jm}^{j_1, j_2}((\\theta_1, \\phi_1), (\\theta_2, \\phi_2))`` and
``Y_{jm}^{j_1, j_2}((\\theta_2, \\phi_2), (\\theta_1, \\phi_1))``
in one pass, utilizing symmetries of the Clebsch-Gordean coefficients.
"""
biposh_flippoints

function _biposh_flippoints(SHT::SHType, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm...)
    B12 = monopolarharmonics(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes...);
    B21 = monopolarharmonics(SHT, θ2, ϕ2, θ1, ϕ1, j1j2modes...);
    biposh_flippoints(B12, B21, θ1, ϕ1, θ2, ϕ2, jm..., j1j2modes...)
end
biposh_flippoints(SHT::SHType, θ1, ϕ1, θ2, ϕ2, j, m, j1j2modes...) =
    _biposh_flippoints(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes, j, m)
biposh_flippoints(SHT::SHType, θ1, ϕ1, θ2, ϕ2, jm::LM, j1j2modes...) =
    _biposh_flippoints(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm)

function _biposh_flippoints_j1j2(B12, B21, θ1, ϕ1, θ2, ϕ2, j1, j2, jm...)
    Y12 = biposh(B12, θ1, ϕ1, θ2, ϕ2, jm..., j1, j2)
    Y21 = biposh(B21, θ2, ϕ2, θ1, ϕ1, jm..., j1, j2)
    (Y12 === nothing || Y21 === nothing) && return nothing
    return Y12, Y21
end
biposh_flippoints(B12::BSHCache, B21::BSHCache, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2) =
    _biposh_flippoints_j1j2(B12, B21, θ1, ϕ1, θ2, ϕ2, j1, j2, j, m)
biposh_flippoints(B12::BSHCache, B21::BSHCache, θ1, ϕ1, θ2, ϕ2, jm::LM, j1, j2) =
    _biposh_flippoints_j1j2(B12, B21, θ1, ϕ1, θ2, ϕ2, j1, j2, jm)

@inline function _biposh_flippoints_j1j2modes(B12::BSHCache{T}, B21::BSHCache{T}, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm...) where {T}
    jm_modes = _modes(jm..., j1j2modes)
    TM = _maybestripwrapper(T, _zerostype(T, jm_modes), jm...)
    Y12 = Vector{Union{TM, Nothing}}(undef, length(j1j2modes))
    Y21 = similar(Y12)
    for (indj1j2, (j1, j2)) in enumerate(j1j2modes)
        if (j2,j1) in j1j2modes && modeindex(j1j2modes, j2, j1) < indj1j2
            indj2j1 = modeindex(j1j2modes, j2, j1)
            # in this case the bipolar harmonics for (j2,j1) have been computed
            # we may use these to compute the ones for (j1,j2)
            _permuteterms!(Y21, indj1j2, Y12[indj2j1], SHType(B12), j1, j2, jm_modes)
            _permuteterms!(Y12, indj1j2, Y21[indj2j1], SHType(B21), j1, j2, jm_modes)
        else
            Y12[indj1j2] = biposh(B12, θ1, ϕ1, θ2, ϕ2, jm..., j1, j2)
            Y21[indj1j2] = biposh(B21, θ2, ϕ2, θ1, ϕ1, jm..., j1, j2)
        end
    end
    _maybewrapj2j1(Y12, j1j2modes), _maybewrapj2j1(Y21, j1j2modes)
end
biposh_flippoints(B12::BSHCache, B21::BSHCache, θ1, ϕ1, θ2, ϕ2, j, m, j1j2modes) =
    _biposh_flippoints_j1j2modes(B12, B21, θ1, ϕ1, θ2, ϕ2, j1j2modes, j, m)
biposh_flippoints(B12::BSHCache, B21::BSHCache, θ1, ϕ1, θ2, ϕ2, jm::LM, j1j2modes) =
    _biposh_flippoints_j1j2modes(B12, B21, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm)


"""
    biposh(SHT, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)

Evaluate the bipolar harmonic ``Y_{jm}^{j_1, j_2}((\\theta_1, \\phi_1), (\\theta_2, \\phi_2))``. `SHT` may be one of `SH()`,
`GSH()` and `VSH(YT(), B())` where `YT` and `B` are vector harmonic and basis types offered by
[`VectorSphericalHarmonics.jl`](https://github.com/jishnub/VectorSphericalHarmonics.jl).

    biposh(SHT, θ1, ϕ1, θ2, ϕ2, j, m, j1j2modes)

Evaluate the bipolar harmonic ``Y_{jm}^{j_1, j_2}((\\theta_1, \\phi_1), (\\theta_2, \\phi_2))`` for all `(j1,j2)` in
`j1j2modes`.

    biposh(SHT, θ1, ϕ1, θ2, ϕ2, jmcoll::SphericalHarmonicModes.LM, j1j2...)

Evaluate the bipolar harmonics ``Y_{jm}^{j_1, j_2}((\\theta_1, \\phi_1), (\\theta_2, \\phi_2))`` for all `(j,m)` in `jmcoll`,
where `j1j2` may be either a 2-Tuple `(j1,j2)` or a collection of 2-Tuples.
"""
biposh

function _biposh(SHT::SHType, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm...)
    B = monopolarharmonics(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes...)
    biposh(B, θ1, ϕ1, θ2, ϕ2, jm..., j1j2modes...)
end
biposh(SHT::SHType, θ1, ϕ1, θ2, ϕ2, j, m, j1j2modes...) = _biposh(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes, j, m)
biposh(SHT::SHType, θ1, ϕ1, θ2, ϕ2, jm::LM, j1j2modes...) = _biposh(SHT, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm)

@inline function _biposh_j1j2modes(B::BSHCache{T}, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm...) where {T}
    jm_modes = _modes(jm..., j1j2modes)
    TM = _maybestripwrapper(T, _zerostype(T, jm_modes), jm...)
    M = Vector{Union{TM, Nothing}}(undef, length(j1j2modes))
    for (indj1j2, (j1, j2)) in enumerate(j1j2modes)
        Bel = biposh(B, θ1, ϕ1, θ2, ϕ2, jm..., j1, j2)
        M[indj1j2] = Bel
    end
    _maybewrapj2j1(M, j1j2modes)
end
biposh(B::BSHCache, θ1, ϕ1, θ2, ϕ2, j, m, j1j2modes) = _biposh_j1j2modes(B, θ1, ϕ1, θ2, ϕ2, j1j2modes, j, m)
biposh(B::BSHCache, θ1, ϕ1, θ2, ϕ2, jm::LM, j1j2modes) = _biposh_j1j2modes(B, θ1, ϕ1, θ2, ϕ2, j1j2modes, jm)

@inline function _biposh_j1j2(B, θ1, ϕ1, θ2, ϕ2, j1, j2, jm...)
    jm_filt = _modes_intersect(jm..., j1, j2)
    jm_filt === nothing && return nothing
    Y = _zeros(eltypeY(B), jm_filt)
    biposh!(parent(Y), B, θ1, ϕ1, θ2, ϕ2, jm..., j1, j2)
    return _maybeonly(Y, jm...)
end
biposh(B::BSHCache, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2) = _biposh_j1j2(B, θ1, ϕ1, θ2, ϕ2, j1, j2, j, m)
biposh(B::BSHCache, θ1, ϕ1, θ2, ϕ2, jm::LM, j1, j2) = _biposh_j1j2(B, θ1, ϕ1, θ2, ϕ2, j1, j2, jm)

const BasisPhase = SVector{9,Int}([(-1)^(α1 + α2) for α1 in -1:1, α2 in -1:1])
const VSHIndexPhase = BasisPhase'
const BasisPhase_VSHIndexPhase = BasisPhase*VSHIndexPhase

_reverse(Y::SArray; dims = :) = _reverse(Y, dims)
# special case this to use the reverse(::StaticArray) method defined by StaticArrays
_reverse(Y::SArray, ::Colon) = reverse(Y)
_reverse(Y::SArray, dims) = reverse(Y, dims = dims)
_reverse(Y::Diagonal{<:Any,<:SVector}) = Diagonal(reverse(diag(Y)))

_conjphase(Y, phase, ::SH) = conj(Y) .* phase
_conjphase(Y, phase, ::GSH) = conj(_reverse(Y)) .* BasisPhase .* phase
_conjphase(Y, phase, ::VSH{PB, SphericalCovariant}) = conj(_reverse(Y)) .* BasisPhase .* phase
function _conjphase(Y, phase, ::VSH{PB, HelicityCovariant})
    D = conj(_reverse(Y))
    Diagonal(diag(D) .* BasisPhase .* phase)
end
_conjphase(Y, phase, ::VSH{PB, <:Union{Cartesian, Polar}}) = (A = conj!(_reverse(Y, dims = 2)); A .*= phase; A)
function _conjphase(Y, phase, ::VSH{<:Any, <:Union{SphericalCovariant, HelicityCovariant}})
    A = conj!(_reverse(Y, dims = 1))
    A .*= BasisPhase_VSHIndexPhase * phase
    A
end
_conjphase(Y, phase, ::VSH{<:Any, <:Union{Cartesian, Polar}}) = conj(Y) .* VSHIndexPhase .* phase

"""
    biposh!(Y12::AbstractVetor, B::BSHCache, θ1, ϕ1, θ2, ϕ2, j, m, j1, j2)

Evaluate the bipolar harmonic ``Y_{jm}^{j_1, j_2}((\\theta_1, \\phi_1), (\\theta_2, \\phi_2))`` and store the result in `Y12`.
The cache `B` determines the type of harmonic evaluated.
"""
biposh!

biposh!(Y12::AbstractVector{T}, B::BSHCache{T}, θ1, ϕ1, θ2, ϕ2, j, m, j1::Integer, j2::Integer) where {T} =
    biposh!(Y12, B, θ1, ϕ1, θ2, ϕ2, _modes(j, m, j1, j2), j1, j2)

function biposh!(Y12::AbstractVector{T}, B::BSHCache{T}, θ1, ϕ1, θ2, ϕ2, jm::LM, j1::Integer, j2::Integer) where {T}
    jm_filt = _modes_intersect(jm, j1, j2)
    (jm_filt === nothing || isempty(jm_filt)) && return Y12
    length(Y12) >= length(jm_filt) || throw(ArgumentError("Y must have a a length >= length($jm_filt)"))
    mrange = m_range(jm_filt)

    Y1, Y2 = monopolarharmonics(B)

    j1 in l_range(first(SphericalHarmonicArrays.modes(Y1))) || throw("please compute monopolar harmonics for for j1 = $j1 first")
    j2 in l_range(first(SphericalHarmonicArrays.modes(Y2))) || throw("please compute monopolar harmonics for for j2 = $j2 first")

    Y = @view Y12[begin .+ (0:length(jm_filt)-1)]
    Y .= zero.(Y)

    for m in _nonnegative(mrange)
        jrange_m = l_range(jm_filt, m)
        modes_m = LM(jrange_m, m)
        modeindrange = modeindex(jm_filt, modes_m)
        Y12_m = @view Y12[modeindrange]
        biposh!(Y12_m, B, θ1, ϕ1, θ2, ϕ2, modes_m, j1, j2)
    end

    YS = SHArray(Y, jm_filt)
    for m in _negative(mrange)
        if -m in mrange
            jrange_m = l_range(jm_filt, m)
            jmin = minimum(jrange_m)
            # overall phase of (-1)^(j1 + j2 - j + m) appears in all terms
            overallphase = neg1pow(j1 + j2 + m - jmin)

            for j in jrange_m
                Yjminm = YS[(j,-m)]
                YS[(j,m)] = _conjphase(Yjminm, overallphase, SHType(B))
                overallphase *= -1
            end
        else
            jrange_m = l_range(jm_filt, m)
            modes_m = LM(jrange_m, m)
            modeindrange = modeindex(jm_filt, modes_m)
            Y12_m = @view Y12[modeindrange]
            biposh!(Y12_m, B, θ1, ϕ1, θ2, ϕ2, modes_m, j1, j2)
        end
    end

    return Y12
end

@inline function C_view_j1j2m_j(C, j1, j2, m, jrange)
    jmin_valid = minimum(degreerange(j1, j2, m))
    jmin = minimum(jrange)
    jmin_offset = jmin - jmin_valid
    Cv = @view C[(begin + jmin_offset) .+ (0:length(jrange)-1)]
end

function biposh!(Y12::AbstractVector{T}, B::BSHCache{T}, θ1, ϕ1, θ2, ϕ2, jm::LM{<:Any,SingleValuedRange}, j1::Integer, j2::Integer) where {T}
    jm_filt = _modes_intersect(jm, j1, j2)
    isempty(jm_filt) && return Y12
    jrange = l_range(jm_filt)
    length(Y12) >= length(jrange) || throw(ArgumentError("vector must have at least $(length(jrange)) elements, received $(length(Y12))"))
    Y = @view Y12[begin .+ (0:length(jrange)-1)]
    Y .= zero.(Y)

    Y1, Y2 = monopolarharmonics(B)

    j1 in l_range(first(SphericalHarmonicArrays.modes(Y1))) || throw("please compute monopolar harmonics for for j1 = $j1 first")
    j2 in l_range(first(SphericalHarmonicArrays.modes(Y2))) || throw("please compute monopolar harmonics for for j2 = $j2 first")

    m = only(m_range(jm_filt))
    Cv = C_view_j1j2m_j(B.C, j1, j2, m, jrange)

    for m1 in -j1:j1
        m2 = m - m1
        abs(m2) > j2 && continue
        clebschgordan!(B.C, B.W3j, j1, m1, j2, m)
        Y1_j1m1 = _parent(Y1[(j1,m1)])
        Y2_j2m2 = _parent(Y2[(j2,m2)])
        Y1Y2 = _kron(Y1_j1m1, Y2_j2m2)
        for ind in eachindex(Y, Cv)
            Y[ind] += Cv[ind] * Y1Y2
        end
    end

    return Y12
end

# All j for one m for (j1,j2) at the NorthPole
function biposh!(Y12::AbstractVector{T}, B::BSHCache{T,SH}, θ1::NorthPole, ϕ1, θ2, ϕ2, jm::LM{<:Any,SingleValuedRange}, j1::Integer, j2::Integer) where {T}
    jm_filt = _modes_intersect(jm, j1, j2)
    isempty(jm_filt) && return Y12
    jrange = l_range(jm_filt)
    length(Y12) >= length(jrange) || throw(ArgumentError("vector must have at least $(length(jrange)) elements, received $(length(Y12))"))
    Y12_section = @view Y12[begin .+ (0:length(jrange)-1)]

    m = only(m_range(jm_filt))
    if abs(m) > j2
        # in this case the value is known to be zero, as the spherical harmonic Ylm(0,ϕ1) == 0 for m != 0
        fill!(Y12_section, zero(eltype(Y12_section)))
        return Y12
    end
    _, Y2 = monopolarharmonics(B)
    clebschgordan!(B.C, B.W3j, j1, 0, j2, m)
    Cv = C_view_j1j2m_j(B.C, j1, j2, m, jrange)
    Y2_j2m = Y2[(j2,m)]
    Y1_l0_NP = √((2j1+1)/4pi)
    @. Y12_section = Cv * kron(Y1_l0_NP, Y2_j2m)
    return Y12
end

function biposh!(Y12::AbstractVector{T}, B::BSHCache{T,SH}, θ1::NorthPole, ϕ1, θ2::NorthPole, ϕ2, jm::LM{<:Any,SingleValuedRange}, j1::Integer, j2::Integer) where {T}
    jm_filt = _modes_intersect(jm, j1, j2)
    isempty(jm_filt) && return Y12
    jrange = l_range(jm_filt)
    length(Y12) >= length(jrange) || throw(ArgumentError("vector must have at least $(length(jrange)) elements, received $(length(Y12))"))
    Y12_section = @view Y12[begin .+ (0:length(jrange)-1)]

    m = only(m_range(jm_filt))
    if m != 0
        # in this case the value is known to be zero, as the spherical harmonic Ylm(0,ϕ1) == 0 for m != 0
        fill!(Y12_section, zero(eltype(Y12_section)))
        return Y12
    end
    clebschgordan!(B.C, B.W3j, j1, 0, j2, 0)
    Cv = C_view_j1j2m_j(B.C, j1, j2, m, jrange)
    Y1_j10_NP = √((2j1+1)/4pi)
    Y2_j20_NP = √((2j2+1)/4pi)
    @. Y12_section = Cv * kron(Y1_j10_NP, Y2_j20_NP)
    return Y12
end

function biposh!(Y12::AbstractVector{T}, B::BSHCache{T}, θ1, ϕ1, θ2::NorthPole, ϕ2, jm::LM{<:Any,SingleValuedRange}, j1::Integer, j2::Integer) where {T}
    biposh!(Y12, flip(B), θ2, ϕ2, θ1, ϕ1, jm, j2, j1)
    jm_filt = _modes_intersect(jm, j1, j2)
    Y12_section = @view Y12[begin .+ (0:length(jm_filt)-1)]
    Y = SHArray(Y12_section, jm_filt)

    @inbounds for m in m_range(jm_filt)
        jrange = l_range(jm_filt, m)
        jmin = minimum(jrange)
        phase = neg1pow(j1 + j2 - jmin)
        for j in jrange
            Y[(j,m)] *= phase
            phase *= -1
        end
    end
    return Y12
end

##################################################################################################

@inline function clebschgordan!(C::AbstractVector, w3j::AbstractVector, j1::Integer, m1::Integer, j2::Integer, m::Integer)
    m2 = m - m1
    # Evaluate the Wigner 3j symbol
    #   /  j  j1  j2 \
    #   \ -m  m1  m2 /
    _, jmin, jmax = SHTOOLS.Wigner3j!(w3j, j1, j2, -m, m1, m2)
    jrange = jmin:jmax
    @assert length(C) >= length(jrange) "Clebsch-Gordan vector does not contain enough elements"

    phase = neg1pow(j1 - j2 + m)
    @inbounds for (ind, j) in enumerate(jrange)
        C[begin - 1 + ind] = w3j[begin - 1 + ind] * √(2j+1) * phase
    end

    return C
end

include("precompile.jl")

end
