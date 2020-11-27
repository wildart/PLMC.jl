"""
    regret(P::MultivariateDistribution, MC::Vector{MultivariateDistribution}, x::AbstractVector)

The regret value of the probability distribution `P` w.r.t the model class `MC`
in the point `x`.
"""
regret(P, MC, x::AbstractVector) = -logpdf(P, x) - minimum(map(p->-logpdf(p,x), MC))
regret(P, MC, X::AbstractMatrix) = -logpdf(P, X) .- min.((-logpdf(p, X) for p in MC)...)

"""
    regretmax(P::MultivariateDistribution, MC::Vector{MultivariateDistribution}, X::AbstractMatrix)

The maximal regret of the probability distribution `P` w.r.t the model class `MC`
for all points of `X`.
"""
regretmax(P, MC, X) = maximum(regret(P,MC,X))

"""
    refinedmdl(meta::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix{T}) -> (T, T, T)

Calculate a refined MDL value for the metaclustering `meta` w.r.t. to the model class 'mcr'
for all points of `X`. Returns tuple of MDL, likelihood, and complexity values.
"""
function refinedmdl(meta::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix{T}) where {T <: AbstractFloat}
    logps = hcat((logpdf(p, X) for p in models(mcr))...)
    cs = counts(mcr)
    ps = map(i->cs[i]./T(sum(cs[i])), meta)
    logpms = mixlogpdf(logps, ps, meta)
    return _refinedmdl(logpms)
end

function _refinedmdl(logpds::AbstractMatrix{T}) where {T <: AbstractFloat}
    n,k = size(logpds)
    minll = minimum(-logpds, dims=2)
    NLL = minll |> sum
    Rₘₐₓ = maximum(-logpds .- minll, dims=1) |> maximum # REG_max(p, M)
    return NLL + Rₘₐₓ, NLL, Rₘₐₓ
end


"""
    nml(meta::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix{T}) -> (T, T, T)

Calculate a normalized message length for the metaclustering `meta` w.r.t. to the model class 'mcr'
for all points of `X`. Returns tuple of NML, likelihood, and complexity values.
"""
function nml(meta::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix{T}) where {T <: AbstractFloat}
    logps = hcat((logpdf(p, X) for p in models(mcr))...)
    cs = counts(mcr)
    ps = map(i->cs[i]./T(sum(cs[i])), meta)
    logpms = mixlogpdf(logps, ps, meta)
    return _nml(logpms)
end

function _nml(logpds::AbstractMatrix)
    n,k = size(logpds)
    minll = minimum(-logpds, dims=2)
    NLL = minll |> sum
    COMP = exp.(-minll) |> sum |> log
    return NLL + n*COMP, NLL, COMP
end


"""
    nll(meta::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix{T}) -> (T, T, T)

Calculate a negative log-likelyhood for the metaclustering `meta` w.r.t. to the model class 'mcr'
for all points of `X`.
"""
function nll(meta::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix{T}) where {T <: AbstractFloat}
    logps = hcat((logpdf(p, X) for p in models(mcr))...)
    cs = counts(mcr)
    ps = map(i->cs[i]./T(sum(cs[i])), meta)
    logpms = mixlogpdf(logps, ps, meta)
    minll = minimum(-logpms, dims=2)
    NLL = minll |> sum
    return NLL, NLL, zero(T)
end


"""
    mdldiff(P′::AbstractMixtureModel, MC::Vector{MultivariateDistribution}, X::AbstractMatrix) -> Float64

Returns MDL difference between original and merged P′ = {Pᵢ ∪ Pⱼ} clusterings.
"""
function mdldiff(meta::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix; kwargs...)
    logps = hcat((logpdf(p, X) for p in models(mcr))...)
    return mdldiff(meta, logps, assignments(mcr); kwargs...)
end

function mdldiff(meta::Vector{Vector{Int}}, logps::AbstractMatrix{T},
                 assign::Vector{Int}; kwargs...) where {T <: AbstractFloat}
    n = size(logps,2)
    cs = zeros(Int, n)
    for i in assign
        cs[i] += 1
    end
    sz = map(i->sum(cs[i]), meta)
    ps = map(i->cs[i]./sum(cs[i]), meta)
    mrgidxs = vcat((findall(i-> i ∈ m, assign) for m in meta)...)
    vlogps = view(logps, mrgidxs, :)
    logpms = mixlogpdf(vlogps, ps, meta)
    return _mdldiff(logpms, vlogps, meta, sz)
end

function _mdldiff(logP′::AbstractMatrix{T}, logPs::AbstractMatrix{T},
                  meta::Vector{Vector{Int}}, sz::Vector{Int}) where {T <: AbstractFloat}
    n = sum(sz)
    NLL = maximum(logP′, dims=2) |> sum
    pv = exp.(logP′)
    s1 = max.((minimum(view(-logPs,:,i), dims=2) for i in meta)...)
    s2 = -log.(sz[2]./view(pv,:,1) .+ sz[1]./view(pv,:,2))
    return NLL + (maximum(s1 .- s2) - log(n))*n  # normalize cost
end
