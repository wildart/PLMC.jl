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
refinedmdl(LC::Vector{MultivariateDistribution}, MC::ClusterComplex.MahalonobisClusteringResult, X::AbstractMatrix) -> (Float64, Float64, Float64)

Calculate a refined MDL value for the model class `LC` w.r.t. to the model class 'MC'
for all points of `X`. Returns tuple of MDL, likelihood, and complexity values.
"""
function refinedmdl(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix)
    logps = hcat((PLMC.logpdf(p, X) for p in models(mcr))...)
    cs = counts(mcr)
    ps = map(i->cs[i]./sum(cs[i]), mrg)
    logpms = mixlogpdf(logps, ps, mrg)
    return _refinedmdl(logpms)
end

function _refinedmdl(logpds::Matrix{T}) where {T <: AbstractFloat}
    n,k = size(logpds)
    minll = minimum(-logpds, dims=2)
    NLL = minll |> sum
    Rₘₐₓ = maximum(-logpds .- minll, dims=1) |> maximum # REG_max(p, M)
    return NLL + Rₘₐₓ, NLL, Rₘₐₓ
end


"""
    nml(LC::Vector{MultivariateDistribution}, MC::ClusterComplex.MahalonobisClusteringResult, X::AbstractMatrix) -> (Float64, Float64, Float64)

Calculate a normalized message length for the model class `LC` w.r.t. to the model class 'MC'
for all points of `X`. Returns tuple of MDL, likelihood, and complexity values.
"""
function nml(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix)
    logps = hcat((PLMC.logpdf(p, X) for p in models(mcr))...)
    cs = counts(mcr)
    ps = map(i->cs[i]./sum(cs[i]), mrg)
    logpms = mixlogpdf(logps, ps, mrg)
    return _nml(logpms)
end

function _nml(logpds::Matrix{T}) where {T <: AbstractFloat}
    n,k = size(logpds)
    minll = minimum(-logpds, dims=2)
    NLL = minll |> sum
    COMP = exp.(-minll) |> sum |> log
    return NLL + n*COMP, NLL, COMP
end


"""
    mdldiff(P′::AbstractMixtureModel, MC::Vector{MultivariateDistribution}, X::AbstractMatrix) -> Float64

Returns MDL difference between original and merged P′ = {Pᵢ ∪ Pⱼ} clusterings.
"""
function mdldiff(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix)
    logps = hcat((PLMC.logpdf(p, X) for p in models(mcr))...)
    cs = counts(mcr)
    ps = map(i->cs[i]./sum(cs[i]), mrg)
    logpms = mixlogpdf(logps, ps, mrg)
    mrgidxs = map(m->findall(i-> i ∈ m, assignments(mcr)), mrg)
    return _mdldiff(logpms, logps, mrg, mrgidxs)
end

function _mdldiff(logP′::AbstractMatrix{T}, logPs::AbstractMatrix{T},
                  mrg::Vector{Vector{Int}}, mrgidxs::Vector{Vector{Int}}) where {T <: AbstractFloat}
    sz = map(length, mrgidxs)
    n = sum(sz)
    allidxs = vcat(mrgidxs...)
    NLL = maximum(view(logP′,allidxs,:), dims=2) |> sum
    s1 = max.(map(i-> minimum(view(-logPs,allidxs,i), dims=2), mrg)...)
    s2 = -log.(sz[2]./exp.(view(logP′,allidxs,1)) .+ sz[1]./exp.(view(logP′,allidxs,2)))
    return NLL - log(n) + maximum(s1 .- s2)
end
