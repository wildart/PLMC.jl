import Distributions: ContinuousMultivariateDistribution, MixtureModel, estimate, logpdf
import MultivariateStats: MultivariateStats, fit
import Clustering: Clustering, assignments, counts, nclusters
import LMCLUS
import LinearManifoldModels: NonparametricEstimator, EmpiricalLinearManifold
import KernelDensity
import StatsBase

const CMDist = ContinuousMultivariateDistribution
const MINP = 1e-300
const MAXLOGL = 100000 #Inf #100000 # -log(MINP)

##########################
# Redundancy calculation #
##########################

function redundancy(p̄::M1, p::M2, x::AbstractVector{T};
                    cap=false) where {M1 <: CMDist, M2 <: CMDist, T <: Real}
    lp̄ = -logpdf(p̄, x)
    lp = -logpdf(p, x)
    return (cap ? min(MAXLOGL, lp̄) : lp̄) - (cap ? min(MAXLOGL, lp) : lp)
end

function redundancy(p̄::M1, p::M2, X::AbstractMatrix{T};
                    cap=false) where {M1 <: CMDist, M2 <: CMDist, T <: Real}
    lp̄ = -logpdf(p̄, X)
    lp = -logpdf(p, X)
    return (cap ? min.(MAXLOGL, lp̄) : lp̄) .- (cap ? min.(MAXLOGL, lp) : lp)
end

######################
# Regret calculation #
######################

function regret(p̄::D1, M::Vector{D2}, x::AbstractVector{T};
                cap=false) where {D1 <: CMDist, D2 <: CMDist, T <: Real}
    lp̄ = -logpdf(p̄, x)
    lp = minimum.(-logpdf(p, x) for p in M)
    return (cap ? min(MAXLOGL, lp̄) : lp̄) - (cap ? min(MAXLOGL, lp) : lp)
end

function regret(p̄::D1, M::Vector{D2}, X::AbstractMatrix{T};
                cap=false) where {D1 <: CMDist, D2 <: CMDist, T <: Real}
    lp̄ = -logpdf(p̄, X)
    lp = mapslices(minimum, hcat([-logpdf(p, X) for p in M]...), 2)
    return (cap ? min.(MAXLOGL, lp̄) : lp̄) .- (cap ? min.(MAXLOGL, lp) : lp)
end

######################
# Model construction #
######################

"""Model-based clustering results
"""
struct ModelResult <: Clustering.ClusteringResult
    models::Vector{CMDist}
    assignments::Vector{Int}
end
assignments(clust::ModelResult) = clust.assignments
nclusters(clust::ModelResult) = length(clust.models)
counts(clust::ModelResult) = map(i->count(clust.assignments .== i), 1:nclusters(clust))

function clusteringmodel(clust::ModelResult, p::AbstractArray{Int} = 1:nclusters(clust))
    cnts = counts(clust)
    return MixtureModel(clust.models[p], cnts[p]./sum(cnts[p]))
end

regret(p̄::D, M::ModelResult, X; cap=false) where {D <: CMDist} = regret(p̄, M.models, X, cap=cap)

function model(res::C, X::AbstractMatrix{T}, MT::Symbol, Q=1,
               parts::Vector{Vector{Int}} = [[i] for i in 1:nclusters(res)] ) where {C <: Clustering.ClusteringResult, T <: Real}
    DS = CMDist[]
    A = Clustering.assignments(res)
    Anew = similar(A)
    for (n,p) in enumerate(parts)
        ll = find(e->e ∈ convert(Vector{Int}, p), A)
        Anew[ll] = n
        dist = if MT == :FA
            fa = fit(MultivariateStats.FactorAnalysis, X[:,ll], maxoutdim=Q, method=:em)
            Distributions.MvNormal(mean(fa), diagm(var(fa)))
        elseif MT == :KDE && isa(res, LMCLUS.LMCLUSResult)
            QT = KernelDensity.UnivariateKDE
            estimate(NonparametricEstimator(EmpiricalLinearManifold, QT), LMCLUS.manifold(res,i), X[:,ll], nquants=Q)
        elseif MT == :HIST && isa(res, LMCLUS.LMCLUSResult)
            QT = StatsBase.Histogram
            estimate(NonparametricEstimator(EmpiricalLinearManifold, QT), LMCLUS.manifold(res,i), X[:,ll], nquants=Q)
        else
            error("Unsupported model type $MT for $C")
        end
        push!(DS, dist)
    end
    return ModelResult(convert(Vector{CMDist}, DS), Anew)
end
