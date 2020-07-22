""" Abstract type for metaclustering techniques """
abstract type AbstarctMetaClustering end

struct Topological <: AbstarctMetaClustering end
struct PHomology <: AbstarctMetaClustering end
struct Spectral <: AbstarctMetaClustering end
struct MDL <: AbstarctMetaClustering end
struct InformationBottleneck <: AbstarctMetaClustering end

""" Convert the model-based clustering `mcr` into a mixture model. """
function Distributions.MixtureModel(mcr::ModelClusteringResult,
                                    p::AbstractVector{Int} = 1:nclusters(mcr))
    cnts = counts(mcr)
    return MixtureModel(mcr.models[p], cnts[p]./sum(cnts[p]))
end

""" Create a model class from the model-based clustering `mcr`. """
function modelclass(mcr::ModelClusteringResult,
                    clsidxs::Array{Array{Int64,1},1})
    return  [
        length(clidxs) == 1 ?
        mcr.models[clidxs][] :
        MixtureModel(mcr, clidxs) for clidxs in clsidxs
    ]
end

struct Agglomeration
    clusters::Vector{Vector{Vector{Int}}}
    mergers::Vector{Vector{Vector{Int}}}
    costs::Vector{Float64}
end
Agglomeration(clusters::Vector{Vector{Int}}) = Agglomeration([clusters], [Vector{Int}[]], [0.0])
Base.count(agg::Agglomeration) = sum(agg.costs)
Base.length(agg::Agglomeration) = length(agg.clusters)
Base.show(io::IO, agg::Agglomeration) = print(io, "Agglomeration of $(length(agg.clusters)) mergers")
Base.last(agg::Agglomeration) = last(agg.clusters)
function Base.push!(agg::Agglomeration, jn::Pair{Vector{Vector{Int}},Float64})
    # get last merge state
    laststate = agg.clusters[end]
    # find indexes of merging clusters
    idxs = findall(a-> a ∈ first(jn), laststate)
    newstate = deepcopy(laststate)
    # merge clusters
    i = idxs[1]
    for j in idxs[2:end]
        append!(newstate[i], newstate[j])
        deleteat!(newstate, j)
    end
    # update agglomeration
    push!(agg.clusters, newstate)
    push!(agg.mergers, first(jn))
    push!(agg.costs,   last(jn))
    return agg
end


"""PLMC partitioning
"""
struct PLMClusteringResult <: ClusteringResult
    models::ModelClusteringResult
    clusters::Vector{Vector{Int}}
    complex::SimplicialComplex
    ϵ::Number
end
Base.show(io::IO, R::PLMClusteringResult) =
    print(io, "PL Manifold Clustering (clusters = $(nclusters(R)), ϵ = $(R.ϵ))")

# Clustering.ClusteringResult interface
nclusters(R::PLMClusteringResult) = length(R.clusters)
function counts(R::PLMClusteringResult)
    massign = assignments(R.models)
    return [ sum( count(massign .== ci) for ci in cls ) for cls in R.clusters ]
end
function assignments(R::PLMClusteringResult)
    massign = assignments(R.models)
    assgn = similar(massign)
    for (i, cls) in enumerate(R.clusters)
        for ci in cls
            assgn[massign .== ci] .= i
        end
    end
    return assgn
end
models(R::PLMClusteringResult) = models(R.models)
