""" Convert the model-based clustering `mcr` into a mixture model. """
function MixtureModel(mcr::ModelClusteringResult,
                      p::Vector{Int} = collect(1:nclusters(mcr)))
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
    marks::Vector{Float64}
end
Base.show(io::IO, A::Agglomeration) = print(io, "Agglomeration of $(length(A.clusters)) mergers")


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

@recipe function f(R::T) where {T<:PLMClusteringResult}
    χ = isinf(R.ϵ) ? 2.0 : R.ϵ
    for (i,idxs) in enumerate(R.clusters)
        addlabel = true
        for c in idxs
            @series begin
                label --> (addlabel ? "MC$i" : "")
                linecolor --> i
                models(R)[c], χ
            end
            addlabel = false
        end
    end
    if length(size(R.complex)) > 0
        D = hcat(map(mean, models(R))...)'
        R.complex, D
    end
end
