"""PLMC partitioning
"""
struct PLMCResultClusteringResult <: ClusteringResult
    manifolds::Vector{MahalonobisCluster}
    clusters::Vector{Vector{Int}}
    complex::SimplicialComplex
    ϵ::Number
end
function Base.show(io::IO, R::PLMCResultClusteringResult)
    print(io, "PL Manifold Clustering (clusters = $(nclusters(R)), ϵ = $(R.ϵ))")
end

# Clustering.ClusteringResult interface
nclusters(R::PLMCResultClusteringResult) = length(R.clusters)
counts(R::PLMCResultClusteringResult) = [ sum( length(R.manifolds[c].idx) for c in cls ) for cls in R.clusters ]
function assignments(R::PLMCResultClusteringResult)
    tot = sum(map(c->length(c.idx), R.manifolds))
    asgn = zeros(Int, tot)
    for (i, cls) in enumerate(R.clusters)
        for c in cls
            asgn[R.manifolds[c].idx] .= i
        end
    end
    return asgn
end
