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

struct Agglomeration
    clusters::Vector{Vector{Vector{Int}}}
    mergers::Vector{Vector{Vector{Int}}}
    marks::Vector{Float64}
end
Base.show(io::IO, A::Agglomeration) = print(io, "Agglomeration of $(length(A.clusters)) mergers")

@recipe function f(plmc::T) where {T<:PLMCResultClusteringResult}
    χ = isinf(plmc.ϵ) ? 2.0 : plmc.ϵ
    for (i,idxs) in enumerate(plmc.clusters)
        for c in idxs
            @series begin
                label --> "MC$i"
                linecolor --> i
                plmc.manifolds[c], χ
            end
        end
    end
    if length(size(plmc.complex)) > 0
        D = hcat(map(m->m.mu, plmc.manifolds)...)'
        plmc.complex, D
    end
end
