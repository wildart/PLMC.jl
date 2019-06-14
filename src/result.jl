"""PLMC partitioning
"""
struct PLMCResult <: Clustering.ClusteringResult
    manifolds::Vector{LMCLUS.Manifold}
    clusters::Vector{Vector{Int}}
    complex::ComputationalHomology.SimplicialComplex
    ϵ::Number
end
function Base.show(io::IO, R::PLMCResult)
    print(io, "PL Manifold Clustering (clusters = $(nclusters(R)), ϵ = $(R.ϵ))")
end

# Clustering.ClusteringResult interface
nclusters(R::PLMCResult) = length(R.clusters)
counts(R::PLMCResult) = [ sum( LMCLUS.outdim(R.manifolds[c]) for c in cls ) for cls in R.clusters ]
function assignments(R::PLMCResult)
    tot = sum(map(LMCLUS.outdim, R.manifolds))
    asgn = zeros(Int, tot)
    for (i, cls) in enumerate(R.clusters)
        for c in cls
            asgn[LMCLUS.points(R.manifolds[c])] = i
        end
    end
    return asgn
end

# Misc
function LMCLUS.manifolds(cls::Clustering.KmeansResult, data::AbstractMatrix; bounds = false)
    M⁺ = LMCLUS.Manifold[]
    A = assignments(cls)
    for i in 1:nclusters(cls)
        m = LMCLUS.Manifold(1, zeros(1), zeros(1, 0), findall(A.==i))
        LMCLUS.adjustbasis!(m, data)
        if bounds
            m.θ = maximum(LMCLUS.distance_to_manifold(data, m))
            m.σ = maximum(LMCLUS.distance_to_manifold(data, m, ocss=true))
        end
        push!(M⁺, m)
    end
    return M⁺
end
