"""Create Gaussian distribution from the Mahalanobis cluster parameters.
"""
MvNormal(mc::MahalonobisCluster) = MvNormal(mc.mu, mc.sigma)

"""Create Gaussian mixture model from the Mahalanobis clusters.
"""
function mixture(mcs::Vector{<:MahalonobisCluster})
    c = map(c->length(c.idx), mcs)
    MixtureModel(map(c -> MvNormal(c), mcs), c/sum(c))
end

function modelclass(mcs::MahalonobisClusteringResult,
                    clsidxs::Array{Array{Int64,1},1})
    mcl = clusters(mcs)
    return  [length(clidxs) == 1 ?
        MvNormal(mcl[clidxs][]) :
        mixture(mcl[clidxs]) for clidxs in clsidxs]
end

function show_vector(io, v::AbstractVector)
    print(io, "[")
    for (i, elt) in enumerate(v)
        i > 1 && print(io, ", ")
        print(io, elt)
    end
    print(io, "]")
end
