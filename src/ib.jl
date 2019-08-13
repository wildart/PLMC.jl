posterior(Cidx::Int, Pidxs::Vector{Int}, mc::ClusterComplex.MahalonobisClusteringResult) =
    Cidx ∉ Pidxs ? 0.0 : length(mc.clusters[Cidx].idx)/sum(i->length(mc.clusters[i].idx), Pidxs)

function posterior(Cidx::Int, Pidxs::Vector{Int},
                   mc::ClusterComplex.MahalonobisClusteringResult,
                   X::AbstractMatrix)
    Ppts = mapreduce(i->mc.clusters[i].idx, vcat, Pidxs)
    mcm = mixture(ClusterComplex.clusters(mc))
    Cdist = components(mcm)[Cidx]
    pc = probs(mcm)[Cidx]
    return sum(i->
        let x = X[:,i]
            pcx = pdf(Cdist, x)
            px = pdf(mcm, x)
            pcx*pc/px
        end, Ppts)/length(Ppts)
end

function ibdiff(P′::Vector{Vector{Int}},
                mc::ClusterComplex.MahalonobisClusteringResult,
                X::AbstractMatrix; hard::Bool=false, mstate=[Int[]])
    k = nclusters(mc)
    n = size(X,2)
    Cprior = map(c->length(c.idx)/n, clusters(mc))
    ibd = 0.0
    for C in 1:k
        # print("C: $C, ")
        p′post = hard ? posterior(C, vcat(P′...), mc) : posterior(C, vcat(P′...), mc, X)
        for P in P′
            # print("P: $P, ")
            ppost = hard ? posterior(C, P, mc) : posterior(C, P, mc, X)
            # print("($ppost, $p′post) ")
            Dₖₗ = ppost*log(ppost/p′post)
            if ppost == 0
                Dₖₗ = 0.0
            end
            # print("Dₖₗ: $Dₖₗ, ")
            ibd += Cprior[C]*Dₖₗ
            # print("IB: $ibd, ")
        end
        # println()
    end
    return ibd
end
