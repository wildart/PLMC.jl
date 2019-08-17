posterior(Cidx::Int, Pidxs::Vector{Int}, mcr::ModelClusteringResult) =
    Cidx ∉ Pidxs ? 0.0 : length(clusters(mcr)[Cidx].idx)/sum(i->length(clusters(mcr)[i].idx), Pidxs)

"""
    posterior(Cidx::Int, Pidxs::Vector{Int}, mcr::ModelClusteringResult, X::AbstractMatrix) -> Float64

Calculate posterior of the clusrer `C` given the PLM cluster `P`, p(C|P).
Clusters are given by their respective indices in the model class presented as model-base clustering 'mcr'.
"""
function posterior(Cidx::Int, Pidxs::Vector{Int},
                   mcr::ModelClusteringResult,
                   X::AbstractMatrix)
    Ppts = findall(i-> i ∈ Pidxs, assignments(mcr))
    # Ppts = mapreduce(i->clusters(mcr)[i].idx, vcat, Pidxs)
    mcm = MixtureModel(mcr)
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
                mcr::ModelClusteringResult,
                X::AbstractMatrix; hard::Bool=false, mstate=[Int[]])
    k = nclusters(mcr)
    n = size(X,2)
    Cprior = map(c->c/n, counts(mcr))
    ibd = 0.0
    for C in 1:k
        # print("C: $C, ")
        p′post = hard ? posterior(C, vcat(P′...), mcr) : posterior(C, vcat(P′...), mcr, X)
        for P in P′
            # print("P: $P, ")
            ppost = hard ? posterior(C, P, mcr) : posterior(C, P, mcr, X)
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
