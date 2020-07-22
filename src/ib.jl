function posterior(Cidx::Int, Pidxs::Vector{Int}, mcr::ModelClusteringResult)
    cnt = counts(mcr)
    Cidx ∉ Pidxs ? 0.0 : cnt[Cidx]/sum(i->cnt[i], Pidxs)
end

"""
    posterior(Cidx::Int, Pidxs::Vector{Int}, mcr::ModelClusteringResult, X::AbstractMatrix) -> Float64

Calculate posterior of the clusrer `C` given the PLM cluster `P`, p(C|P).
Clusters are given by their respective indices in the model class presented as model-base clustering 'mcr'.
"""
function posterior(Cidx::Int, Pidxs::Vector{Int},
                   mcr::ModelClusteringResult,
                   X::AbstractMatrix,
                   marginal::MixtureModel)
    Ppts = findall(i-> i ∈ Pidxs, assignments(mcr))
    # Ppts = mapreduce(i->clusters(mcr)[i].idx, vcat, Pidxs)
    Cdist = components(marginal)[Cidx]
    pc = probs(marginal)[Cidx]
    return sum(i->
        let x = X[:,i]
            pcx = pdf(Cdist, x)
            px = pdf(marginal, x)
            pcx*pc/px
        end, Ppts)/length(Ppts)
end

function posterior(Cidx::Int, marginal::MixtureModel, X::AbstractMatrix)
    pn = size(X,2)
    Cdist = components(marginal)[Cidx]
    pc = probs(marginal)[Cidx]
    pc*sum(pdf(Cdist, X) ./ pdf(marginal, X))/pn
end

function ibdiff(P′::Vector{Vector{Int}},
                mcr::ModelClusteringResult,
                X::AbstractMatrix; hard::Bool=true, kwargs...)
    k = nclusters(mcr)
    n = size(X,2)
    cnts = counts(mcr)
    pC = map(c->c/n, cnts)
    marginal = PLMC.MixtureModel(mcr)
    ibd = 0.0
    assign = assignments(mcr)
    P′idxs = vcat(P′...)
    pP′ = sum(sum(pC[P]) for P in P′)
    for C in 1:k
        pCP′ = if hard
            posterior(C, P′idxs, mcr)
        else
            P′pts = map(i-> i ∈ P′idxs, assign)
            PLMC.posterior(C, marginal, view(X,:,P′pts))
        end
        Dₖₗ2 = pCP′ == 0.0 ? 0.0 : pCP′*log(pCP′/pP′)
        for P in P′
            pP  = sum(pC[P])
            pCP = if hard
                posterior(C, P, mcr)
            else
                Ppts = map(i-> i ∈ P, assign)
                PLMC.posterior(C, marginal, view(X,:,Ppts))
            end
            Dₖₗ1 = pCP == 0.0 ? 0.0 : pCP*log(pCP/pP)
            ibd += pC[C]*(Dₖₗ1 - Dₖₗ2)
        end
        @debug "IB Difference" cluster=C IB=ibd
    end
    return ibd
end
