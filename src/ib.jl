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

posterior(Cidx::Int, Pidxs::Vector{Int}, cnt::Vector{Int}) =
    Cidx ∉ Pidxs ? 0.0 : cnt[Cidx]/sum(i->cnt[i], Pidxs)

function posterior(Cidx::Int, priorC::AbstractVector{T}, logCs::AbstractMatrix{T},
                   mgnl::AbstractVector{T}, idxs::Vector{Bool}) where {T <: AbstractFloat}
    n = sum(idxs)
    logCP = view(logCs, idxs, :)
    priorC[Cidx]*sum(exp.(view(logCP,:,Cidx))./view(mgnl,idxs))/n
end

function ibdiff(P′::Vector{Vector{Int}},
                mcr::ModelClusteringResult,
                X::AbstractMatrix; kwargs...)
    logps = hcat((logpdf(p, X) for p in models(mcr))...)
    return ibdiff(P′, logps, assignments(mcr); kwargs...)
end

function ibdiff(P′::Vector{Vector{Int}}, logCs::AbstractMatrix{T},
                assign::Vector{Int}; hard=true,
                marginal::Union{AbstractVector{T}, Nothing}=nothing,
                kwargs...)  where {T <: AbstractFloat}
    # precompute distributions
    n, k = size(logCs)
    cs = zeros(Int, k)
    for i in assign
        cs[i] += 1
    end
    pC = map(c->c/n, cs)
    pP′ = sum(sum(pC[P]) for P in P′)
    if marginal === nothing && !hard
        marginal = sum(exp.(logCs' .+ log.(pC)), dims=1)'
    end
    # calculate IB difference
    P′idxs = vcat(P′...)
    ibd = 0.0
    for C in 1:k
        pCP′ = if hard
            posterior(C, P′idxs, cs)
        else
            P′pts = map(i-> i ∈ P′idxs, assign)
            posterior(C, pC, logCs, marginal, P′pts)
        end
        Dₖₗ2 = pCP′ == 0.0 ? 0.0 : pCP′*log(pCP′/pP′)
        for P in P′
            pP  = sum(pC[P])
            pCP = if hard
                posterior(C, P, cs)
            else
                Ppts = map(i-> i ∈ P, assign)
                posterior(C, pC, logCs, marginal, Ppts)
            end
            Dₖₗ1 = pCP == 0.0 ? 0.0 : pCP*log(pCP/pP)
            ibd += pC[C]*(Dₖₗ1 - Dₖₗ2)
        end
        # @debug "IB Difference" cluster=C IB=ibd
    end
    return ibd
end
