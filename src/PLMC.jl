module PLMC

using StatsBase
using Distributions
using Clustering
using ClusterComplex
using Random
using ComputationalHomology
using Optim: optimize

import Clustering: assignments, counts, nclusters
import ClusterComplex: models

export PLMClusteringResult, modelclass,
       refinedmdl, ibdiff, mdldiff,
       Agglomeration,
       agglomerate, agglomerateph, testmerges, plmc,
       #re-export,
       MvNormal, MixtureModel

include("types.jl")
include("mdl.jl")
include("ib.jl")
include("spectral.jl")
include("plots.jl")
include("utils.jl")

# Agglomerative Clustering

function show_vector(io, v::AbstractVector)
    print(io, "[")
    for (i, elt) in enumerate(v)
        i > 1 && print(io, ", ")
        print(io, elt)
    end
    print(io, "]")
end

function testmerges(scorefunc, mtree::Vector{Vector{Vector{Int}}},
                    mcr::ModelClusteringResult, X::AbstractMatrix)
    c = length(mtree)
    MDL = fill(Inf, c)
    LL = fill(Inf, c)
    Rₘₐₓ = fill(Inf, c)
    for i in 1:c
        MDL[i], LL[i], Rₘₐₓ[i] = scorefunc(mtree[i], mcr, X)
        @debug begin
            io = IOBuffer()
            show_vector(io, mtree[i])
            strcls = String(take!(io))
            "$i -> $strcls\nMDL = $(MDL[i])\nLL = $(LL[i])\nRₘₐₓ = $(Rₘₐₓ[i])"
        end
    end
    return MDL, LL, Rₘₐₓ
end
"""
    agglomerate(flt::Filtration) -> Agglomeration

Construct agglomerative pairwise merge tree from the filtration `flt`.
"""
function agglomerate(flt::Filtration; individual=true, kwargs...)
    mtree   = Vector{Vector{Int}}[]
    mergers = Vector{Vector{Int}}[]
    mvals   = valtype(flt)[]

    # use individual clusters as PLM clusters
    if individual
        insert!(mtree, 1, [[i] for i in 1:size(complex(flt),0)])
        push!(mvals, zero(eltype(mvals)))
    end
    cplx = complex(flt)

    ST = Vector{Int}[]
    for (v, splxs) in flt
        merged = false
        for (d,sid) in splxs
            simplex = cplx[sid, d]
            # join simplex points
            # println("$v => $simplex")
            pts = values(simplex)
            if length(pts) == 1
                # println("[] <= $(pts[1])")
                push!(ST, pts)
            else
                j = pts[1]
                jj = findfirst(a-> j ∈ a, ST)
                for k in pts[2:end]
                    kk = findfirst(a-> k ∈ a, ST)
                    # println("$j <= $k : $jj <= $kk")
                    jj == kk && continue
                    # println("$jj <= $kk : $(ST[jj]) <= $(ST[kk])")
                    push!(mergers, [deepcopy(ST[jj]), deepcopy(ST[kk])])
                    append!(ST[jj], ST[kk])
                    deleteat!(ST, kk)
                    merged = true
                end
            end
        end
        if merged
            push!(mtree, deepcopy(ST))
            push!(mvals, v)
            # println("$(mtree[end]) => $v")
        end
    end
    return Agglomeration(mtree, mergers, mvals)
end

"""
    agglomerate(flt::Filtration, mcr::ModelClusteringResult, X::AbstractMatrix) -> (PLMClusteringResult, Agglomeration)

Return agglomerative piecewise clustering from the filtration `flt` by finding a corrspondent minimum MDL value.
"""
function agglomerate(flt::Filtration, mcr::ModelClusteringResult,
                     X::AbstractMatrix; individual=true)
    agg = agglomerate(flt, individual=individual)
    return plmc(agg, mcr, X, filtration=flt), agg
end

function agglomerateph(flt::Filtration; individual=true)
    mtree   = Vector{Vector{Int}}[]
    mergers = Vector{Vector{Int}}[]
    mvals   = valtype(flt)[]

    # use individual clusters as PLM clusters
    if individual
        insert!(mtree, 1, [[i] for i in 1:size(complex(flt),0)])
        push!(mvals, zero(eltype(mvals)))
    end

    cplx = complex(flt)
    ph = persistenthomology(PersistentCocycleReduction{Float64}, flt)
    dgm = diagram(ph)
    dims = sort!(collect(keys(dgm)))
    fvals = values(ph)

    ST = Vector{Int}[]
    # i = 16
    for i in 1:length(fvals)-1
        cv, nv = fvals[i:i+1]
        itr = collect(Iterators.flatten(filter(x-> cv < death(x) <=nv, dgm[d]) for d in dims))
        gs = ComputationalHomology.generator.(itr)
        # g = gs[2]
        for g in gs
            ComputationalHomology.dim(g) > 0 && continue

            splxs = unique(Iterators.flatten(values(cplx[sid, 0]) for sid in keys(g)))
            mrg = deepcopy(mtree[end])
            mrgidx = map(c->any(s-> s ∈ c, splxs), mrg)
            push!(mergers, mrg[mrgidx])
            merged = vcat(mrg[mrgidx]...)
            deleteat!(mrg, mrgidx)
            append!(mrg, [merged])

            push!(mtree, mrg)
            push!(mvals, cv)
            @debug "Merging" at=mvals[end] state=mtree[end]' generator=g
        end
    end
    return Agglomeration(mtree, mergers, mvals)
end

"""
    agglomerate(flt::Filtration, mcr::ModelClusteringResult, X::AbstractMatrix) -> (PLMClusteringResult, Agglomeration)

Return agglomerative piecewise clustering from the filtration `flt` by finding a corrspondent minimum MDL value.
"""
function agglomerateph(flt::Filtration, mcr::ModelClusteringResult,
                       X::AbstractMatrix; individual=true)
    agg = agglomerateph(flt, individual=individual)
    return plmc(agg, mcr, X, filtration=flt), agg
end

function mergecost(nodes::Vector{Vector{Int}}, measure::Function,
                   mcr::ModelClusteringResult, X::AbstractMatrix, β::Int;
                   kwargs...)
    logps = hcat((logpdf(p, X) for p in models(mcr))...)
    assign = assignments(mcr)
    return mergecost(nodes, measure, logps, assign, β; kwargs...)
end

function mergecost(nodes::Vector{Vector{Int}}, measure::Function,
                   logps::AbstractMatrix{T}, assign::Vector{Int}, β::Int;
                   kwargs...) where {T <: AbstractFloat}
    n = length(nodes)
    # println("nodes: ", n)
    ds = [let nds = nodes[[i,j]]; nds=>measure(nds, logps, assign; kwargs...); end for i in 1:n, j in 1:n if i < j]
    dsi = sortperm(ds, by=last)
    mi = min(β, length(dsi))
    return ds[dsi[1:mi]]
end

"""
    agglomerate(measure::Function, mcr::ModelClusteringResult, X::AbstractMatrix; β=1) -> Agglomeration

Construct agglomerative pairwise clustering through the beam search using `measure` function. `β` is a length of the beam.
"""
function agglomerate(measure::Function, mcr::ModelClusteringResult,
                     X::AbstractMatrix{T}; kwargs...) where {T <: AbstractFloat}
    k = nclusters(mcr)
    args = Dict(kwargs...)
    β = get(args, :β, 1)
    βsize = get(args, :βsize, β)

    assign = assignments(mcr)
    priorC = map(c->c/length(assign), counts(mcr))
    logps = hcat((logpdf(p, X) for p in models(mcr))...)
    logpx = sum(exp.(logps' .+ log.(priorC)), dims=1)' |> vec

    # initialize beam storage
    beam = Agglomeration[]
    push!(beam, Agglomeration([[i] for i in 1:k]))
    bcount = β
    cls = last(first(beam))
    terminate = false
    while length(cls) > 1 && !terminate
        # fill beam structure with possible options
        while bcount > 0
            tmp = Agglomeration[]
            while length(beam) > 0
                agg = pop!(beam)
                mcosts = mergecost(last(agg), measure, logps, assign, β; marginal=logpx, kwargs...)
                for mrg in mcosts
                    @debug "Merging" at=mrg aggregation=agg
                    if isinf(mrg.second)
                        push!(tmp, agg)
                        terminate = true
                        break
                    else
                        newagg = deepcopy(agg)
                        push!(newagg, mrg)
                        push!(tmp, newagg)
                    end
                end
            end
            unique!(a->last(a), tmp)
            beam = tmp
            bcount -= 1
        end
        bcount = 1

        # prune beam structure
        idxs = sortperm(map(count, beam))
        mi = min(β^βsize, length(idxs))
        beam = beam[idxs[1:mi]]
        # map(a->a.clusters[end] => count(a), beam)
        cls = last(first(beam))
    end
    return first(beam)
end

"""
    plmc(agg::Agglomeration, mcr::ModelClusteringResult, X::AbstractMatrix) -> PLMClusteringResult

Construct a piecewise clustering from the agglomerative merge tree `agg` by finding a corrspondent minimum MDL value.
"""
function plmc(agg::Agglomeration, mcr::ModelClusteringResult,
              X::AbstractMatrix, filtration=nothing; kwargs...)
    args = Dict(kwargs...)
    scorefunc = get(args, :score, refinedmdl)
    findscore = get(args, :find, findglobalmin)

    MDL, _ = testmerges(scorefunc, agg.clusters, mcr, X)
    mdlidx = findscore(MDL)

    ϵ = Inf
    scplx = SimplicialComplex()
    if filtration !== nothing
        ϵ = agg.costs[mdlidx]
        scplx = complex(filtration, ϵ)
    end
    return PLMClusteringResult(mcr, agg.clusters[mdlidx], scplx, ϵ)
end

"""
    plmc(AlgorithmType, flt::Filtration, mcr::ModelClusteringResult, X::AbstractMatrix) -> PLMClusteringResult

Construct a piecewise clustering of the dataset `X` using `AlgorithmType` algorithm given the cover represented by a filtration `flt` and a mode-based clustering `mcr`.
"""
function plmc(::Type{Topological}, X::AbstractMatrix, flt::Filtration,
              mcr::ModelClusteringResult; kwargs...)
    agg = agglomerate(flt; kwargs...)
    plmc(agg, mcr, X, flt; kwargs...)
end

function plmc(::Type{PHomology}, X::AbstractMatrix, flt::Filtration,
              mcr::ModelClusteringResult; kwargs...)
    agg = agglomerateph(flt; kwargs...)
    plmc(agg, mcr, X, flt; kwargs...)
end

function plmc(::Type{MDL}, X::AbstractMatrix, flt::Filtration,
              mcr::ModelClusteringResult; kwargs...)
    agg = agglomerate(mdldiff, mcr, X; kwargs...)
    plmc(agg, mcr, X; kwargs...)
end

function plmc(::Type{InformationBottleneck}, X::AbstractMatrix, flt::Filtration,
              mcr::ModelClusteringResult; kwargs...)
    agg = agglomerate(ibdiff, mcr, X; kwargs...)
    plmc(agg, mcr, X; kwargs...)
end

end
