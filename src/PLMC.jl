module PLMC

using StatsBase
using Distributions
using LinearAlgebra, SparseArrays, Arpack
using Clustering
using ClusterComplex
using Combinatorics
using Random
using ComputationalHomology

import Distributions: MvNormal
import Clustering: assignments, counts, nclusters

export MvNormal, PLMCResultClusteringResult,
       assignments, counts, nclusters,
       refinedmdl, ibdiff, mdldiff,
       agglomerate, testmerges, plmc

include("utils.jl")
include("types.jl")
include("mdl.jl")
include("ib.jl")
include("spectral.jl")
# include("model.jl")

# Agglomerative Clustering

function testmerges(mtree::Vector{Vector{Vector{Int}}},
                    mcr::MahalonobisClusteringResult,
                    X::AbstractMatrix)
    c = length(mtree)
    MDL = fill(0.0, c)
    LL = fill(0.0, c)
    Rₘₐₓ = fill(0.0, c)
    for i in 1:c
        strcls = let io = IOBuffer()
            show_vector(io, mtree[i])
            String(take!(io))
        end
        MDL[i], LL[i], Rₘₐₓ[i] = refinedmdl(mtree[i], mcr, X)
        # print("$i -> ", mtree[i], "=> ")
        # println("MDL: ", MDL[i], ", LL: ", LL[i], ", Rₘₐₓ: ", Rₘₐₓ[i])
        @debug "$i -> $strcls" MDL=MDL[i] LL=LL[i] Rₘₐₓ=Rₘₐₓ[i]
    end
    return MDL, LL, Rₘₐₓ
end

## Construct merging tree from the filtration complex
"""
    agglomerate(flt::Filtration) -> (Vector{Vector{Vector{Int}}}, Vector{Vector{Vector{Int}}})

Construct agglomerative pairwise clustering from the filtration `flt`.
"""
function agglomerate(flt::Filtration; individual=true)
    mtree   = Vector{Vector{Int}}[]
    mergers = Vector{Vector{Int}}[]
    mvals   = valtype(flt)[]

    # use individual clusters as PLM clusters
    if individual
        insert!(mtree, 1, [[i] for i in 1:size(complex(flt),0)])
        push!(mvals, zero(eltype(mvals)))
    end

    ST = Vector{Int}[]
    for (v, splxs) in ComputationalHomology.simplices(flt)
        merged = false
        for simplex in splxs
            # join simplex points
            println("$v => $simplex")
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
            println("$(mtree[end]) => $v")
        end
    end
    return Agglomeration(mtree, mergers, mvals)
end

"""
    agglomerate(measure::Function, mcr::MahalonobisClusteringResult, X::AbstractMatrix) -> (Vector{Vector{Vector{Int}}}, Vector{Vector{Vector{Int}}})

Construct agglomerative pairwise clustering from the filtration `flt`.
"""
function agglomerate(measure::Function,
                     mcr::MahalonobisClusteringResult,
                     X::AbstractMatrix)
    mtree   = Vector{Vector{Int}}[]
    mergers = Vector{Vector{Int}}[]
    mvals   = Float64[]
    k = nclusters(mcr)

    insert!(mtree, 1, [[i] for i in 1:k])

    while length(mtree[end]) > 1
        ds = [P′=>measure(P′, mcr, X) for P′ in combinations(mtree[end], 2)]
        dsi = sortperm(ds, by=last)
        minds = ds[dsi[1]]
        println("Merged: ", minds)
        push!(mergers, first(minds))

        i, j = findall(a-> a ∈ first(minds), mtree[end])
        ST = deepcopy(mtree[end])
        append!(ST[i], ST[j])
        deleteat!(ST, j)
        push!(mtree, ST)
        println(ST)
    end

    return Agglomeration(mtree, mergers, mvals)
end

function plmc(agg::Agglomeration,
              mcr::MahalonobisClusteringResult,
              X::AbstractMatrix; filtration=nothing)
    MDL, _ = testmerges(agg.clusters, mcr, X)
    mdlval, mdlidx = findmin(MDL)
    ϵ = Inf
    scplx = SimplicialComplex()
    if filtration !== nothing
        ϵ = agg.marks[mdlidx]
        scplx = complex(filtration, ϵ)
    end
    return PLMCResultClusteringResult(clusters(mcr), agg.clusters[mdlidx], scplx, ϵ)
end

end
