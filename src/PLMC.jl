module PLMC

import Combinatorics
import Distances
import Distributions
import MultivariateStats: MultivariateStats, fit, outdim, indim, projection
import LMCLUS: LMCLUS, outdim, indim, projection, manifolds, points
import Clustering: Clustering, assignments, counts, nclusters
import ComputationalHomology

export ptm_dist, partition, lmcc, plmclus

include("utils.jl")
include("result.jl")
include("mdl.jl")
include("spectral.jl")
include("model.jl")

"""Partition data using LMCLUS
"""
function partition(X::AbstractMatrix, params::LMCLUS.Parameters;
                   refine=false, refinetol=10, dropnoise=nothing,
                   display = :iter, method=:lmclus)
    # set log level
    loglev = Clustering.display_level(display)
    # Generate clusters
    clust = if method == :lmclus
        LMCLUS.loglevel!(loglev) # set logging level
        LMCLUS.lmclus(X, params)
    elseif method == :kmeans
        srand(params.random_seed)
        res = Clustering.kmeans(X, params.number_of_clusters, display=display)
        ms = manifolds(res, X)
        LMCLUS.LMCLUSResult(ms, LMCLUS.Separation[])
    else
        error("Uknown partition method: $method")
    end

    # Perform LMCLUS clustering refinement
    if method == :lmclus
        ms = manifolds(clust)

        # fix basis of noise manifold
        M = ms[end]
        if indim(M) == 0 && dropnoise !== nothing && !dropnoise
            R = fit(MultivariateStats.PCA, X[:, points(M)]; maxoutdim = 1)
            M.d = outdim(R)
            M.μ = mean(R)
            M.proj = projection(R)
        end

        # Refine clusters
        dfun = (X,m)->first(ptm_dist(X, m))
        if refine
            efun = (X,m) -> LMCLUS.MDL.calculate(params.mdl_algo, m, X, params.mdl_model_precision, params.mdl_data_precision)
            clust = LMCLUS.refine(clust, X, dfun, efun, debug=(loglev>0), tol=refinetol, drop_last=dropnoise, min_cluster_size=params.min_cluster_size)
        else
            if dropnoise === nothing
                clust = LMCLUS.clearoutliers(clust, X, dfun)
            else
                dropnoise && pop!(ms)
            end
        end
    end

    return clust
end


"""Construct filtration from the linear manifold clustering
"""
function lmcc(ms::Vector{LMCLUS.Manifold}, X::AbstractMatrix{Float64}, ϵ::Float64; maxdim = 3, ν=0, debug = false)
    D = hcat(map(m->first(ptm_dist(X, m)), ms)...)
    cplx, w = ComputationalHomology.witness(D', ϵ, ν=ν, maxdim=maxdim)
    return ComputationalHomology.filtration(cplx, w)
end
lmcc(CLS::LMCLUS.LMCLUSResult, X::Matrix{Float64}, ϵ::Float64; kwargs...) = lmcc(manifolds(CLS), X, ϵ; kwargs...)
lmcc(CLS::Clustering.KmeansResult, X::Matrix{Float64}, ϵ::Float64; kwargs...) = lmcc(manifolds(CLS, X), X, ϵ; kwargs...)


"""Perform piecewise linear manifold clustering
"""
function plmclus(X::AbstractMatrix{T}, params::LMCLUS.Parameters;
                 display = :iter, ϵ::Real = 10.0, ν=0,
                 partitioning=:lmclus, refine=true, refinetol=10, dropnoise=nothing,
                 clustering=:mdl, similarity=:adjacency, k=0, ɛ::Real=1e-2,
                 modeltype=:NONE, q=1, perf=:regret) where T<:AbstractFloat
    # set log level
    loglev = Clustering.display_level(display)
    # Partition dataset on LM clusters
    clust = partition(X, params; method=partitioning, refine=refine, refinetol=refinetol, dropnoise=dropnoise, display=display)
    ms = manifolds(clust)

    # Construct filtration from LM clusters
    D = hcat(map(m->first(ptm_dist(X, m)), ms)...)
    # cplx, w = ComputationalHomology.witness(D', ϵ, ν=ν, maxdim=1)
    cplx, w = ComputationalHomology.witness(D', maximum(D), ν=ν, maxdim=1)
    flt = ComputationalHomology.filtration(cplx, w)

    # Find in maximal (finite) filtration value of zero persistent homology group (corresponds to a connected component)
    itr = ComputationalHomology.intervals(flt)[0]
    ϵ′ = mapreduce(last, (v0,v)-> max(v0, isinf(v)? -v : v), itr)

    # Find complex to corresponding filtration value
    loglev > 0 && println("Find largest connected component: ")
    ccplx = last(first(flt))
    for (v, cplx) in flt
        loglev > 1 && println("\t$v => $cplx")
        if v > ϵ′
            ccplx = cplx
            break
        end
    end
    loglev > 0 && println("Largest: $v => $cplx")

    # Perform piecewise linear clustering
    plmclust, plmdlvals = if clustering == :mdl
        srand(params.random_seed)
        plmclus(X, ms, flt, params.mdl_algo, params.mdl_model_precision, params.mdl_data_precision, debug=(loglev > 0))
    else
        # create similarity matrix
        A = if similarity == :adjacency
            adjacency_matrix(ccplx, Float64)
        elseif similarity == :ptmdist
            f = ComputationalHomology.Filtration(ccplx, ComputationalHomology.order(flt))
            similarity_matrix(f)
        end

        # construct clustering configurations
        conf = if clustering == :mcl
            srand(params.random_seed)
            sclust = Clustering.mcl(A, display=(loglev > 2 ? :verbose : :none))
            Vector{Vector{Int}}[[find(assignments(sclust) .== i) for i in 1:nclusters(sclust)]]
        elseif clustering == :every
            n = nclusters(clust)
            # 1:sum( length(Combinatorics.partitions(1:n, i)) for i in 1:n )
            vcat([collect(Combinatorics.partitions(1:n, i)) for i in 1:n]...)
        elseif k == 1
            Vector{Vector{Int}}[[collect(1:nclusters(clust))]]
        elseif k > 1
            srand(params.random_seed)
            sclust = spectralclust(A, k, display=display, init=:rand)
            Vector{Vector{Int}}[[find(assignments(sclust) .== i) for i in 1:nclusters(sclust)]]
        else
            conf = Vector{Vector{Int}}[[collect(1:nclusters(clust))]]
            for i in 2:nclusters(clust)-1
                srand(params.random_seed)
                sclust = spectralclust(A, i, display=display, init=:rand)
                cls = [find(assignments(sclust) .== i) for i in 1:nclusters(sclust)]
                push!(conf, cls)
                loglev > 1 && println("Π($i): $cls")
            end
            unique(conf)
        end

        loglev > 0 && println("\nConfigurations:")
        # convert partition to set of model
        mclust = modeltype == :NONE ? clust : model(clust, X, modeltype, q)
        # calculate MDL or RED values for each configurations
        mdls = Pair{Number,PLMCResult}[]
        for c in conf
            R = PLMCResult(ms, c, ccplx, ϵ′)
            COMP = if modeltype == :NONE
                plmdl3(R, X, params.mdl_algo, params.mdl_model_precision, params.mdl_data_precision, ɛ=ɛ, debug=(loglev > 1)) |> log2
            else
                plmclust = model(clust, X, modeltype, q, R.clusters)
                if perf == :Kregret
                    # calculate maximal regret
                    maximum(regret(clusteringmodel(plmclust), mclust, X))
                elseif perf == :Kredundancy
                    # calculate maximal redundancy
                    maximum(redundancy(clusteringmodel(plmclust), clusteringmodel(mclust), X))
                elseif perf == :Γredundancy
                    a = assignments(plmclust)
                    Km = clusteringmodel(mclust)
                    sum(maximum(redundancy(plmclust.models[i], Km, X[:,find(a .== i)])) for i in 1:nclusters(plmclust))
                elseif perf == :Γredundancy2
                    Km = PLMC.clusteringmodel(mclust)
                    a = assignments(R)
                    sum(maximum(PLMC.redundancy(PLMC.clusteringmodel(mclust, R.clusters[i]), Km, X[:,find(a .== i)])) for i in 1:nclusters(R))
                else
                    error("Uknown performance measure")
                end
            end
            n = nclusters(clust)
            # Lm = sum( l*log(m) for (l,m) in zip(map(length, R.clusters), map(maximum, R.clusters)) )
            # Lm = sum( l*log(n) for l in map(length, R.clusters) )
            # K size + sum(Γ_i size)
            Lm = length(R.clusters)*log2(n) + sum( l*log2(n) for l in map(length, R.clusters) )
            loglev > 0 && println("$(R.clusters) => $COMP + Lm ($Lm)\n")
            push!(mdls, (modeltype == :NONE ? COMP : COMP+Lm)=>R)
        end

        # find optimal value
        # si = modeltype == :NONE || k > 0 ? 1 : 2
        # vals = collect( first(v) for v in mdls[si:end] )
        vals = collect( first(v) for v in mdls )
        v, i = findmin(vals)
        # i = findlast(x->x == v, vals) # select larger clustering if MDL is the same
        # last(mdls[si:end][i]), vals
        last(mdls[i]), vals
    end

    return plmclust, plmdlvals
end
plmclus(X::AbstractMatrix{Float64}) = plmclus(X, LMCLUS.Parameters(size(X,1)-1); kwargs...)

end
