using LinearAlgebra
using SparseArrays
using Arpack

"""Laplacian of an adjacency matrix `A`"""
laplacian(A::AbstractSparseMatrix) = spdiagm(0=>vec(sum(A, dims=2))) - A

"""Normalized laplacian of an adjacency matrix `A`"""
function normalized_laplacian(A::AbstractSparseMatrix)
    D = spdiagm(0=>1.0./sqrt.(vec(sum(A, dims=2))))
    return sparse(I, size(D)) - D*A*D
end

"""Spectral clustering results
"""
struct SpectralResult <: ClusteringResult
    res::KmeansResult
    λ::AbstractVector{<:Real}
end
assignments(clust::SpectralResult) = assignments(clust.res)
nclusters(clust::SpectralResult) = nclusters(clust.res)
counts(clust::SpectralResult) = counts(clust.res)

"""Spectral clustering using of square similarity (adjacency) matrix

Accepts sparse matrix `M` and `k` number of expected clusters as parϵ = Infameters.
Parameter `method` specifies clustering algorithm:

- `:njw` for "On Spectral Clustering: Analysis and an algorithm" by Nj, Jordan, Weiss (2002) (default)
- `:sm` for "Normalized Cuts and Image Segmentation" by Shi and Malik (2000)
"""
function spectralclust(A::AbstractSparseMatrix, k::Int; method=:njw, init=:kmpp)
    @assert size(A,1) == size(A,2) "Matrix should be square"
    @assert size(A,1) > k "Number of clusters cannot be more then observations"
    @assert k > 1 "Number of clusters should be more then one"

    if method == :sm
        L = laplacian(A)
        D = Diagonal(L)
        λ, ϕ = eigs(L, D, which=:SR, nev=min(2*k, size(A,1)-1))
        idxs = findall(real(λ).> eps())
        ϕ = ϕ[:,idxs]
        λ = λ[idxs]
    elseif method == :njw
        L = normalized_laplacian(A)
        kk = min(k+1, size(A,1)-1)
        λ, ϕ = eigs(L, which=:SR, nev=kk)
        k = min(length(λ), kk-1)
        ϕ = ϕ[:,1:k]
        λ = λ[1:k]
    else
        throw(ArgumentError("Invalid method `$method`"))
    end

    # normalize
    ϕ = real(ϕ)
    ϕ ./= mapslices(norm,ϕ,dims=2)

    KM = kmeans(ϕ', k, init=init)
    return SpectralResult(KM, real(λ))
end
spectralclust(cplx::AbstractComplex, k::Int; kwargs...) =
    spectralclust(ComputationalHomology.adjacency_matrix(cplx), k; kwargs...)

function plmc(::Type{Spectral},
              flt::Filtration,
              mcr::ModelClusteringResult;
              k = round(Int, sqrt(size(complex(flt),0))), kwargs...)
    cplx = complex(flt)
    scr = spectralclust(cplx, k; kwargs...)
    meta = [Int[] for i in 1:nclusters(scr)]
    # group clustres in a metacluster
    for (i, a) in enumerate(assignments(scr))
        push!(meta[a], i)
    end
    return PLMClusteringResult(mcr, meta, cplx, Inf)
end
