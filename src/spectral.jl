import ComputationalHomology: AbstractComplex, Filtration

"""Adjacency matrix created from 1-subcomplex of simplicial complex `cplx`.
"""
function adjacency_matrix(cplx::AbstractComplex, T::DataType=Int)
    nv = size(cplx,0)
    ne = size(cplx,1)
    @assert nv > 0 "Complex should not be empty"
    S = eltype(ComputationalHomology.celltype(cplx)) # get complex cell's type
    I = zeros(S, 2*ne)
    J = zeros(S, 2*ne)
    V = zeros(T, 2*ne)
    for (i,s) in enumerate(ComputationalHomology.simplices(cplx, 1))
        vs = s[:values]
        I[2*i-1] = vs[1]
        J[2*i-1] = vs[2]
        I[2*i] = vs[2]
        J[2*i] = vs[1]
        V[2*i-1] = one(T)
        V[2*i] = one(T)
    end
    return sparse(I, J, V, nv, nv)
end

"""Similarity matrix created from 1-subcomplex of simplicial complex `cplx` and distance weights.
"""
function similarity_matrix(flt::Filtration)
    cplx = complex(flt)
    ord = ComputationalHomology.order(flt)
    nv = size(cplx,0)
    ne = size(cplx,1)
    @assert nv > 0 "Complex should not be empty"
    S = eltype(ComputationalHomology.celltype(cplx)) # get complex cell's type
    I = zeros(S, 2*ne)
    J = zeros(S, 2*ne)
    V = zeros(valtype(flt), 2*ne)
    for (i,s) in enumerate(ComputationalHomology.simplices(cplx, 1))
        vs = s[:values]
        I[2*i-1] = vs[1]
        J[2*i-1] = vs[2]
        I[2*i] = vs[2]
        J[2*i] = vs[1]
        idx = findfirst(v->v[1]==1 && v[2] == s[:index], ord)
        if idx > 0
            V[2*i-1] = ord[idx][3]
            V[2*i] = ord[idx][3]
        else
            error("No filtration value for simplex: $s")
        end
    end
    return sparse(I, J, V, nv, nv)
end

"""Laplacian of an adjacency matrix `A`"""
laplacian(A::AbstractSparseMatrix) = sparse(Diagonal(sum(A,2)[:])) - A

"""Normalized laplacian of an adjacency matrix `A`"""
function normalized_laplacian(A::AbstractSparseMatrix)
    D = 1./sqrt.(sum(A,2)[:])
    return speye(A) - Diagonal(D)*A*Diagonal(D)
end


"""Spectral clustering results
"""
struct SpectralResult <: Clustering.ClusteringResult
    res::Clustering.KmeansResult
    λ::Vector{Float64}
end
assignments(clust::SpectralResult) = assignments(clust.res)
nclusters(clust::SpectralResult) = nclusters(clust.res)
counts(clust::SpectralResult) = counts(clust.res)

"""Spectral clustering using of square similarity (adjacency) matrix

Accepts sparse matrix `M` and `k` number of expected clusters as parameters.
Parameter `method` specifies clustering algorithm:

- `:njw` for "On Spectral Clustering: Analysis and an algorithm" by Nj, Jordan, Weiss (2002) (default)
- `:sm` for "Normalized Cuts and Image Segmentation" by Shi and Malik (2000)
"""
function spectralclust(A::AbstractSparseMatrix, k::Int; method=:njw, display=:none, init=:kmpp)
    @assert size(A,1) == size(A,2) "Matrix should be square"
    @assert size(A,1) > k "Number of clusters cannot be more then observations"
    @assert k > 1 "Number of clusters should be more then one"

    if method == :sm
        L = laplacian(A)
        # (display == :iter) && println("Laplacian:\n", collect(L))
        D = sparse(Diagonal(diag(L)))
        λ, ϕ = eigs(L, D, which=:SR, nev=min(2*k, size(A,1)-1))
        idxs = find(real(λ).> eps())
        ϕ = ϕ[:,idxs]
        λ = λ[idxs]
    elseif method == :njw
        L = normalized_laplacian(A)
        # (display == :iter) && println("Laplacian:\n", collect(L))
        kk = min(k+1, size(A,1)-1)
        λ, ϕ = eigs(L, which=:SR, nev=kk)
        k = min(length(λ), kk-1)
        ϕ = ϕ[:,1:k]
        λ = λ[1:k]
    else
        error("Invalid method: $method")
    end

    # normalize
    ϕ = real(ϕ)
    ϕ ./= mapslices(norm,ϕ,2)

    KM = Clustering.kmeans(ϕ', k, init=init, display=display)
    return SpectralResult(KM, real(λ))
end
spectralclust(cplx::AbstractComplex, k::Int; kwargs...) = spectralclust(adjacency_matrix(cplx, Float64), k; kwargs...)
