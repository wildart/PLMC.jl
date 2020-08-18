function markminima(J, tol=1e-4; debug=false)
    N = length(J)
    M = zeros(Bool,N)
    P = zeros(N)
    # find all minima
    M[1] = J[1] < J[2]
    prev = J[2] - J[1]
    curr = 0.0
    P[1] = prev
    for i=2:(N-1)
        curr = J[i+1] - J[i]
        curr = (abs(curr) < tol) ? 0 : curr
        debug && println("$i: $prev<=0 && $curr>=0")
        M[i] = prev<=0 && curr>=0
        prev=curr
        P[i] = prev
    end
    M[end] = J[end] < J[end-1]
    debug && println(M)
    debug && println(P[M])
    return (M, P)
end

function findglobalmin(J, tol=1e-4; debug=false)
    N = length(J)
    N == 1 && return 1
    # find all minima
    M, P = markminima(J; debug=false)
    # zero minima below tolerance (skip first & last)
    minima, MI = if sum(M[2:end-1]) > 0
        Mnew = copy(M)
        Mnew[1] = Mnew[end] = false
        mval, mi = findmin(J[Mnew])
        map(v->(abs(v) < tol) ? 0 : v, J[Mnew].-mval), Mnew
    else
        J[M], M
    end
    debug && println(minima)
    # get minima index
    findall(MI)[(last∘findmin)(minima)]
end

function findglobalmin2(J; debug=false)
    N = length(J)
    N == 1 && return 1
    f(x) = J[round(Int,x)]
    r = optimize(f, 1, N)
    debug && println(r)
    idxs = sortperm(J)
    si = findfirst(x-> x>=minimum(r), J[idxs])
    si === nothing ? N : si
    idxs[si]
end

function findlocalmin(J)
    findmin(J)[2]
end

function findlocalminrev(J)
    c = length(J)
    Jmax = Inf
    midx = c
    for i in c:-1:1
        if J[c] < Jmax
            Jmax = J[c]
        else
            break
        end
        midx = i
    end
    return idx
end

function findfirstminimum(J; debug=false)
    N = length(J)
    N == 1 && return 1
    # find all minima
    M, P = markminima(J; debug=debug)
    m, midx = Inf, 0
    for i in findall(M)
        J[i] |> println
        if J[i] < m
            m = J[i]
            midx = i
        else
            break
        end
    end
    return midx
end

function mixlogpdf(logps::AbstractMatrix{T}, ps::Vector{Vector{T}},
                   idxs::Vector{Vector{Int}}) where {T <: AbstractFloat}
    # using the formula below for numerical stability
    #
    # logpdf(d, x) = log(sum_i pri[i] * pdf(cs[i], x))
    #              = log(sum_i pri[i] * exp(logpdf(cs[i], x)))
    #              = log(sum_i exp(logpri[i] + logpdf(cs[i], x)))
    #              = m + log(sum_i exp(logpri[i] + logpdf(cs[i], x) - m))
    #
    #  m is chosen to be the maximum of logpri[i] + logpdf(cs[i], x)
    #  such that the argument of exp is in a reasonable range
    #
    M = length(idxs)
    N, L = size(logps)

    @assert all(e->length(e[1]) == length(e[2]), zip(idxs, ps)) "Sizes of indexes and priors must be equal"

    lps = Matrix{T}(undef, N, M)

    @inbounds for k in eachindex(idxs)
        p = ps[k]
        ids = idxs[k]
        K = length(p)
        lp = Vector{T}(undef, K)

        j = 1
        for logp in eachrow(logps)
            m = -Inf   # m <- the maximum of log(p(cs[i], x)) + log(pri[i])
            for i in eachindex(p)
                pi = p[i]
                if pi > 0.0
                    # lp[i] <- log(p(cs[i], x)) + log(pri[i])
                    lp_i = logp[ids[i]] + log(pi)
                    lp[i] = lp_i
                    if lp_i > m
                        m = lp_i
                    end
                end
            end
            v = 0.0
            for i = 1:K
                if p[i] > 0.0
                    v += exp(lp[i] - m)
                end
                lp[i] = 0
            end
            lps[j, k] = m + log(v)
            j+=1
        end
    end
    return lps
end

function score(fn, cl::C, data::AbstractMatrix) where {C <: ClusteringResult}
    assign = assignments(cl)
    clusters = [[i] for i in 1:nclusters(cl)]
    models = [model(data, findall(i .∈  assign)) for i in 1:nclusters(cl)]
    mcr = ModelClusteringResult(models, assign)
    fn(clusters, mcr, data)
end
score(fn, cl::PLMClusteringResult, data::AbstractMatrix) = fn(cl.clusters, cl.models, data)
