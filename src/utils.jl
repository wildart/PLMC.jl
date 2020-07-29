function findglobalmin(J, tol=1e-4; debug=false)
    N = length(J)
    N == 1 && return 1
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
    # debug && println(P[M])
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
    idx = c
    for i in c:-1:1
        if J[c] < Jmax
            Jmax = J[c]
        else
            break
        end
        idx = i
    end
    return idx
end
