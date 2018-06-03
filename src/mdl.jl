"""Caluclate MDL value of the piecewise linear manifold cluster
"""
function plmdl(ms::Vector{LMCLUS.Manifold}, cplx::ComputationalHomology.AbstractComplex,
               plcids::Vector{Int}, X::AbstractMatrix{T},
               ::Type{MT}, Pm::Int, Pd::Int;
               debug = false, ɛ::T=1e-2, tot::Int = 1000,
               tol::Real = 1e-8) where {MT <: LMCLUS.MDL.MethodType, T <: Real}
    # Find how clusters are connected piecewise linear cluster
    # by constructing connectivity profile
    K = Set(plcids)
    Π = Vector{Int}[]
    for s in get(ComputationalHomology.cells(cplx, 1))
        vs = s[:values]
        all(i ∈ K for i in vs) && push!(Π, vs)
    end
    debug && println("K: $K")
    debug && println("Π: $Π")

    # find clusters that impose their models
    Iᵢ = Set{Int}()
    Iₐ = Set{Int}()
    for p in 1:length(Π)
        (i,j) = Π[p]
        cpidx = vcat(LMCLUS.points(ms[i]), LMCLUS.points(ms[j]))
        ipmdl = LMCLUS.MDL.calculate(MT, ms[i], X[:,cpidx], Pm, Pd)
        jpmdl = LMCLUS.MDL.calculate(MT, ms[j], X[:,cpidx], Pm, Pd)
        debug && println("i: $ipmdl, j: $jpmdl => ", (ipmdl<jpmdl ? "$i->$j" : "$i<-$j"))
        i, j = ipmdl < jpmdl ? (i, j) : (j, i)
        push!(Iᵢ,i)
        push!(Iₐ,j)
        Π[p] = [i,j] # correct profile
    end
    debug && println("Corrected Π: $Π")
    debug && println("Iᵢ: $Iᵢ")
    debug && println("Iₐ: $Iₐ")

    # Remove cycles from the graph
    for cycl in Iₐ ∩ Iᵢ
        impd = find(π -> π[1] == cycl,  Π)
        acpt = find(π -> π[2] == cycl,  Π)
        for p in impd
            (i,j) = Π[p]
            for (k, ii) in Π[acpt]
                cpidx = vcat(LMCLUS.points(ms[k]), LMCLUS.points(ms[j]))
                kpmdl = LMCLUS.MDL.calculate(MT, ms[k], X[:,cpidx], Pm, Pd)
                jpmdl = LMCLUS.MDL.calculate(MT, ms[j], X[:,cpidx], Pm, Pd)
                if kpmdl < jpmdl # from i->j & k->i => k->i & k->j
                    Π[p] = [k,j]
                    delete!(Iᵢ, i)
                else # from i->j & k->i => j->k->i->j (cycle)
                    error("Cycle: $j->$k->$i->$j")
                end
            end
        end
    end
    Π = unique(Π)
    debug && println("Correct incompatible models in Π: $Π")
    debug && println("Iᵢ: $Iᵢ")
    debug && println("Iₐ: $Iₐ")

    debug && println("Calculate model MDL from Iᵢ: $Iᵢ")
    L_Γ_Iᵢ = 0
    for i in Iᵢ
        L_Γ_Iᵢ += LMCLUS.MDL.calculate(MT, ms[i], X, Pm, Pd)
    end
    debug && println("L(Γ | Iᵢ): $L_Γ_Iᵢ")

    debug && println("Iₐ: $Iₐ")
    debug && println("Calculate data MDL from Iₐ: $Iₐ")
    L_Γ_Iₐ_Iᵢ_Π = 0
    for j in Iₐ
        datal = typemax(Int)
        for idx in find(π->π[2]==j, Π)
            i = Π[idx][1]
            Ld = LMCLUS.MDL.datadl(MT, ms[i], X[:,LMCLUS.points(ms[j])], Pd, ɛ, tot, tol)
            debug && println(j, " <- ", i, " ", Ld)
            if Ld < datal
                datal = Ld
            end
        end
        debug && println(j, " <- * ", datal)
        L_Γ_Iₐ_Iᵢ_Π += datal
    end
    debug && println("L(Γ | Iₐ,Iᵢ,Π): $L_Γ_Iₐ_Iᵢ_Π")

    R = symdiff(K, Iᵢ ∪ Iₐ)
    debug && println("K\(Iᵢ ∪ Iₐ): $R")
    Lrest = 0
    for i in R
        Lrest += LMCLUS.MDL.calculate(MT, ms[i], X, Pm, Pd)
    end
    debug && println("Lrest: $Lrest")

    # encoding of cluster indexes
    # L_plmidx = ceil(Int, log2(maximum(plcids)))*length(plcids)
    # L_plmidx = Pm*length(plcids)
    # debug && println("MDL of cluster indexes: $L_plmidx")

    plmdl = L_Γ_Iᵢ + L_Γ_Iₐ_Iᵢ_Π + Lrest #+ L_plmidx
    debug && println(plmdl)
    debug && println("----------")

    return plmdl
end

function plmdl2(ms::Vector{LMCLUS.Manifold}, cplx::ComputationalHomology.AbstractComplex,
                plcids::Vector{Int}, X::AbstractMatrix{T},
                ::Type{MT}, Pm::Int, Pd::Int;
                debug = false, ɛ::T=1e-2, tot::Int = 1000,
                tol::Real = 1e-8) where {MT<:LMCLUS.MDL.MethodType, T<:AbstractFloat}
    # Find how clusters are connected piecewise linear cluster
    # by constructing connectivity profile
    K = Set(plcids)
    Π = Vector{Int}[]
    for s in get(ComputationalHomology.cells(cplx, 1))
        vs = s[:values]
        all(i ∈ K for i in vs) && push!(Π, vs)
    end
    debug && println("K: $K")
    debug && println("Π: $Π")

    # find clusters that impose their models
    Iᵢ = Set{Int}()
    Iₐ = Set{Int}()
    for p in 1:length(Π)
        (i,j) = Π[p]
        cpidx = vcat(LMCLUS.points(ms[i]), LMCLUS.points(ms[j]))
        ipmdl = LMCLUS.MDL.calculate(MT, ms[i], X[:,cpidx], Pm, Pd)
        jpmdl = LMCLUS.MDL.calculate(MT, ms[j], X[:,cpidx], Pm, Pd)
        debug && println("i: $ipmdl, j: $jpmdl => ", (ipmdl<jpmdl ? "$i->$j" : "$i<-$j"))
        i, j = ipmdl < jpmdl ? (i, j) : (j, i)
        push!(Iᵢ,i)
        push!(Iₐ,j)
        Π[p] = [i,j] # correct profile
    end
    debug && println("Corrected Π: $Π")
    debug && println("Iᵢ: $Iᵢ")
    debug && println("Iₐ: $Iₐ")

    # Remove cycles from the graph
    for cycl in Iₐ ∩ Iᵢ
        impd = find(π -> π[1] == cycl,  Π)
        acpt = find(π -> π[2] == cycl,  Π)
        for p in impd
            (i,j) = Π[p]
            for (k, ii) in Π[acpt]
                cpidx = vcat(LMCLUS.points(ms[k]), LMCLUS.points(ms[j]))
                kpmdl = LMCLUS.MDL.calculate(MT, ms[k], X[:,cpidx], Pm, Pd)
                jpmdl = LMCLUS.MDL.calculate(MT, ms[j], X[:,cpidx], Pm, Pd)
                if kpmdl < jpmdl # from i->j & k->i => k->i & k->j
                    Π[p] = [k,j]
                    delete!(Iᵢ, i)
                else # from i->j & k->i => j->k->i->j (cycle)
                    error("Cycle: $j->$k->$i->$j in $Π")
                end
            end
        end
    end
    Π = unique(Π)
    debug && println("Correct incompatible models in Π: $Π")
    debug && println("Iᵢ: $Iᵢ")
    debug && println("Iₐ: $Iₐ")

    II = collect(Iᵢ)
    L_Γ_Iₐ_Iᵢ_Π = 0
    if length(II) > 0
        # L_Γ_Iₐ_Iᵢ_Π, minidx = findmin( sum( LMCLUS.MDL.datadl(MT, ms[i], X[:,LMCLUS.points(ms[j])], Pd, ɛ, tot, tol) for j in Iₐ ∪ setdiff(Iᵢ, i) ) for i in II )
        L_Γ_Iₐ_Iᵢ_Π, minidx = findmin( let ld = [j=>LMCLUS.MDL.datadl(MT, ms[i], X[:,LMCLUS.points(ms[j])], Pd, ɛ, tot, tol) for j in Iₐ ∪ setdiff(Iᵢ, i)];  debug && println("$i => $ld"); sum(map(last,ld)); end for i in II )
        i = II[minidx]
    else
        i = first(K)
    end

    debug && println("Calculate model MDL for $i")
    L_Γ_Iᵢ = LMCLUS.MDL.calculate(MT, ms[i], X[:,LMCLUS.points(ms[i])], Pm, Pd)
    debug && println("L(Γ | $i): $L_Γ_Iᵢ")
    debug && println("L(Γ | $(setdiff(K, i)), $i, Π): $L_Γ_Iₐ_Iᵢ_Π")

    plmdl = L_Γ_Iᵢ + L_Γ_Iₐ_Iᵢ_Π
    debug && println(plmdl)
    debug && println("----------")

    return plmdl
end

"""Calculate MDL value of the piecewise linear manifold clustering
"""
function plmdl(R::PLMCResult, X::AbstractMatrix{T}, ::Type{MT}, Pm::Int, Pd::Int;
               kwargs...) where {MT <: LMCLUS.MDL.MethodType, T <: Real}
    return sum( plmdl(R.manifolds, R.complex, cl, X, MT, Pm, Pd; kwargs...) for cl in R.clusters )
end

function plmdl2(R::PLMCResult, X::AbstractMatrix{T}, ::Type{MT}, Pm::Int, Pd::Int;
    kwargs...) where {MT <: LMCLUS.MDL.MethodType, T <: Real}
    return sum( plmdl2(R.manifolds, R.complex, cl, X, MT, Pm, Pd; kwargs...) for cl in R.clusters )
end
