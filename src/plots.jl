using RecipesBase

@recipe function f(R::T) where {T<:PLMClusteringResult}
    χ = isinf(R.ϵ) ? 2.0 : R.ϵ
    for (i,idxs) in enumerate(R.clusters)
        addlabel = true
        for c in idxs
            @series begin
                label --> (addlabel ? "MC$i" : "")
                linecolor --> i
                models(R)[c], χ
            end
            addlabel = false
        end
    end
    if length(size(R.complex)) > 0
        D = hcat(map(mean, models(R))...)'
        R.complex, D
    end
end
