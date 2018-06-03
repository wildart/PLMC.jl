"""Calculate Mahalanobis distance in point-to-manifold distance space.
"""
function ptm_dist(X::AbstractMatrix, M::LMCLUS.Manifold)
    D1 = LMCLUS.distance_to_manifold(X, mean(M), projection(M))
    D2 = LMCLUS.distance_to_manifold(X, mean(M), projection(M); ocss=true)
    δ = hcat(D1, D2)'
    δs = δ[:,points(M)]
    S = cov(δs, 2)
    IS = try
        inv(S)
    catch
        pinv(S) # use pseudoinverse
    end
    return Distances.colwise(Distances.Mahalanobis(IS), δ, zeros(δ)), δ, S
end

function elliptical_bound(S::AbstractMatrix{<:Real}, χ::Real)
    F = eigfact(Symmetric(S))
    λ = F[:values]
    FI = sortperm(λ, rev=true)
    EV = F[:vectors][:,FI[1]]
    ϕ = atan2(EV[FI[2]], EV[FI[1]])
    if ϕ < 0
        ϕ += 2π
    end
    R = [ cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ) ]

    θ = linspace(0, 2π)
    a, b = χ*sqrt(λ[FI[1]]), χ*sqrt(λ[FI[2]])
    x, y  = a.*cos.(θ), b.*sin.(θ)
    ellipse = R * hcat(x, y)'
    return ellipse
end
