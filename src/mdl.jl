"""
    regret(P::MultivariateDistribution, MC::Vector{MultivariateDistribution}, x::AbstractVector)

The regret value of the probability distribution `P` w.r.t the model class `MC`
in the point `x`.
"""
regret(P, MC, x) = -logpdf(P, x) - minimum(map(p->-logpdf(p,x), MC))

"""
    regretmax(P::MultivariateDistribution, MC::Vector{MultivariateDistribution}, X::AbstractMatrix)

The maximal regret of the probability distribution `P` w.r.t the model class `MC`
for all points of `X`.
"""
regretmax(P, MC, X) = mapslices(x->regret(P, MC, x), X, dims=1) |> maximum

"""
refinedmdl(LC::Vector{MultivariateDistribution}, MC::ClusterComplex.MahalonobisClusteringResult, X::AbstractMatrix) -> (Float64, Float64, Float64)

Calculate a refined MDL value for the model class `LC` w.r.t. to the model class 'MC'
for all points of `X`. Returns tuple of MDL, likelihood, and complexity values.
"""
function refinedmdl(LC::Vector{<:MultivariateDistribution},
                    MC::Vector{<:MultivariateDistribution},
                    X::AbstractMatrix)
    LKH = sum(minimum(mapslices(x->[-logpdf(P, x) for P in LC], X, dims=1), dims=1))
    Rₘₐₓ = sum([regretmax(P, MC, X) for P in LC])
    return LKH + Rₘₐₓ, LKH, Rₘₐₓ
end

refinedmdl(mrg::Vector{Vector{Int}}, mcr::ClusterComplex.MahalonobisClusteringResult, X::AbstractMatrix) =
    refinedmdl(modelclass(mcr, mrg), map(MvNormal, clusters(mcr)), X)

"""
    mdldiff(P′::AbstractMixtureModel, MC::Vector{MultivariateDistribution}, X::AbstractMatrix) -> Float64

Returns MDL difference between original and merged P′ = {Pᵢ ∪ Pⱼ} clusterings.
"""
function mdldiff(P′::AbstractMixtureModel,
                 MCL::Vector{<:MultivariateDistribution},
                 X::AbstractMatrix)
    Pᵢ = components(P′)[1]
    Pⱼ = components(P′)[2]
    # map(p->-logpdf(p, XX), components(P′))
    # map(extrema, map(p->logpdf(p, XX), components(P′)))
    # mapslices(x->max(-logpdf(Pᵢ,x), -logpdf(Pⱼ,x)), XX, dims=1)
    # mapslices(x->logpdf(Pᵢ,x) + logpdf(Pⱼ,x) + min(-logpdf(Pᵢ,x), -logpdf(Pⱼ,x)), XX, dims=1) |> sum
    # mapslices(x->logpdf(Pᵢ,x) + logpdf(Pⱼ,x) + min(-logpdf(Pᵢ,x), -logpdf(Pⱼ,x)), XX, dims=1) |> extrema
    # mapslices(x->logpdf(Pᵢ,x) + logpdf(Pⱼ,x) + min(-logpdf(Pᵢ,x), -logpdf(Pⱼ,x)), Xn, dims=1) |> sum
    LL = mapslices(x->max(-logpdf(Pᵢ,x), -logpdf(Pⱼ,x)), X, dims=1) |> sum
    R′Rᵢ = mapslices(x->logpdf(P′,x)-logpdf(Pᵢ,x), X, dims=1) |> maximum
    # R′ = PLMC.regretmax(P′, MCL, X)
    # Rᵢ = PLMC.regretmax(Pᵢ, MCL, X)
    Rⱼ = PLMC.regretmax(Pⱼ, MCL, X)
    # println("LL[$LL] + max(R′-Rᵢ)[$(R′Rᵢ)] + Rⱼ[$(Rⱼ)] = $(LL + R′Rᵢ + Rⱼ)")
    return LL + R′Rᵢ + Rⱼ
end

function mdldiff(mrg::Vector{Vector{Int}}, mcr::ClusterComplex.MahalonobisClusteringResult, X::AbstractMatrix)
    # print("$mrg => ")
    mrgidxs = vcat(mrg...)
    idxs = mapreduce(c->c.idx, vcat, clusters(mcr)[mrgidxs])
    mdldiff(modelclass(mcr, [mrgidxs])[], map(MvNormal, clusters(mcr)), view(X ,:,idxs))
end
