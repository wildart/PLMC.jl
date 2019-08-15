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

refinedmdl(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix) =
    refinedmdl(modelclass(mcr, mrg), mcr.models, X)

"""
    mdldiff(P′::AbstractMixtureModel, MC::Vector{MultivariateDistribution}, X::AbstractMatrix) -> Float64

Returns MDL difference between original and merged P′ = {Pᵢ ∪ Pⱼ} clusterings.
"""
function mdldiff(P′::AbstractMixtureModel,
                 MCL::Vector{<:MultivariateDistribution},
                 X::AbstractMatrix)
    Pᵢ = components(P′)[1]
    Pⱼ = components(P′)[2]
    LL = mapslices(x->max(-logpdf(Pᵢ,x), -logpdf(Pⱼ,x)), X, dims=1) |> sum
    R′Rᵢ = mapslices(x->logpdf(P′,x)-logpdf(Pᵢ,x), X, dims=1) |> maximum
    Rⱼ = PLMC.regretmax(Pⱼ, MCL, X)
    # println("LL[$LL] + max(R′-Rᵢ)[$(R′Rᵢ)] + Rⱼ[$(Rⱼ)] = $(LL + R′Rᵢ + Rⱼ)")
    return LL + R′Rᵢ + Rⱼ
end

function mdldiff(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix)
    # print("$mrg => ")
    mrgidxs = vcat(mrg...)
    idxs = findall(i-> i ∈ mrgidxs, assignments(mcr))
    mdldiff(modelclass(mcr, [mrgidxs])[], mcr.models, view(X ,:,idxs))
end
