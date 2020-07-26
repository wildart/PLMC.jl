"""
    regret(P::MultivariateDistribution, MC::Vector{MultivariateDistribution}, x::AbstractVector)

The regret value of the probability distribution `P` w.r.t the model class `MC`
in the point `x`.
"""
regret(P, MC, x::AbstractVector) = -logpdf(P, x) - minimum(map(p->-logpdf(p,x), MC))
regret(P, MC, X::AbstractMatrix) = -logpdf(P, X) .- min.((-logpdf(p, X) for p in MC)...)

"""
    regretmax(P::MultivariateDistribution, MC::Vector{MultivariateDistribution}, X::AbstractMatrix)

The maximal regret of the probability distribution `P` w.r.t the model class `MC`
for all points of `X`.
"""
regretmax(P, MC, X) = maximum(regret(P,MC,X))

"""
refinedmdl(LC::Vector{MultivariateDistribution}, MC::ClusterComplex.MahalonobisClusteringResult, X::AbstractMatrix) -> (Float64, Float64, Float64)

Calculate a refined MDL value for the model class `LC` w.r.t. to the model class 'MC'
for all points of `X`. Returns tuple of MDL, likelihood, and complexity values.
"""
function refinedmdl(LC::Vector{<:MultivariateDistribution},
                    MS::Vector{<:MultivariateDistribution},
                    X::AbstractMatrix)
    LKH = min.((-logpdf(p, X) for p in LC)...) |> sum
    Rₘₐₓ = maximum(regretmax(P, MS, X) for P in LC)
    return LKH + Rₘₐₓ, LKH, Rₘₐₓ
end

refinedmdl(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix) =
    refinedmdl(modelclass(mcr, mrg), models(mcr), X)

"""
    mdldiff(P′::AbstractMixtureModel, MC::Vector{MultivariateDistribution}, X::AbstractMatrix) -> Float64

Returns MDL difference between original and merged P′ = {Pᵢ ∪ Pⱼ} clusterings.
"""
function mdldiff(P′::AbstractMixtureModel,
                 MCL::Vector{<:MultivariateDistribution},
                 X::AbstractMatrix)
    n = size(X, 2)
    Pᵢ = components(P′)[1]
    Pⱼ = components(P′)[2]
    LL = mapslices(x->max(-logpdf(Pᵢ,x), -logpdf(Pⱼ,x)), X, dims=1) |> sum
    R′Rᵢ = mapslices(x->logpdf(P′,x)-logpdf(Pᵢ,x), X, dims=1) |> maximum
    Rⱼ = PLMC.regretmax(Pⱼ, MCL, X)
    # println("LL[$LL] + max(R′-Rᵢ)[$(R′Rᵢ)] + Rⱼ[$(Rⱼ)] = $(LL + R′Rᵢ + Rⱼ)")
    return LL + R′Rᵢ + Rⱼ
end

function mdldiff(P′::Vector{<:MultivariateDistribution}, sz::Vector{Int},
                 X::AbstractMatrix)
    n = sum(sz)
    logPᵢ = logpdf(P′[1],X)
    logPⱼ = logpdf(P′[2],X)
    LL = sum(max(pᵢ, pⱼ) for (pᵢ, pⱼ) in zip(logPᵢ, logPⱼ))
    s1 = maximum.(eachcol([minimum(map(p-> -logpdf(p, x), components(p))) for p in P′, x in eachcol(X)]))
    # s1 = maximum(min.((-logpdf(p, X) for p in LC)...))
    s2 = -log.(sz[2]./exp.(logPᵢ) .+ sz[1]./exp.(logPⱼ))
    # println("LL[$LL] + max[$(maximum(s1 .- s2))]")
    return LL - log(n) + maximum(s1 .- s2)
end

function mdldiff(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult,
                 X::AbstractMatrix; kwargs...)
    assign = assignments(mcr)
    mrgidxs = map(m->findall(i-> i ∈ m, assign), mrg)
    idxs = vcat(mrgidxs...)
    mdldiff(modelclass(mcr, mrg), map(length, mrgidxs), view(X ,:,idxs))
end

# function mdldiff2(mrg::Vector{Vector{Int}}, mcr::ModelClusteringResult, X::AbstractMatrix)
#     mrgidxs = vcat(mrg...)
#     idxs = findall(i-> i ∈ mrgidxs, assignments(mcr))
#     mdldiff(first(modelclass(mcr, [mrgidxs])), models(mcr), view(X ,:,idxs))
# end
