using PLMC
using Clustering
using ClusterComplex
using Distributions
using ComputationalHomology
using LinearAlgebra
using Statistics
using Test

@testset "Agglomeration" begin
    n = 3
    cls = [[i] for i in 1:n]
    agg = Agglomeration(cls)
    @test count(agg) == 0.0
    @test length(agg) == 1
    @test length(agg.clusters) == length(agg.mergers) == length(agg.costs)

    push!(agg, cls[1:2]=>1.0)
    @test count(agg) == 1.0
    @test length(agg) == 2
    @test length(agg.clusters) == length(agg.mergers) == length(agg.costs)
end

@testset "Model-based Clustering" begin
    d1, l1 = MvNormal([0., 0.], Diagonal([1., 1.])), fill(1,10)
    d2, l2 = MvNormal([5., 5.], Diagonal([0.5, 0.5])), fill(2,10)
    d3, l3 = MvNormal([2., 5.], Diagonal([1.5, 0.5])), fill(3,10)

    mcr = ModelClusteringResult([d1,d2,d3], vcat(l1,l2,l3))
    @test nclusters(mcr) == 3
    @test counts(mcr) == [10, 10, 10]
    @test assignments(mcr) == vcat(l1,l2,l3)
    @test models(mcr)[1] == d1

    clidxs = [[1,2], [3]]
    MCs = modelclass(mcr, clidxs)
    @test MCs[1] isa MixtureModel
    @test length(components(MCs[1])) == length(clidxs[1])
    @test MCs[2] isa MvNormal

    plmcs = PLMClusteringResult(mcr, clidxs, SimplicialComplex(), 4.0)
    @test nclusters(plmcs) == length(clidxs)
    @test counts(plmcs) == [20, 10]
    @test assignments(plmcs) == vcat(fill(1, 20), fill(2, 10))
    @test models(plmcs)[1] == d1
end

@testset "IT Measures" begin
    @testset "Result Types" for T in [Float64, Float32],
                                itm in [PLMC.refinedmdl, PLMC.nml, PLMC.nll]
        mu = [0, T(0)]
        d1, l1 = MvNormal(mu.+5, Diagonal(mu.+1)), fill(1,10)
        d2, l2 = MvNormal(mu.+5, Diagonal((mu.+1)/2)), fill(2,10)
        mcr = ModelClusteringResult([d1,d2], vcat(l1,l2))
        X = [rand(d1, 10) rand(d2, 10)]
        @test itm([[1,2]], mcr, X) |> eltype == T
    end
end
