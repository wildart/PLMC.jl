using PLMC
using Clustering
using ClusterComplex
using Distributions
using ComputationalHomology
using LinearAlgebra
using Statistics
using Test

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
