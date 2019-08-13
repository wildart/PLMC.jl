using PLMC
using ClusterComplex
using Distributions
using ComputationalHomology
using LinearAlgebra
using Test

@testset "Mahalonobis Clustering" begin
    c1 = MahalonobisCluster([0., 0.], Diagonal([1., 1.]), collect(1:10))
    c2 = MahalonobisCluster([5., 5.], Diagonal([0.5, 0.5]), collect(11:20))
    c3 = MahalonobisCluster([2., 5.], Diagonal([1.5, 0.5]), collect(21:30))
    d1 = MvNormal(c1)
    dt = MvNormal([0., 0.], Diagonal([1., 1.]))
    @test mean(d1) == mean(dt)
    @test cov(d1) == cov(dt)

    @test PLMC.mixture([c1,c2]) isa MixtureModel

    clidxs = [[1,2], [3]]
    mcr = MahalonobisClusteringResult([c1,c2,c3])

    MCs = PLMC.modelclass(mcr,  clidxs)
    @test MCs[1] isa MixtureModel
    @test MCs[2] isa MvNormal

    plmcs = PLMCResultClusteringResult([c1,c2,c3], clidxs, SimplicialComplex(), 4.0)
    @test nclusters(plmcs) == length(clidxs)
    @test counts(plmcs) == [20, 10]
    @test assignments(plmcs) == vcat(fill(1, 20), fill(2, 10))
end
