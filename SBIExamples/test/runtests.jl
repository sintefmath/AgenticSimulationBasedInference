using Test, SBIExamples, Random, Statistics, LinearAlgebra

@testset "Forward models" begin
    p = example(:elliptic)
    x = (1:p.n)/(p.n+1)
    @test forward(p, [1.2]) ≈ x .* (1 .- x) / 2.4
    p = example(:nonlinear_elliptic)
    u = forward(p, p.truth)
    h = 1/(p.n+1)
    residual = (2u - [0; u[1:end-1]] - [u[2:end]; 0])/h^2 + 4u.^3 .- 10
    @test norm(residual, Inf) < 1e-8
    @test all(>(0), u)
    p = example(:burgers)
    u = forward(p, p.truth)
    @test abs(sum(u)) < 1e-12
    @test maximum(abs, u) < p.truth[2]
    p = example(:graph_transport)
    u = reshape(forward(p, p.truth), p.n, 3)
    @test minimum(u) >= 0
    @test u[:,2] ≈ (0.65/0.35)*u[:,3]
    initialmass = sum(exp(-(((i-0.5)/p.n-0.7)/0.12)^2) for i in 1:p.n)
    @test sum(u) <= initialmass + 1e-12
    @test sum(u) > 0.99initialmass
    for name in example_names()
        p = example(name)
        @test simulate(p; rng=MersenneTwister(2)) == simulate(p; rng=MersenneTwister(2))
        @test all(isfinite, forward(p, p.truth))
        @test_throws DimensionMismatch forward(p, Float64[])
    end
    @test_throws ArgumentError example(:unknown)
    @test_throws ArgumentError example(:burgers; noise=0)
    @test_throws ArgumentError example(:elliptic; n=2)
end

@testset "GpABC integration and parameter recovery" begin
    for name in example_names()
        Random.seed!(42)
        p = example(name; n=16)
        observed = simulate(p)
        result = infer(p, observed; nparticles=12, tolerance=3.0, maxiterations=20_000)
        @test result.n_accepted == 12
        @test all(d -> isfinite(d) && d <= 3.0, result.distances)
        @test all(q -> all(i -> minimum(p.priors[i]) <= q[i] <= maximum(p.priors[i]), eachindex(p.truth)), eachrow(result.population))
    end
    Random.seed!(123)
    p = example(:elliptic; noise=0.002)
    result = infer(p, simulate(p); nparticles=100, tolerance=1.8)
    samples = result.population[:,1]
    @test abs(mean(samples)-p.truth[1]) < 0.1
    @test std(samples) < 0.15
end
