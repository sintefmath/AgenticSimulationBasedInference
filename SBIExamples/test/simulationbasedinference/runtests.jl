using Test, Random, Statistics, LinearAlgebra, Distributions
import SimulationBasedInference as SBI
using SciMLBase: solve, EnsembleSerial

# Both suites execute exactly the same model source. Loading the parent package
# would also load GpABC, whose StatsBase 0.33 bound conflicts with SBI's 0.34.
module Models
using Distributions, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "..", "src", "models.jl"))
end

function inference_problem(p, data)
    prediction = θ -> Models.sensors(p, Models.forward(p, θ))
    observable = SBI.SimulatorObservable(identity, size(data); name=:sensors)
    forward_problem = SBI.SimulatorForwardProblem(prediction, p.truth, observable)
    model_prior = SBI.prior(; (name => dist for (name, dist) in zip(p.parameters, p.priors))...)
    # EnIS uses the likelihood prior's median to form its observation covariance.
    # A point mass fixes σ to the benchmark's known measurement noise.
    likelihood = SBI.IsotropicGaussianLikelihood(observable, data, SBI.prior(σ=Dirac(p.noise)))
    problem = SBI.SimulatorInferenceProblem(forward_problem, nothing, model_prior, likelihood)
    return problem, prediction
end

function run_ensemble(problem; seed=42, ensemble_size=512)
    solve(problem, SBI.EnIS(), EnsembleSerial(); ensemble_size,
          rng=MersenneTwister(seed), verbose=false)
end

@testset "SimulationBasedInference: all forward models and EnIS" begin
    for name in Models.example_names()
        @testset "$name" begin
            p = Models.example(name; n=16, noise=0.02)
            observed = Models.simulate(p; rng=MersenneTwister(21))
            problem, prediction = inference_problem(p, observed)
            solution = run_ensemble(problem)
            draws = SBI.get_transformed_ensemble(solution)
            weights = SBI.get_weights(solution)
            @test size(draws) == (length(p.truth), 512)
            @test length(weights) == 512
            @test all(isfinite, weights) && all(>=(0), weights)
            @test sum(weights) ≈ 1.0
            @test all(i -> all(x -> insupport(p.priors[i], x), draws[i,:]), eachindex(p.truth))
            # Check the actual framework weights against the known Gaussian
            # observation model, catching noise-scale, parameter-order and
            # observable-adapter errors (rather than only checking shapes).
            logweights = [-sum(abs2, (prediction(θ)-observed)/p.noise)/2 for θ in eachcol(draws)]
            expected = exp.(logweights .- maximum(logweights))
            expected ./= sum(expected)
            @test weights ≈ expected rtol=1e-10 atol=1e-12
            @test inv(sum(abs2, weights)) > 2
            repeated = run_ensemble(problem)
            @test SBI.get_transformed_ensemble(repeated) == draws
            @test SBI.get_weights(repeated) == weights
        end
    end
end

@testset "SimulationBasedInference: elliptic posterior vs quadrature" begin
    p = Models.example(:elliptic; n=16, noise=0.005)
    observed = Models.simulate(p; rng=MersenneTwister(123))
    problem, prediction = inference_problem(p, observed)
    solution = run_ensemble(problem; seed=123, ensemble_size=4096)
    draws = vec(SBI.get_transformed_ensemble(solution))
    weights = SBI.get_weights(solution)
    posterior_mean = dot(draws, weights)
    posterior_std = sqrt(dot((draws .- posterior_mean).^2, weights))

    # Independent trapezoidal quadrature using the analytic solution -k u'' = 1.
    lo, hi = extrema(p.priors[1])
    grid = range(lo, hi; length=20_001)
    x = Models.sensor_indices(p) ./ (p.n+1)
    logweights = [-sum(abs2, (x .* (1 .- x) ./ (2k) - observed)/p.noise)/2 for k in grid]
    exact_weights = exp.(logweights .- maximum(logweights))
    exact_weights[[1,end]] ./= 2 # trapezoidal endpoint weights
    exact_weights ./= sum(exact_weights)
    reference_mean = dot(grid, exact_weights)
    reference_std = sqrt(dot((grid .- reference_mean).^2, exact_weights))
    @test abs(posterior_mean-reference_mean) < 0.01
    @test abs(posterior_std-reference_std) < 0.01
    @test abs(posterior_mean-p.truth[1]) < 0.1
    @test posterior_std < std(p.priors[1])/3
    @test inv(sum(abs2, weights)) > 100
end
