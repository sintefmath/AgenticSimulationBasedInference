module SBIExamples

using GpABC, Distributions, LinearAlgebra, Random, Statistics
export SBIProblem, example, forward, simulate, infer, example_names

include("models.jl")

"""
    infer(problem, observed; tolerance=2.0, nparticles=100, maxiterations=100_000)

Run GpABC rejection ABC using noise-normalized sensor RMSE. Returns the
native GpABC result. Seed `Random.seed!` before calling for repeatability;
the backend uses Julia's default RNG. Sampling is serial. Tolerance is in noise
standard deviations, and finite tolerance introduces ABC approximation error.
"""
function infer(p::SBIProblem, observed::AbstractVector; tolerance::Real=2.0,
               nparticles::Int=100, maxiterations::Int=100_000)
    expected = p.name == :graph_transport ? 24 : 8
    length(observed) == expected || throw(DimensionMismatch("expected $expected sensors"))
    all(isfinite, observed) || throw(ArgumentError("observations must be finite"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("invalid tolerance"))
    nparticles > 0 && maxiterations >= nparticles || throw(ArgumentError("invalid sampling budget"))
    simulator = θ -> reshape(simulate(p, θ), :, 1)
    distance = (a, b) -> sqrt(mean(abs2, (a-b)/p.noise))
    SimulatedABCRejection(reshape(Float64.(observed), :, 1), simulator, p.priors,
                         Float64(tolerance), nparticles; distance_function=distance,
                         max_iter=maxiterations, write_progress=false)
end

end
