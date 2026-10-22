# SBIExamples.jl

Small PDE inverse problems for simulation-based inference in Julia. Each forward
solve uses 32 spatial cells/nodes by default, one or two unknown parameters, and
8 noisy point sensors per edge. No PDE framework or neural-network training is
needed. This is an educational benchmark package, not a high-fidelity PDE solver.

## Run

From the repository root with Julia 1.10 or later:

```sh
julia --project=SBIExamples -e 'using Pkg; Pkg.instantiate()'
```

Run all four examples and the tests from the repository root:

```sh
julia --project=SBIExamples SBIExamples/examples/run_all.jl
julia --project=SBIExamples SBIExamples/examples/run_all.jl elliptic
julia --project=SBIExamples -e 'using Pkg; Pkg.test()'
```

```julia
using SBIExamples, Random
Random.seed!(2026)
p = example(:burgers)
y = simulate(p)                    # synthetic observations at known truth
posterior = infer(p, y)             # GpABC result, equally weighted particles
θ = posterior.population[1, :]
state = forward(p, θ)               # noiseless full spatial solution
```

## Models

| Name | Equation and domain | Unknowns (uniform prior); truth |
| --- | --- | --- |
| `burgers` | u_t + (u²/2)_x = νu_xx on periodic [0,1]; u(x,0)=A sin(2πx); observe t=0.2 | ν∈[0.01,0.08], A∈[0.6,1.2]; (0.03,0.9) |
| `graph_transport` | u_t + c u_x=0 on three directed unit edges, one parent feeding two daughters; observe t=0.65 | c∈[0.6,1.3], split s∈[0.2,0.8]; (0.9,0.65) |
| `elliptic` | −k u″=1 on [0,1], u(0)=u(1)=0 | k∈[0.5,2]; 1.2 |
| `nonlinear_elliptic` | −u″+λu³=10 on [0,1], u(0)=u(1)=0 | λ∈[0.5,8]; 4 |

Burgers uses conservative Rusanov flux plus centered diffusion and a combined
advection/diffusion time-step bound. The graph is a minimal hyperbolic conservation
law on a network: an upwind finite-volume discretization with outgoing junction
fluxes s c u_parent and (1−s)c u_parent. Its initial parent pulse is
exp(−((x−0.7)/0.12)²); daughters start at zero. External inflow is zero and daughter
outlets are free outflow. Graph states are flattened edge by edge. Elliptic models
use centered differences at interior nodes; the nonlinear solve uses damped Newton
with an explicit residual convergence check.

## Inference and limitations

[GpABC.jl](https://github.com/tanhevg/GpABC.jl) supplies rejection
ABC. It fits these fast, low-dimensional examples without evaluating a likelihood.
Every simulated observation includes independent Gaussian noise with standard
deviation `p.noise` (default 0.01), including synthetic target observations. The
summary is the sensor vector itself. Distance is sensor RMSE divided by the noise
standard deviation. The default acceptance tolerance is 2.0; this produces an
approximate posterior and is not a claim of calibrated uncertainty. Smaller
tolerances increase simulation cost. These additive-noise examples also have a
tractable likelihood, making them useful future validation baselines for SBI.

`infer(p, y; nparticles=100, tolerance=2.0, maxiterations=100_000)` controls cost.
GpABC uses the default RNG; seed it before inference. Execution is serial.
Inspect the returned particle count and backend warnings if the simulation budget
is exhausted. Rejection particles have equal weight. The runner prints means and
central 90% intervals; they are ABC intervals, not exact Bayesian intervals.

Use `example(name; n=64, noise=0.005)` to change resolution and noise. Changing `n`
changes sensor locations slightly (rounded evenly spaced grid indices). Generated
data and inference use the same discretization; test refinement separately before
interpreting a posterior physically. The one-parameter elliptic problems keep the
forcing fixed to avoid confounding conductivity with forcing amplitude.

## Dependency compatibility

GpABC 0.1.1 references `OrdinaryDiffEqAlgorithm`, which newer OrdinaryDiffEq
releases no longer expose. `Project.toml` constrains OrdinaryDiffEq to the 6.98
series to preserve this API while permitting current LinearSolve dependencies.
The complete test suite and all four default inference runs were verified with
Julia 1.13.0, GpABC 0.1.1, and OrdinaryDiffEq 6.98.0 on Apple Silicon. First-time
installation/compilation can take several minutes because GpABC loads SciML.

## SimulationBasedInference.jl tests

A second integration suite runs all four models through
[SimulationBasedInference.jl](https://github.com/bgroenks96/SimulationBasedInference.jl)
using `SimulatorForwardProblem`, `SimulatorInferenceProblem`, and ensemble
importance sampling (`EnIS`). It checks normalized posterior weights against the
known Gaussian observation model, prior bounds, effective sample size, and seeded
reproducibility. For the linear elliptic problem it also checks posterior mean
and standard deviation against numerical quadrature of the analytic solution.
This is likelihood-based importance sampling; it complements the GpABC
likelihood-free rejection tests and does not assert identical posteriors.

Run from the repository root:

```sh
julia --project=SBIExamples/test/simulationbasedinference -e 'using Pkg; Pkg.instantiate()'
julia --project=SBIExamples/test/simulationbasedinference SBIExamples/test/simulationbasedinference/runtests.jl
```

The suite has a separate environment because GpABC 0.1 requires StatsBase 0.33,
whereas SimulationBasedInference requires 0.34. Both suites load the same
`src/models.jl` source; the isolated suite does not import the parent package or
its GpABC backend. The regular `Pkg.test()` command still runs the original
numerical and GpABC tests; run both commands to cover both frameworks. EnIS uses
the noiseless sensor prediction with the known Gaussian noise covariance; it
keeps prior ensemble members and assigns posterior weights, so posterior
statistics must be weighted. Tests use serial execution and local seeded RNGs.
