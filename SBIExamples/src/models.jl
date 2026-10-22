"""A small inverse problem with independent uniform priors and Gaussian sensor noise."""
struct SBIProblem
    name::Symbol
    parameters::Vector{Symbol}
    truth::Vector{Float64}
    priors::Vector{Uniform{Float64}}
    noise::Float64
    n::Int
end

example_names() = (:burgers, :graph_transport, :elliptic, :nonlinear_elliptic)

"""Construct a benchmark; `n` is the number of cells (interior nodes for elliptic models)."""
function example(name::Symbol; n::Int=32, noise::Real=0.01)
    n >= 8 || throw(ArgumentError("n must be at least 8"))
    isfinite(noise) && noise > 0 || throw(ArgumentError("noise must be finite and positive"))
    pars, truth, priors = if name == :burgers
        ([:viscosity, :amplitude], [0.03, 0.9], [Uniform(0.01, 0.08), Uniform(0.6, 1.2)])
    elseif name == :graph_transport
        ([:speed, :split], [0.9, 0.65], [Uniform(0.6, 1.3), Uniform(0.2, 0.8)])
    elseif name == :elliptic
        ([:conductivity], [1.2], [Uniform(0.5, 2.0)])
    elseif name == :nonlinear_elliptic
        ([:reaction], [4.0], [Uniform(0.5, 8.0)])
    else
        throw(ArgumentError("unknown example $name; choose from $(example_names())"))
    end
    SBIProblem(name, pars, truth, priors, Float64(noise), n)
end

# Periodic finite volumes: Rusanov convection and centered viscous flux.
function burgers(ν, amplitude, n)
    dx = 1 / n
    u = [amplitude * sin(2π * (i - 0.5) * dx) for i in 1:n]
    t = 0.0
    while t < 0.2
        a = maximum(abs, u)
        dt = min(0.4 / (a / dx + 2ν / dx^2), 0.2 - t)
        flux = [(u[i]^2 + u[mod1(i+1,n)]^2)/4 -
                a/2 * (u[mod1(i+1,n)]-u[i]) -
                ν/dx * (u[mod1(i+1,n)]-u[i]) for i in 1:n]
        u -= (dt/dx) .* (flux .- circshift(flux, 1))
        t += dt
    end
    u
end

# Three unit-length edges: parent -> two daughters. Zero external inflow,
# free outflow; the junction distributes exactly the parent's outgoing flux.
function graph_transport(c, split, n)
    dx = 1/n
    u = zeros(n, 3)
    u[:,1] = [exp(-(( (i-0.5)*dx - 0.7)/0.12)^2) for i in 1:n]
    t = 0.0
    while t < 0.65
        dt = min(0.8dx/c, 0.65-t)
        incoming = [0.0, split*c*u[end,1], (1-split)*c*u[end,1]]
        old = copy(u)
        for edge in 1:3, i in 1:n
            left = i == 1 ? incoming[edge] : c*old[i-1,edge]
            u[i,edge] = old[i,edge] - dt/dx*(c*old[i,edge]-left)
        end
        t += dt
    end
    vec(u)
end

function elliptic(k, n)
    h = 1/(n+1)
    SymTridiagonal(fill(2k/h^2,n), fill(-k/h^2,n-1)) \ ones(n)
end

# -u'' + λu³ = 10 with homogeneous Dirichlet conditions; damped Newton.
function nonlinear_elliptic(λ, n)
    h = 1/(n+1)
    A = SymTridiagonal(fill(2/h^2,n), fill(-1/h^2,n-1))
    u = A \ fill(10.0,n)
    for _ in 1:50
        r = A*u + λ*u.^3 .- 10
        norm(r, Inf) < 1e-9 && return u
        step = (A + Diagonal(3λ*u.^2)) \ r
        α = 1.0
        while norm(A*(u-α*step) + λ*(u-α*step).^3 .- 10) >= norm(r)
            α /= 2
            α > 1e-10 || error("Newton line search failed")
        end
        u -= α*step
    end
    error("Nonlinear elliptic solver did not converge")
end

"""Return the noiseless full spatial state at the observation time (or steady state)."""
function forward(p::SBIProblem, θ::AbstractVector)
    length(θ) == length(p.truth) || throw(DimensionMismatch("wrong parameter count"))
    all(isfinite, θ) || throw(ArgumentError("parameters must be finite"))
    all(>(0), θ) || throw(ArgumentError("parameters must be positive"))
    if p.name == :burgers
        burgers(θ..., p.n)
    elseif p.name == :graph_transport
        θ[2] < 1 || throw(ArgumentError("split must lie in (0,1)"))
        graph_transport(θ..., p.n)
    elseif p.name == :elliptic
        elliptic(θ[1], p.n)
    else
        nonlinear_elliptic(θ[1], p.n)
    end
end

sensor_indices(p) = round.(Int, range(1, p.n; length=8))
function sensors(p, state)
    idx = sensor_indices(p)
    p.name == :graph_transport ? state[vcat(idx, idx .+ p.n, idx .+ 2p.n)] : state[idx]
end

"""Simulate eight sensors per edge with independent N(0, noise²) observation errors."""
function simulate(p::SBIProblem, θ::AbstractVector=p.truth; rng=Random.default_rng())
    y = sensors(p, forward(p, θ))
    y + p.noise * randn(rng, length(y))
end

