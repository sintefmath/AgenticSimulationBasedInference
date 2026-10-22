using SBIExamples, Random, Statistics

# Optional first argument selects one example, e.g. elliptic.
names = isempty(ARGS) ? example_names() : (Symbol(ARGS[1]),)
for name in names
    Random.seed!(2026)
    problem = example(name)
    observed = simulate(problem)
    posterior = infer(problem, observed; nparticles=100)
    posterior.n_accepted > 0 || error("No accepted particles; increase the budget or tolerance")
    println("\n", name, " (", posterior.n_accepted, " accepted particles)")
    for (i, parameter) in enumerate(problem.parameters)
        draws = posterior.population[:,i]
        println(parameter, ": truth=", problem.truth[i], " mean=", round(mean(draws); digits=4),
                " 90% interval=", round.(quantile(draws, [0.05, 0.95]); digits=4))
    end
end
