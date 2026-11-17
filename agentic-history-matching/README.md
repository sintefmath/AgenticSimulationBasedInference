# Agentic history matching: a simple model-discovery example

This experiment adapts ideas from [Murphy's (2026) Model Discovery Agent (MDA)](#relationship-to-murphy-2026) to a simple history-matching problem: inferring a hidden permeability field from synthetic pressure observations.

The LLM proposes geological **model structures**, while **ES-MDA** fits their permeability parameters. Residuals guide revisions, validation scores rank models, and protected final observations test the frozen model's pressure forecasts under predefined changes in injection and production rates.

The experiment:

1. Generates the hidden field and noisy pressure observations.
2. Fits and assesses an initial homogeneous model structure.
3. Lets the agent propose alternative model structures, then uses ES-MDA to fit each structure's permeability parameters to the conditioning observations.
4. Locks in the selected model and its fitted ensemble before the final test.
5. Grades against truth and final observations withheld during search.

**Operator-only:** this README describes the hidden truth. Do not give it to the hypothesis-proposing agent, called the *proposer* in the code.

## Quick start

Requires `uv`, Python 3.13, and credentials for an OpenAI or OpenAI-compatible endpoint.

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in this directory's `.env` (see [`.env.example`](.env.example)).

Only OpenAI provider dependencies are installed by default. To use Anthropic or Google, add the corresponding extra to `pydantic-ai-slim` in `pyproject.toml`, regenerate `uv.lock` with `uv lock --project agentic-history-matching` from the repository root, and configure the provider's model and credentials.

From the repository root:

```bash
uv sync --project agentic-history-matching --locked
uv --directory agentic-history-matching run --locked --env-file .env python -m controller configs/run.yaml
```

## Fixed hidden geology

![Top-down map of the 100 m × 100 m reservoir, with a 100 mD background, a full-height 1 mD band spanning x = 60–70 m, and four circle-marked wells (west, east, south, north) with diamond-marked pressure gauges at their centers. Three wells and gauges lie west of the band and one lies east.](assets/reservoir.svg)

Circles mark wells; diamonds at their centers mark colocated pressure gauges. Idealized reservoir pressure is measured.

- **Background:** 100 mD.
- **Vertical band:** a low-permeability strip at 1 mD, 10 m wide, centered at x = 65 m and spanning the full y-direction. “Vertical” refers to the 2D map, not depth.

**Wells:** Well locations stay fixed, but their roles change over time. Each well injects fluid, produces fluid, or is off according to the schedule. See [Wells and controls](#wells-and-controls) for details.

## Which measurements can the agent see?

The agent can see the fixed well locations and injection/production schedule, but cannot change them. The true permeability field and parameters are stored under `protected/` with the final observations and are never sent to the proposer.

All four gauges are observed at the end of every one-hour simulation step: 256 observations in total.

| Measurements | Purpose | What the agent receives |
| --- | --- | --- |
| **Conditioning**, 0–32 hours | Fit each hypothesis's parameters to these 128 observations. | Raw observations, times/locations, predictions, and residuals. |
| **Validation**, 32–48 hours | Rank hypotheses using 64 observations excluded from parameter fitting. | Validation scores, not raw observations. These scores guide search. |
| **Protected final**, 48–64 hours | Grade the frozen selection using 64 further observations. | Nothing during search. |

## How the experiment works

The experiment separates **model-structure search** from **parameter fitting** through two loops:

- **Outer loop:** The LLM proposes geological model structures using observations and numerical feedback.
- **Inner loop:** The controller holds each structure fixed, fits its parameter ensemble with ES-MDA, and scores its forecasts.

A **hypothesis** is a proposed model structure. Each **candidate** pairs one structure with a parameter ensemble for numerical evaluation.

| Term | Meaning here |
| --- | --- |
| **Model structure**, `m` | The assumed layout: the same permeability everywhere (**homogeneous**), or one vertical or horizontal low-permeability strip (**band**). For a band, the structure also specifies its position and width. The pressure equation stays fixed. |
| **Model parameters**, `θ` | The background permeability and, for a band, its permeability relative to the background (the band/background ratio). ES-MDA adjusts these numbers to fit the pressure measurements. |

The code calls the three pattern types **families**: `homogeneous`, `vertical_band`, and `horizontal_band`.

For example, a background permeability of **100 mD** and a band/background ratio of **0.02** give a band permeability of **2 mD**.

### A complete agent run

```text
Setup:
  Prepare hidden truth and noisy observations
  Keep truth and protected-final observations unavailable to the proposer
  Construct SearchController

Outer loop — run_agent_search(search, model):
  Evaluate the predeclared homogeneous baseline using SearchController.evaluate(...)
  Give the proposer public evidence and the completed baseline result

  While search is open and model requests remain:
    Supply current status/budgets; retain earlier evidence/results in history
    LLM chooses one action with a rationale:
      evaluate_hypothesis(structure):
        SearchController.evaluate(structure) → completed feedback → next turn
      finish_search(adequate | inadequate | uncertain):
        Rank completed candidates by validation CRPS (lower is better)
        Reject an adequate claim unless the top-ranked candidate has
          conditioning and validation RMSE <= 2 × measurement-noise std
        If rejected, allow correction; otherwise freeze the outcome
          and selected structure/fitted ensemble

  If the model-request limit is reached before an accepted finish:
    Freeze as uncertain using the same validation-ranking rule

Candidate evaluation — SearchController.evaluate(structure), called above:
  Validate proposal; return existing result for an equivalent structure
  Otherwise check the remaining budget and sample one parameter ensemble from fixed priors
  Inner loop — repeat 4 times, keeping the structure fixed:
    Generate fields and simulate conditioning history
    Update parameters with ES-MDA using conditioning observations only
  Forecast full history with the fitted ensemble
  Save fitted parameters, simulated responses, and noise-added predictions
  Return conditioning RMSE/residuals, validation scores,
         parameter summaries, and the adequacy-check result

Protected assessment — after selection is frozen:
  Read the frozen selection and saved ensembles from the same run
  Grade saved final-period forecasts and truth recovery, without refitting or reselection
```

When search freezes, `search/selection.json` records the validation ranking and selected candidate. `search/run.json` records inference settings, budgets, and the adequacy rule. The proposer transcript, `search/proposer/messages.jsonl`, records the session, stop reason, and final search state.

Only completed candidate evaluations are recorded. Numerical failures abort the experiment rather than leaving failed candidates in the ranking.

### What instructs the agent?

The LLM's instructions are `SYSTEM_PROMPT` in [`controller/proposer.py`](controller/proposer.py). Tool returns add completed evaluations, and each turn receives a compact current-status and budget update.

The agent uses **pressure mismatches**—differences between ensemble-mean simulated pressure and measured pressure, divided by the measurement-noise standard deviation—to propose a different permeability layout. The prompt asks it to explain these mismatches using the conditioning observations and known well controls. Validation scores also guide its proposals.

The agent has two tools:

- `evaluate_hypothesis`: validate, fit, and score one candidate before returning completed results.
- `finish_search`: freeze an `adequate`, `inadequate`, or `uncertain` outcome. An `adequate` claim must pass the numerical error check below; `uncertain` allows insufficient evidence or budget.

The **hypothesis grammar**, the allowed format for model-structure proposals, permits homogeneous permeability or **one** low-permeability band. The LLM chooses orientation, center, and width. For example:

```json
{
  "hypothesis": {
    "kind": "band",
    "orientation": "horizontal",
    "center_fraction": 0.4,
    "width_fraction": 0.2
  },
  "reason": "Test whether a horizontal band explains the directional pressure mismatches."
}
```

The band's orientation, position, and width stay fixed while ES-MDA fits the permeability values. ES-MDA updates background permeability and **contrast**, the band/background permeability ratio. This lets the prior describe the band's relative permeability rather than an independent absolute value. Log and logit transforms enforce positive permeability and a less-permeable band without clipping; this is a modeling choice, not an ES-MDA requirement.

Priors, likelihood, observation error, solver, cutoffs, and scoring remain controller-owned and fixed. Diagonal bands, multiple bands, and new pattern types are unsupported.

## What the simulator calculates

The simulator uses [FiPy](https://www.ctcms.nist.gov/fipy/) to solve pressure diffusion:

$$
S\frac{\partial p}{\partial t}
=\nabla\cdot\left(\frac{k(x;m,\theta)}{\mu}\nabla p\right)+q(x,t)
$$

| Term | Meaning |
| --- | --- |
| $p(x,t)$ | Fluid pressure (Pa) at spatial position $x$ (2D, in m) and time $t$ (s). |
| $k(x;m,\theta)$ | Permeability field (m²), determined by layout $m$ and parameters $\theta$. |
| $m$ | Layout: homogeneous or a band with specified orientation, position, and width. |
| $\theta$ | Permeability parameters: background permeability and, for a band, the band/background ratio. |
| $S$ | Storage coefficient: fluid volume stored per bulk volume per unit pressure increase; fixed at `1e-9` Pa⁻¹. |
| $\mu$ | Fluid dynamic viscosity; fixed at `1e-3` Pa·s. |
| $q(x,t)$ | Injection/production rate per bulk volume (s⁻¹); positive injects, negative produces. |
| $\partial p/\partial t$ | Pressure-change rate (Pa/s). |
| $\nabla p$ | Spatial pressure gradient (Pa/m). |
| $\nabla\cdot$ | Divergence, measuring a vector field's net outward flux per unit volume. |

The equation predicts pressure changes caused by flow, injection, and production. It combines fluid conservation with **Darcy's law**, $\mathbf{u}=-(k/\mu)\nabla p$, where $\mathbf{u}$ is volumetric flow per bulk cross-sectional area (m/s), directed from high to low pressure.

Storage, viscosity, and initial pressure are fixed; exterior boundaries are no-flow. Gravity and multiphase flow are excluded; this is a simplified pressure-inference example, not a realistic reservoir model.

### Wells and controls

Coordinates are fractions of domain length, in `(x, y)` order. Gauges are always colocated with wells, so only well coordinates are stored. Wells and pressure observations follow the order below:

| Well and gauge | Fractional coordinates |
| --- | --- |
| `west` | `(0.20, 0.25)` |
| `east` | `(0.80, 0.25)` |
| `south` | `(0.25, 0.20)` |
| `north` | `(0.25, 0.80)` |

Controls are balanced source/sink pairs. Rates below are multipliers of **`Q = 1e-4` m³/s**, in west/east/south/north well order. Positive rates inject; negative rates produce. Each row applies for eight one-hour steps.

The table below shows the well schedule for the 64-hour experiment, split into conditioning, validation, and protected final periods.

| Time (hours) | Steps | West / Q | East / Q | South / Q | North / Q | Use |
| --- | --- | --- | --- | --- | --- | --- |
| 0–8 | 1–8 | +1.0 | -1.0 | 0 | 0 | Conditioning |
| 8–16 | 9–16 | 0 | 0 | 0 | 0 | Conditioning |
| 16–24 | 17–24 | 0 | 0 | +1.0 | -1.0 | Conditioning |
| 24–32 | 25–32 | 0 | 0 | 0 | 0 | Conditioning |
| 32–40 | 33–40 | -0.6 | +0.6 | 0 | 0 | Validation |
| 40–48 | 41–48 | 0 | 0 | 0 | 0 | Validation |
| 48–56 | 49–56 | +0.4 | -0.4 | -0.8 | +0.8 | Protected final |
| 56–64 | 57–64 | 0 | 0 | 0 | 0 | Protected final |

## How parameter fitting works

For each proposed layout, ES-MDA keeps the geometry fixed and jointly updates these parameters:

| Parameter | Prior | Interpretation |
| --- | --- | --- |
| Background permeability, `k_bg` | `log(k_bg / mD) ~ Normal(log(100), 0.7)` | Positive permeability with prior median **100 mD**. |
| Band/background permeability ratio, `r`, when a band is present | `logit(r) ~ Normal(logit(0.02), 1.3)` | Ratio between zero and one, with prior median **0.02**. |

Band permeability is derived as `k_band = r × k_bg`; it is not fitted independently.

## Relationship to Murphy (2026)

Murphy's MDA learns **mechanistic world models**: scientifically meaningful models that explain observations and predict responses to interventions. This experiment keeps the pressure-diffusion equation fixed and searches only the permeability representation. It adapts MDA's separation between LLM structure proposals and numerical parameter fitting, not the full algorithm (Section 3; Appendix A.3 and A.8). MDA names Murphy's discovery agent, not the ES-MDA inference algorithm.

- **Bounded hypothesis-space expansion:** in Murphy's **M-open** setting, the current hypothesis class can be insufficient and is expanded through new proposals. Here the evaluated candidate pool can grow from the homogeneous model to band structures, but the reviewed grammar never changes; the fixed truth lies within that grammar. This is a bounded analogue of M-open expansion, not open-ended mechanism discovery.
- **Parameter fitting and ranking models by validation scores:** ES-MDA fits the permeability values while holding each proposed layout fixed. Validation **CRPS** (continuous ranked probability score) ranks candidates; the controller does not compute marginal likelihoods, posterior model probabilities, or Bayesian model averaging.
- **Fixed experiment design:** well locations, gauges, and injection/production controls are prescribed. MDA's Value-of-Information experiment selection—choosing what to measure next for its expected benefit—is not implemented.
- **Independent final assessment:** validation scores guide proposals and therefore participate in search. Only protected final observations remain independent of search and assess the frozen selection without refitting.

Bayesian model weighting, adaptive experiment design, and proposals outside the grammar are possible future extensions, not implemented features.

## References

- Murphy, Kevin (2026). *Model Discovery Agent: LLM-Assisted Bayesian Experiment Design for Data-Efficient Discovery of Mechanistic World Models*. Section 3 and Appendix A.3/A.8. Reference key: `murphy2026`; source document: `murphy2026_model_discovery_agent.pdf`.
