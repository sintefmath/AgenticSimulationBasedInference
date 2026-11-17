"""Condition one reviewed hypothesis and expose public scores and residual diagnostics.

ES-MDA (Ensemble Smoother with Multiple Data Assimilation) fits the parameters using
conditioning residuals and sampled observation-error perturbations.
"""

from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
from iterative_ensemble_smoother import ESMDA
from numpy.typing import NDArray

from cases.pressure import Phase, PressureCase
from controller.artifacts import write_arrays, write_json
from controller.hypotheses import Hypothesis
from controller.local import LocalExecutor, realization_ids
from controller.progress import logger
from numerical.evaluation import (
    conditioning_rmse,
    parameter_summary,
    predictive_ensemble,
    predictive_scores,
)
from numerical.geology import decode_parameters, parameter_names, prior_specification, sample_prior

NUMBER_OF_ITERATIONS = 4


def condition_candidate(
    hypothesis: Hypothesis,
    executor: LocalExecutor,
    conditioning: dict[str, NDArray[np.generic]],
    validation: dict[str, NDArray[np.generic]],
    directory: Path,
    realizations: int,
    seed: int,
) -> dict[str, object]:
    """Condition one fixed-geometry hypothesis with ES-MDA updates, then forecast and validate.

    Only conditioning observations enter updates. The final posterior.npz stores
    realization IDs, fitted parameters, and deterministic/noisy forecasts.
    """
    case = executor.model.case
    family = hypothesis.family
    logger.info(
        "%s: sampling initial parameter ensemble (%d realizations)", directory.name, realizations
    )
    prior_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    update_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    parameters = sample_prior(family, realizations, prior_rng)
    names = np.array(parameter_names(family))
    smoother = ESMDA(
        covariance=conditioning["variance_pa2"],
        observations=conditioning["values_pa"],
        alpha=NUMBER_OF_ITERATIONS,  # An integer gives equal inflation in every update.
        seed=update_rng,
    )
    calls_before = executor.simulations
    started = perf_counter()
    write_json(
        directory / "candidate.json",
        {
            **prior_specification(family),
            "case": case.model_dump(mode="json"),
            "hypothesis": hypothesis.specification(),
            "realizations": realizations,
            "prior_seed": [seed, 0],
            "update_seed": [seed, 1],
            "prior_rng_state_after": prior_rng.bit_generator.state,
            "algorithm": "ES-MDA",
            "inflation": smoother.alpha.tolist(),
            "truncation": 1.0,
            "localization": None,
            "failed_realization_policy": (
                "abort experiment; do not drop, retry, or zero-fill realizations"
            ),
        },
    )
    history: list[float] = []
    updates = smoother.num_assimilations()
    for step in range(updates):
        stage = f"fitting update {step + 1}/{updates}"
        logger.info("%s: %s; simulating conditioning responses", directory.name, stage)
        responses = executor.batch(
            family, parameters, case.conditioning_end_step, hypothesis.geometry
        )
        history.append(conditioning_rmse(responses, conditioning["values_pa"]))
        perturbations = update_rng.normal(0, case.observation_std_pa, size=responses.shape)
        logger.info(
            "%s: %s; updating parameters; pre-update conditioning RMSE=%.2f Pa",
            directory.name,
            stage,
            history[-1],
        )
        smoother.prepare_assimilation(
            Y=responses, truncation=1.0, observation_perturbations=perturbations
        )
        parameters = smoother.assimilate_batch(X=parameters)
        # Reject non-finite updates and band ratios that round to 0 or 1.
        decode_parameters(family, parameters)
        logger.info("%s: %s complete", directory.name, stage)
    logger.info(
        "%s: forecasting full history (%d time steps, %d realizations)",
        directory.name,
        case.total_steps,
        realizations,
    )
    responses = executor.batch(family, parameters, case.total_steps, hypothesis.geometry)
    predictive_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    predictions = predictive_ensemble(responses, case.observation_std_pa, predictive_rng)
    posterior_checksum = write_arrays(
        directory / "posterior.npz",
        parameters=parameters,
        responses_pa=responses,
        predictions_pa=predictions,
        realization_ids=realization_ids(realizations),
        parameter_names=names,
        observation_ids=case.observation_ids,
    )
    logger.info("%s: forecast saved; scoring validation observations", directory.name)
    record = {
        "family": family.value,
        "hypothesis": hypothesis.specification(),
        "conditioning_rmse_history_pa": history,
        "conditioning_rmse_final_pa": conditioning_rmse(
            responses[case.phase_mask(Phase.CONDITIONING)], conditioning["values_pa"]
        ),
        "validation": predictive_scores(
            predictions[case.phase_mask(Phase.VALIDATION)], validation["values_pa"]
        ),
        "parameters": parameter_summary(family, parameters),
        "posterior_sha256": posterior_checksum,
        "predictive_seed": [seed, 2],
        "predictive_rng_state_after": predictive_rng.bit_generator.state,
        "diagnostics": residual_diagnostics(
            responses[case.phase_mask(Phase.CONDITIONING)], conditioning["values_pa"], case
        ),
    }
    record["simulator_calls"] = executor.simulations - calls_before
    record["wall_seconds"] = perf_counter() - started
    record["max_balance_error_m3"] = executor.max_balance_error_m3
    write_json(directory / "result.json", record)
    logger.info("%s: result saved to %s", directory.name, directory / "result.json")
    return record


def residual_diagnostics(
    responses: NDArray[np.float64], observations: NDArray[np.float64], case: PressureCase
) -> dict[str, object]:
    """Return signed, noise-standardized conditioning residuals by gauge, time, and pulse.

    Residual = (predicted mean - observation) / observation std; positive means overprediction.
    Observed and predicted pressure changes are in bar.
    """
    times_s = case.observation_times_s
    times_s = times_s[times_s <= case.conditioning_end_s]
    shape = (times_s.size, len(case.gauge_ids))
    mean = responses.mean(axis=1)
    residual = ((mean - observations) / case.observation_std_pa).reshape(shape)
    observed = case.pressure_change_bar(observations).reshape(shape)
    predicted = case.pressure_change_bar(mean).reshape(shape)
    periods = []
    boundaries = {0.0, case.conditioning_end_s}
    for control in case.well_controls:
        boundaries.update(
            time_s
            for time_s in (control.start_s, control.end_s)
            if time_s < case.conditioning_end_s
        )
    for start_s, end_s in pairwise(sorted(boundaries)):
        values = residual[(times_s > start_s) & (times_s <= end_s)]
        periods.append(
            {
                "period": f"{start_s / 3600:g}-{end_s / 3600:g} hours",
                "mean_signed_residual_std": values.mean(axis=0).tolist(),
                "rmse_std": np.sqrt(np.mean(values**2, axis=0)).tolist(),
            }
        )
    return {
        "residual_definition": "(predicted mean - observation) / observation standard deviation",
        "gauge_ids": list(case.gauge_ids),
        "times_hours": (times_s / 3600).tolist(),
        "observed_pressure_change_bar": observed.tolist(),
        "predicted_pressure_change_bar": predicted.tolist(),
        "signed_residual_std": residual.tolist(),
        "periods": periods,
    }
