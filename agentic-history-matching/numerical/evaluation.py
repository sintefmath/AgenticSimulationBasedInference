"""Score predictive ensembles independently of conditioning and summarize posterior parameters."""

import numpy as np
from numpy.typing import NDArray

from numerical.geology import Family, decode_parameters


def predictive_ensemble(
    responses_pa: NDArray[np.float64], std_pa: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Include declared measurement error in forecasts of noisy observations."""
    return responses_pa + rng.normal(0, std_pa, size=responses_pa.shape)


def predictive_scores(
    predictions_pa: NDArray[np.float64], observations_pa: NDArray[np.float64]
) -> dict[str, float]:
    """Compute scores for a predictive ensemble.

    Parameters
    ----------
    predictions_pa : numpy.ndarray
        Predictions with observation rows and realization columns.

    Returns
    -------
    dict
        CRPS, RMSE, and 90% predictive-interval coverage and width. CRPS, RMSE,
        and interval width are in Pa.

    Notes
    -----
    CRPS (continuous ranked probability score) scores the ensemble, not just its
    mean. Intervals use the 5th and 95th percentiles.
    """
    realizations = predictions_pa.shape[1]
    ordered = np.sort(predictions_pa, axis=1)
    # Sorting gives half the mean pairwise distance without allocating an N-by-N array.
    weights = 2 * np.arange(1, realizations + 1) - realizations - 1
    half_pairwise_distance = np.sum(ordered * weights, axis=1) / realizations**2
    crps = np.mean(np.abs(predictions_pa - observations_pa[:, np.newaxis]), axis=1)
    crps -= half_pairwise_distance
    lower, upper = np.quantile(predictions_pa, [0.05, 0.95], axis=1)
    return {
        "crps_pa": float(crps.mean()),
        "rmse_pa": float(np.sqrt(np.mean((predictions_pa.mean(axis=1) - observations_pa) ** 2))),
        "coverage_90": float(np.mean((observations_pa >= lower) & (observations_pa <= upper))),
        "interval_width_90_pa": float(np.mean(upper - lower)),
    }


def conditioning_rmse(
    responses_pa: NDArray[np.float64], observations_pa: NDArray[np.float64]
) -> float:
    """Return conditioning RMSE of the ensemble-mean response."""
    return float(np.sqrt(np.mean((responses_pa.mean(axis=1) - observations_pa) ** 2)))


def parameter_summary(
    family: Family, parameters: NDArray[np.float64]
) -> dict[str, dict[str, float]]:
    """Summarize the posterior in physical units.

    Parameters
    ----------
    parameters : numpy.ndarray
        Posterior parameters.

    Returns
    -------
    dict
        Parameter summaries, including joint band permeability.

    Notes
    -----
    Band permeability is computed per realization before summarizing to
    preserve parameter dependence.
    """
    physical = decode_parameters(family, parameters)
    rows = {"background_md": physical[0]}
    if family != Family.HOMOGENEOUS:
        rows["band_ratio"] = physical[1]
        rows["band_md"] = physical[0] * physical[1]
    return {
        name: {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
        for name, values in rows.items()
    }
