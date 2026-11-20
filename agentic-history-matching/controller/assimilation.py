"""Fit ensemble parameters with ES-MDA (Ensemble Smoother with Multiple Data Assimilation).

The updates use conditioning residuals and sampled observation-error perturbations.
"""

import numpy as np
from iterative_ensemble_smoother import ESMDA
from numpy.typing import NDArray

NUMBER_OF_ITERATIONS = 4


def create_smoother(
    observations: NDArray[np.float64],
    variance: NDArray[np.float64],
    rng: np.random.Generator,
) -> ESMDA:
    """Prepare equal-inflation updates from validated observation vectors."""
    return ESMDA(
        covariance=variance,
        observations=observations,
        alpha=NUMBER_OF_ITERATIONS,
        seed=rng,
    )


def update_parameters(
    smoother: ESMDA,
    parameters: NDArray[np.float64],
    responses: NDArray[np.float64],
    perturbations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Update parameters without mutating inputs or masking invalid responses."""
    if parameters.ndim != 2 or parameters.shape[1] < 2:
        raise ValueError(
            "parameters must have shape (parameters, realizations), with at least two realizations"
        )
    expected = (smoother.observations.size, parameters.shape[1])
    if responses.shape != expected or perturbations.shape != expected:
        raise ValueError("responses and perturbations must have shape (observations, realizations)")
    for values in (parameters, responses, perturbations):
        if values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("assimilation arrays must contain finite float64 values")
    smoother.prepare_assimilation(
        Y=responses,
        truncation=1.0,
        observation_perturbations=perturbations,
    )
    updated = smoother.assimilate_batch(X=parameters)
    if not np.isfinite(updated).all():
        raise ValueError("ES-MDA produced non-finite parameters")
    return updated
