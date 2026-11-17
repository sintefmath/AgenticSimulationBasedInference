"""Generate homogeneous or single-band permeability fields from continuous parameters."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from pressure_case import PressureCase

MILLIDARCY_M2 = 9.869233e-16
# Fixed Gaussian priors in log/logit parameter space.
LOG_BACKGROUND_PRIOR_MEAN = float(np.log(100.0))
LOG_BACKGROUND_PRIOR_STD = 0.7
LOGIT_RATIO_PRIOR_MEAN = float(logit(0.02))
LOGIT_RATIO_PRIOR_STD = 1.3


class Family(StrEnum):
    """Name the three allowed permeability patterns; conditioning never blends family IDs."""

    HOMOGENEOUS = "homogeneous"
    VERTICAL_BAND = "vertical_band"
    HORIZONTAL_BAND = "horizontal_band"


@dataclass(frozen=True)
class BandGeometry:
    """Fix band geometry throughout conditioning.

    Parameters
    ----------
    center_fraction, width_fraction : float
        Band center and width as fractions of the domain.
    """

    center_fraction: float
    width_fraction: float


def band_cells(case: PressureCase, geometry: BandGeometry) -> NDArray[np.bool_]:
    """Select grid rows/columns whose cell centers lie inside the band."""
    centers = (np.arange(case.grid_size) + 0.5) / case.grid_size
    return np.abs(centers - geometry.center_fraction) < geometry.width_fraction / 2


def parameter_names(family: Family) -> tuple[str, ...]:
    """Return the names of the transformed permeability parameters.

    Returns
    -------
    tuple of str
        Names for log background permeability and, for banded families, the
        logit band-to-background permeability ratio.
    """
    if family == Family.HOMOGENEOUS:
        return ("log_background_md",)
    return ("log_background_md", "logit_band_ratio")


def sample_prior(
    family: Family, realizations: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sample fixed Gaussian priors in log/logit parameter space."""
    background = rng.normal(LOG_BACKGROUND_PRIOR_MEAN, LOG_BACKGROUND_PRIOR_STD, size=realizations)
    if family == Family.HOMOGENEOUS:
        return background[np.newaxis, :]
    contrast = rng.normal(LOGIT_RATIO_PRIOR_MEAN, LOGIT_RATIO_PRIOR_STD, size=realizations)
    return np.vstack((background, contrast))


def encode_parameters(
    family: Family, background_md: float, band_ratio: float
) -> NDArray[np.float64]:
    """Convert physical values to log/logit parameters for one realization.

    Parameters
    ----------
    background_md : float
        Background permeability in mD.
    band_ratio : float
        Band-to-background permeability ratio, used for banded families.

    Returns
    -------
    numpy.ndarray
        ``[log(background_md)]`` for the homogeneous family or
        ``[log(background_md), logit(band_ratio)]`` for banded families.
        ``log`` denotes the natural logarithm.

    Notes
    -----
    These transforms let ES-MDA update the parameters without directly enforcing
    physical bounds. They are a modeling choice, not an ES-MDA requirement.
    """
    if not np.isfinite(background_md) or background_md <= 0:
        raise ValueError("background permeability must be finite and positive")
    if family == Family.HOMOGENEOUS:
        return np.array([np.log(background_md)])
    if not np.isfinite(band_ratio) or not 0 < band_ratio < 1:
        raise ValueError(
            "band/background permeability ratio must be greater than 0 and less than 1"
        )
    return np.array([np.log(background_md), logit(band_ratio)])


def decode_parameters(family: Family, parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert log/logit parameters back to physical values.

    Parameters
    ----------
    parameters : numpy.ndarray
        Log/logit parameters with one column per realization.

    Returns
    -------
    numpy.ndarray
        One column per realization, with rows ordered as ``[background_md]`` for
        the homogeneous family or ``[background_md, band_ratio]`` for banded
        families. ``background_md`` is background permeability in mD, and
        ``band_ratio`` is the band-to-background permeability ratio.
    """
    if parameters.ndim != 2 or parameters.shape[0] != len(parameter_names(family)):
        raise ValueError(
            "parameter shape must be (parameters, realizations) for the chosen pattern type"
        )
    if not np.isfinite(parameters).all():
        raise ValueError("parameters must be finite")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        background = np.exp(parameters[0])
    if family == Family.HOMOGENEOUS:
        return background[np.newaxis, :]
    ratio = expit(parameters[1])
    if not ((ratio > 0) & (ratio < 1)).all():
        raise ValueError(
            "converted band/background ratio rounded to 0 or 1; it must be strictly between them"
        )
    return np.vstack((background, ratio))


def permeability_field(
    family: Family,
    parameters: NDArray[np.float64],
    case: PressureCase,
    geometry: BandGeometry | None = None,
) -> NDArray[np.float64]:
    """Generate a 2D intrinsic permeability field in square meters (m^2).

    Convert decoded permeability from millidarcies (mD) to SI units.

    Parameters
    ----------
    geometry : BandGeometry or None, optional
        Required for banded families; not needed for homogeneous fields.

    Returns
    -------
    field : numpy.ndarray
        Permeability indexed as ``field[y, x]``, with rows along y and columns
        along x.
    """
    physical = decode_parameters(family, parameters[:, np.newaxis])[:, 0]
    field = np.full((case.grid_size, case.grid_size), physical[0] * MILLIDARCY_M2)
    match family:
        case Family.VERTICAL_BAND:
            field[:, band_cells(case, geometry)] *= physical[1]
        case Family.HORIZONTAL_BAND:
            field[band_cells(case, geometry), :] *= physical[1]
    if not np.isfinite(field).all() or not (field > 0).all():
        raise ValueError("generated permeability must be finite and positive")
    return field


def prior_specification(family: Family) -> dict[str, object]:
    """Describe parameter priors; std denotes standard deviation, not variance."""
    return {
        "family": family.value,
        "recipe_id": "fixed_permeability_patterns_v1",
        "parameter_names": list(parameter_names(family)),
        "background": {
            "distribution": "normal in natural log(permeability / mD)",
            "mean": LOG_BACKGROUND_PRIOR_MEAN,
            "std": LOG_BACKGROUND_PRIOR_STD,
        },
        "band_ratio": (
            None
            if family == Family.HOMOGENEOUS
            else {
                "distribution": "normal in logit(ratio) = log(ratio / (1-ratio))",
                "mean": LOGIT_RATIO_PRIOR_MEAN,
                "std": LOGIT_RATIO_PRIOR_STD,
            }
        ),
        "millidarcy_m2": MILLIDARCY_M2,
    }
