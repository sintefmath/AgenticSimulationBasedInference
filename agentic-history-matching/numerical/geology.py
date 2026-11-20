"""Generate homogeneous or single-band permeability fields from continuous parameters."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from cases.pressure import PressureCase

MILLIDARCY_M2 = 9.869233e-16


class Family(StrEnum):
    """Name the three allowed permeability patterns; conditioning never blends family IDs."""

    HOMOGENEOUS = "homogeneous"
    VERTICAL_BAND = "vertical_band"
    HORIZONTAL_BAND = "horizontal_band"


@dataclass(frozen=True)
class BandGeometry:
    """Fix band center and width as domain fractions throughout conditioning."""

    center_fraction: float
    width_fraction: float


def band_cells(case: PressureCase, geometry: BandGeometry) -> NDArray[np.bool_]:
    """Select grid rows/columns whose cell centers lie inside the band."""
    centers = (np.arange(case.grid_size) + 0.5) / case.grid_size
    return np.abs(centers - geometry.center_fraction) < geometry.width_fraction / 2


def parameter_names(family: Family) -> tuple[str, ...]:
    """Name transformed parameters: log background permeability and logit band/background ratio."""
    if family == Family.HOMOGENEOUS:
        return ("log_background_md",)
    return ("log_background_md", "logit_band_ratio")


def sample_prior(
    family: Family, realizations: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sample fixed Gaussian priors in log/logit parameter space."""
    if realizations < 2:
        raise ValueError("at least two ensemble realizations are required")
    background = rng.normal(np.log(100.0), 0.7, size=realizations)
    if family == Family.HOMOGENEOUS:
        return background[np.newaxis, :]
    contrast = rng.normal(logit(0.02), 1.3, size=realizations)
    return np.vstack((background, contrast))


def encode_parameters(
    family: Family, background_md: float, band_ratio: float = 0.01
) -> NDArray[np.float64]:
    """Encode permeability in mD and band/background ratio as log/logit parameters.

    These coordinates allow unconstrained ES-MDA updates while decoding enforces
    positive permeability and a band/background ratio in (0, 1), without clipping.
    The transforms are a modeling choice, not an ES-MDA requirement.
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
    """Decode a parameters-by-realizations matrix; reject non-finite values and invalid ratios."""
    if parameters.ndim != 2 or parameters.shape[0] != len(parameter_names(family)):
        raise ValueError(
            "parameter shape must be (parameters, realizations) for the chosen pattern type"
        )
    if parameters.dtype != np.float64 or not np.isfinite(parameters).all():
        raise ValueError("parameters must be finite float64 values")
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
    The returned array is indexed as field[y, x]: rows along y, columns along x.
    Band families require explicit geometry; homogeneous fields need none.
    """
    physical = decode_parameters(family, parameters[:, np.newaxis])[:, 0]
    field = np.full((case.grid_size, case.grid_size), physical[0] * MILLIDARCY_M2)
    if family != Family.HOMOGENEOUS:
        if geometry is None:
            raise ValueError("band families require explicit geometry")
        band = band_cells(case, geometry)
        match family:
            case Family.VERTICAL_BAND:
                field[:, band] *= physical[1]
            case Family.HORIZONTAL_BAND:
                field[band, :] *= physical[1]
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
            "mean": float(np.log(100)),
            "std": 0.7,
        },
        "band_ratio": (
            None
            if family == Family.HOMOGENEOUS
            else {
                "distribution": "normal in logit(ratio) = log(ratio / (1-ratio))",
                "mean": float(logit(0.02)),
                "std": 1.3,
            }
        ),
        "millidarcy_m2": MILLIDARCY_M2,
    }
