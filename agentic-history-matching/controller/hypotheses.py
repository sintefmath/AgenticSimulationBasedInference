"""Check allowed permeability patterns and identify proposals covering the same grid cells."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter

from numerical.geology import BandGeometry, Family, band_cells
from pressure_case import PressureCase

MAX_BAND_WIDTH = 0.5


class HomogeneousProposal(BaseModel, extra="forbid", strict=True):
    """Propose the same permeability in every cell."""

    kind: Literal["homogeneous"]


class BandProposal(BaseModel, extra="forbid", strict=True, allow_inf_nan=False):
    """Propose one low-permeability band, with controller-checked grid geometry."""

    kind: Literal["band"]
    orientation: Literal["vertical", "horizontal"]
    center_fraction: float = Field(ge=0, le=1)
    width_fraction: float = Field(gt=0, le=MAX_BAND_WIDTH)


_PROPOSAL_ADAPTER = TypeAdapter(HomogeneousProposal | BandProposal)


@dataclass(frozen=True)
class Hypothesis:
    """Select a hypothesis without changing inference policy.

    Parameters
    ----------
    family : Family
        Reviewed permeability family.
    geometry : BandGeometry or None, optional
        Fixed geometry.
    """

    family: Family
    geometry: BandGeometry | None = None

    def specification(self) -> dict[str, object]:
        if self.family == Family.HOMOGENEOUS:
            return {"kind": "homogeneous"}
        return {
            "kind": "band",
            "orientation": "vertical" if self.family == Family.VERTICAL_BAND else "horizontal",
            "center_fraction": self.geometry.center_fraction,
            "width_fraction": self.geometry.width_fraction,
        }

    def identifier(self, case: PressureCase) -> str:
        """Reuse equivalent rasterized geometries, not just identical submitted JSON."""
        identity = {"family": self.family.value}
        if self.geometry is not None:
            identity["cells"] = np.flatnonzero(band_cells(case, self.geometry)).tolist()
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return "h_" + hashlib.sha256(encoded).hexdigest()[:24]


def parse_hypothesis(value: object, case: PressureCase) -> Hypothesis:
    """Validate a proposed hypothesis.

    Parameters
    ----------
    value : object
        Submitted proposal. Code, unsupported fields, and invalid band
        geometry are rejected.
    """
    proposal = _PROPOSAL_ADAPTER.validate_python(value)
    if isinstance(proposal, HomogeneousProposal):
        return Hypothesis(Family.HOMOGENEOUS)
    geometry = BandGeometry(
        center_fraction=proposal.center_fraction,
        width_fraction=proposal.width_fraction,
    )
    validate_band_geometry(geometry, case)
    family = Family.VERTICAL_BAND if proposal.orientation == "vertical" else Family.HORIZONTAL_BAND
    return Hypothesis(family, geometry)


def band_width_bounds(case: PressureCase) -> tuple[float, float]:
    """Return the narrowest resolvable and widest allowed band width, as domain fractions."""
    return 2 / case.grid_size, MAX_BAND_WIDTH


def validate_band_geometry(geometry: BandGeometry, case: PressureCase) -> None:
    """Validate a band's geometry.

    Parameters
    ----------
    geometry : BandGeometry
        Band geometry must be finite, within the domain, and sufficiently resolved.
    """
    center = geometry.center_fraction
    width = geometry.width_fraction
    narrowest, widest = band_width_bounds(case)
    if not narrowest - 1e-12 <= width <= widest:
        raise ValueError(f"band width must lie in [{narrowest}, {widest}]")
    if not width / 2 <= center <= 1 - width / 2:
        raise ValueError("band must lie inside the domain")
    cells = np.flatnonzero(band_cells(case, geometry))
    if cells.size < 2 or cells[0] == 0 or cells[-1] == case.grid_size - 1:
        raise ValueError(
            "band must span at least two grid rows/columns with background material on both sides"
        )
