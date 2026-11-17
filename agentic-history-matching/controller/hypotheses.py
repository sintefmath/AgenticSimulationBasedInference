"""Check allowed permeability patterns and identify proposals covering the same grid cells."""

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter

from cases.pressure import PressureCase
from numerical.geology import BandGeometry, Family, band_cells


class HomogeneousProposal(BaseModel, extra="forbid", strict=True):
    """Propose the same permeability in every cell."""

    kind: Literal["homogeneous"]


class BandProposal(BaseModel, extra="forbid", strict=True, allow_inf_nan=False):
    """Propose one low-permeability band, with controller-checked grid geometry."""

    kind: Literal["band"]
    orientation: Literal["vertical", "horizontal"]
    center_fraction: float = Field(ge=0, le=1)
    width_fraction: float = Field(gt=0, le=0.5)


_PROPOSAL_ADAPTER = TypeAdapter(HomogeneousProposal | BandProposal)


@dataclass(frozen=True)
class Hypothesis:
    """Select a reviewed family and fixed geometry without changing inference policy."""

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
    """Reject code, unsupported fields, and invalid band geometry at the proposal boundary."""
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


def validate_band_geometry(geometry: BandGeometry, case: PressureCase) -> None:
    """Reject non-finite, out-of-domain, or insufficiently resolved band geometry."""
    center = geometry.center_fraction
    width = geometry.width_fraction
    if not (isfinite(center) and isfinite(width)):
        raise ValueError("center_fraction and width_fraction must be finite numbers")
    if not 2 / case.grid_size - 1e-12 <= width <= 0.5:
        raise ValueError(f"band width must lie in [{2 / case.grid_size}, 0.5]")
    if not width / 2 <= center <= 1 - width / 2:
        raise ValueError("band must lie inside the domain")
    cells = np.flatnonzero(band_cells(case, geometry))
    if cells.size < 2 or cells[0] == 0 or cells[-1] == case.grid_size - 1:
        raise ValueError(
            "band must span at least two grid rows/columns with background material on both sides"
        )
