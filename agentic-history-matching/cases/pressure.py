"""Define pressure-case geometry, controls, observations, and data cutoffs."""

from enum import StrEnum
from math import isclose
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Phase(StrEnum):
    """Separate conditioning, validation, and protected final observations."""

    CONDITIONING = "conditioning"
    VALIDATION = "validation"
    FINAL = "final"


type FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
type PositiveFiniteFloat = Annotated[FiniteFloat, Field(gt=0)]


class WellControlInterval(BaseModel):
    """Schedule balanced west/east/south/north rate multipliers over [start_s, end_s)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    start_s: FiniteFloat = Field(ge=0)
    end_s: PositiveFiniteFloat
    multipliers: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def _validate_interval(self) -> Self:
        if self.start_s >= self.end_s:
            raise ValueError("well-control intervals must end after they start")
        if not isclose(sum(self.multipliers), 0.0, abs_tol=1e-12):
            raise ValueError("well-control rates must sum to zero")
        return self


class PressureCase(BaseModel):
    """Specify known grid, physics, controls, and observation settings, in SI units."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    grid_size: int = Field(default=20, strict=True, ge=4, multiple_of=2)
    length_m: PositiveFiniteFloat = 100.0
    thickness_m: PositiveFiniteFloat = 10.0
    storage_per_pa: PositiveFiniteFloat = 1e-9
    viscosity_pa_s: PositiveFiniteFloat = 1e-3
    initial_pressure_pa: PositiveFiniteFloat = 2e7
    time_step_s: PositiveFiniteFloat = 3600.0
    conditioning_end_s: ClassVar[float] = 32 * 3600.0
    validation_end_s: ClassVar[float] = 48 * 3600.0
    final_end_s: ClassVar[float] = 64 * 3600.0
    rate_m3_s: PositiveFiniteFloat = 1e-4
    observation_std_pa: PositiveFiniteFloat = 1000.0
    well_controls: tuple[WellControlInterval, ...] = (
        WellControlInterval(start_s=0.0, end_s=8 * 3600.0, multipliers=(1.0, -1.0, 0.0, 0.0)),
        WellControlInterval(
            start_s=16 * 3600.0, end_s=24 * 3600.0, multipliers=(0.0, 0.0, 1.0, -1.0)
        ),
        WellControlInterval(
            start_s=32 * 3600.0, end_s=40 * 3600.0, multipliers=(-0.6, 0.6, 0.0, 0.0)
        ),
        WellControlInterval(
            start_s=48 * 3600.0, end_s=56 * 3600.0, multipliers=(0.4, -0.4, -0.8, 0.8)
        ),
    )

    @model_validator(mode="after")
    def _validate_schedule(self) -> Self:
        for name, cutoff_s in (
            ("conditioning_end_s", self.conditioning_end_s),
            ("validation_end_s", self.validation_end_s),
            ("final_end_s", self.final_end_s),
        ):
            steps = cutoff_s / self.time_step_s
            if not steps.is_integer():
                raise ValueError(f"{name} must align with a simulation time step")
        previous_end_s = 0.0
        for control in self.well_controls:
            if control.end_s > self.final_end_s:
                raise ValueError("well-control intervals must lie within the experiment")
            if control.start_s < previous_end_s:
                raise ValueError("well-control intervals must be ordered and non-overlapping")
            if any(
                not (time_s / self.time_step_s).is_integer()
                for time_s in (control.start_s, control.end_s)
            ):
                raise ValueError("well-control intervals must align with simulation time steps")
            previous_end_s = control.end_s
        return self

    @property
    def conditioning_end_step(self) -> int:
        """Return the validated simulation-step index of the conditioning cutoff."""
        return int(self.conditioning_end_s / self.time_step_s)

    @property
    def total_steps(self) -> int:
        """Return the validated number of simulation steps through the final cutoff."""
        return int(self.final_end_s / self.time_step_s)

    @property
    def well_positions(self) -> NDArray[np.float64]:
        """Return fractional well coordinates in west/east/south/north observation order."""
        return np.array([[0.2, 0.25], [0.8, 0.25], [0.25, 0.2], [0.25, 0.8]])

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        """Return colocated gauge names in west/east/south/north well order."""
        return ("west", "east", "south", "north")

    @property
    def rates(self) -> NDArray[np.float64]:
        """Expand [start, end) controls into balanced per-step rates; gaps have all wells off."""
        rates = np.zeros((self.total_steps, len(self.well_positions)), dtype=np.float64)
        for control in self.well_controls:
            start = int(control.start_s / self.time_step_s)
            end = int(control.end_s / self.time_step_s)
            rates[start:end] = np.asarray(control.multipliers) * self.rate_m3_s
        return rates

    @property
    def observation_steps(self) -> NDArray[np.int64]:
        """Return every completed simulation-step index, excluding the initial state."""
        return np.arange(1, self.total_steps + 1)

    @property
    def observation_times_s(self) -> NDArray[np.float64]:
        """Return the elapsed physical time of each observation sample."""
        return self.observation_steps * self.time_step_s

    @property
    def observation_ids(self) -> NDArray[np.str_]:
        """Label observations in time-major, then well order."""
        return np.array(
            [
                f"step_{step:04d}:{gauge}"
                for step in self.observation_steps
                for gauge in self.gauge_ids
            ]
        )

    def phase_mask(self, phase: Phase) -> NDArray[np.bool_]:
        """Select observations for conditioning, validation, or final evaluation."""
        times_s = np.repeat(self.observation_times_s, len(self.gauge_ids))
        match phase:
            case Phase.CONDITIONING:
                return times_s <= self.conditioning_end_s
            case Phase.VALIDATION:
                return (times_s > self.conditioning_end_s) & (times_s <= self.validation_end_s)
            case Phase.FINAL:
                return times_s > self.validation_end_s

    def specification(self) -> dict[str, object]:
        """Record case settings, units, controls, observation order, and cutoffs."""
        return {
            "schema_version": 6,
            "case_id": "pressure_diffusion_v1",
            "settings": self.model_dump(mode="json"),
            "boundary_condition": "zero_flux",
            "pressure_unit": "Pa",
            "permeability_unit": "m2",
            "rate_unit": "m3/s",
            "cell_order": "C: y rows, x columns",
            "observation_order": "time, then well",
            "observation_ids": self.observation_ids.tolist(),
            "well_positions_fraction": self.well_positions.tolist(),
            "rates_m3_s": self.rates.tolist(),
            "conditioning_end_step": self.conditioning_end_step,
            "validation_end_step": int(self.validation_end_s / self.time_step_s),
            "final_end_step": self.total_steps,
            # Keep the fixed label: load_case validates it as part of the artifact format.
            "observation_error": "independent Gaussian, fixed standard deviation",
        }
