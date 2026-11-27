"""Define fixed pressure-case geometry, controls, observations, and data cutoffs."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class Phase(StrEnum):
    """Separate conditioning, validation, and protected final observations."""

    CONDITIONING = "conditioning"
    VALIDATION = "validation"
    FINAL = "final"


@dataclass(frozen=True)
class WellControlInterval:
    """Schedule well controls over an interval.

    Parameters
    ----------
    start_s, end_s : float
        Bounds of the half-open interval ``[start_s, end_s)``.
    multipliers : tuple of float
        Balanced rate multipliers in west/east/south/north order.
    """

    start_s: float
    end_s: float
    multipliers: tuple[float, float, float, float]


class PressureCase:
    """Describe the fixed pressure experiment in SI units."""

    boundary_condition = "zero_flux"
    grid_size = 20
    length_m = 100.0
    thickness_m = 10.0
    storage_per_pa = 1e-9
    viscosity_pa_s = 1e-3
    initial_pressure_pa = 2e7
    time_step_s = 3600.0
    conditioning_end_s = 32 * 3600.0
    validation_end_s = 48 * 3600.0
    final_end_s = 64 * 3600.0
    rate_m3_s = 1e-4
    observation_std_pa = 1000.0
    well_controls = (
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

    @property
    def conditioning_end_step(self) -> int:
        """Return the simulation-step index of the conditioning cutoff."""
        return int(self.conditioning_end_s / self.time_step_s)

    @property
    def total_steps(self) -> int:
        """Return the number of simulation steps through the final cutoff."""
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
        """Select observations for the requested phase.

        Parameters
        ----------
        phase : Phase
            Conditioning, validation, or final evaluation.

        Returns
        -------
        numpy.ndarray
            Mask selecting observations in that phase.
        """
        times_s = np.repeat(self.observation_times_s, len(self.gauge_ids))
        match phase:
            case Phase.CONDITIONING:
                return times_s <= self.conditioning_end_s
            case Phase.VALIDATION:
                return (times_s > self.conditioning_end_s) & (times_s <= self.validation_end_s)
            case Phase.FINAL:
                return times_s > self.validation_end_s

    def pressure_change_bar(self, pressure_pa: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert pressure to a change from the initial pressure.

        Parameters
        ----------
        pressure_pa : numpy.ndarray
            Absolute pressure in Pa.

        Returns
        -------
        numpy.ndarray
            Change from initial pressure in bar.
        """
        return (pressure_pa - self.initial_pressure_pa) / 1e5

    def settings(self) -> dict[str, object]:
        """Return JSON-ready settings shared by artifacts and the proposer."""
        return {
            "grid_size": self.grid_size,
            "length_m": self.length_m,
            "thickness_m": self.thickness_m,
            "storage_per_pa": self.storage_per_pa,
            "viscosity_pa_s": self.viscosity_pa_s,
            "initial_pressure_pa": self.initial_pressure_pa,
            "time_step_s": self.time_step_s,
            "rate_m3_s": self.rate_m3_s,
            "observation_std_pa": self.observation_std_pa,
            "well_controls": [
                {
                    "start_s": control.start_s,
                    "end_s": control.end_s,
                    "multipliers": list(control.multipliers),
                }
                for control in self.well_controls
            ],
        }
