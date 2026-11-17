"""Execute labeled ensemble realizations serially and account for every attempted simulator call."""

import numpy as np
from numpy.typing import NDArray

from controller.progress import logger
from numerical.geology import BandGeometry, Family, permeability_field
from numerical.pressure import PressureModel


def realization_ids(count: int) -> NDArray[np.str_]:
    """Label ensemble columns in submission order; artifacts and failures share these IDs."""
    return np.array([f"realization_{realization:04d}" for realization in range(count)])


class RealizationFailure(RuntimeError):
    """Identify a failed realization; abort its candidate without dropping or zero-filling it."""

    def __init__(self, family: Family, realization_id: str, reason: str) -> None:
        super().__init__(f"{family.value}/{realization_id}: {reason}")


class LocalExecutor:
    """Own serial ensemble execution and simulator-call accounting for one candidate."""

    def __init__(self, model: PressureModel, max_simulations: int) -> None:
        self.model = model
        self.max_simulations = max_simulations
        self.simulations = 0
        self.max_balance_error_m3 = 0.0

    def batch(
        self,
        family: Family,
        parameters: NDArray[np.float64],
        steps: int,
        geometry: BandGeometry | None = None,
    ) -> NDArray[np.float64]:
        """Return observations by realizations, or fail with a stable realization ID."""
        observations = steps * len(self.model.case.gauge_ids)
        realizations = parameters.shape[1]
        responses = np.empty((observations, realizations))
        milestone = max(1, (realizations + 4) // 5)
        for realization, values in enumerate(parameters.T):
            try:
                field = permeability_field(family, values, self.model.case, geometry)
                self.simulations += 1
                result = self.model.simulate(field, steps)
                responses[:, realization] = result.responses_pa
                self.max_balance_error_m3 = max(
                    self.max_balance_error_m3, result.max_balance_error_m3
                )
            except (ValueError, FloatingPointError, RuntimeError) as error:
                realization_id = realization_ids(realizations)[realization]
                raise RealizationFailure(family, realization_id, str(error)) from error
            completed = realization + 1
            if completed % milestone == 0 or completed == realizations:
                logger.info(
                    "%s batch: %d/%d realizations complete; candidate simulator calls=%d/%d",
                    family.value,
                    completed,
                    realizations,
                    self.simulations,
                    self.max_simulations,
                )
        return responses
