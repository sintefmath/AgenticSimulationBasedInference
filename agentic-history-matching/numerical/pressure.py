"""Assemble pressure diffusion with FiPy and reuse its sparse solve per realization."""

from dataclasses import dataclass

import numpy as np
from fipy import CellVariable, DiffusionTerm, Grid2D, TransientTerm
from fipy.solvers.scipy import LinearLUSolver
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from cases.pressure import PressureCase


@dataclass(frozen=True)
class Simulation:
    """Store gauge responses and maximum volume-balance error."""

    responses_pa: NDArray[np.float64]
    max_balance_error_m3: float


class PressureModel:
    """Simulate one permeability field with fixed controls and no-flow exterior boundaries."""

    def __init__(self, case: PressureCase) -> None:
        self.case = case
        self.mesh = Grid2D(
            nx=case.grid_size,
            ny=case.grid_size,
            dx=case.length_m / case.grid_size,
            dy=case.length_m / case.grid_size,
        )
        self.cell_areas_m2 = np.asarray(self.mesh.cellVolumes, dtype=np.float64)
        self.well_cells = self.cell_indices(case.well_positions)
        self.source_rhs = np.zeros((case.total_steps, self.mesh.numberOfCells))
        for well, cell in enumerate(self.well_cells):
            # FiPy's 2D "volumes" are areas, so m3/s well rates need the physical thickness.
            self.source_rhs[:, cell] += case.rates[:, well] / case.thickness_m

    def cell_indices(self, positions: NDArray[np.float64]) -> NDArray[np.int64]:
        """Map fractional coordinates to C-ordered cells containing each point."""
        xy = np.floor(positions * self.case.grid_size).astype(np.int64)
        return xy[:, 1] * self.case.grid_size + xy[:, 0]

    def simulate(self, permeability_m2: NDArray[np.float64], steps: int) -> Simulation:
        """Return time-major gauge responses and verify volume conservation."""
        case = self.case
        permeability = CellVariable(mesh=self.mesh, value=permeability_m2.ravel(order="C"))
        pressure_change = CellVariable(mesh=self.mesh, value=0.0, hasOld=True)
        # Harmonic face mobility preserves resistance at low-permeability interfaces.
        equation = TransientTerm(coeff=case.storage_per_pa) == DiffusionTerm(
            coeff=permeability.harmonicFaceValue / case.viscosity_pa_s
        )
        equation.cacheMatrix()
        equation.justResidualVector(
            var=pressure_change,
            dt=case.time_step_s,
            solver=LinearLUSolver(tolerance=1e-12),
        )
        indptr, indices, data = equation.matrix.CSR
        matrix = csr_matrix(
            (data, indices, indptr), shape=(self.mesh.numberOfCells, self.mesh.numberOfCells)
        )
        # Fixed permeability and time step allow exact reuse of the LU factorization.
        factorization = splu(matrix.tocsc())
        mass = case.storage_per_pa * self.cell_areas_m2 / case.time_step_s
        change = np.zeros(self.mesh.numberOfCells)
        responses = np.empty((steps, len(case.gauge_ids)))
        injected_m3 = 0.0
        balance_error = 0.0
        for step in range(steps):
            change = factorization.solve(mass * change + self.source_rhs[step])
            absolute_pressure = change + case.initial_pressure_pa
            if not np.isfinite(absolute_pressure).all() or not (absolute_pressure > 0).all():
                raise ValueError(f"invalid physical pressure at step {step + 1}")
            injected_m3 += self.source_rhs[step].sum() * case.thickness_m * case.time_step_s
            stored_m3 = float(
                np.sum(change * self.cell_areas_m2) * case.storage_per_pa * case.thickness_m
            )
            balance_error = max(balance_error, abs(stored_m3 - injected_m3))
            responses[step] = absolute_pressure[self.well_cells]

        if balance_error > 1e-8:
            raise ValueError(f"volume balance error exceeds tolerance: {balance_error} m3")
        return Simulation(
            responses_pa=responses.ravel(order="C"),
            max_balance_error_m3=balance_error,
        )
