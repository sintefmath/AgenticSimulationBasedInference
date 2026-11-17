"""The one-rock example and layout shared by the notebook and its animation."""

import matplotlib.pyplot as plt
import numpy as np

from simulator_1d import CELLS, T_END, permeability, simulate

EXAMPLE_SEED = 2026


def plot_one_rock(theta):
    """Plot permeability and the two-year water front without showing the figure."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for level in range(3):
        xc = (np.arange(CELLS[level]) + 0.5) / CELLS[level]
        _, S = simulate(theta, level, t_end=0.2 * T_END, return_state=True)
        ax[0].step(xc, permeability(theta, level), where="mid", label=f"level {level} ({CELLS[level]} cells)")
        ax[1].step(xc, S, where="mid", label=f"level {level} ({CELLS[level]} cells)")
    ax[0].set(yscale="log", xlim=(0, 1), xlabel="injector → producer", ylabel="permeability",
              title="The rock: permeability along the tube\n(input, does not change in time)")
    ax[1].set(xlim=(0, 1), ylim=(0.15, 0.85), xlabel="injector → producer", ylabel="water saturation",
              title="Snapshot after 2 years:\nwater saturation along the tube")
    ax[0].legend(); ax[1].legend()
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.82, wspace=0.3)
    return fig, ax
