"""Make simulator_diagram.png for mlmc_reservoir.ipynb: what goes in and out of simulate(theta, level).

Run from the repo root:
    uv run --project MLMCReservoir python MLMCReservoir/figures/simulator_diagram.py
"""
from pathlib import Path

import matplotlib
import numpy as np

import simulator_1d as sim

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent

LEVEL = 0
cmap = LinearSegmentedColormap.from_list("oil_water", ["#1a7f2e", "#8fd3f4"])   # green = oil, blue = water
theta = np.random.default_rng(7).standard_normal(sim.N_THETA)
n_fine, n = sim.CELLS[-1], sim.CELLS[LEVEL]
k_fine = np.exp(sim.KL @ theta)
k_level = sim.permeability(theta, LEVEL)
P = sim.simulate(theta, LEVEL)
YEARS = [2, 5, 10]
S_t = np.array([sim.simulate(theta, LEVEL, t_end=y / 10 * sim.T_END, return_state=True)[1] for y in YEARS])

fig = plt.figure(figsize=(15, 3.8))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1.2, 1.2, 1.2, 0.8], wspace=0.85, left=0.035, right=0.985, top=0.68, bottom=0.18)
ax = [fig.add_subplot(gs[0, i]) for i in range(5)]

ax[0].bar(np.arange(1, theta.size + 1), theta, color="#7f7f7f")
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set(title="Input θ\n30 random numbers", xlabel="i", xticks=[1, 10, 20, 30], yticks=[])

ax[1].plot((np.arange(n_fine) + 0.5) / n_fine, k_fine, color="#8c564b")
ax[1].set(title=f"Permeability\nfine grid, {n_fine} cells", yscale="log", xlim=(0, 1), xlabel="injector → producer", yticks=[])

ax[2].plot((np.arange(n_fine) + 0.5) / n_fine, k_fine, color="#8c564b", alpha=0.25)
ax[2].stairs(k_level, np.linspace(0, 1, n + 1), color="#8c564b", lw=2)
ax[2].set(title=f"Averaged to the grid\nlevel {LEVEL}: {n} cells", yscale="log", xlim=(0, 1), xlabel="injector → producer", yticks=[])

ax[3].imshow(S_t, cmap=cmap, vmin=sim.SWC, vmax=1 - sim.SOR, aspect="auto", extent=[0, 1, len(YEARS), 0])
ax[3].vlines(np.arange(1, n) / n, 0, len(YEARS), color="white", lw=0.6)
ax[3].hlines(np.arange(1, len(YEARS)), 0, 1, color="white", lw=3)
ax[3].set(title="Water–oil flow\nblue = water, green = oil", xticks=[], yticks=[], xlabel="injector → producer")
for r, y in enumerate(YEARS):
    ax[3].text(0.97, r + 0.5, f"{y} years", color="white", ha="right", va="center", fontsize=9, fontweight="bold")

ax[4].axis("off")
ax[4].set_title("Output P\nrecovery factor")
ax[4].text(0.5, 0.5, f"{P:.1f} %", ha="center", va="center", fontsize=22, fontweight="bold", transform=ax[4].transAxes,
           bbox=dict(boxstyle="round,pad=0.5", fc="#f2f2f2", ec="#7f7f7f"))

for a in ax[1:3]:                                                                # log axes: no tick labels
    a.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.tick_params(axis="y", which="both", left=False)

# arrows between the panels, with the method above and a short detail below
ARROW_LABELS = [
    ("KL expansion", r"$k = e^{\Phi\theta}$"),
    ("harmonic average", f"{n_fine // n} fine cells → 1"),
    ("finite volumes", "(upwind)"),
    ("oil produced", "/ oil in place"),
]
for (a, b), (above, below) in zip(zip(ax[:-1], ax[1:]), ARROW_LABELS):
    x0, x1 = a.get_position().x1 + 0.005, b.get_position().x0 - 0.005
    y = (a.get_position().y0 + a.get_position().y1) / 2
    fig.add_artist(matplotlib.patches.FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color="black"))
    fig.text((x0 + x1) / 2, y + 0.03, above, ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.text((x0 + x1) / 2, y - 0.03, below, ha="center", va="top", fontsize=9)

# the second input: the level, entering at the averaging step
p = ax[2].get_position()
fig.text((p.x0 + p.x1) / 2, 0.985, "Input level ℓ (0, 1 or 2)", ha="center", va="top", fontsize=10, fontweight="bold")
fig.add_artist(matplotlib.patches.FancyArrowPatch(((p.x0 + p.x1) / 2, 0.92), ((p.x0 + p.x1) / 2, p.y1 + 0.16), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, color="black"))
fig.text(0.5, 0.02, "Fixed inside the simulator: porosity, viscosities, pressure difference between the wells, 10 years",
         ha="center", fontsize=9, color="#555555")

fig.savefig(HERE / "simulator_diagram.png", dpi=90, facecolor="white")
print("saved", HERE / "simulator_diagram.png", f"P = {P:.2f} %")
