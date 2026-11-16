"""Make tube_levels_1d.png for mlmc_reservoir.ipynb: the same rock on the three 1D grids (levels 0, 1, 2).

Run from the repo root:
    .venv/bin/python literature/figures/tube_levels_1d.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent

# 1D simulator: taken from the notebook, so the figure always matches it
nb = json.loads((HERE.parent / "mlmc_reservoir.ipynb").read_text(encoding="utf-8"))
sim = {}
exec(next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "def simulate" in "".join(c["source"])), sim)

YEARS = 5
cmap = LinearSegmentedColormap.from_list("oil_water", ["#1a7f2e", "#8fd3f4"])   # green = oil, blue = water
theta = np.random.default_rng(7).standard_normal(sim["N_THETA"])                # one rock, used on all three grids

fig, ax = plt.subplots(3, 1, figsize=(11, 3.6))
for level in range(3):
    n = sim["CELLS"][level]
    _, S = sim["simulate"](theta, level, t_end=YEARS / 10 * sim["T_END"], return_state=True)
    im = ax[level].imshow(S[None, :], cmap=cmap, vmin=sim["SWC"], vmax=1 - sim["SOR"], aspect="auto", extent=[0, 1, 0, 1])
    if n <= 64:                                                                  # cell borders (too dense to draw for 256)
        ax[level].vlines(np.arange(1, n) / n, 0, 1, color="white", lw=0.6 if n == 16 else 0.3)
    ax[level].set(xticks=[], yticks=[], xlim=(0, 1))
    ax[level].set_ylabel(f"level {level}\n{n} cells", rotation=0, ha="right", va="center")
ax[0].set_title(f"One rock θ on the three grids, after {YEARS} years")
ax[2].set_xlabel("injector → producer")
fig.subplots_adjust(left=0.1, right=0.84, top=0.88, bottom=0.12, hspace=0.25)
# colour bar from the top of level 0 to the bottom of level 2, with a gap of 0.05 (figure width) to the tubes
top, bottom = ax[0].get_position().y1, ax[2].get_position().y0
cax = fig.add_axes([0.89, bottom, 0.012, top - bottom])
cb = fig.colorbar(im, cax=cax, ticks=[sim["SWC"], 1 - sim["SOR"]])
cb.ax.set_yticklabels(["oil", "water"])
fig.savefig(HERE / "tube_levels_1d.png", dpi=90, facecolor="white")
print("saved", HERE / "tube_levels_1d.png")
