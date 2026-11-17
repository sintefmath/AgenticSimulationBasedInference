"""Make water_front_levels.gif for mlmc_reservoir.ipynb: the same rock as the notebook's
"One rock θ on the three grids" cell, from 0 to 10 years.

The GIF uses the same canvas, panel layout and crop as that cell's output, so its two panels sit exactly
under the two panels of the cell (same width and height on screen).
Run from the repo root:
    uv run --project MLMCReservoir python MLMCReservoir/figures/water_front_levels.py
"""
import io
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from reservoir_plots import EXAMPLE_SEED, plot_one_rock
from simulator_1d import CELLS, N_THETA, SWC, T_END, simulate

HERE = Path(__file__).parent
DPI = 100                                            # same as the notebook's inline figures
YEARS = np.linspace(0, 10, 51)                       # one frame every 0.2 years
FRAME_MS, PAUSE_MS = 120, 2000                       # ~6 s of flow, then hold the last frame for 2 s

# Use the same rock and static layout as the notebook's one-rock example.
theta = np.random.default_rng(EXAMPLE_SEED).standard_normal(N_THETA)
static, _ = plot_one_rock(theta)
crop = static.get_tightbbox(static.canvas.get_renderer()).padded(plt.rcParams["savefig.pad_inches"])

# Water saturation and recovery so far, per frame and level
sat = [[None] * 3 for _ in YEARS]
rec = np.zeros((len(YEARS), 3))
for k, yr in enumerate(YEARS):
    for level in range(3):
        if yr == 0:
            sat[k][level] = np.full(CELLS[level], SWC)
        else:
            rec[k, level], sat[k][level] = simulate(theta, level, t_end=yr / 10 * T_END, return_state=True)

fig, ax = plt.subplots(1, 2, figsize=static.get_size_inches())
fig.subplots_adjust(**{k: getattr(static.subplotpars, k) for k in ["left", "right", "bottom", "top", "wspace"]})
plt.close(static)
ax_s, ax_p = ax                                      # left: water (the cause), right: recovery (its sum over time)
s_lines = [ax_s.step((np.arange(CELLS[l]) + 0.5) / CELLS[l], sat[0][l], where="mid", label=f"level {l} ({CELLS[l]} cells)")[0] for l in range(3)]
p_lines = [ax_p.plot([], [], label=f"level {l} ({CELLS[l]} cells)")[0] for l in range(3)]
ax_s.set(xlim=(0, 1), ylim=(0.15, 0.85), xlabel="injector → producer", ylabel="water saturation")
ax_p.set(xlim=(0, 10), ylim=(0, 60), xlabel="years", ylabel="recovery factor P (%)")
ax_s.legend()

frames = []
for k, yr in enumerate(YEARS):
    for l in range(3):
        s_lines[l].set_ydata(sat[k][l])
        p_lines[l].set_data(YEARS[: k + 1], rec[: k + 1, l])
        p_lines[l].set_label(f"level {l} ({CELLS[l]} cells): {rec[k, l]:4.1f} %")
    ax_s.set_title(f"Water saturation along the tube\nafter {yr:4.1f} years")
    ax_p.set_title(f"Recovery factor so far\nafter {yr:4.1f} years")
    ax_p.legend(loc="lower right", prop={"family": "monospace", "size": 9})
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=crop, facecolor="white")
    frames.append(Image.open(buf).convert("RGB"))

frames[0].save(HERE / "water_front_levels.gif", save_all=True, append_images=frames[1:], loop=0,
               duration=[FRAME_MS] * (len(frames) - 1) + [PAUSE_MS])
print("saved", HERE / "water_front_levels.gif", frames[0].size, "| P after 10 years:", np.round(rec[-1], 2))
