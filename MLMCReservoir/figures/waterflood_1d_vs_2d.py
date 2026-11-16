"""Make waterflood_1d_vs_2d.png for mlmc_reservoir.ipynb: the same waterflood in 1D and in 2D.

Run from the repo root (needs scipy, which the notebook itself does not):
    uv run --with scipy python literature/figures/waterflood_1d_vs_2d.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent

# 1D simulator: taken from the notebook, so the figure always matches it
nb = json.loads((HERE.parent / "mlmc_reservoir.ipynb").read_text(encoding="utf-8"))
sim1d = {}
exec(next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "def simulate" in "".join(c["source"])), sim1d)
PHI, SWC, SOR, T_END, DFMAX, mobilities = (sim1d[k] for k in ["PHI", "SWC", "SOR", "T_END", "DFMAX", "mobilities"])

# 2D: same fluids, square slice, p = 1 on the left edge (injector), p = 0 on the right edge (producer), no flow top/bottom
N_THETA, CORR_LEN, N_FINE = 30, 0.2, 64
_x = (np.arange(N_FINE) + 0.5) / N_FINE
_w, _v = np.linalg.eigh(np.exp(-np.abs(_x[:, None] - _x[None, :]) / CORR_LEN))
_w, _v = _w[::-1], _v[:, ::-1]
# separable exponential covariance: 2D KL modes are products of 1D modes; keep the N_THETA largest
_modes = sorted(((_w[i] * _w[j], i, j) for i in range(20) for j in range(20)), reverse=True)[:N_THETA]


def simulate_2d(theta, t_end):
    n, h = N_FINE, 1 / N_FINE
    k = np.exp(sum(t * np.sqrt(lam) * np.outer(_v[:, i], _v[:, j]) for t, (lam, i, j) in zip(theta, _modes)))
    S = np.full((n, n), SWC)
    idx = np.arange(n * n).reshape(n, n)
    rows = np.concatenate([idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel(), idx.ravel()])
    cols = np.concatenate([idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel(), idx.ravel()])
    t = 0.0
    while t < t_end - 1e-12:
        lw, lt = mobilities(S)
        lam = k * lt
        tx, ty = 2 / (1 / lam[:, :-1] + 1 / lam[:, 1:]), 2 / (1 / lam[:-1, :] + 1 / lam[1:, :])   # face transmissibilities
        tl, tr = 2 * lam[:, 0], 2 * lam[:, -1]
        diag = np.zeros((n, n))
        diag[:, :-1] += tx; diag[:, 1:] += tx; diag[:-1, :] += ty; diag[1:, :] += ty; diag[:, 0] += tl; diag[:, -1] += tr
        vals = np.concatenate([-tx.ravel(), -tx.ravel(), -ty.ravel(), -ty.ravel(), diag.ravel()])
        b = np.zeros((n, n)); b[:, 0] = tl
        p = spla.spsolve(sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n)), b.ravel()).reshape(n, n)
        qx = np.zeros((n, n + 1)); qy = np.zeros((n + 1, n))
        qx[:, 1:-1] = tx * (p[:, :-1] - p[:, 1:]); qx[:, 0] = tl * (1 - p[:, 0]); qx[:, -1] = tr * p[:, -1]
        qy[1:-1, :] = ty * (p[:-1, :] - p[1:, :])
        fw = lw / lt
        fwx = np.concatenate([np.ones((n, 1)), np.where(qx[:, 1:-1] > 0, fw[:, :-1], fw[:, 1:]), fw[:, -1:]], axis=1) * qx
        fwy = np.zeros((n + 1, n)); fwy[1:-1, :] = np.where(qy[1:-1, :] > 0, fw[:-1, :], fw[1:, :]) * qy[1:-1, :]
        dt = min(0.45 * PHI * h * h / (max(np.abs(qx).max(), np.abs(qy).max()) * DFMAX), t_end - t)
        S = S - dt / (PHI * h * h) * (np.diff(fwx, axis=1) + np.diff(fwy, axis=0))
        t += dt
    return S


cmap = LinearSegmentedColormap.from_list("oil_water", ["#1a7f2e", "#8fd3f4"])   # green = oil, blue = water
norm = dict(cmap=cmap, vmin=SWC, vmax=1 - SOR)
rng = np.random.default_rng(7)
theta_1d, theta_2d = rng.standard_normal(sim1d["N_THETA"]), rng.standard_normal(N_THETA)
years = [2, 5, 10]

# All panels in a column span exactly injector (x = 0) to producer (x = 1), and all columns have the same width;
# the colour bar gets its own narrow column so it does not shrink the last scenario.
fig, ax = plt.subplots(3, 4, figsize=(13, 9), gridspec_kw={"height_ratios": [0.3, 1, 1], "width_ratios": [1, 1, 1, 0.04]})
for j, yr in enumerate(years):
    _, S1 = sim1d["simulate"](theta_1d, 2, t_end=yr / 10 * T_END, return_state=True)
    ax[0, j].imshow(S1[None, :], aspect="auto", extent=[0, 1, 0, 1], **norm)
    ax[0, j].set(xticks=[], yticks=[], title=f"1D tube, after {yr} years")
    ax[1, j].plot((np.arange(S1.size) + 0.5) / S1.size, S1, color="#1f77b4")
    ax[1, j].set(xlim=(0, 1), ylim=(0.15, 0.85), xlabel="injector → producer", title="1D: water saturation along the tube")
    im = ax[2, j].imshow(simulate_2d(theta_2d, yr / 10 * T_END), origin="lower", aspect="auto", extent=[0, 1, 0, 1], **norm)
    ax[2, j].set(xticks=[], yticks=[], xlabel="injector → producer", title=f"2D slice, after {yr} years")
ax[1, 0].set_ylabel("water saturation")
ax[0, 3].axis("off"); ax[1, 3].axis("off")
cb = fig.colorbar(im, cax=ax[2, 3], ticks=[SWC, 1 - SOR])
cb.ax.set_yticklabels(["oil", "water"])
fig.tight_layout()
fig.savefig(HERE / "waterflood_1d_vs_2d.png", dpi=90, facecolor="white")
print("saved", HERE / "waterflood_1d_vs_2d.png")
