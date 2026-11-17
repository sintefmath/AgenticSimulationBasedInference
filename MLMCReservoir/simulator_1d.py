"""Toy 1D waterflood simulator shared by the MLMC notebook and figure scripts."""

import numpy as np

PHI, SWC, SOR = 0.2, 0.2, 0.2      # porosity, connate water, residual oil
MU_W, MU_O = 0.5, 2.0              # water and oil viscosity (cP)
DP, T_END = 1.0, 0.18              # pressure drop and "10 years", in model units
CELLS = [16, 64, 256]              # grid of level 0, 1, 2
M = 4                              # refinement factor between levels

# Random rock: log-permeability = Gaussian random field built from theta (Karhunen-Loeve expansion)
N_THETA, SIGMA, CORR_LEN = 30, 1.0, 0.2
x = (np.arange(CELLS[-1]) + 0.5) / CELLS[-1]
w, v = np.linalg.eigh(SIGMA**2 * np.exp(-np.abs(x[:, None] - x[None, :]) / CORR_LEN))
KL = v[:, ::-1][:, :N_THETA] * np.sqrt(w[::-1][:N_THETA])


def permeability(theta, level):
    # Fine permeability from theta, upscaled to the level's grid by harmonic averaging
    k_fine = np.exp(KL @ theta)
    r = CELLS[-1] // CELLS[level]
    return r / np.sum(1 / k_fine.reshape(-1, r), axis=1)


def mobilities(S):
    Sn = np.clip((S - SWC) / (1 - SWC - SOR), 0, 1)
    lw, lo = Sn**2 / MU_W, (1 - Sn) ** 2 / MU_O
    return lw, lw + lo


_S = np.linspace(SWC, 1 - SOR, 2001)
_lw, _lt = mobilities(_S)
DFMAX = np.max(np.diff(_lw / _lt) / np.diff(_S))   # fastest wave speed, for a stable time step


def simulate(theta, level, t_end=T_END, return_state=False):
    # Waterflood on one grid; returns the oil recovery factor (%) at t_end (default: 10 years)
    k = permeability(theta, level)
    dx = 1 / k.size
    S = np.full(k.size, SWC)
    t = 0.0
    while t < t_end - 1e-12:
        lw, lt = mobilities(S)
        q = DP / np.sum(dx / (k * lt))                   # total rate set by the pressure drop
        dt = min(0.9 * PHI * dx / (q * DFMAX), t_end - t)
        flux = q * np.concatenate(([1.0], lw / lt))      # water flux; pure water enters on the left
        S = S - dt / (PHI * dx) * np.diff(flux)
        t += dt
    rf = 100 * np.sum(S - SWC) * dx / (1 - SWC)
    return (rf, S) if return_state else rf
