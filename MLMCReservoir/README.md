# MLMCReservoir

Multilevel Monte Carlo (MLMC) step by step on a small toy problem: water pushes oil through a 1D tube of rock with uncertain permeability, simulated on three grids (16, 64 and 256 cells). It follows Algorithm 1 in M. B. Giles (2015), [Multilevel Monte Carlo methods](https://doi.org/10.1017/S096249291500001X), *Acta Numerica* 24, 259–328. One run takes milliseconds, but the steps are the same for a real reservoir simulator.

## Files

| File | Content |
| --- | --- |
| `mlmc_reservoir.ipynb` | the notebook: problem, toy simulator, MLMC in four steps (test runs, number of samples, estimate, bias check), comparison with plain Monte Carlo |
| `figures/` | figures used in the notebook; each `*.py` script makes the image with the same name |
| `presentation/mlmc_reservoir_slides.pdf` | short talk (14 slides + 7 backup slides) based on the notebook |

## Run

Python 3.12 or later, managed with [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --project MLMCReservoir
```

Open `mlmc_reservoir.ipynb` in VS Code or Jupyter and select the kernel in `MLMCReservoir/.venv`. To run the whole notebook from the command line instead:

```sh
uv run --project MLMCReservoir jupyter nbconvert --to notebook --execute --inplace MLMCReservoir/mlmc_reservoir.ipynb
```

The run times (and so the number of samples and the speed-up) change a little from run to run, because the cost per sample is measured. To remake a figure, e.g. `uv run --project MLMCReservoir python MLMCReservoir/figures/water_front_levels.py`.
