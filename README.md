# AgenticSimulationBasedInference

Experiments in simulation-based inference.

## Python projects

| Project | Description |
| --- | --- |
| [agentic-history-matching](agentic-history-matching/README.md) | Agent-guided model-structure search and ensemble history matching |
| [MLMCReservoir](MLMCReservoir/README.md) | Multilevel Monte Carlo on a toy reservoir simulator |

These are independent [uv](https://docs.astral.sh/uv/) projects, not members of a shared uv workspace. Each has its own `pyproject.toml`, `uv.lock`, and `.venv`, so their Python and dependency requirements can differ. See each project's README for prerequisites and usage.

## Working with uv

Run the following commands from the repository root. No virtual environment activation is required. If another project's environment is already activated in your shell, run `deactivate` first to avoid an environment-mismatch warning. Do not use `--active` to sync one project's dependencies into another project's environment.

### Install dependencies

Select the project explicitly:

```bash
uv sync --project agentic-history-matching --locked
uv sync --project MLMCReservoir --locked
```

`--locked` checks that the project's lockfile is up to date instead of changing it.

### Run commands

Use `uv run` rather than activating an environment and calling `python` directly. For example, after configuring `agentic-history-matching/.env` as described in its README:

```bash
uv --directory agentic-history-matching run --locked --env-file .env python -m controller configs/run.yaml
```

To execute the MLMC notebook:

```bash
uv --directory MLMCReservoir run --locked jupyter nbconvert --to notebook --execute --inplace mlmc_reservoir.ipynb
```

- `--project PATH` selects the project and its environment without changing the working directory.
- `--directory PATH` changes the working directory first, so module imports, `.env`, configuration files, and other relative paths are resolved from that project directory.
- If you are already inside the project directory, omit `--project` or `--directory` and use `uv sync --locked` or `uv run --locked ...` directly. Keep any project-specific flags shown in its README.
- In an editor or notebook, select the interpreter or kernel from the relevant project's `.venv`.

Commit each project's `pyproject.toml` and `uv.lock` when dependencies change. Keep `.venv` directories local; both projects already ignore them in Git.
