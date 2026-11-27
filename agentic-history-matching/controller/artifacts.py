"""Read and write local experiment artifacts."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_arrays(path: Path, **arrays: NDArray[np.generic]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, allow_pickle=False, **arrays)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_arrays(path: Path) -> dict[str, NDArray[np.generic]]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
