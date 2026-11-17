"""Publish immutable JSON/NumPy artifacts and verify SHA-256 checksums on loading."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray


def write_json(path: Path, value: Mapping[str, object]) -> str:
    """Atomically publish JSON and return its SHA-256 checksum."""
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return write_bytes(path, content)


def write_bytes(path: Path, content: bytes) -> str:
    """Atomically publish bytes without replacing an existing artifact."""
    return _publish(path, lambda handle: handle.write(content))


def write_arrays(path: Path, **arrays: NDArray[np.generic]) -> str:
    """Save named arrays to a compressed .npz file and return its SHA-256 checksum.

    Refuse to overwrite an existing file; publish only after writing is complete.
    Reject object-dtype arrays so the archive can be loaded without pickle.
    """
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("object arrays are not supported artifact contents")
    return _publish(path, lambda handle: np.savez_compressed(handle, **arrays))


def read_json(path: Path) -> dict[str, object]:
    """Load an artifact and require a JSON object."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_arrays(path: Path, expected_checksum: str) -> dict[str, NDArray[np.generic]]:
    """Verify artifact integrity before loading arrays with pickle disabled."""
    if checksum(path) != expected_checksum:
        raise ValueError(f"artifact checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def checksum(path: Path) -> str:
    """Return the artifact's SHA-256 checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(path: Path, writer: Callable[[BinaryIO], object]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite saved file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if another writer already claimed this name.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return checksum(path)
