"""Utility helpers shared across backend modules.

Purpose:
- Create runtime output folders.
- Convert between grid and world coordinate conventions.
- Convert numpy arrays to JSON-safe payload types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_directories() -> None:
    """Create runtime output directories if they do not already exist."""
    Path("backend/generated/models").mkdir(parents=True, exist_ok=True)
    Path("backend/generated/grids").mkdir(parents=True, exist_ok=True)


def to_serializable_path(path: Iterable[tuple[int, int]]) -> list[dict[str, int]]:
    """Convert `(row, col)` tuples to JSON-friendly dictionary objects."""
    return [{"row": int(r), "col": int(c)} for r, c in path]


def grid_to_world(row: int, col: int, cell_size_m: float) -> tuple[float, float]:
    """Map grid coordinates `(row, col)` to world-space `(x, z)` in meters."""
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")
    x = col * cell_size_m
    z = row * cell_size_m
    return float(x), float(z)


def world_to_grid(x: float, z: float, cell_size_m: float, rows: int, cols: int) -> tuple[int, int]:
    """Map world-space `(x, z)` to clamped grid coordinates `(row, col)`."""
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be > 0")

    col = int(round(x / cell_size_m))
    row = int(round(z / cell_size_m))
    row = max(0, min(rows - 1, row))
    col = max(0, min(cols - 1, col))
    return row, col


def json_grid(grid: np.ndarray) -> list[list[int]]:
    """Convert a numpy occupancy grid to nested Python int lists."""
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("grid must be a 2D numpy array")
    return grid.astype(int).tolist()
