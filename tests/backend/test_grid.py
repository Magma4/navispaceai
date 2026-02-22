"""Unit tests for backend.grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.grid import export_grid_json, walls_to_occupancy_grid


def test_walls_to_occupancy_grid_no_walls_is_empty_grid() -> None:
    """No walls should produce a fully free occupancy grid."""
    grid, inflated = walls_to_occupancy_grid((40, 40), walls=[])

    assert grid.shape == (10, 10)
    assert np.all(grid == 0)
    assert inflated.shape == (40, 40)


def test_walls_to_occupancy_grid_marks_obstacles() -> None:
    """A simple vertical wall should occupy some grid cells."""
    walls = [{"x1": 20, "y1": 5, "x2": 20, "y2": 35}]
    grid, _ = walls_to_occupancy_grid((40, 40), walls=walls, cell_size_px=4)
    assert np.any(grid == 1)


def test_walls_to_occupancy_grid_invalid_cell_size_raises() -> None:
    """Cell size must be strictly positive."""
    with pytest.raises(ValueError, match="cell_size_px"):
        walls_to_occupancy_grid((40, 40), walls=[], cell_size_px=0)


def test_walls_to_occupancy_grid_hole_fill_reduces_internal_voids() -> None:
    """Tiny enclosed holes should be filled during wall cleanup."""
    walls = [
        {"x1": 8, "y1": 8, "x2": 32, "y2": 8},
        {"x1": 32, "y1": 8, "x2": 32, "y2": 32},
        {"x1": 32, "y1": 32, "x2": 8, "y2": 32},
        {"x1": 8, "y1": 32, "x2": 8, "y2": 8},
    ]

    _, inflated = walls_to_occupancy_grid(
        (40, 40),
        walls=walls,
        wall_thickness_px=1,
        inflation_radius_px=0,
        fill_hole_max_area_px=1000,
    )

    assert inflated[20, 20] > 0


def test_export_grid_json_writes_expected_payload(tmp_path: Path) -> None:
    """Grid export should serialize metadata and occupancy values."""
    grid = np.zeros((3, 4), dtype=np.uint8)
    out = tmp_path / "grid.json"

    output_path = export_grid_json(out, grid, cell_size_px=4, model_scale_m_per_px=0.05)
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert data["rows"] == 3
    assert data["cols"] == 4
    assert data["cell_size_m"] == pytest.approx(0.2)
