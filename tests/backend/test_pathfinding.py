"""Unit tests for backend.pathfinding."""

from __future__ import annotations

import numpy as np
import pytest

from backend.pathfinding import astar


def test_astar_returns_path_on_open_grid() -> None:
    """A* should find a valid route on an empty grid."""
    grid = np.zeros((8, 8), dtype=np.uint8)
    path = astar(grid, (0, 0), (7, 7), diagonal=True)

    assert path[0] == (0, 0)
    assert path[-1] == (7, 7)
    assert len(path) > 1


def test_astar_start_occupied_raises() -> None:
    """If start cell is blocked, astar should fail fast."""
    grid = np.zeros((4, 4), dtype=np.uint8)
    grid[0, 0] = 1

    with pytest.raises(ValueError, match="Start cell is occupied"):
        astar(grid, (0, 0), (3, 3))


def test_astar_no_path_returns_empty_list() -> None:
    """Completely sealed goal should return no path."""
    grid = np.ones((5, 5), dtype=np.uint8)
    grid[0, 0] = 0
    grid[4, 4] = 0

    path = astar(grid, (0, 0), (4, 4), diagonal=True)
    assert path == []


def test_astar_non_binary_grid_raises() -> None:
    """Grid values must be binary occupancy values."""
    grid = np.array([[0, 2], [0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="only 0 .* 1"):
        astar(grid, (0, 0), (1, 1))
