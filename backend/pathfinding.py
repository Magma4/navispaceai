"""A* pathfinding implementation for occupancy grids.

Purpose:
- Compute shortest free-space routes in occupancy grids.
- Support 4-way and 8-way movement for indoor navigation.

Usage example:
    >>> import numpy as np
    >>> from backend.pathfinding import astar
    >>> grid = np.zeros((10, 10), dtype=np.uint8)
    >>> astar(grid, (0, 0), (9, 9), diagonal=True)
"""

from __future__ import annotations

import heapq
import math

import numpy as np

GridPoint = tuple[int, int]


def _validate_grid(grid: np.ndarray) -> None:
    """Validate occupancy grid contract (2D, non-empty, binary-ish)."""
    if not isinstance(grid, np.ndarray):
        raise ValueError("Grid must be a numpy array")
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("Grid must be a non-empty 2D array")
    if np.any((grid != 0) & (grid != 1)):
        raise ValueError("Grid must contain only 0 (free) and 1 (occupied) values")


def _heuristic(a: GridPoint, b: GridPoint, diagonal: bool = True) -> float:
    """Compute admissible heuristic distance between two points."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if diagonal:
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)
    return float(dr + dc)


def _neighbors(point: GridPoint, grid: np.ndarray, diagonal: bool = True) -> list[tuple[GridPoint, float]]:
    """Return valid neighboring cells and move costs."""
    r, c = point
    rows, cols = grid.shape

    directions = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if diagonal:
        directions.extend(
            [
                (-1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (1, 1, math.sqrt(2)),
            ]
        )

    result: list[tuple[GridPoint, float]] = []
    for dr, dc, cost in directions:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        if grid[nr, nc] == 1:
            continue

        # Prevent diagonal corner cutting through blocked cells.
        if diagonal and dr != 0 and dc != 0:
            if grid[r + dr, c] == 1 and grid[r, c + dc] == 1:
                continue

        result.append(((nr, nc), cost))

    return result


def astar(grid: np.ndarray, start: GridPoint, goal: GridPoint, diagonal: bool = True) -> list[GridPoint]:
    """Compute shortest path via A*.

    Args:
        grid: 2D occupancy grid, with 0 as free and 1 as occupied.
        start: Start point `(row, col)`.
        goal: Goal point `(row, col)`.
        diagonal: Enable 8-direction movement when True.

    Returns:
        List of grid points from start to goal. Empty list if no path exists.

    Raises:
        ValueError: If grid/start/goal are invalid.
    """
    _validate_grid(grid)

    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("Start is out of grid bounds")
    if not (0 <= gr < rows and 0 <= gc < cols):
        raise ValueError("Goal is out of grid bounds")
    if grid[sr, sc] == 1:
        raise ValueError("Start cell is occupied")
    if grid[gr, gc] == 1:
        raise ValueError("Goal cell is occupied")

    open_heap: list[tuple[float, GridPoint]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: dict[GridPoint, GridPoint] = {}
    g_score: dict[GridPoint, float] = {start: 0.0}
    closed: set[GridPoint] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        for neighbor, step_cost in _neighbors(current, grid, diagonal=diagonal):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(neighbor, goal, diagonal)
                heapq.heappush(open_heap, (f, neighbor))

    return []
