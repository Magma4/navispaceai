"""3D A* pathfinding for multi-floor occupancy grids.

Supports:
- 8-direction movement on each floor
- Vertical transitions through registered connector nodes
- World-space start/goal in meters
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from backend.grid3D import ConnectorNode, grid3d_to_world, world_to_grid3d


Grid3DPoint = tuple[int, int, int]  # (floor_idx, row, col)


@dataclass(slots=True)
class Path3DResult:
    """Structured 3D path result payload."""

    grid_path: list[Grid3DPoint]
    world_path: list[dict[str, float]]


def _heuristic(a: Grid3DPoint, b: Grid3DPoint, floor_height_m: float, cell_size_m: float) -> float:
    """Euclidean heuristic in meter space for admissible 3D A*."""
    df = (a[0] - b[0]) * floor_height_m
    dr = (a[1] - b[1]) * cell_size_m
    dc = (a[2] - b[2]) * cell_size_m
    return math.sqrt(df * df + dr * dr + dc * dc)


def _build_connector_lookup(connector_nodes: list[ConnectorNode]) -> dict[str, list[ConnectorNode]]:
    """Group connector nodes by connector_id."""
    lookup: dict[str, list[ConnectorNode]] = {}
    for node in connector_nodes:
        lookup.setdefault(node.connector_id, []).append(node)
    return lookup


def _is_inside(grid3d: np.ndarray, node: Grid3DPoint) -> bool:
    f, r, c = node
    return 0 <= f < grid3d.shape[0] and 0 <= r < grid3d.shape[1] and 0 <= c < grid3d.shape[2]


def _is_free(grid3d: np.ndarray, node: Grid3DPoint) -> bool:
    return _is_inside(grid3d, node) and int(grid3d[node[0], node[1], node[2]]) == 0


def _neighbors(
    grid3d: np.ndarray,
    node: Grid3DPoint,
    connector_nodes: list[ConnectorNode],
    cell_size_m: float,
    floor_height_m: float,
) -> list[tuple[Grid3DPoint, float]]:
    """Enumerate horizontal and vertical neighbors with edge costs."""
    f, r, c = node

    moves_2d = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    out: list[tuple[Grid3DPoint, float]] = []

    # Horizontal movement on current floor.
    for dr, dc, base_cost in moves_2d:
        nbr = (f, r + dr, c + dc)
        if _is_free(grid3d, nbr):
            out.append((nbr, base_cost * cell_size_m))

    # Vertical movement only through connector neighborhoods.
    for node_a in connector_nodes:
        if node_a.floor_index != f:
            continue

        if abs(node_a.row - r) > node_a.radius_cells or abs(node_a.col - c) > node_a.radius_cells:
            continue

        for node_b in connector_nodes:
            if node_b.connector_id != node_a.connector_id:
                continue
            if node_b.floor_index == node_a.floor_index:
                continue

            target = (node_b.floor_index, node_b.row, node_b.col)
            if _is_free(grid3d, target):
                floor_delta = abs(node_b.floor_index - node_a.floor_index)
                out.append((target, floor_delta * floor_height_m))

    return out


def astar_3d(
    grid3d: np.ndarray,
    start: Grid3DPoint,
    goal: Grid3DPoint,
    connector_nodes: list[ConnectorNode],
    cell_size_m: float,
    floor_height_m: float,
) -> list[Grid3DPoint]:
    """Run A* on 3D occupancy + connector graph.

    Args:
        grid3d: Occupancy tensor (F, R, C), 0 free / 1 occupied.
        start: Start index (f, r, c).
        goal: Goal index (f, r, c).
        connector_nodes: Inter-floor connector anchors.
        cell_size_m: Horizontal cell scale.
        floor_height_m: Vertical floor spacing.

    Returns:
        Ordered node list from start to goal.
    """
    if grid3d.ndim != 3 or grid3d.size == 0:
        raise ValueError("grid3d must be a non-empty 3D occupancy tensor")
    if cell_size_m <= 0 or floor_height_m <= 0:
        raise ValueError("cell_size_m and floor_height_m must be > 0")
    if not _is_free(grid3d, start):
        raise ValueError("Start node is occupied or out of bounds")
    if not _is_free(grid3d, goal):
        raise ValueError("Goal node is occupied or out of bounds")

    open_heap: list[tuple[float, Grid3DPoint]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: dict[Grid3DPoint, Grid3DPoint] = {}
    g_score: dict[Grid3DPoint, float] = {start: 0.0}
    closed: set[Grid3DPoint] = set()

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

        for neighbor, step_cost in _neighbors(
            grid3d=grid3d,
            node=current,
            connector_nodes=connector_nodes,
            cell_size_m=cell_size_m,
            floor_height_m=floor_height_m,
        ):
            if neighbor in closed:
                continue

            tentative = g_score[current] + step_cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + _heuristic(neighbor, goal, floor_height_m, cell_size_m)
                heapq.heappush(open_heap, (f_score, neighbor))

    return []


def find_path_3d_world(
    grid3d: np.ndarray,
    start_m: tuple[float, float, float],
    goal_m: tuple[float, float, float],
    connector_nodes: list[ConnectorNode],
    cell_size_m: float,
    floor_height_m: float,
) -> Path3DResult:
    """Compute 3D path using world coordinates in meters.

    Args:
        grid3d: 3D occupancy tensor.
        start_m: Start world coordinate (x, y, z).
        goal_m: Goal world coordinate (x, y, z).
        connector_nodes: Connector anchors for vertical transitions.
        cell_size_m: Horizontal cell size.
        floor_height_m: Floor spacing in meters.

    Returns:
        Path3DResult with both grid and world path representations.
    """
    floors, rows, cols = grid3d.shape

    start_idx = world_to_grid3d(
        x_m=float(start_m[0]),
        y_m=float(start_m[1]),
        z_m=float(start_m[2]),
        floor_height_m=floor_height_m,
        cell_size_m=cell_size_m,
        floor_count=floors,
        rows=rows,
        cols=cols,
    )

    goal_idx = world_to_grid3d(
        x_m=float(goal_m[0]),
        y_m=float(goal_m[1]),
        z_m=float(goal_m[2]),
        floor_height_m=floor_height_m,
        cell_size_m=cell_size_m,
        floor_count=floors,
        rows=rows,
        cols=cols,
    )

    grid_path = astar_3d(
        grid3d=grid3d,
        start=start_idx,
        goal=goal_idx,
        connector_nodes=connector_nodes,
        cell_size_m=cell_size_m,
        floor_height_m=floor_height_m,
    )

    world_path: list[dict[str, float]] = []
    for f, r, c in grid_path:
        x_m, y_m, z_m = grid3d_to_world(
            floor_idx=f,
            row=r,
            col=c,
            floor_height_m=floor_height_m,
            cell_size_m=cell_size_m,
        )
        world_path.append({"x": x_m, "y": y_m, "z": z_m})

    return Path3DResult(grid_path=grid_path, world_path=world_path)
