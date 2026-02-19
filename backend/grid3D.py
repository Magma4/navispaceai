"""3D occupancy grid construction for multi-floor navigation.

Grid convention:
- grid[floor_index, row, col] == 0 => free
- grid[floor_index, row, col] == 1 => occupied

All coordinate conversion helpers use meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class ConnectorNode:
    """Discrete connector anchor mapped into grid coordinates."""

    connector_id: str
    connector_type: str
    floor_index: int
    row: int
    col: int
    radius_cells: int = 1


def normalize_floor_grids(floor_grids: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Normalize per-floor 2D grids into one aligned 3D tensor.

    Floors are sorted by floor number and padded to max row/col size.
    Padded cells are treated as occupied to avoid out-of-bounds traversal.

    Args:
        floor_grids: Mapping floor_number -> 2D occupancy grid.

    Returns:
        Tuple of:
            - 3D grid tensor with shape (F, R, C)
            - Ordered floor numbers (index -> floor number)
    """
    if not floor_grids:
        raise ValueError("floor_grids cannot be empty")

    ordered_floors = sorted(floor_grids.keys())
    max_rows = max(int(floor_grids[f].shape[0]) for f in ordered_floors)
    max_cols = max(int(floor_grids[f].shape[1]) for f in ordered_floors)

    grid3d = np.ones((len(ordered_floors), max_rows, max_cols), dtype=np.uint8)

    for idx, floor in enumerate(ordered_floors):
        floor_grid = floor_grids[floor]
        if floor_grid.ndim != 2:
            raise ValueError(f"Floor {floor} grid must be 2D")

        rows, cols = floor_grid.shape
        grid3d[idx, :rows, :cols] = floor_grid.astype(np.uint8)

    return grid3d, ordered_floors


def inflate_3d_obstacles(grid3d: np.ndarray, radius_cells: int = 1) -> np.ndarray:
    """Inflate obstacles on each floor for agent body clearance.

    Args:
        grid3d: 3D occupancy tensor (F, R, C).
        radius_cells: Dilation radius in cells.

    Returns:
        Inflated occupancy tensor with same shape.
    """
    if grid3d.ndim != 3:
        raise ValueError("grid3d must be 3D")
    if radius_cells < 0:
        raise ValueError("radius_cells must be >= 0")

    if radius_cells == 0:
        return grid3d.copy()

    kernel_size = radius_cells * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    out = np.zeros_like(grid3d)
    for f in range(grid3d.shape[0]):
        floor_occ = (grid3d[f] > 0).astype(np.uint8) * 255
        dilated = cv2.dilate(floor_occ, kernel, iterations=1)
        out[f] = (dilated > 0).astype(np.uint8)

    return out


def build_connector_nodes(
    ordered_floors: list[int],
    connectors: list[dict],
    cell_size_m: float,
) -> list[ConnectorNode]:
    """Convert world-space connectors into per-floor grid nodes.

    Args:
        ordered_floors: Floor number order corresponding to 3D grid indices.
        connectors: Connector metadata dict list with keys:
            - connector_id
            - connector_type
            - floors
            - position_m (x, y, z)
            - radius_m
        cell_size_m: Grid cell size in meters.

    Returns:
        List of ConnectorNode entries.
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")

    floor_to_index = {floor_num: idx for idx, floor_num in enumerate(ordered_floors)}
    nodes: list[ConnectorNode] = []

    for connector in connectors:
        position_m = connector["position_m"]
        x_m = float(position_m[0])
        z_m = float(position_m[2])
        col = int(round(x_m / cell_size_m))
        row = int(round(z_m / cell_size_m))

        radius_cells = max(1, int(round(float(connector.get("radius_m", 0.5)) / cell_size_m)))

        for floor_num in connector["floors"]:
            if floor_num not in floor_to_index:
                continue

            nodes.append(
                ConnectorNode(
                    connector_id=connector["connector_id"],
                    connector_type=connector["connector_type"],
                    floor_index=floor_to_index[floor_num],
                    row=row,
                    col=col,
                    radius_cells=radius_cells,
                )
            )

    return nodes


def carve_connector_clearance(grid3d: np.ndarray, connector_nodes: list[ConnectorNode]) -> np.ndarray:
    """Carve free traversal area around connector nodes.

    This ensures stairs/elevator cells are passable even if local wall inflation marked
    the area occupied.
    """
    out = grid3d.copy()
    floors, rows, cols = out.shape

    for node in connector_nodes:
        if node.floor_index < 0 or node.floor_index >= floors:
            continue

        r0 = max(0, node.row - node.radius_cells)
        r1 = min(rows, node.row + node.radius_cells + 1)
        c0 = max(0, node.col - node.radius_cells)
        c1 = min(cols, node.col + node.radius_cells + 1)

        out[node.floor_index, r0:r1, c0:c1] = 0

    return out


def world_to_grid3d(
    x_m: float,
    y_m: float,
    z_m: float,
    floor_height_m: float,
    cell_size_m: float,
    floor_count: int,
    rows: int,
    cols: int,
) -> tuple[int, int, int]:
    """Convert world meters to clamped 3D grid index (f, r, c)."""
    if floor_height_m <= 0 or cell_size_m <= 0:
        raise ValueError("floor_height_m and cell_size_m must be > 0")

    floor_idx = int(round(y_m / floor_height_m))
    row = int(round(z_m / cell_size_m))
    col = int(round(x_m / cell_size_m))

    floor_idx = max(0, min(floor_count - 1, floor_idx))
    row = max(0, min(rows - 1, row))
    col = max(0, min(cols - 1, col))
    return floor_idx, row, col


def grid3d_to_world(
    floor_idx: int,
    row: int,
    col: int,
    floor_height_m: float,
    cell_size_m: float,
) -> tuple[float, float, float]:
    """Convert 3D grid index (f, r, c) to world-space meters (x, y, z)."""
    x_m = col * cell_size_m
    y_m = floor_idx * floor_height_m
    z_m = row * cell_size_m
    return float(x_m), float(y_m), float(z_m)
