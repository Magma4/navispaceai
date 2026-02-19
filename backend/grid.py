"""Occupancy grid generation from vector wall segments.

Purpose:
- Rasterize vector walls to occupancy masks.
- Inflate obstacles to account for agent clearance.
- Export grid and metadata as JSON for clients.

Usage example:
    >>> grid, mask = walls_to_occupancy_grid((512, 512), walls)
    >>> export_grid_json("backend/generated/grids/grid.json", grid, 4, 0.05)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

WallSegment = dict[str, int]
DoorCandidate = dict[str, int]


def _validate_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize input image shape."""
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a tuple of (height, width)")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive")
    return height, width


def _validate_wall_segment(seg: WallSegment) -> None:
    """Ensure wall segment has required coordinates."""
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(seg.keys()):
        raise ValueError("Wall segment must include x1, y1, x2, y2")


def walls_to_occupancy_grid(
    image_shape: tuple[int, int],
    walls: list[WallSegment],
    cell_size_px: int = 4,
    wall_thickness_px: int = 2,
    inflation_radius_px: int = 3,
    door_candidates: list[DoorCandidate] | None = None,
    door_clearance_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize walls and build an inflated occupancy grid.

    Args:
        image_shape: Source image size as `(height, width)`.
        walls: List of wall segments.
        cell_size_px: Cell pooling size in pixels.
        wall_thickness_px: Line thickness when rasterizing walls.
        inflation_radius_px: Pixel dilation radius around walls.
        door_candidates: Optional detected door bounding boxes (`x, y, w, h`).
        door_clearance_px: Expansion margin for carving door passages.

    Returns:
        Tuple `(grid, inflated_mask)` where:
        - `grid`: coarse occupancy array (1 = occupied, 0 = free)
        - `inflated_mask`: full-resolution obstacle mask

    Raises:
        ValueError: If shape, wall data, or parameters are invalid.
    """
    height, width = _validate_shape(image_shape)
    if cell_size_px <= 0:
        raise ValueError("cell_size_px must be > 0")
    if wall_thickness_px <= 0:
        raise ValueError("wall_thickness_px must be > 0")
    if inflation_radius_px < 0:
        raise ValueError("inflation_radius_px must be >= 0")
    if door_clearance_px < 0:
        raise ValueError("door_clearance_px must be >= 0")

    wall_mask = np.zeros((height, width), dtype=np.uint8)

    for seg in walls:
        _validate_wall_segment(seg)
        cv2.line(
            wall_mask,
            (int(seg["x1"]), int(seg["y1"])),
            (int(seg["x2"]), int(seg["y2"])),
            color=255,
            thickness=wall_thickness_px,
        )

    kernel_size = max(1, inflation_radius_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inflated = cv2.dilate(wall_mask, kernel, iterations=1)

    # Door-aware carving: restore passable openings that inflation may have sealed.
    if door_candidates:
        for door in door_candidates:
            if not {"x", "y", "w", "h"}.issubset(door.keys()):
                continue

            x = int(door["x"])
            y = int(door["y"])
            w = int(max(1, door["w"]))
            h = int(max(1, door["h"]))

            x0 = max(0, x - door_clearance_px)
            y0 = max(0, y - door_clearance_px)
            x1 = min(width, x + w + door_clearance_px)
            y1 = min(height, y + h + door_clearance_px)

            if x0 >= x1 or y0 >= y1:
                continue

            # Only carve where inflated walls are present to avoid random open-space edits.
            patch = inflated[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            if np.mean(patch > 0) >= 0.05:
                inflated[y0:y1, x0:x1] = 0

    rows = int(np.ceil(height / cell_size_px))
    cols = int(np.ceil(width / cell_size_px))
    grid = np.zeros((rows, cols), dtype=np.uint8)

    # Pool each cell with max value: any occupied pixel marks the cell as blocked.
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_size_px
            y1 = min((r + 1) * cell_size_px, height)
            x0 = c * cell_size_px
            x1 = min((c + 1) * cell_size_px, width)
            block = inflated[y0:y1, x0:x1]
            grid[r, c] = 1 if np.any(block > 0) else 0

    return grid, inflated


def export_grid_json(
    output_path: str | Path,
    grid: np.ndarray,
    cell_size_px: int,
    model_scale_m_per_px: float,
) -> str:
    """Export occupancy grid and metadata to a JSON file.

    Args:
        output_path: Destination JSON path.
        grid: Occupancy grid array.
        cell_size_px: Grid cell size in source pixels.
        model_scale_m_per_px: Scale used to map pixels to meters.

    Returns:
        String path to exported JSON file.

    Raises:
        ValueError: If grid or numeric parameters are invalid.
    """
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty 2D numpy array")
    if cell_size_px <= 0:
        raise ValueError("cell_size_px must be > 0")
    if model_scale_m_per_px <= 0:
        raise ValueError("model_scale_m_per_px must be > 0")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "grid": grid.astype(int).tolist(),
        "rows": int(grid.shape[0]),
        "cols": int(grid.shape[1]),
        "cell_size_px": int(cell_size_px),
        "cell_size_m": float(cell_size_px * model_scale_m_per_px),
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    return str(output)
