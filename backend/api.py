"""FastAPI routes for single-floor and multi-floor blueprint navigation pipelines.

This module now supports two modes:
- Single-floor flow (`/process-blueprint`, `/find-path`)
- Multi-floor flow (`/process-building`, `/floors`, `/rooms`, `/find-path-3d`)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from backend.building_manager import BuildingManager, RoomMeta
from backend.detection import (
    adaptive_hough_params,
    detect_doors,
    detect_walls,
    filter_door_candidates,
    filter_wall_segments,
)
from backend.geometry_validation import validate_scene_graph
from backend.grid import export_grid_json, walls_to_occupancy_grid
from backend.grid3D import (
    ConnectorNode,
    build_connector_nodes,
    carve_connector_clearance,
    inflate_3d_obstacles,
    normalize_floor_grids,
)
from backend.modeling import build_model_from_walls
from backend.pathfinding import astar, smooth_path
from backend.pathfinding3D import find_path_3d_world
from backend.preprocessing import preprocess_blueprint
from backend.scene_graph import build_scene_graph
from backend.room_connectivity import build_room_connectivity, shortest_room_route
from backend.utils import ensure_directories, grid_to_world, json_grid, to_serializable_path

MODEL_SCALE_M_PER_PX = 0.05
GRID_CELL_SIZE_PX = 4


@dataclass
class ProcessingState:
    """In-memory state for latest processed navigation payloads."""

    # Single-floor state.
    grid: np.ndarray | None = None
    cell_size_m: float = GRID_CELL_SIZE_PX * MODEL_SCALE_M_PER_PX
    model_url: str | None = None

    # Multi-floor state.
    building_manager: BuildingManager | None = None
    grid3d: np.ndarray | None = None
    ordered_floors: list[int] = field(default_factory=list)
    connector_nodes: list[ConnectorNode] = field(default_factory=list)
    room_connectivity: dict[str, Any] | None = None


STATE = ProcessingState()


class GridPoint(BaseModel):
    """Grid coordinate with row/column indexing."""

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)


class PathRequest(BaseModel):
    """Request payload for 2D A* path computation."""

    start: GridPoint
    goal: GridPoint
    diagonal: bool = True


class PathResponse(BaseModel):
    """Response payload for 2D pathfinding requests."""

    path: list[dict[str, int]]
    world_path: list[dict[str, float]]
    smoothed_path: list[dict[str, int]] | None = None
    smoothed_world_path: list[dict[str, float]] | None = None


class WorldPoint(BaseModel):
    """World coordinate in meters."""

    x: float
    y: float
    z: float


class Path3DRequest(BaseModel):
    """Request payload for 3D A* path computation.

    Provide either:
    - start + goal world points, or
    - start_room_id + goal_room_id
    """

    start: WorldPoint | None = None
    goal: WorldPoint | None = None
    start_room_id: str | None = None
    goal_room_id: str | None = None

    @model_validator(mode="after")
    def validate_inputs(self) -> "Path3DRequest":
        """Ensure caller provides endpoint coordinates or room IDs."""
        has_points = self.start is not None and self.goal is not None
        has_rooms = bool(self.start_room_id and self.goal_room_id)
        if not (has_points or has_rooms):
            raise ValueError(
                "Provide either start/goal world coordinates or start_room_id/goal_room_id"
            )
        return self


class Path3DGridPoint(BaseModel):
    """Discrete 3D path point in building floor grid."""

    floor: int
    row: int
    col: int


class Path3DResponse(BaseModel):
    """Response payload for 3D pathfinding requests."""

    grid_path: list[Path3DGridPoint]
    world_path: list[dict[str, float]]


def _decode_upload_image(raw_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV BGR image."""
    if not raw_bytes:
        raise ValueError("Uploaded file is empty")

    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported or corrupted image format")
    return image


def _safe_stem(value: str) -> str:
    """Create filesystem-safe identifier segment."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-") or "building"


def _serialize_room(room: RoomMeta) -> dict[str, Any]:
    """Serialize RoomMeta to JSON-safe dictionary."""
    return {
        "room_id": room.room_id,
        "floor_number": room.floor_number,
        "name": room.name,
        "centroid_m": room.centroid_m,
        "polygon_m": room.polygon_m,
        "area_m2": room.area_m2,
    }


def _parse_floor_numbers(raw: str | None, file_count: int) -> list[int]:
    """Parse floor number mapping aligned with uploaded files."""
    if file_count <= 0:
        raise ValueError("At least one file is required")

    if not raw:
        return list(range(file_count))

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("floor_numbers must be valid JSON (e.g. [0,1,2])") from exc

    if not isinstance(parsed, list):
        raise ValueError("floor_numbers must be a JSON list")

    try:
        values = [int(v) for v in parsed]
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("floor_numbers must contain integers") from exc

    if len(values) != file_count:
        raise ValueError("floor_numbers length must match uploaded files count")

    if len(set(values)) != len(values):
        raise ValueError("floor_numbers must be unique")

    return values


def _parse_connectors(raw: str | None) -> list[dict[str, Any]]:
    """Parse connector JSON payload.

    Expected schema per item:
      {
        "connector_id": "C1",
        "connector_type": "stairs"|"elevator",
        "floors": [0, 1, 2],
        "position_m": [x, y, z],
        "radius_m": 0.6
      }
    """
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("connector_json must be valid JSON") from exc

    if not isinstance(parsed, list):
        raise ValueError("connector_json must be a JSON list")

    connectors: list[dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"connector[{idx}] must be an object")

        required = {"connector_id", "connector_type", "floors", "position_m"}
        if not required.issubset(item.keys()):
            raise ValueError(
                f"connector[{idx}] must include connector_id, connector_type, floors, position_m"
            )

        position = item["position_m"]
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"connector[{idx}].position_m must be [x,y,z]")

        connectors.append(
            {
                "connector_id": str(item["connector_id"]),
                "connector_type": str(item["connector_type"]),
                "floors": [int(f) for f in item["floors"]],
                "position_m": (float(position[0]), float(position[1]), float(position[2])),
                "radius_m": float(item.get("radius_m", 0.6)),
            }
        )

    return connectors


def _default_connector_for_building(manager: BuildingManager) -> dict[str, Any] | None:
    """Create one conservative auto-connector for buildings with 2+ floors.

    This provides a usable default vertical transition in demo flow when explicit
    connectors were not supplied by caller.
    """
    floors = sorted(manager.meta.floors.keys())
    if len(floors) < 2:
        return None

    centers_x: list[float] = []
    centers_z: list[float] = []

    for floor in floors:
        floor_meta = manager.get_floor(floor)
        centers_x.append(float(floor_meta.grid.shape[1] * floor_meta.cell_size_m * 0.5))
        centers_z.append(float(floor_meta.grid.shape[0] * floor_meta.cell_size_m * 0.5))

    x_m = float(np.mean(centers_x)) if centers_x else 0.0
    z_m = float(np.mean(centers_z)) if centers_z else 0.0

    return {
        "connector_id": "AUTO_STAIRS_CORE",
        "connector_type": "stairs",
        "floors": floors,
        "position_m": (x_m, manager.meta.origin_m[1], z_m),
        "radius_m": max(0.6, manager.default_cell_size_m * 2.5),
    }


def _latest_building_or_400() -> BuildingManager:
    """Get latest multi-floor building state or raise 400."""
    if STATE.building_manager is None:
        raise HTTPException(status_code=400, detail="No processed building available yet")
    return STATE.building_manager


def _nearest_free_cell(
    grid: np.ndarray,
    row: int,
    col: int,
    max_radius: int = 12,
) -> tuple[int, int] | None:
    """Find nearest free cell to a requested start/goal cell.

    Search expands in square rings around the requested cell and returns the free
    cell with minimum Euclidean distance in the first ring that contains any free
    candidates.
    """
    rows, cols = grid.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return None

    if int(grid[row, col]) == 0:
        return row, col

    for radius in range(1, max_radius + 1):
        r0 = max(0, row - radius)
        r1 = min(rows - 1, row + radius)
        c0 = max(0, col - radius)
        c1 = min(cols - 1, col + radius)

        candidates: list[tuple[float, int, int]] = []

        # Top and bottom rows.
        for c in range(c0, c1 + 1):
            if int(grid[r0, c]) == 0:
                candidates.append(((r0 - row) ** 2 + (c - col) ** 2, r0, c))
            if r1 != r0 and int(grid[r1, c]) == 0:
                candidates.append(((r1 - row) ** 2 + (c - col) ** 2, r1, c))

        # Left and right columns excluding corners already checked.
        for r in range(r0 + 1, r1):
            if int(grid[r, c0]) == 0:
                candidates.append(((r - row) ** 2 + (c0 - col) ** 2, r, c0))
            if c1 != c0 and int(grid[r, c1]) == 0:
                candidates.append(((r - row) ** 2 + (c1 - col) ** 2, r, c1))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            _, best_r, best_c = candidates[0]
            return best_r, best_c

    return None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    ensure_directories()

    app = FastAPI(title="NavispaceAI API", version="1.2.0")

    raw_origins = os.getenv("NAVISPACE_CORS_ORIGINS", "*").strip()
    if raw_origins == "*":
        cors_origins = ["*"]
        allow_credentials = False
    else:
        cors_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/generated", StaticFiles(directory="backend/generated"), name="generated")

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Health endpoint with runtime capability metadata."""
        ml_enabled = os.getenv("NAVISPACE_ENABLE_ML", "auto").lower() != "false"
        model_path = os.getenv("NAVISPACE_SEG_MODEL", "").strip()
        ml_engine_loaded = False
        ml_reason: str | None = None
        ml_device = os.getenv("NAVISPACE_SEG_DEVICE", "cpu").strip() or "cpu"

        try:
            from backend.ml.infer import get_segmentation_runtime_status

            status = get_segmentation_runtime_status()
            ml_enabled = bool(status.enabled)
            ml_engine_loaded = bool(status.engine_loaded)
            ml_reason = status.reason
            ml_device = status.device
            model_path = status.model_path
        except Exception as exc:
            ml_reason = f"runtime-status-unavailable: {exc}"

        return {
            "status": "ok",
            "version": app.version,
            "ml_enabled": ml_enabled,
            "ml_model_configured": bool(model_path),
            "ml_engine_loaded": ml_engine_loaded,
            "ml_device": ml_device,
            "ml_reason": ml_reason,
            "mode": "multi" if STATE.building_manager is not None else "single",
        }

    @app.post("/process-blueprint")
    async def process_blueprint(file: UploadFile = File(...)) -> dict:
        """Process one uploaded blueprint and return 2D occupancy + model metadata."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file name provided")

        try:
            raw = await file.read()
            image = _decode_upload_image(raw)

            pre = preprocess_blueprint(image)
            hough_threshold, hough_min_len, hough_max_gap = adaptive_hough_params(pre["gray"].shape)
            walls = detect_walls(
                pre["edges"],
                threshold=hough_threshold,
                min_line_length=hough_min_len,
                max_line_gap=hough_max_gap,
            )
            walls = filter_wall_segments(
                walls,
                pre["gray"].shape,
                primary_bbox=pre.get("primary_bbox"),
                bbox_margin_px=max(18, int(min(pre["gray"].shape) * 0.015)),
                support_mask=pre.get("denoised"),
                min_support_ratio=0.1 if bool(pre.get("ml_used")) else 0.2,
                min_median_half_thickness_px=0.85 if bool(pre.get("ml_used")) else 1.2,
            )
            doors = detect_doors(
                pre["binary"],
                walls,
                door_mask=pre.get("door_mask"),
                prefer_ml_only=bool(pre.get("ml_used")),
            )
            doors = filter_door_candidates(
                doors,
                pre["gray"].shape,
                primary_bbox=pre.get("primary_bbox"),
                bbox_margin_px=max(18, int(min(pre["gray"].shape) * 0.015)),
            )

            grid, _ = walls_to_occupancy_grid(
                image_shape=pre["gray"].shape,
                walls=walls,
                cell_size_px=GRID_CELL_SIZE_PX,
                inflation_radius_px=1,
                door_candidates=doors,
                door_clearance_px=3,
            )

            ts = int(time.time() * 1000)
            grid_path = f"backend/generated/grids/grid_{ts}.json"
            export_grid_json(
                grid_path,
                grid,
                cell_size_px=GRID_CELL_SIZE_PX,
                model_scale_m_per_px=MODEL_SCALE_M_PER_PX,
            )

            model_filename = f"model_{ts}.glb"
            model_disk_path = f"backend/generated/models/{model_filename}"
            build_model_from_walls(
                walls=walls,
                image_shape=pre["gray"].shape,
                output_path=model_disk_path,
                wall_height_m=3.0,
                model_scale_m_per_px=MODEL_SCALE_M_PER_PX,
                door_candidates=doors,
            )

            STATE.grid = grid
            STATE.cell_size_m = GRID_CELL_SIZE_PX * MODEL_SCALE_M_PER_PX
            STATE.model_url = f"/generated/models/{model_filename}"

            # Clear stale multi-floor state to avoid mode ambiguity.
            STATE.building_manager = None
            STATE.grid3d = None
            STATE.ordered_floors = []
            STATE.connector_nodes = []

            rows, cols = grid.shape
            return {
                "message": "Blueprint processed successfully",
                "walls": walls,
                "doors": doors,
                "grid": json_grid(grid),
                "grid_shape": {"rows": rows, "cols": cols},
                "cell_size_m": STATE.cell_size_m,
                "model_url": STATE.model_url,
                "grid_url": f"/generated/grids/{Path(grid_path).name}",
                "ml_used": bool(pre.get("ml_used")),
                "ml_enabled": bool(pre.get("ml_enabled")),
                "ml_engine_loaded": bool(pre.get("ml_engine_loaded")),
                "ml_device": pre.get("ml_device"),
                "ml_reason": pre.get("ml_reason"),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Blueprint processing failed: {exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - safety net
            raise HTTPException(status_code=500, detail=f"Unexpected processing error: {exc}") from exc

    @app.post("/find-path", response_model=PathResponse)
    async def find_path(payload: PathRequest) -> PathResponse:
        """Compute 2D A* path between two grid points on latest single-floor grid."""
        if STATE.grid is None:
            raise HTTPException(status_code=400, detail="No processed blueprint available yet")

        rows, cols = STATE.grid.shape
        start_req = (payload.start.row, payload.start.col)
        goal_req = (payload.goal.row, payload.goal.col)

        if not (0 <= start_req[0] < rows and 0 <= start_req[1] < cols):
            raise HTTPException(status_code=400, detail="Invalid path query: Start is out of grid bounds")
        if not (0 <= goal_req[0] < rows and 0 <= goal_req[1] < cols):
            raise HTTPException(status_code=400, detail="Invalid path query: Goal is out of grid bounds")

        start = _nearest_free_cell(STATE.grid, start_req[0], start_req[1], max_radius=14)
        goal = _nearest_free_cell(STATE.grid, goal_req[0], goal_req[1], max_radius=14)

        if start is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid path query: Start is blocked and no nearby navigable cell was found",
            )
        if goal is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid path query: Goal is blocked and no nearby navigable cell was found",
            )

        try:
            path = astar(
                grid=STATE.grid,
                start=start,
                goal=goal,
                diagonal=payload.diagonal,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid path query: {exc}") from exc
        except Exception as exc:  # pragma: no cover - safety net
            raise HTTPException(status_code=500, detail=f"Unexpected pathfinding error: {exc}") from exc

        if not path:
            raise HTTPException(status_code=404, detail="No navigable path found")

        world_path = []
        for row, col in path:
            x, z = grid_to_world(row, col, STATE.cell_size_m)
            world_path.append({"x": x, "z": z})

        smoothed = smooth_path(STATE.grid, path)
        smoothed_world_path = []
        for row, col in smoothed:
            x, z = grid_to_world(row, col, STATE.cell_size_m)
            smoothed_world_path.append({"x": x, "z": z})

        return PathResponse(
            path=to_serializable_path(path),
            world_path=world_path,
            smoothed_path=to_serializable_path(smoothed),
            smoothed_world_path=smoothed_world_path,
        )

    @app.post("/process-building")
    async def process_building(
        files: list[UploadFile] = File(..., description="Blueprint images, one per floor"),
        floor_numbers: str | None = Form(
            default=None,
            description="JSON array floor numbers aligned with files, e.g. [0,1,2]",
        ),
        building_id: str = Form(default="building-1"),
        floor_height_m: float = Form(default=3.2),
        cell_size_m: float = Form(default=0.2),
        connector_json: str | None = Form(
            default=None,
            description="Optional connector JSON list",
        ),
    ) -> dict[str, Any]:
        """Process multiple floor blueprints and initialize 3D navigation state.

        Returns floor metadata, room indexing data, and 3D occupancy summary.
        """
        if not files:
            raise HTTPException(status_code=400, detail="At least one blueprint file is required")
        if floor_height_m <= 0:
            raise HTTPException(status_code=400, detail="floor_height_m must be > 0")
        if cell_size_m <= 0:
            raise HTTPException(status_code=400, detail="cell_size_m must be > 0")

        try:
            parsed_floor_numbers = _parse_floor_numbers(floor_numbers, len(files))
            parsed_connectors = _parse_connectors(connector_json)

            manager = BuildingManager(
                building_id=_safe_stem(building_id),
                floor_height_m=floor_height_m,
                default_cell_size_m=cell_size_m,
                model_scale_m_per_px=MODEL_SCALE_M_PER_PX,
            )

            ts = int(time.time() * 1000)
            floor_artifacts: dict[int, dict[str, Any]] = {}

            for idx, upload in enumerate(files):
                if not upload.filename:
                    raise ValueError(f"File at index {idx} has no filename")

                floor_number = parsed_floor_numbers[idx]
                image = _decode_upload_image(await upload.read())

                floor_meta = manager.process_floor_blueprint(
                    floor_number=floor_number,
                    image_bgr=image,
                    cell_size_m=cell_size_m,
                    hough_threshold=70,
                )

                # Build floor-specific GLB for stacked rendering in frontend.
                pre = preprocess_blueprint(image)
                hough_threshold, hough_min_len, hough_max_gap = adaptive_hough_params(pre["gray"].shape)
                walls = detect_walls(
                    pre["edges"],
                    threshold=hough_threshold,
                    min_line_length=hough_min_len,
                    max_line_gap=hough_max_gap,
                )
                walls = filter_wall_segments(
                    walls,
                    pre["gray"].shape,
                    primary_bbox=pre.get("primary_bbox"),
                    bbox_margin_px=max(18, int(min(pre["gray"].shape) * 0.015)),
                    support_mask=pre.get("denoised"),
                    min_support_ratio=0.1 if bool(pre.get("ml_used")) else 0.2,
                    min_median_half_thickness_px=0.85 if bool(pre.get("ml_used")) else 1.2,
                )
                doors = detect_doors(
                    pre["binary"],
                    walls,
                    door_mask=pre.get("door_mask"),
                    prefer_ml_only=bool(pre.get("ml_used")),
                )
                doors = filter_door_candidates(
                    doors,
                    pre["gray"].shape,
                    primary_bbox=pre.get("primary_bbox"),
                    bbox_margin_px=max(18, int(min(pre["gray"].shape) * 0.015)),
                )

                floor_artifacts[int(floor_number)] = {
                    "walls": walls,
                    "doors": doors,
                }

                model_filename = f"model_{_safe_stem(building_id)}_f{floor_number}_{ts}_{idx}.glb"
                model_path = f"backend/generated/models/{model_filename}"
                build_model_from_walls(
                    walls=walls,
                    image_shape=pre["gray"].shape,
                    output_path=model_path,
                    wall_height_m=3.0,
                    model_scale_m_per_px=MODEL_SCALE_M_PER_PX,
                    door_candidates=doors,
                )
                floor_meta.model_url = f"/generated/models/{model_filename}"

            # Register explicit connectors, or auto-seed one if floor count > 1.
            if parsed_connectors:
                for connector in parsed_connectors:
                    manager.add_connector(
                        connector_id=connector["connector_id"],
                        connector_type=connector["connector_type"],
                        floors=connector["floors"],
                        position_m=connector["position_m"],
                        radius_m=connector["radius_m"],
                    )
            else:
                auto_connector = _default_connector_for_building(manager)
                if auto_connector:
                    manager.add_connector(
                        connector_id=auto_connector["connector_id"],
                        connector_type=auto_connector["connector_type"],
                        floors=auto_connector["floors"],
                        position_m=auto_connector["position_m"],
                        radius_m=auto_connector["radius_m"],
                    )

            floor_grids = {f: manager.get_floor(f).grid for f in sorted(manager.meta.floors.keys())}
            raw_grid3d, ordered_floors = normalize_floor_grids(floor_grids)
            inflated_grid3d = inflate_3d_obstacles(raw_grid3d, radius_cells=1)

            connector_dicts = [
                {
                    "connector_id": c.connector_id,
                    "connector_type": c.connector_type,
                    "floors": c.floors,
                    "position_m": c.position_m,
                    "radius_m": c.radius_m,
                }
                for c in manager.meta.connectors
            ]
            connector_nodes = build_connector_nodes(
                ordered_floors=ordered_floors,
                connectors=connector_dicts,
                cell_size_m=manager.default_cell_size_m,
            )
            nav_grid3d = carve_connector_clearance(inflated_grid3d, connector_nodes)

            floors_payload = manager.get_floors_payload()["floors"]
            rooms_payload = manager.get_rooms_payload()["rooms"]
            scene_graph = build_scene_graph(
                building_id=manager.meta.building_id,
                floor_height_m=manager.meta.floor_height_m,
                model_scale_m_per_px=MODEL_SCALE_M_PER_PX,
                floors_payload=floors_payload,
                rooms_payload=rooms_payload,
                connectors_payload=connector_dicts,
                floor_artifacts=floor_artifacts,
            )
            validation_report = validate_scene_graph(scene_graph)
            room_connectivity = build_room_connectivity(rooms_payload, connector_dicts)

            STATE.building_manager = manager
            STATE.grid3d = nav_grid3d
            STATE.ordered_floors = ordered_floors
            STATE.connector_nodes = connector_nodes
            STATE.room_connectivity = room_connectivity

            # Keep single-floor state populated for compatibility with existing UI.
            first_floor = manager.get_floor(ordered_floors[0])
            STATE.grid = first_floor.grid
            STATE.cell_size_m = first_floor.cell_size_m
            STATE.model_url = first_floor.model_url

            return {
                "message": "Building processed successfully",
                "building_id": manager.meta.building_id,
                "floor_count": len(ordered_floors),
                "room_count": len(manager.list_rooms()),
                "grid3d_shape": {
                    "floors": int(nav_grid3d.shape[0]),
                    "rows": int(nav_grid3d.shape[1]),
                    "cols": int(nav_grid3d.shape[2]),
                },
                "cell_size_m": manager.default_cell_size_m,
                "floor_height_m": manager.meta.floor_height_m,
                "floors": floors_payload,
                "connectors": connector_dicts,
                "rooms": rooms_payload,
                "scene_graph": scene_graph,
                "validation_report": validation_report,
                "room_connectivity": room_connectivity,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Building processing failed: {exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - safety net
            raise HTTPException(status_code=500, detail=f"Unexpected building processing error: {exc}") from exc

    @app.get("/floors")
    async def get_floors() -> dict[str, Any]:
        """Return latest building and floor metadata."""
        manager = _latest_building_or_400()
        return manager.get_floors_payload()

    @app.get("/rooms")
    async def get_rooms(floor_number: int | None = Query(default=None)) -> dict[str, Any]:
        """Return indexed room metadata for latest processed building."""
        manager = _latest_building_or_400()

        if floor_number is None:
            return manager.get_rooms_payload()

        try:
            rooms = manager.list_rooms(floor_number=floor_number)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "building_id": manager.meta.building_id,
            "rooms": [_serialize_room(room) for room in rooms],
        }

    @app.get("/room-connectivity")
    async def get_room_connectivity() -> dict[str, Any]:
        """Return latest room-level connectivity graph for semantic routing."""
        _latest_building_or_400()
        if STATE.room_connectivity is None:
            raise HTTPException(status_code=400, detail="Room connectivity graph is not initialized")
        return STATE.room_connectivity

    @app.get("/room-route")
    async def get_room_route(start_room_id: str = Query(...), goal_room_id: str = Query(...)) -> dict[str, Any]:
        """Return semantic room-to-room route over connectivity graph."""
        _latest_building_or_400()
        if STATE.room_connectivity is None:
            raise HTTPException(status_code=400, detail="Room connectivity graph is not initialized")

        route = shortest_room_route(STATE.room_connectivity, start_room_id=start_room_id, goal_room_id=goal_room_id)
        if not route:
            raise HTTPException(status_code=404, detail="No room route found")

        return {
            "start_room_id": start_room_id,
            "goal_room_id": goal_room_id,
            "room_route": route,
            "hops": max(0, len(route) - 1),
        }

    @app.post("/find-path-3d", response_model=Path3DResponse)
    async def find_path_3d(payload: Path3DRequest) -> Path3DResponse:
        """Compute 3D A* path across floors using world-space or room endpoints."""
        manager = _latest_building_or_400()

        if STATE.grid3d is None:
            raise HTTPException(status_code=400, detail="3D occupancy grid is not initialized")

        def resolve_world_point(point: WorldPoint | None, room_id: str | None, label: str) -> tuple[float, float, float]:
            """Resolve endpoint from explicit point or room centroid."""
            if room_id:
                room = manager.find_room_by_id(room_id)
                if room is None:
                    raise HTTPException(status_code=404, detail=f"{label}_room_id '{room_id}' was not found")
                return (
                    float(room.centroid_m[0]),
                    float(room.centroid_m[1]),
                    float(room.centroid_m[2]),
                )

            if point is None:
                raise HTTPException(status_code=400, detail=f"{label} world coordinate is required")

            return float(point.x), float(point.y), float(point.z)

        try:
            start_m = resolve_world_point(payload.start, payload.start_room_id, "start")
            goal_m = resolve_world_point(payload.goal, payload.goal_room_id, "goal")

            result = find_path_3d_world(
                grid3d=STATE.grid3d,
                start_m=start_m,
                goal_m=goal_m,
                connector_nodes=STATE.connector_nodes,
                cell_size_m=manager.default_cell_size_m,
                floor_height_m=manager.meta.floor_height_m,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid 3D path query: {exc}") from exc
        except Exception as exc:  # pragma: no cover - safety net
            raise HTTPException(status_code=500, detail=f"Unexpected 3D pathfinding error: {exc}") from exc

        if not result.grid_path:
            raise HTTPException(status_code=404, detail="No navigable 3D path found")

        grid_steps: list[Path3DGridPoint] = []
        for floor_idx, row, col in result.grid_path:
            floor_number = (
                STATE.ordered_floors[floor_idx]
                if 0 <= floor_idx < len(STATE.ordered_floors)
                else int(floor_idx)
            )
            grid_steps.append(
                Path3DGridPoint(
                    floor=int(floor_number),
                    row=int(row),
                    col=int(col),
                )
            )

        world_path = [
            {"x": float(step["x"]), "y": float(step["y"]), "z": float(step["z"])}
            for step in result.world_path
        ]

        return Path3DResponse(grid_path=grid_steps, world_path=world_path)

    return app
