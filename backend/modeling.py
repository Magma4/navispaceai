"""3D model generation from detected wall segments.

Purpose:
- Convert 2D wall vectors into 3D wall meshes.
- Export a GLB model for frontend visualization.

Usage example:
    >>> from backend.modeling import build_model_from_walls
    >>> build_model_from_walls(walls, (512, 512), "backend/generated/models/model.glb")
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import trimesh

WallSegment = dict[str, int]
DoorCandidate = dict[str, int]
StairCandidate = dict[str, int]


def _validate_modeling_inputs(
    walls: list[WallSegment],
    image_shape: tuple[int, int],
    model_scale_m_per_px: float,
    wall_height_m: float,
    wall_thickness_m: float,
) -> None:
    """Validate 2D geometry and scaling parameters used for mesh generation."""
    if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("image_shape must be (height, width) with positive values")
    if model_scale_m_per_px <= 0:
        raise ValueError("model_scale_m_per_px must be > 0")
    if wall_height_m <= 0:
        raise ValueError("wall_height_m must be > 0")
    if wall_thickness_m <= 0:
        raise ValueError("wall_thickness_m must be > 0")

    for seg in walls:
        if not {"x1", "y1", "x2", "y2"}.issubset(seg.keys()):
            raise ValueError("Each wall segment must include x1, y1, x2, y2")


def _segment_length_and_angle(x1: float, z1: float, x2: float, z2: float) -> tuple[float, float]:
    """Compute segment length and yaw angle in XZ plane."""
    dx = x2 - x1
    dz = z2 - z1
    length = math.hypot(dx, dz)
    angle = math.atan2(dz, dx)
    return length, angle


def _apply_face_color(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> None:
    """Apply a solid RGBA color to all mesh faces."""
    color = np.array(rgba, dtype=np.uint8)
    mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))


def _compute_wall_bounds_m(
    walls: list[WallSegment],
    model_scale_m_per_px: float,
) -> tuple[float, float, float, float] | None:
    """Compute min/max XZ bounds in meters from wall segments."""
    if not walls:
        return None

    xs: list[float] = []
    zs: list[float] = []
    for seg in walls:
        if not {"x1", "y1", "x2", "y2"}.issubset(seg.keys()):
            continue
        xs.append(float(seg["x1"]) * model_scale_m_per_px)
        xs.append(float(seg["x2"]) * model_scale_m_per_px)
        zs.append(float(seg["y1"]) * model_scale_m_per_px)
        zs.append(float(seg["y2"]) * model_scale_m_per_px)

    if not xs or not zs:
        return None
    return min(xs), max(xs), min(zs), max(zs)


def _raster_wall_run_meshes(
    wall_mask: np.ndarray,
    *,
    model_scale_m_per_px: float,
    wall_height_m: float,
    wall_thickness_m: float,
    min_run_px: int = 8,
) -> list[trimesh.Trimesh]:
    """Create wall boxes from horizontal and vertical runs on a denoised binary wall mask."""
    if wall_mask is None or wall_mask.size == 0 or wall_mask.ndim != 2:
        return []

    # Mild downsample to reduce mesh count while preserving global structure.
    down = 2
    resized = cv2.resize(
        (wall_mask > 0).astype(np.uint8) * 255,
        (max(1, wall_mask.shape[1] // down), max(1, wall_mask.shape[0] // down)),
        interpolation=cv2.INTER_NEAREST,
    )
    scale = model_scale_m_per_px * down

    meshes: list[trimesh.Trimesh] = []
    rows, cols = resized.shape

    # Horizontal runs
    for r in range(rows):
        row = resized[r]
        c = 0
        while c < cols:
            if row[c] == 0:
                c += 1
                continue
            start = c
            while c < cols and row[c] > 0:
                c += 1
            end = c - 1
            run_len = end - start + 1
            if run_len < int(min_run_px):
                continue

            length_m = max(0.08, run_len * scale)
            center_x = ((start + end) * 0.5) * scale
            center_z = r * scale

            wall_mesh = trimesh.creation.box(extents=(length_m, wall_height_m, wall_thickness_m))
            wall_mesh.apply_translation((center_x, wall_height_m / 2.0, center_z))
            meshes.append(wall_mesh)

    # Vertical runs
    for c in range(cols):
        col = resized[:, c]
        r = 0
        while r < rows:
            if col[r] == 0:
                r += 1
                continue
            start = r
            while r < rows and col[r] > 0:
                r += 1
            end = r - 1
            run_len = end - start + 1
            if run_len < int(min_run_px):
                continue

            length_m = max(0.08, run_len * scale)
            center_x = c * scale
            center_z = ((start + end) * 0.5) * scale

            wall_mesh = trimesh.creation.box(extents=(wall_thickness_m, wall_height_m, length_m))
            wall_mesh.apply_translation((center_x, wall_height_m / 2.0, center_z))
            meshes.append(wall_mesh)

    return meshes


def extrude_walls_to_scene(
    walls: list[WallSegment],
    image_shape: tuple[int, int],
    model_scale_m_per_px: float = 0.05,
    wall_height_m: float = 3.0,
    wall_thickness_m: float = 0.15,
    door_candidates: list[DoorCandidate] | None = None,
    staircase_candidates: list[StairCandidate] | None = None,
    wall_mask: np.ndarray | None = None,
    floor_color_rgba: tuple[int, int, int, int] = (192, 203, 214, 255),
    wall_color_rgba: tuple[int, int, int, int] = (88, 95, 108, 255),
    door_color_rgba: tuple[int, int, int, int] = (230, 120, 40, 255),
    stair_color_rgba: tuple[int, int, int, int] = (78, 170, 255, 255),
) -> trimesh.Scene:
    """Convert 2D wall line segments into a trimesh scene.

    Args:
        walls: List of 2D wall segments in image pixel coordinates.
        image_shape: Source image shape `(height, width)`.
        model_scale_m_per_px: Pixel-to-meter scale.
        wall_height_m: Wall extrusion height in meters.
        wall_thickness_m: Wall thickness in meters.
        door_candidates: Optional door bounding boxes (`x`, `y`, `w`, `h`) in source pixels.
        staircase_candidates: Optional staircase bounding boxes (`x`, `y`, `w`, `h`) in source pixels.
        floor_color_rgba: Floor mesh color.
        wall_color_rgba: Wall mesh color.
        door_color_rgba: Door mesh color.

    Returns:
        trimesh.Scene containing floor and wall meshes.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_modeling_inputs(walls, image_shape, model_scale_m_per_px, wall_height_m, wall_thickness_m)

    height_px, width_px = image_shape
    scene = trimesh.Scene()

    # Add a thin floor slab sized to detected structure bounds (not full page extent),
    # so camera fitting and rendering focus on the real building footprint.
    bounds = _compute_wall_bounds_m(walls, model_scale_m_per_px)
    if bounds is None:
        floor_w = max(1.0, width_px * model_scale_m_per_px)
        floor_d = max(1.0, height_px * model_scale_m_per_px)
        floor_cx = floor_w / 2.0
        floor_cz = floor_d / 2.0
    else:
        min_x, max_x, min_z, max_z = bounds
        pad = max(0.6, wall_thickness_m * 4.0)
        floor_w = max(1.0, (max_x - min_x) + pad * 2.0)
        floor_d = max(1.0, (max_z - min_z) + pad * 2.0)
        floor_cx = (min_x + max_x) / 2.0
        floor_cz = (min_z + max_z) / 2.0

    floor = trimesh.creation.box(extents=(floor_w, 0.05, floor_d))
    floor.apply_translation((floor_cx, -0.025, floor_cz))
    _apply_face_color(floor, floor_color_rgba)
    scene.add_geometry(floor, geom_name="floor")

    stair_rects = [s for s in (staircase_candidates or []) if {"x", "y", "w", "h"}.issubset(s.keys())]

    wall_mask_for_mesh = wall_mask if wall_mask is not None else np.zeros(image_shape, dtype=np.uint8)
    if stair_rects and wall_mask_for_mesh.size > 0:
        wall_mask_for_mesh = wall_mask_for_mesh.copy()
        for stair in stair_rects:
            sx = int(stair["x"])
            sy = int(stair["y"])
            sw = max(1, int(stair["w"]))
            sh = max(1, int(stair["h"]))
            pad = 3
            x0 = max(0, sx - pad)
            y0 = max(0, sy - pad)
            x1 = min(wall_mask_for_mesh.shape[1], sx + sw + pad)
            y1 = min(wall_mask_for_mesh.shape[0], sy + sh + pad)
            wall_mask_for_mesh[y0:y1, x0:x1] = 0

    raster_meshes = _raster_wall_run_meshes(
        wall_mask_for_mesh,
        model_scale_m_per_px=model_scale_m_per_px,
        wall_height_m=wall_height_m,
        wall_thickness_m=wall_thickness_m,
    )
    for idx, wall_mesh in enumerate(raster_meshes):
        _apply_face_color(wall_mesh, wall_color_rgba)
        scene.add_geometry(wall_mesh, geom_name=f"wall_raster_{idx}")

    def _segment_midpoint_in_stairs(seg: WallSegment) -> bool:
        if not stair_rects:
            return False
        mx = 0.5 * (float(seg["x1"]) + float(seg["x2"]))
        my = 0.5 * (float(seg["y1"]) + float(seg["y2"]))
        for stair in stair_rects:
            sx = float(stair["x"])
            sy = float(stair["y"])
            sw = max(1.0, float(stair["w"]))
            sh = max(1.0, float(stair["h"]))
            if sx <= mx <= sx + sw and sy <= my <= sy + sh:
                return True
        return False

    # Keep vector walls too; they add directional fidelity on top of raster runs.
    for idx, seg in enumerate(walls):
        x1 = seg["x1"] * model_scale_m_per_px
        z1 = seg["y1"] * model_scale_m_per_px
        x2 = seg["x2"] * model_scale_m_per_px
        z2 = seg["y2"] * model_scale_m_per_px

        length, angle = _segment_length_and_angle(x1, z1, x2, z2)
        if length <= 0.18:
            continue
        if _segment_midpoint_in_stairs(seg):
            continue

        wall_mesh = trimesh.creation.box(extents=(length, wall_height_m, wall_thickness_m))

        center_x = (x1 + x2) / 2.0
        center_z = (z1 + z2) / 2.0

        rotation = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        translation = trimesh.transformations.translation_matrix([center_x, wall_height_m / 2.0, center_z])
        transform = np.dot(translation, rotation)
        wall_mesh.apply_transform(transform)
        _apply_face_color(wall_mesh, wall_color_rgba)

        scene.add_geometry(wall_mesh, geom_name=f"wall_vec_{idx}")

    # Add door leaves with a slight open angle for clear visual semantics.
    for idx, door in enumerate(door_candidates or []):
        if not {"x", "y", "w", "h"}.issubset(door.keys()):
            continue

        door_w_px = max(1.0, float(door["w"]))
        door_h_px = max(1.0, float(door["h"]))
        center_x_px = float(door["x"]) + door_w_px / 2.0
        center_z_px = float(door["y"]) + door_h_px / 2.0

        center_x = center_x_px * model_scale_m_per_px
        center_z = center_z_px * model_scale_m_per_px

        clear_width_m = max(0.8, min(1.25, max(door_w_px, door_h_px) * model_scale_m_per_px))
        leaf_width_m = clear_width_m * 0.48
        door_height_m = min(2.2, max(1.95, wall_height_m * 0.72))
        door_thickness_m = min(0.06, wall_thickness_m * 0.45)

        # Orient along longer axis of detected door bbox and rotate leaf open.
        base_yaw = 0.0 if door_w_px >= door_h_px else math.pi / 2.0
        open_yaw = math.radians(28.0)

        # Hinge offset from opening center so the leaf appears to swing.
        hx = center_x - (clear_width_m * 0.25) * math.cos(base_yaw)
        hz = center_z - (clear_width_m * 0.25) * math.sin(base_yaw)

        door_leaf = trimesh.creation.box(extents=(leaf_width_m, door_height_m, door_thickness_m))
        rot_base = trimesh.transformations.rotation_matrix(base_yaw + open_yaw, [0, 1, 0])
        tr = trimesh.transformations.translation_matrix([hx, door_height_m / 2.0, hz])
        door_leaf.apply_transform(np.dot(tr, rot_base))
        _apply_face_color(door_leaf, door_color_rgba)
        scene.add_geometry(door_leaf, geom_name=f"door_{idx}")

    # Add staircase as repeated steps (instead of one block) for realism.
    for idx, stair in enumerate(staircase_candidates or []):
        if not {"x", "y", "w", "h"}.issubset(stair.keys()):
            continue

        stair_w_px = max(1.0, float(stair["w"]))
        stair_h_px = max(1.0, float(stair["h"]))
        center_x_px = float(stair["x"]) + stair_w_px / 2.0
        center_z_px = float(stair["y"]) + stair_h_px / 2.0

        center_x = center_x_px * model_scale_m_per_px
        center_z = center_z_px * model_scale_m_per_px
        run_m = max(0.8, max(stair_w_px, stair_h_px) * model_scale_m_per_px)
        width_m = max(0.8, min(stair_w_px, stair_h_px) * model_scale_m_per_px)
        yaw = 0.0 if stair_w_px >= stair_h_px else math.pi / 2.0

        step_count = int(max(5, min(14, round(run_m / 0.28))))
        tread_m = run_m / max(1, step_count)
        riser_m = min(0.18, max(0.12, wall_height_m * 0.05))

        start_offset = -run_m * 0.5 + tread_m * 0.5
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        for step_idx in range(step_count):
            local_x = start_offset + step_idx * tread_m
            local_z = 0.0
            world_x = center_x + local_x * cos_y - local_z * sin_y
            world_z = center_z + local_x * sin_y + local_z * cos_y
            step_h = riser_m * (step_idx + 1)

            step_mesh = trimesh.creation.box(extents=(tread_m * 0.95, step_h, width_m * 0.96))
            rot = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
            tr = trimesh.transformations.translation_matrix([world_x, step_h / 2.0, world_z])
            step_mesh.apply_transform(np.dot(tr, rot))
            _apply_face_color(step_mesh, stair_color_rgba)
            scene.add_geometry(step_mesh, geom_name=f"stair_{idx}_step_{step_idx}")

    return scene


def export_scene_glb(scene: trimesh.Scene, output_path: str) -> str:
    """Export a trimesh scene to GLB format.

    Args:
        scene: Trimesh scene to export.
        output_path: Destination `.glb` file path.

    Returns:
        String path to exported GLB file.

    Raises:
        ValueError: If scene/output path is invalid.
    """
    if scene is None:
        raise ValueError("scene is required")
    if not output_path or not str(output_path).strip():
        raise ValueError("output_path is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    glb_data = scene.export(file_type="glb")
    output.write_bytes(glb_data)
    return str(output)


def build_model_from_walls(
    walls: list[WallSegment],
    image_shape: tuple[int, int],
    output_path: str,
    wall_height_m: float = 3.0,
    model_scale_m_per_px: float = 0.05,
    door_candidates: list[DoorCandidate] | None = None,
    staircase_candidates: list[StairCandidate] | None = None,
    wall_mask: np.ndarray | None = None,
) -> str:
    """High-level helper to construct and export a 3D model from wall segments.

    Args:
        walls: Wall segment list.
        image_shape: Source image shape `(height, width)`.
        output_path: Export destination path.
        wall_height_m: Wall extrusion height.
        model_scale_m_per_px: Pixel-to-meter scale.
        door_candidates: Optional detected door boxes.

    Returns:
        Exported `.glb` file path.
    """
    scene = extrude_walls_to_scene(
        walls=walls,
        image_shape=image_shape,
        model_scale_m_per_px=model_scale_m_per_px,
        wall_height_m=wall_height_m,
        door_candidates=door_candidates,
        staircase_candidates=staircase_candidates,
        wall_mask=wall_mask,
    )
    return export_scene_glb(scene, output_path)
