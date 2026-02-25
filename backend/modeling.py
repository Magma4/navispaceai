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


def _segment_support_in_mask(seg: WallSegment, mask: np.ndarray, samples: int = 28) -> float:
    """Estimate what fraction of a segment lies on non-zero mask pixels."""
    if mask is None or mask.size == 0:
        return 0.0
    h, w = mask.shape[:2]
    x1 = float(seg["x1"])
    y1 = float(seg["y1"])
    x2 = float(seg["x2"])
    y2 = float(seg["y2"])

    total = 0
    hit = 0
    for i in range(max(2, int(samples))):
        t = i / max(1, samples - 1)
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        total += 1
        if int(mask[y, x]) > 0:
            hit += 1

    if total == 0:
        return 0.0
    return float(hit) / float(total)


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
    # Bridge tiny gaps so long walls don't render as broken fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel, iterations=1)
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
    use_simple_stair_mesh: bool = True,
    render_door_meshes: bool = False,
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
    stair_rects_world: list[tuple[float, float, float, float]] = []

    # Stairwell lock mask in source-pixel space (hard no-wall zone).
    stair_lock_mask = np.zeros(image_shape, dtype=np.uint8)

    for stair in stair_rects:
        sx = float(stair["x"])
        sy = float(stair["y"])
        sw = max(1.0, float(stair["w"]))
        sh = max(1.0, float(stair["h"]))
        stair_rects_world.append((
            sx * model_scale_m_per_px,
            sy * model_scale_m_per_px,
            (sx + sw) * model_scale_m_per_px,
            (sy + sh) * model_scale_m_per_px,
        ))

        pad = 24.0
        x0 = max(0, int(math.floor(sx - pad)))
        y0 = max(0, int(math.floor(sy - pad)))
        x1 = min(stair_lock_mask.shape[1], int(math.ceil(sx + sw + pad)))
        y1 = min(stair_lock_mask.shape[0], int(math.ceil(sy + sh + pad)))
        stair_lock_mask[y0:y1, x0:x1] = 255

    wall_mask_for_mesh = wall_mask if wall_mask is not None else np.zeros(image_shape, dtype=np.uint8)
    if stair_rects and wall_mask_for_mesh.size > 0:
        wall_mask_for_mesh = wall_mask_for_mesh.copy()
        wall_mask_for_mesh[stair_lock_mask > 0] = 0

    def _mesh_overlaps_stairs_world(mesh: trimesh.Trimesh) -> bool:
        if not stair_rects_world:
            return False
        bounds = mesh.bounds
        if bounds is None or len(bounds) != 2:
            return False
        min_x, _, min_z = float(bounds[0][0]), float(bounds[0][1]), float(bounds[0][2])
        max_x, _, max_z = float(bounds[1][0]), float(bounds[1][1]), float(bounds[1][2])
        pad_m = max(0.16, model_scale_m_per_px * 20.0)
        for sx0, sz0, sx1, sz1 in stair_rects_world:
            rx0 = sx0 - pad_m
            rz0 = sz0 - pad_m
            rx1 = sx1 + pad_m
            rz1 = sz1 + pad_m
            if not (max_x < rx0 or min_x > rx1 or max_z < rz0 or min_z > rz1):
                return True
        return False

    raster_meshes = _raster_wall_run_meshes(
        wall_mask_for_mesh,
        model_scale_m_per_px=model_scale_m_per_px,
        wall_height_m=wall_height_m,
        wall_thickness_m=wall_thickness_m,
    )
    raster_idx = 0
    for wall_mesh in raster_meshes:
        if _mesh_overlaps_stairs_world(wall_mesh):
            continue
        _apply_face_color(wall_mesh, wall_color_rgba)
        scene.add_geometry(wall_mesh, geom_name=f"wall_raster_{raster_idx}")
        raster_idx += 1

    has_stair_lock = stair_lock_mask.size > 0 and np.count_nonzero(stair_lock_mask) > 0

    def _segment_overlaps_stairs(seg: WallSegment) -> bool:
        if not has_stair_lock:
            return False

        x1 = float(seg["x1"])
        y1 = float(seg["y1"])
        x2 = float(seg["x2"])
        y2 = float(seg["y2"])

        # Sample along the segment in pixel space and reject if any sample enters
        # the locked stairwell zone.
        seg_len = max(abs(x2 - x1), abs(y2 - y1))
        steps = max(3, int(seg_len / 3.0))
        h, w = stair_lock_mask.shape[:2]
        for i in range(steps + 1):
            t = i / max(1, steps)
            sx = int(round(x1 + (x2 - x1) * t))
            sy = int(round(y1 + (y2 - y1) * t))
            if 0 <= sx < w and 0 <= sy < h and int(stair_lock_mask[sy, sx]) > 0:
                return True
        return False

    # Keep vector walls for directional fidelity, but avoid duplicating walls
    # already well-covered by raster runs.
    for idx, seg in enumerate(walls):
        x1 = seg["x1"] * model_scale_m_per_px
        z1 = seg["y1"] * model_scale_m_per_px
        x2 = seg["x2"] * model_scale_m_per_px
        z2 = seg["y2"] * model_scale_m_per_px

        length, angle = _segment_length_and_angle(x1, z1, x2, z2)
        if length <= 0.18:
            continue
        if _segment_overlaps_stairs(seg):
            continue

        # Conflict resolver: if raster wall mask already strongly supports this segment,
        # skip vector duplicate to prevent over-thick/double walls.
        if wall_mask_for_mesh is not None and wall_mask_for_mesh.size > 0:
            support = _segment_support_in_mask(seg, wall_mask_for_mesh)
            if support >= 0.72:
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

    # Optional door meshes. Disabled by default to avoid orange stub artifacts.
    if render_door_meshes:
        for idx, door in enumerate(door_candidates or []):
            if not {"x", "y", "w", "h"}.issubset(door.keys()):
                continue

            door_w_px = max(1.0, float(door["w"]))
            door_h_px = max(1.0, float(door["h"]))
            center_x_px = float(door["x"]) + door_w_px / 2.0
            center_z_px = float(door["y"]) + door_h_px / 2.0

            center_x = center_x_px * model_scale_m_per_px
            center_z = center_z_px * model_scale_m_per_px

            clear_width_m = max(0.82, min(1.15, max(door_w_px, door_h_px) * model_scale_m_per_px))
            leaf_width_m = clear_width_m * 0.9
            door_height_m = min(2.15, max(1.98, wall_height_m * 0.74))
            door_thickness_m = min(0.05, wall_thickness_m * 0.35)

            # Orient along longer axis of detected door bbox and rotate leaf open.
            base_yaw = 0.0 if door_w_px >= door_h_px else math.pi / 2.0
            open_yaw = math.radians(52.0)

            # Hinge offset from opening center so the leaf appears to swing.
            hx = center_x - (clear_width_m * 0.45) * math.cos(base_yaw)
            hz = center_z - (clear_width_m * 0.45) * math.sin(base_yaw)

            door_leaf = trimesh.creation.box(extents=(leaf_width_m, door_height_m, door_thickness_m))
            rot_base = trimesh.transformations.rotation_matrix(base_yaw + open_yaw, [0, 1, 0])
            tr = trimesh.transformations.translation_matrix([hx, door_height_m / 2.0, hz])
            door_leaf.apply_transform(np.dot(tr, rot_base))
            _apply_face_color(door_leaf, door_color_rgba)
            scene.add_geometry(door_leaf, geom_name=f"door_{idx}")

    # Stair meshes: default to simple block for stability while detection quality improves.
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

        if use_simple_stair_mesh:
            stair_height_m = min(0.45, max(0.22, wall_height_m * 0.14))
            stair_mesh = trimesh.creation.box(extents=(run_m * 0.98, stair_height_m, width_m * 0.98))
            rot = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
            tr = trimesh.transformations.translation_matrix([center_x, stair_height_m / 2.0, center_z])
            stair_mesh.apply_transform(np.dot(tr, rot))
            _apply_face_color(stair_mesh, stair_color_rgba)
            scene.add_geometry(stair_mesh, geom_name=f"stair_{idx}")
            continue

        # Optional detailed stepped mesh.
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

    _prune_tiny_scene_geometry(scene, min_xy_extent_m=0.12)
    return scene


def _prune_tiny_scene_geometry(scene: trimesh.Scene, min_xy_extent_m: float = 0.12) -> None:
    """Remove tiny disconnected artifacts that commonly come from noisy detections."""
    if scene is None:
        return

    remove_names: list[str] = []
    for name, geom in scene.geometry.items():
        # Keep semantic non-wall items; culling targets micro wall artifacts.
        if not str(name).startswith(("wall_", "wall")):
            continue
        bounds = getattr(geom, "bounds", None)
        if bounds is None or len(bounds) != 2:
            continue
        extents = np.asarray(bounds[1]) - np.asarray(bounds[0])
        if extents.size < 3:
            continue
        if float(max(extents[0], extents[2])) < float(min_xy_extent_m):
            remove_names.append(name)

    for name in remove_names:
        scene.delete_geometry(name)


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
    use_simple_stair_mesh: bool = False,
) -> str:
    """High-level helper to construct and export a 3D model from wall segments.

    Args:
        walls: Wall segment list.
        image_shape: Source image shape `(height, width)`.
        output_path: Export destination path.
        wall_height_m: Wall extrusion height.
        model_scale_m_per_px: Pixel-to-meter scale.
        door_candidates: Optional detected door boxes.
        use_simple_stair_mesh: If True, use simple stair block meshes; otherwise
            export stepped stairs for more realistic geometry.

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
        use_simple_stair_mesh=use_simple_stair_mesh,
    )
    return export_scene_glb(scene, output_path)
