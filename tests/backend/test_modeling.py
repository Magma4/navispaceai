"""Unit tests for backend.modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import pytest

from backend.modeling import build_model_from_walls, export_scene_glb, extrude_walls_to_scene


def test_extrude_walls_to_scene_with_no_walls_creates_floor() -> None:
    """Even with no walls, scene should contain floor geometry."""
    scene = extrude_walls_to_scene(walls=[], image_shape=(64, 64))
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) >= 1


def test_extrude_walls_invalid_shape_raises() -> None:
    """Modeling should reject invalid image shape values."""
    with pytest.raises(ValueError, match="image_shape"):
        extrude_walls_to_scene(walls=[], image_shape=(0, 64))


def test_export_scene_glb_writes_file(tmp_path: Path) -> None:
    """Export should produce a non-empty GLB file."""
    scene = extrude_walls_to_scene(
        walls=[{"x1": 5, "y1": 5, "x2": 50, "y2": 5}],
        image_shape=(64, 64),
    )
    output = tmp_path / "model.glb"
    path = export_scene_glb(scene, str(output))

    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_build_model_from_walls_creates_glb(tmp_path: Path) -> None:
    """High-level builder should return exported GLB path."""
    output = tmp_path / "building.glb"
    path = build_model_from_walls(
        walls=[{"x1": 10, "y1": 10, "x2": 50, "y2": 10}],
        image_shape=(64, 64),
        output_path=str(output),
    )
    assert Path(path).exists()


def test_extrude_walls_to_scene_uses_wall_mask_runs() -> None:
    """Raster wall mask should generate additional wall geometry."""
    wall_mask = np.zeros((64, 64), dtype=np.uint8)
    wall_mask[20:24, 8:56] = 255

    scene = extrude_walls_to_scene(
        walls=[],
        image_shape=(64, 64),
        wall_mask=wall_mask,
    )

    assert isinstance(scene, trimesh.Scene)
    assert any(name.startswith("wall_raster_") for name in scene.geometry.keys())
