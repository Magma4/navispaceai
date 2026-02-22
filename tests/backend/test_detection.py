"""Unit tests for backend.detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.detection import (
    detect_doors,
    detect_staircases,
    detect_walls,
    extract_walls_from_mask,
    filter_door_candidates,
    merge_collinear_wall_segments,
)


def test_detect_walls_finds_segments_on_simple_line() -> None:
    """A clear line should produce at least one detected wall segment."""
    edges = np.zeros((128, 128), dtype=np.uint8)
    cv2.line(edges, (10, 64), (118, 64), 255, 2)

    walls = detect_walls(edges, threshold=20, min_line_length=20, max_line_gap=2)

    assert len(walls) >= 1
    assert {"x1", "y1", "x2", "y2"}.issubset(walls[0].keys())


def test_detect_walls_invalid_input_raises() -> None:
    """Non-2D image input should raise validation error."""
    with pytest.raises(ValueError, match="2D"):
        detect_walls(np.zeros((16, 16, 3), dtype=np.uint8))


def test_detect_doors_returns_list_for_blank_binary() -> None:
    """Blank binary input should return an empty candidate list."""
    binary = np.zeros((64, 64), dtype=np.uint8)
    doors = detect_doors(binary, walls=[])
    assert isinstance(doors, list)


def test_detect_doors_invalid_input_raises() -> None:
    """Door detection should validate image dimensionality."""
    with pytest.raises(ValueError, match="2D"):
        detect_doors(np.zeros((8, 8, 3), dtype=np.uint8), walls=[])


def test_detect_doors_uses_ml_mask_when_available() -> None:
    """ML door mask should be converted into door candidates."""
    binary = np.zeros((64, 64), dtype=np.uint8)
    door_mask = np.zeros((64, 64), dtype=np.uint8)
    door_mask[20:32, 28:36] = 255

    doors = detect_doors(binary=binary, walls=[], door_mask=door_mask)
    assert len(doors) >= 1


def test_extract_walls_from_mask_returns_segments() -> None:
    """Contour vectorization should produce wall segments on a rectangular mask."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (108, 108), 255, 3)

    walls = extract_walls_from_mask(mask)
    assert len(walls) >= 4
    assert {"x1", "y1", "x2", "y2"}.issubset(walls[0].keys())


def test_merge_collinear_wall_segments_merges_fragments() -> None:
    """Near-collinear contiguous fragments should collapse into fewer longer segments."""
    walls = [
        {"x1": 10, "y1": 10, "x2": 40, "y2": 10},
        {"x1": 41, "y1": 10, "x2": 80, "y2": 11},
        {"x1": 82, "y1": 11, "x2": 120, "y2": 11},
    ]

    merged = merge_collinear_wall_segments(walls, endpoint_gap_px=6.0)
    assert len(merged) < len(walls)


def test_detect_staircases_from_parallel_runs() -> None:
    """Repeated parallel treads should yield staircase candidates."""
    binary = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(binary, (20, 20), (105, 95), 255, 1)
    for y in range(24, 92, 10):
        cv2.line(binary, (24, y), (101, y), 255, 2)

    stairs = detect_staircases(binary)
    assert isinstance(stairs, list)
    assert len(stairs) >= 1


def test_filter_door_candidates_prefers_near_wall_openings() -> None:
    """Doors far from any wall segment should be filtered out."""
    doors = [
        {"x": 50, "y": 50, "w": 8, "h": 12},
        {"x": 10, "y": 10, "w": 8, "h": 12},
    ]
    walls = [{"x1": 0, "y1": 14, "x2": 30, "y2": 14}]

    kept = filter_door_candidates(
        doors,
        image_shape=(100, 100),
        walls=walls,
        max_wall_gap_px=10.0,
    )

    assert len(kept) == 1
    assert kept[0]["x"] == 10
