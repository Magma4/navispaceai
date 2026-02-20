"""Unit tests for scene graph building and validation."""

from __future__ import annotations

from backend.geometry_validation import validate_scene_graph
from backend.scene_graph import build_scene_graph


def _sample_scene_graph() -> dict:
    floors = [
        {
            "floor_number": 0,
            "elevation_m": 0.0,
            "cell_size_m": 0.2,
            "rows": 100,
            "cols": 100,
            "model_url": "/generated/models/f0.glb",
            "room_count": 1,
        }
    ]
    rooms = [
        {
            "room_id": "F0-R001",
            "floor_number": 0,
            "name": "Room 1",
            "centroid_m": [1.0, 0.0, 1.0],
            "polygon_m": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            "area_m2": 4.0,
        }
    ]
    connectors = [
        {
            "connector_id": "C1",
            "connector_type": "stairs",
            "floors": [0, 1],
            "position_m": [1.0, 0.0, 1.0],
            "radius_m": 0.5,
        }
    ]
    floor_artifacts = {
        0: {
            "walls": [
                {"x1": 0, "y1": 0, "x2": 40, "y2": 0},
                {"x1": 40, "y1": 0, "x2": 40, "y2": 40},
            ],
            "doors": [{"x": 20, "y": 1, "w": 4, "h": 3}],
        }
    }

    return build_scene_graph(
        building_id="b1",
        floor_height_m=3.2,
        model_scale_m_per_px=0.05,
        floors_payload=floors,
        rooms_payload=rooms,
        connectors_payload=connectors,
        floor_artifacts=floor_artifacts,
    )


def test_build_scene_graph_has_expected_sections() -> None:
    graph = _sample_scene_graph()

    assert graph["schema_version"] == "1.0"
    assert graph["building_id"] == "b1"
    assert len(graph["levels"]) == 1
    assert len(graph["levels"][0]["rooms"]) == 1
    assert len(graph["levels"][0]["walls"]) == 2
    assert len(graph["levels"][0]["openings"]) == 1
    assert len(graph["connectivity"]) == 1


def test_validate_scene_graph_flags_distant_door() -> None:
    graph = _sample_scene_graph()
    graph["levels"][0]["openings"][0]["pose_m"]["x"] = 200.0
    graph["levels"][0]["openings"][0]["pose_m"]["z"] = 200.0

    report = validate_scene_graph(graph)

    assert report["ok"] is True
    assert report["summary"]["warnings"] >= 1
    assert any(issue["kind"] == "door_clearance" for issue in report["issues"])
