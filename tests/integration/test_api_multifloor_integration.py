"""Integration tests for multi-floor FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import STATE, create_app


def _sample_blueprint_bytes() -> bytes:
    """Load bundled sample blueprint bytes for upload tests."""
    return Path("assets/sample_blueprint.png").read_bytes()


def test_multi_floor_endpoints_require_building_state() -> None:
    """GET /floors and /rooms should return 400 before building processing."""
    STATE.building_manager = None
    STATE.grid3d = None
    client = TestClient(create_app())

    floors_res = client.get("/floors")
    rooms_res = client.get("/rooms")

    assert floors_res.status_code == 400
    assert rooms_res.status_code == 400


def test_process_building_returns_floors_rooms_and_connectors() -> None:
    """POST /process-building should initialize floors, rooms, and connectors."""
    client = TestClient(create_app())
    floor_a = ("files", ("f0.png", _sample_blueprint_bytes(), "image/png"))
    floor_b = ("files", ("f1.png", _sample_blueprint_bytes(), "image/png"))

    process_res = client.post("/process-building", files=[floor_a, floor_b])
    assert process_res.status_code == 200
    body = process_res.json()

    assert body["floor_count"] == 2
    assert len(body["floors"]) == 2
    assert len(body["rooms"]) > 0
    assert len(body["connectors"]) >= 1
    assert "scene_graph" in body
    assert "validation_report" in body
    assert "room_connectivity" in body
    assert len(body["scene_graph"]["levels"]) == 2
    assert body["room_connectivity"]["node_count"] >= 2

    floors_res = client.get("/floors")
    rooms_res = client.get("/rooms")
    assert floors_res.status_code == 200
    assert rooms_res.status_code == 200
    assert len(floors_res.json()["floors"]) == 2
    assert len(rooms_res.json()["rooms"]) > 0


def test_find_path_3d_by_room_ids_returns_world_path() -> None:
    """POST /find-path-3d should return path payload for valid room IDs."""
    client = TestClient(create_app())
    floor_a = ("files", ("f0.png", _sample_blueprint_bytes(), "image/png"))
    floor_b = ("files", ("f1.png", _sample_blueprint_bytes(), "image/png"))
    process_res = client.post("/process-building", files=[floor_a, floor_b])
    assert process_res.status_code == 200

    room_start = "F0-R001"
    room_goal = "F1-R001"
    path_res = client.post(
        "/find-path-3d",
        json={"start_room_id": room_start, "goal_room_id": room_goal},
    )

    assert path_res.status_code == 200
    payload = path_res.json()
    assert len(payload["grid_path"]) > 0
    assert len(payload["world_path"]) == len(payload["grid_path"])


def test_room_connectivity_and_route_endpoints() -> None:
    """Room connectivity graph and semantic room route should be available."""
    client = TestClient(create_app())
    floor_a = ("files", ("f0.png", _sample_blueprint_bytes(), "image/png"))
    floor_b = ("files", ("f1.png", _sample_blueprint_bytes(), "image/png"))
    process_res = client.post("/process-building", files=[floor_a, floor_b])
    assert process_res.status_code == 200

    graph_res = client.get("/room-connectivity")
    assert graph_res.status_code == 200
    graph = graph_res.json()
    assert graph["node_count"] >= 2
    assert graph["edge_count"] >= 1

    route_res = client.get("/room-route", params={"start_room_id": "F0-R001", "goal_room_id": "F1-R001"})
    assert route_res.status_code == 200
    route = route_res.json()
    assert len(route["room_route"]) >= 2
    assert route["room_route"][0] == "F0-R001"
    assert route["room_route"][-1] == "F1-R001"
