"""Integration tests for FastAPI blueprint processing and pathfinding endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import STATE, create_app


def _sample_blueprint_bytes() -> bytes:
    """Load bundled sample blueprint bytes for API upload tests."""
    return Path("assets/sample_blueprint.png").read_bytes()


def test_process_blueprint_returns_grid_and_model_paths() -> None:
    """POST /process-blueprint should return structured output and generated URLs."""
    STATE.grid = None
    client = TestClient(create_app())

    files = {"file": ("sample_blueprint.png", _sample_blueprint_bytes(), "image/png")}
    res = client.post("/process-blueprint", files=files)

    assert res.status_code == 200
    body = res.json()

    assert isinstance(body["grid"], list)
    assert body["grid_shape"]["rows"] > 0
    assert body["grid_shape"]["cols"] > 0
    assert body["model_url"].startswith("/generated/models/model_")
    assert body["grid_url"].startswith("/generated/grids/grid_")


def test_find_path_valid_and_invalid_coordinates() -> None:
    """POST /find-path should return path for valid coords and 400 for invalid ones."""
    STATE.grid = None
    client = TestClient(create_app())

    files = {"file": ("sample_blueprint.png", _sample_blueprint_bytes(), "image/png")}
    process_res = client.post("/process-blueprint", files=files)
    assert process_res.status_code == 200

    body = process_res.json()
    rows = body["grid_shape"]["rows"]
    cols = body["grid_shape"]["cols"]

    # Pick corners and rely on retry fallback if corner is occupied.
    start = {"row": 0, "col": 0}
    goal = {"row": rows - 1, "col": cols - 1}

    valid_res = client.post("/find-path", json={"start": start, "goal": goal, "diagonal": True})
    if valid_res.status_code == 404:
        # If corners are blocked in this sample, use close interior points.
        valid_res = client.post(
            "/find-path",
            json={"start": {"row": 2, "col": 2}, "goal": {"row": max(3, rows - 3), "col": max(3, cols - 3)}},
        )

    assert valid_res.status_code == 200
    payload = valid_res.json()
    assert len(payload["path"]) > 0
    assert len(payload["world_path"]) == len(payload["path"])

    invalid_res = client.post(
        "/find-path",
        json={"start": {"row": rows + 10, "col": 0}, "goal": {"row": 1, "col": 1}},
    )
    assert invalid_res.status_code == 400
    assert "Invalid path query" in invalid_res.json()["detail"]
