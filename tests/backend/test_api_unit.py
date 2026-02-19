"""Unit-level API tests for direct endpoint behavior and error handling."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.api import STATE, create_app


def test_health_endpoint() -> None:
    """Health endpoint should report API availability."""
    client = TestClient(create_app())
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_find_path_without_processed_blueprint_returns_400() -> None:
    """Pathfinding should reject requests before a blueprint is processed."""
    STATE.grid = None
    client = TestClient(create_app())

    res = client.post("/find-path", json={"start": {"row": 0, "col": 0}, "goal": {"row": 1, "col": 1}})
    assert res.status_code == 400
    assert "No processed blueprint" in res.json()["detail"]


def test_find_path_no_path_returns_404() -> None:
    """When no route exists in current grid, API should return 404."""
    STATE.grid = np.ones((6, 6), dtype=np.uint8)
    STATE.grid[0, 0] = 0
    STATE.grid[5, 5] = 0

    client = TestClient(create_app())
    res = client.post("/find-path", json={"start": {"row": 0, "col": 0}, "goal": {"row": 5, "col": 5}})

    assert res.status_code == 404
    assert res.json()["detail"] == "No navigable path found"
