"""Pytest global fixtures and test isolation hooks."""

from __future__ import annotations

import numpy as np
import pytest

from backend.api import STATE


@pytest.fixture(autouse=True)
def reset_processing_state() -> None:
    """Reset in-memory API processing state before each test."""
    STATE.grid = None
    STATE.model_url = None
    STATE.cell_size_m = 0.2
    STATE.building_manager = None
    STATE.grid3d = None
    STATE.ordered_floors = []
    STATE.connector_nodes = []


@pytest.fixture()
def open_grid() -> np.ndarray:
    """Provide a simple reusable free-space grid."""
    return np.zeros((10, 10), dtype=np.uint8)
