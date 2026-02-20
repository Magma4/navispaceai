"""Unit tests for backend.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing import load_blueprint, preprocess_blueprint


def test_load_blueprint_invalid_path_raises() -> None:
    """Loading a nonexistent image should raise a ValueError."""
    with pytest.raises(ValueError, match="Unable to load blueprint"):
        load_blueprint("/tmp/does-not-exist.png")


def test_preprocess_blueprint_success() -> None:
    """Preprocessing should return expected keys and image shapes."""
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    outputs = preprocess_blueprint(img)

    required = {"gray", "binary", "denoised", "edges", "door_mask"}
    assert required.issubset(set(outputs.keys()))
    assert outputs["gray"].shape == (64, 64)
    assert outputs["edges"].shape == (64, 64)
    assert outputs["door_mask"].shape == (64, 64)


def test_preprocess_blueprint_empty_raises() -> None:
    """Empty arrays are invalid input."""
    with pytest.raises(ValueError, match="empty"):
        preprocess_blueprint(np.array([]))


def test_preprocess_blueprint_wrong_shape_raises() -> None:
    """2D arrays should fail because preprocessing expects BGR input."""
    with pytest.raises(ValueError, match=r"shape \(H, W, 3\)"):
        preprocess_blueprint(np.zeros((64, 64), dtype=np.uint8))
