"""Blueprint preprocessing pipeline.

Purpose:
- Load and normalize blueprint image input for downstream detection.
- Generate derived images used by wall/door extraction.

Usage example:
    >>> import cv2
    >>> from backend.preprocessing import preprocess_blueprint
    >>> image = cv2.imread("assets/sample_blueprint.png")
    >>> outputs = preprocess_blueprint(image)
    >>> outputs["edges"].shape
"""

from __future__ import annotations

import cv2
import numpy as np


def _predict_ml_masks(image_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Try ML segmentation inference and return (wall_mask, door_mask).

    Returns (None, None) if ML is disabled/unavailable.
    """
    try:
        from backend.ml.infer import get_segmentation_engine_from_env

        engine = get_segmentation_engine_from_env()
        if engine is None:
            return None, None

        prediction = engine.predict(image_bgr)
        return prediction.wall_mask, prediction.door_mask
    except Exception:
        # Safe fallback to classical pipeline.
        return None, None


def _get_ml_runtime_status() -> dict[str, object]:
    """Read lightweight ML runtime diagnostics for observability."""
    try:
        from backend.ml.infer import get_segmentation_runtime_status

        status = get_segmentation_runtime_status()
        return {
            "ml_enabled": bool(status.enabled),
            "ml_engine_loaded": bool(status.engine_loaded),
            "ml_model_path": str(status.model_path or ""),
            "ml_model_exists": bool(status.model_exists),
            "ml_device": str(status.device),
            "ml_reason": status.reason,
        }
    except Exception as exc:
        return {
            "ml_enabled": False,
            "ml_engine_loaded": False,
            "ml_model_path": "",
            "ml_model_exists": False,
            "ml_device": "cpu",
            "ml_reason": f"runtime-status-error: {exc}",
        }


def _retain_primary_components(binary_mask: np.ndarray, keep_ratio: float = 0.2, max_components: int = 3) -> np.ndarray:
    """Keep only dominant connected components from a binary mask.

    This removes detached annotations/legends that often appear far from the building footprint
    and can distort 3D framing and path planning.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return binary_mask

    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        components.append((label, area))

    if not components:
        return binary_mask

    components.sort(key=lambda x: x[1], reverse=True)
    largest_area = float(components[0][1])

    keep_labels: set[int] = set()
    for idx, (label, area) in enumerate(components):
        if idx >= max_components:
            break
        if area / largest_area < keep_ratio:
            continue
        keep_labels.add(label)

    if not keep_labels:
        keep_labels.add(components[0][0])

    out = np.zeros_like(binary_mask, dtype=np.uint8)
    for label in keep_labels:
        out[labels == label] = 255
    return out


def _largest_component_bbox(binary_mask: np.ndarray) -> dict[str, int] | None:
    """Return bounding box of the largest connected component in a binary mask."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    best_label = -1
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    if best_label <= 0 or best_area <= 0:
        return None

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    return {"x": x, "y": y, "w": w, "h": h}


def _extract_wall_mask(gray: np.ndarray) -> np.ndarray:
    """Build a wall-focused binary mask from grayscale blueprint input.

    Strategy:
    - Keep only dark strokes (walls) and suppress light guide lines.
    - Bridge small wall gaps.
    - Remove tiny text/symbol artifacts with connected-component filtering.
    - Retain only dominant footprint components to ignore detached plan references.
    """
    # Keep strong/dark drafting lines.
    _, dark = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY_INV)

    # Bridge only tiny gaps; larger kernels tend to connect dashed guide lines into fake walls.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Remove tiny artifacts (text/symbol noise) while preserving walls.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((wall_mask > 0).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(wall_mask)
    min_area = max(60, int(gray.shape[0] * gray.shape[1] * 0.00002))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        if max(w, h) < 18:
            continue
        filtered[labels == label] = 255

    # Keep dominant footprint only to suppress detached legends/annotations/dashed references.
    filtered = _retain_primary_components(filtered, keep_ratio=0.35, max_components=1)
    return filtered


def load_blueprint(path: str) -> np.ndarray:
    """Load a blueprint image from disk as a BGR numpy array.

    Args:
        path: File path to blueprint image.

    Returns:
        Loaded image in BGR channel order.

    Raises:
        ValueError: If path is empty or file cannot be decoded as an image.
    """
    if not path or not str(path).strip():
        raise ValueError("Blueprint path is required")

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to load blueprint from path: {path}")
    return image


def preprocess_blueprint(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Run standard preprocessing for blueprint analysis.

    Pipeline:
    1. Convert to grayscale.
    2. Adaptive thresholding.
    3. Morphological denoising.
    4. Edge detection.

    Args:
        image_bgr: Input color image array of shape (H, W, 3).

    Returns:
        Dictionary with keys: `gray`, `binary`, `denoised`, `edges`.

    Raises:
        ValueError: If input is empty, not a numpy array, or not a 3-channel image.
    """
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("Input blueprint image must be a numpy array")
    if image_bgr.size == 0:
        raise ValueError("Input blueprint image is empty")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input blueprint image must have shape (H, W, 3)")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )

    kernel = np.ones((3, 3), np.uint8)
    denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)

    classical_wall_mask = _extract_wall_mask(gray)
    ml_wall_mask, ml_door_mask = _predict_ml_masks(image_bgr)
    ml_used = ml_wall_mask is not None and ml_wall_mask.shape == classical_wall_mask.shape

    if ml_used:
        # ML-first fusion:
        # - keep ML wall structure as primary signal
        # - only keep classical details close to ML walls
        # This suppresses dashed/annotation strokes that classical thresholding often over-detects.
        ml_dilated = cv2.dilate(
            ml_wall_mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=1,
        )
        classical_near_ml = cv2.bitwise_and(classical_wall_mask, ml_dilated)
        wall_mask = cv2.bitwise_or(ml_wall_mask, classical_near_ml)
        wall_mask = cv2.morphologyEx(
            wall_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
    else:
        wall_mask = classical_wall_mask

    # Final dominant-footprint cleanup post-fusion.
    wall_mask = _retain_primary_components(wall_mask, keep_ratio=0.35, max_components=1)
    edges = cv2.Canny(wall_mask, 40, 140)
    primary_bbox = _largest_component_bbox(wall_mask)
    ml_runtime = _get_ml_runtime_status()

    return {
        "gray": gray,
        "binary": binary,
        "denoised": wall_mask,
        "edges": edges,
        "primary_bbox": primary_bbox,
        "ml_used": ml_used,
        "ml_enabled": ml_runtime["ml_enabled"],
        "ml_engine_loaded": ml_runtime["ml_engine_loaded"],
        "ml_model_path": ml_runtime["ml_model_path"],
        "ml_model_exists": ml_runtime["ml_model_exists"],
        "ml_device": ml_runtime["ml_device"],
        "ml_reason": ml_runtime["ml_reason"],
        # Optional ML channel; downstream modules may ignore when not present.
        "door_mask": ml_door_mask if ml_door_mask is not None else np.zeros_like(gray, dtype=np.uint8),
    }
