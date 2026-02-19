"""Wall and door detection utilities.

Purpose:
- Detect wall line segments from preprocessed edge images.
- Identify door-like openings via contour heuristics.

Usage example:
    >>> from backend.detection import detect_walls
    >>> walls = detect_walls(edges)
"""

from __future__ import annotations

import math
import cv2
import numpy as np

WallSegment = dict[str, int]
DoorCandidate = dict[str, int]


def _validate_binary_image(image: np.ndarray, name: str) -> None:
    """Validate a 2D grayscale/binary image array."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if image.size == 0:
        raise ValueError(f"{name} is empty")
    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")


def detect_walls(
    edges: np.ndarray,
    threshold: int = 60,
    min_line_length: int = 30,
    max_line_gap: int = 8,
) -> list[WallSegment]:
    """Detect wall line segments using probabilistic Hough transform.

    Args:
        edges: 2D edge map, typically from Canny.
        threshold: Hough accumulator threshold.
        min_line_length: Minimum segment length in pixels.
        max_line_gap: Max allowed gap to merge colinear segments.

    Returns:
        List of wall segments in dict form: `{x1, y1, x2, y2}`.

    Raises:
        ValueError: If input image or parameters are invalid.
    """
    _validate_binary_image(edges, "edges")
    if threshold <= 0 or min_line_length <= 0 or max_line_gap < 0:
        raise ValueError("Hough parameters must be positive (max_line_gap can be zero)")

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    segments: list[WallSegment] = []
    if lines is None:
        return segments

    for line in lines:
        x1, y1, x2, y2 = line[0]
        segments.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})

    return segments


def adaptive_hough_params(image_shape: tuple[int, int]) -> tuple[int, int, int]:
    """Compute scale-aware Hough parameters for small and large blueprints.

    Returns:
        Tuple (threshold, min_line_length, max_line_gap).
    """
    if len(image_shape) != 2:
        return 80, 40, 6

    h, w = int(image_shape[0]), int(image_shape[1])
    min_dim = max(1, min(h, w))
    threshold = max(90, int(min_dim * 0.09))
    min_line_length = max(55, int(min_dim * 0.08))
    # Keep gap small to avoid linking dashed references into continuous walls.
    max_line_gap = max(2, int(min_dim * 0.0025))
    return threshold, min_line_length, max_line_gap


def _segment_support_ratio(
    seg: WallSegment,
    support_mask: np.ndarray,
    samples: int = 24,
) -> float:
    """Estimate how much of a segment lies on non-zero support pixels."""
    x1 = float(seg["x1"])
    y1 = float(seg["y1"])
    x2 = float(seg["x2"])
    y2 = float(seg["y2"])
    h, w = support_mask.shape[:2]

    total = 0
    hits = 0
    for i in range(max(2, samples)):
        t = i / max(1, samples - 1)
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        total += 1
        if support_mask[y, x] > 0:
            hits += 1
    if total == 0:
        return 0.0
    return hits / total


def _segment_median_thickness(
    seg: WallSegment,
    distance_map: np.ndarray,
    samples: int = 18,
) -> float:
    """Estimate segment stroke half-thickness from distance transform map."""
    x1 = float(seg["x1"])
    y1 = float(seg["y1"])
    x2 = float(seg["x2"])
    y2 = float(seg["y2"])
    h, w = distance_map.shape[:2]
    vals: list[float] = []
    for i in range(max(2, samples)):
        t = i / max(1, samples - 1)
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        vals.append(float(distance_map[y, x]))
    if not vals:
        return 0.0
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def filter_wall_segments(
    walls: list[WallSegment],
    image_shape: tuple[int, int],
    *,
    primary_bbox: dict[str, int] | None = None,
    bbox_margin_px: int = 20,
    min_length_px: int = 30,
    support_mask: np.ndarray | None = None,
    min_support_ratio: float = 0.1,
    min_median_half_thickness_px: float = 0.85,
    connection_tolerance_px: float = 10.0,
    keep_component_ratio: float = 0.18,
    max_components: int = 2,
) -> list[WallSegment]:
    """Filter noisy wall segments by length + dominant footprint bounds + dedup.

    Args:
        walls: Raw segments from Hough transform.
        image_shape: Source (height, width).
        primary_bbox: Optional dominant footprint bbox (`x,y,w,h`).
        bbox_margin_px: Expansion margin for bbox filtering.
        min_length_px: Reject segments shorter than this.
    """
    if not walls:
        return []

    height, width = int(image_shape[0]), int(image_shape[1])
    min_dim = max(1, min(height, width))
    min_len = max(int(min_length_px), int(min_dim * 0.03))

    if primary_bbox:
        bx0 = int(primary_bbox.get("x", 0)) - bbox_margin_px
        by0 = int(primary_bbox.get("y", 0)) - bbox_margin_px
        bx1 = int(primary_bbox.get("x", 0) + primary_bbox.get("w", width)) + bbox_margin_px
        by1 = int(primary_bbox.get("y", 0) + primary_bbox.get("h", height)) + bbox_margin_px
    else:
        bx0, by0, bx1, by1 = 0, 0, width, height

    bx0 = max(0, bx0)
    by0 = max(0, by0)
    bx1 = min(width, bx1)
    by1 = min(height, by1)

    kept: list[WallSegment] = []
    seen: set[tuple[int, int, int, int]] = set()
    dist_map: np.ndarray | None = None
    support_eval: np.ndarray | None = None
    if support_mask is not None and isinstance(support_mask, np.ndarray) and support_mask.ndim == 2:
        support_bin = (support_mask > 0).astype(np.uint8)
        support_bin = cv2.dilate(
            support_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        support_eval = support_bin
        dist_map = cv2.distanceTransform(support_bin, cv2.DIST_L2, 3)

    for seg in walls:
        x1 = int(seg["x1"])
        y1 = int(seg["y1"])
        x2 = int(seg["x2"])
        y2 = int(seg["y2"])

        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue

        # Keep if any endpoint or midpoint is inside footprint bbox.
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        inside = (
            (bx0 <= x1 <= bx1 and by0 <= y1 <= by1)
            or (bx0 <= x2 <= bx1 and by0 <= y2 <= by1)
            or (bx0 <= mx <= bx1 and by0 <= my <= by1)
        )
        if not inside:
            continue

        if support_eval is not None:
            ratio = _segment_support_ratio(seg, support_eval, samples=28)
            if ratio < float(min_support_ratio):
                continue
            if dist_map is not None:
                thickness = _segment_median_thickness(seg, dist_map, samples=18)
                if thickness < float(min_median_half_thickness_px):
                    continue

        # Deduplicate near-identical segments.
        q = 3
        ax, ay, bx, by = x1 // q, y1 // q, x2 // q, y2 // q
        if (ax, ay, bx, by) > (bx, by, ax, ay):
            ax, ay, bx, by = bx, by, ax, ay
        key = (ax, ay, bx, by)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    if not kept:
        return kept

    # Keep dominant connected wall network(s), drop isolated drafting segments.
    n = len(kept)
    endpoints: list[tuple[tuple[float, float], tuple[float, float]]] = [
        ((float(s["x1"]), float(s["y1"])), (float(s["x2"]), float(s["y2"]))) for s in kept
    ]
    lengths: list[float] = [
        math.hypot(float(s["x2"]) - float(s["x1"]), float(s["y2"]) - float(s["y1"])) for s in kept
    ]

    tol2 = float(connection_tolerance_px) * float(connection_tolerance_px)

    def _pt_close(a: tuple[float, float], b: tuple[float, float]) -> bool:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) <= tol2

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        a0, a1 = endpoints[i]
        for j in range(i + 1, n):
            b0, b1 = endpoints[j]
            if _pt_close(a0, b0) or _pt_close(a0, b1) or _pt_close(a1, b0) or _pt_close(a1, b1):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    components: list[tuple[list[int], float]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        total_len = 0.0
        while stack:
            cur = stack.pop()
            comp.append(cur)
            total_len += lengths[cur]
            for nxt in adj[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append((comp, total_len))

    if not components:
        return kept

    components.sort(key=lambda item: item[1], reverse=True)
    largest_len = max(1e-6, components[0][1])

    keep_idxs: set[int] = set()
    for idx, (comp, total_len) in enumerate(components):
        if idx >= max_components:
            break
        if total_len / largest_len < float(keep_component_ratio):
            continue
        keep_idxs.update(comp)

    if not keep_idxs:
        keep_idxs.update(components[0][0])

    return [kept[i] for i in sorted(keep_idxs)]


def filter_door_candidates(
    doors: list[DoorCandidate],
    image_shape: tuple[int, int],
    *,
    primary_bbox: dict[str, int] | None = None,
    bbox_margin_px: int = 20,
) -> list[DoorCandidate]:
    """Filter door candidates to dominant footprint bounds."""
    if not doors:
        return []
    if not primary_bbox:
        return doors

    height, width = int(image_shape[0]), int(image_shape[1])
    x0 = max(0, int(primary_bbox.get("x", 0)) - bbox_margin_px)
    y0 = max(0, int(primary_bbox.get("y", 0)) - bbox_margin_px)
    x1 = min(width, int(primary_bbox.get("x", 0) + primary_bbox.get("w", width)) + bbox_margin_px)
    y1 = min(height, int(primary_bbox.get("y", 0) + primary_bbox.get("h", height)) + bbox_margin_px)

    out: list[DoorCandidate] = []
    for door in doors:
        if not {"x", "y", "w", "h"}.issubset(door.keys()):
            continue
        cx = int(door["x"] + max(1, door["w"]) / 2)
        cy = int(door["y"] + max(1, door["h"]) / 2)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append({"x": int(door["x"]), "y": int(door["y"]), "w": int(door["w"]), "h": int(door["h"])})
    return out


def detect_doors(
    binary: np.ndarray,
    walls: list[WallSegment] | None = None,
    door_mask: np.ndarray | None = None,
    prefer_ml_only: bool = False,
) -> list[DoorCandidate]:
    """Detect door-like openings using contour geometry heuristics.

    Args:
        binary: 2D binary image where walls/structure are highlighted.
        walls: Optional wall segments, retained for future contextual heuristics.
        door_mask: Optional ML-predicted door mask (2D, 0/255).
        prefer_ml_only: If True and door_mask is provided, skip heuristic fallback when ML finds none.

    Returns:
        List of candidate door bounding boxes: `{x, y, w, h}`.

    Raises:
        ValueError: If image input is invalid.
    """
    _validate_binary_image(binary, "binary")

    # ML path: if a segmentation mask is provided, use it as primary door proposal source.
    if door_mask is not None and isinstance(door_mask, np.ndarray) and door_mask.ndim == 2 and door_mask.size > 0:
        door_binary = (door_mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(door_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ml_candidates: list[DoorCandidate] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Keep compact elongated regions typically corresponding to swing/openings.
            if 40 <= area <= 4000 and 4 <= min(w, h) <= 40 and max(w, h) <= 120:
                ml_candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

        if ml_candidates:
            return ml_candidates
        if prefer_ml_only:
            return []

    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[DoorCandidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Door-sized rectangular gaps in blueprint pixel space (heuristic window).
        if 80 <= area <= 2000 and 5 <= min(w, h) <= 25 and max(w, h) <= 80:
            candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return candidates
