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
StairCandidate = dict[str, int]


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


def extract_walls_from_mask(
    wall_mask: np.ndarray,
    *,
    min_contour_area: int = 120,
    approx_eps_ratio: float = 0.01,
) -> list[WallSegment]:
    """Vectorize wall-like contours into line segments.

    This complements Hough output for blueprint styles where long straight segments
    are broken by symbols/text and Hough under-detects structure.
    """
    _validate_binary_image(wall_mask, "wall_mask")

    binary = (wall_mask > 0).astype(np.uint8) * 255
    # Use RETR_LIST to include interior contours as well; external-only misses
    # many interior room walls in dense plans.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    segments: list[WallSegment] = []
    seen: set[tuple[int, int, int, int]] = set()

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_contour_area):
            continue

        peri = cv2.arcLength(contour, True)
        if peri <= 1e-6:
            continue

        approx = cv2.approxPolyDP(contour, max(1.0, float(approx_eps_ratio) * peri), True)
        pts = approx.reshape(-1, 2) if approx is not None else None
        if pts is None or len(pts) < 2:
            continue

        for i in range(len(pts)):
            x1, y1 = int(pts[i][0]), int(pts[i][1])
            x2, y2 = int(pts[(i + 1) % len(pts)][0]), int(pts[(i + 1) % len(pts)][1])

            if math.hypot(x2 - x1, y2 - y1) < 10:
                continue

            q = 3
            ax, ay, bx, by = x1 // q, y1 // q, x2 // q, y2 // q
            if (ax, ay, bx, by) > (bx, by, ax, ay):
                ax, ay, bx, by = bx, by, ax, ay
            key = (ax, ay, bx, by)
            if key in seen:
                continue
            seen.add(key)
            segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return segments


def merge_collinear_wall_segments(
    walls: list[WallSegment],
    *,
    angle_tolerance_deg: float = 8.0,
    endpoint_gap_px: float = 18.0,
    parallel_offset_px: float = 12.0,
    interval_gap_px: float = 25.0,
    min_wall_length_px: float = 35.0,
) -> list[WallSegment]:
    """Merge near-collinear wall segments into longer linked wall runs.

    Heuristics:
    - Orientation compatibility (angle tolerance)
    - Near-coplanar offset (distance from candidate endpoints to base infinite line)
    - Endpoint proximity or projected interval continuity (small gaps)
    - Final pruning of tiny residual segments
    """
    if not walls:
        return []

    def _angle(seg: WallSegment) -> float:
        return math.degrees(math.atan2(float(seg["y2"] - seg["y1"]), float(seg["x2"] - seg["x1"])))

    def _norm(a: float) -> float:
        while a < -90:
            a += 180
        while a > 90:
            a -= 180
        return a

    def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def _segment_len(seg: WallSegment) -> float:
        return math.hypot(float(seg["x2"] - seg["x1"]), float(seg["y2"] - seg["y1"]))

    def _line_offset(px: float, py: float, x1: float, y1: float, ux: float, uy: float) -> float:
        # Distance from point P to line passing through A in direction U.
        vx = px - x1
        vy = py - y1
        return abs(vx * uy - vy * ux)

    def _project_t(px: float, py: float, x1: float, y1: float, ux: float, uy: float) -> float:
        return (px - x1) * ux + (py - y1) * uy

    remaining = [dict(seg) for seg in walls if _segment_len(seg) >= max(2.0, float(min_wall_length_px) * 0.35)]
    merged: list[WallSegment] = []
    gap2 = float(endpoint_gap_px) * float(endpoint_gap_px)

    while remaining:
        base = remaining.pop()
        changed = True

        while changed:
            changed = False
            p1 = (float(base["x1"]), float(base["y1"]))
            p2 = (float(base["x2"]), float(base["y2"]))
            a0 = _norm(_angle(base))

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            ln = math.hypot(dx, dy)
            if ln <= 1e-6:
                break
            ux, uy = dx / ln, dy / ln

            base_t0 = 0.0
            base_t1 = ln

            line_pts = [p1, p2]
            keep: list[WallSegment] = []

            for seg in remaining:
                a1 = _norm(_angle(seg))
                if abs(a0 - a1) > float(angle_tolerance_deg):
                    keep.append(seg)
                    continue

                q1 = (float(seg["x1"]), float(seg["y1"]))
                q2 = (float(seg["x2"]), float(seg["y2"]))

                # Condition A: direct endpoint proximity.
                close_endpoints = (
                    min(_dist2(p1, q1), _dist2(p1, q2), _dist2(p2, q1), _dist2(p2, q2)) <= gap2
                )

                # Condition B: near-coplanar + short projected gap.
                off1 = _line_offset(q1[0], q1[1], p1[0], p1[1], ux, uy)
                off2 = _line_offset(q2[0], q2[1], p1[0], p1[1], ux, uy)
                near_parallel_line = max(off1, off2) <= float(parallel_offset_px)

                tq1 = _project_t(q1[0], q1[1], p1[0], p1[1], ux, uy)
                tq2 = _project_t(q2[0], q2[1], p1[0], p1[1], ux, uy)
                seg_t0 = min(tq1, tq2)
                seg_t1 = max(tq1, tq2)

                if seg_t1 < base_t0:
                    interval_gap = base_t0 - seg_t1
                elif seg_t0 > base_t1:
                    interval_gap = seg_t0 - base_t1
                else:
                    interval_gap = 0.0

                if not close_endpoints and not (near_parallel_line and interval_gap <= float(interval_gap_px)):
                    keep.append(seg)
                    continue

                line_pts.extend([q1, q2])
                changed = True

            if changed:
                # Refit merged segment by farthest pair of collected endpoints.
                best = (p1, p2)
                best_d2 = _dist2(p1, p2)
                for i in range(len(line_pts)):
                    for j in range(i + 1, len(line_pts)):
                        d2 = _dist2(line_pts[i], line_pts[j])
                        if d2 > best_d2:
                            best = (line_pts[i], line_pts[j])
                            best_d2 = d2
                base = {
                    "x1": int(round(best[0][0])),
                    "y1": int(round(best[0][1])),
                    "x2": int(round(best[1][0])),
                    "y2": int(round(best[1][1])),
                }

            remaining = keep

        if _segment_len(base) >= float(min_wall_length_px):
            merged.append(base)

    return merged


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
    connection_tolerance_px: float = 14.0,
    keep_component_ratio: float = 0.08,
    max_components: int = 6,
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
    walls: list[WallSegment] | None = None,
    max_wall_gap_px: float = 14.0,
) -> list[DoorCandidate]:
    """Filter door candidates to footprint bounds and nearby wall context."""
    if not doors:
        return []

    height, width = int(image_shape[0]), int(image_shape[1])
    if primary_bbox:
        x0 = max(0, int(primary_bbox.get("x", 0)) - bbox_margin_px)
        y0 = max(0, int(primary_bbox.get("y", 0)) - bbox_margin_px)
        x1 = min(width, int(primary_bbox.get("x", 0) + primary_bbox.get("w", width)) + bbox_margin_px)
        y1 = min(height, int(primary_bbox.get("y", 0) + primary_bbox.get("h", height)) + bbox_margin_px)
    else:
        x0, y0, x1, y1 = 0, 0, width, height

    def _point_to_seg_dist(px: float, py: float, seg: WallSegment) -> float:
        x1s = float(seg["x1"])
        y1s = float(seg["y1"])
        x2s = float(seg["x2"])
        y2s = float(seg["y2"])
        vx = x2s - x1s
        vy = y2s - y1s
        wx = px - x1s
        wy = py - y1s
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - x1s, py - y1s)
        c2 = vx * vx + vy * vy
        if c2 <= 1e-8:
            return math.hypot(px - x1s, py - y1s)
        if c1 >= c2:
            return math.hypot(px - x2s, py - y2s)
        t = c1 / c2
        proj_x = x1s + t * vx
        proj_y = y1s + t * vy
        return math.hypot(px - proj_x, py - proj_y)

    out: list[DoorCandidate] = []
    for door in doors:
        if not {"x", "y", "w", "h"}.issubset(door.keys()):
            continue
        cx = int(door["x"] + max(1, door["w"]) / 2)
        cy = int(door["y"] + max(1, door["h"]) / 2)
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue

        if walls:
            nearest = min(_point_to_seg_dist(float(cx), float(cy), seg) for seg in walls)
            if nearest > float(max_wall_gap_px):
                continue

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


def detect_staircases(
    binary: np.ndarray,
    *,
    min_area: int = 180,
    max_area: int = 20000,
    min_steps: int = 3,
) -> list[StairCandidate]:
    """Detect staircase-like regions from repeated parallel line patterns.

    The detector is intentionally conservative and returns coarse stair bboxes.
    """
    _validate_binary_image(binary, "binary")

    # Ensure foreground strokes are white for contour/line extraction.
    fg = (binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[StairCandidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = int(w * h)
        if area < int(min_area) or area > int(max_area):
            continue

        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect < 1.08:
            continue

        roi = fg[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        roi_edges = cv2.Canny(roi, 40, 120)
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(10, int(min(w, h) * 0.12)),
            minLineLength=max(6, int(min(w, h) * 0.2)),
            maxLineGap=3,
        )
        if lines is None or len(lines) < int(min_steps):
            continue

        horizontal = 0
        vertical = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ang = abs(math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))))
            if ang <= 15 or ang >= 165:
                horizontal += 1
            if 75 <= ang <= 105:
                vertical += 1

        # Stair signatures usually have repeated near-parallel treads and some riser edges.
        if max(horizontal, vertical) < int(min_steps):
            continue

        candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return candidates
