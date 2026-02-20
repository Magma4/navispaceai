"""Geometry validation checks for BIM scene graph quality gates."""

from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Point


def _line_from_wall(wall: dict[str, Any]) -> LineString:
    line = wall.get("line_m", {})
    return LineString([(float(line["x1"]), float(line["z1"])), (float(line["x2"]), float(line["z2"]))])


def _endpoint_set(line: LineString, precision: int = 4) -> set[tuple[float, float]]:
    coords = list(line.coords)
    if len(coords) < 2:
        return set()
    a = (round(float(coords[0][0]), precision), round(float(coords[0][1]), precision))
    b = (round(float(coords[-1][0]), precision), round(float(coords[-1][1]), precision))
    return {a, b}


def validate_scene_graph(scene_graph: dict[str, Any], door_wall_max_gap_m: float = 0.45) -> dict[str, Any]:
    """Validate wall topology and opening placement consistency."""
    issues: list[dict[str, Any]] = []
    wall_checks = 0
    door_checks = 0

    levels = list(scene_graph.get("levels", []))

    for level in levels:
        floor = int(level.get("floor_number", -1))
        walls = list(level.get("walls", []))
        openings = list(level.get("openings", []))

        wall_lines: list[tuple[str, LineString]] = []
        for wall in walls:
            try:
                wall_lines.append((str(wall.get("id", "")), _line_from_wall(wall)))
            except Exception:
                issues.append(
                    {
                        "kind": "wall_invalid",
                        "severity": "error",
                        "floor": floor,
                        "wall_id": wall.get("id"),
                        "message": "Wall line has invalid coordinates",
                    }
                )

        # Wall intersection checks (excluding shared endpoints).
        for i in range(len(wall_lines)):
            id_a, line_a = wall_lines[i]
            for j in range(i + 1, len(wall_lines)):
                id_b, line_b = wall_lines[j]
                wall_checks += 1
                if not line_a.intersects(line_b):
                    continue

                inter = line_a.intersection(line_b)
                if inter.is_empty:
                    continue

                endpoints = _endpoint_set(line_a) | _endpoint_set(line_b)
                if inter.geom_type == "Point":
                    p = (round(float(inter.x), 4), round(float(inter.y), 4))
                    if p in endpoints:
                        continue

                issues.append(
                    {
                        "kind": "wall_intersection",
                        "severity": "warning",
                        "floor": floor,
                        "wall_a": id_a,
                        "wall_b": id_b,
                        "message": "Walls intersect away from shared endpoints",
                    }
                )

        # Door/opening clearance checks.
        wall_geoms = [geom for _, geom in wall_lines]
        for opening in openings:
            if str(opening.get("type", "")).lower() != "door":
                continue
            door_checks += 1
            pose = opening.get("pose_m", {})
            door_pt = Point(float(pose.get("x", 0.0)), float(pose.get("z", 0.0)))
            near_wall = False
            for wall in wall_geoms:
                if wall.distance(door_pt) <= float(door_wall_max_gap_m):
                    near_wall = True
                    break
            if not near_wall:
                issues.append(
                    {
                        "kind": "door_clearance",
                        "severity": "warning",
                        "floor": floor,
                        "opening_id": opening.get("id"),
                        "message": f"Door is not near any wall within {door_wall_max_gap_m:.2f}m",
                    }
                )

    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")

    return {
        "ok": error_count == 0,
        "summary": {
            "levels": len(levels),
            "wall_checks": wall_checks,
            "door_checks": door_checks,
            "errors": error_count,
            "warnings": warning_count,
        },
        "issues": issues,
    }
