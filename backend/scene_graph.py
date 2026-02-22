"""Canonical BIM-style scene graph builder for multi-floor blueprint processing."""

from __future__ import annotations

from typing import Any


WallSegment = dict[str, int]
DoorCandidate = dict[str, int]
StairCandidate = dict[str, int]


def _segment_to_world(seg: WallSegment, scale_m_per_px: float) -> dict[str, float]:
    return {
        "x1": float(seg["x1"]) * scale_m_per_px,
        "z1": float(seg["y1"]) * scale_m_per_px,
        "x2": float(seg["x2"]) * scale_m_per_px,
        "z2": float(seg["y2"]) * scale_m_per_px,
    }


def _rect_to_world(rect: dict[str, int], scale_m_per_px: float) -> dict[str, float]:
    x = float(rect.get("x", 0))
    y = float(rect.get("y", 0))
    w = max(0.0, float(rect.get("w", 0)))
    h = max(0.0, float(rect.get("h", 0)))
    return {
        "x": x * scale_m_per_px,
        "z": y * scale_m_per_px,
        "width_m": w * scale_m_per_px,
        "height_m": h * scale_m_per_px,
    }


def build_scene_graph(
    *,
    building_id: str,
    floor_height_m: float,
    model_scale_m_per_px: float,
    floors_payload: list[dict[str, Any]],
    rooms_payload: list[dict[str, Any]],
    connectors_payload: list[dict[str, Any]],
    floor_artifacts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Build canonical scene graph used by rendering + navigation subsystems."""
    rooms_by_floor: dict[int, list[dict[str, Any]]] = {}
    for room in rooms_payload:
        floor_num = int(room["floor_number"])
        rooms_by_floor.setdefault(floor_num, []).append(
            {
                "id": room["room_id"],
                "name": room["name"],
                "floor": floor_num,
                "centroid_m": room["centroid_m"],
                "polygon_m": room["polygon_m"],
                "area_m2": float(room["area_m2"]),
            }
        )

    levels: list[dict[str, Any]] = []
    for floor in floors_payload:
        floor_num = int(floor["floor_number"])
        artifacts = floor_artifacts.get(floor_num, {})
        walls_px: list[WallSegment] = list(artifacts.get("walls", []))
        doors_px: list[DoorCandidate] = list(artifacts.get("doors", []))
        stairs_px: list[StairCandidate] = list(artifacts.get("stairs", []))

        walls = [
            {
                "id": f"F{floor_num}-W{idx:04d}",
                "floor": floor_num,
                "line_m": _segment_to_world(seg, model_scale_m_per_px),
            }
            for idx, seg in enumerate(walls_px, start=1)
        ]

        openings = [
            {
                "id": f"F{floor_num}-O{idx:04d}",
                "type": "door",
                "floor": floor_num,
                "pose_m": _rect_to_world(door, model_scale_m_per_px),
            }
            for idx, door in enumerate(doors_px, start=1)
        ]

        openings.extend(
            [
                {
                    "id": f"F{floor_num}-S{idx:04d}",
                    "type": "staircase",
                    "floor": floor_num,
                    "pose_m": _rect_to_world(stair, model_scale_m_per_px),
                }
                for idx, stair in enumerate(stairs_px, start=1)
            ]
        )

        levels.append(
            {
                "id": f"L{floor_num}",
                "floor_number": floor_num,
                "elevation_m": float(floor.get("elevation_m", floor_num * floor_height_m)),
                "cell_size_m": float(floor.get("cell_size_m", 0.2)),
                "rooms": rooms_by_floor.get(floor_num, []),
                "walls": walls,
                "openings": openings,
            }
        )

    connectivity: list[dict[str, Any]] = []
    for connector in connectors_payload:
        floors = sorted({int(v) for v in connector.get("floors", [])})
        for i in range(len(floors) - 1):
            connectivity.append(
                {
                    "type": "vertical",
                    "connector_id": connector["connector_id"],
                    "connector_type": connector.get("connector_type", "stairs"),
                    "from_floor": floors[i],
                    "to_floor": floors[i + 1],
                    "position_m": connector.get("position_m"),
                    "radius_m": float(connector.get("radius_m", 0.5)),
                }
            )

    return {
        "schema_version": "1.0",
        "building_id": building_id,
        "scale_m_per_px": float(model_scale_m_per_px),
        "floor_height_m": float(floor_height_m),
        "levels": sorted(levels, key=lambda lv: int(lv["floor_number"])),
        "connectivity": connectivity,
    }
