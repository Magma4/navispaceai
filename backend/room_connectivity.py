"""Room-level connectivity graph for floor-local and cross-floor routing."""

from __future__ import annotations

from collections import deque
from typing import Any

from shapely.geometry import Point, Polygon


def _room_centroid_xy(room: dict[str, Any]) -> tuple[float, float]:
    c = room.get("centroid_m") or [0.0, 0.0, 0.0]
    return float(c[0]), float(c[2])


def _room_polygon(room: dict[str, Any]) -> Polygon:
    poly = room.get("polygon_m") or []
    if len(poly) < 3:
        x, z = _room_centroid_xy(room)
        return Point(x, z).buffer(0.2)
    return Polygon([(float(p[0]), float(p[1])) for p in poly])


def build_room_connectivity(
    rooms: list[dict[str, Any]],
    connectors: list[dict[str, Any]],
    adjacency_gap_m: float = 0.35,
) -> dict[str, Any]:
    """Build graph with room nodes + horizontal/vertical connectivity edges."""
    room_map: dict[str, dict[str, Any]] = {str(r["room_id"]): r for r in rooms}
    nodes = [
        {
            "room_id": str(r["room_id"]),
            "floor_number": int(r["floor_number"]),
            "name": str(r.get("name", r["room_id"])),
        }
        for r in rooms
    ]

    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()

    def add_edge(a: str, b: str, kind: str, weight: float, meta: dict[str, Any] | None = None) -> None:
        if a == b:
            return
        u, v = sorted([a, b])
        key = (u, v, kind)
        if key in edge_keys:
            return
        edge_keys.add(key)
        payload = {
            "from_room_id": a,
            "to_room_id": b,
            "kind": kind,
            "weight": float(weight),
        }
        if meta:
            payload.update(meta)
        edges.append(payload)

    rooms_by_floor: dict[int, list[dict[str, Any]]] = {}
    for room in rooms:
        rooms_by_floor.setdefault(int(room["floor_number"]), []).append(room)

    for floor, floor_rooms in rooms_by_floor.items():
        polys = [(str(r["room_id"]), _room_polygon(r)) for r in floor_rooms]
        for i in range(len(polys)):
            id_a, poly_a = polys[i]
            for j in range(i + 1, len(polys)):
                id_b, poly_b = polys[j]
                dist = float(poly_a.distance(poly_b))
                if poly_a.intersects(poly_b) or dist <= float(adjacency_gap_m):
                    add_edge(id_a, id_b, "horizontal", max(0.05, dist), {"floor_number": floor})

    # Vertical edges by linking nearest room to each connector per floor.
    for connector in connectors:
        floors = sorted({int(f) for f in connector.get("floors", [])})
        pos = connector.get("position_m") or [0.0, 0.0, 0.0]
        cx, cz = float(pos[0]), float(pos[2])

        nearest_by_floor: dict[int, tuple[str, float]] = {}
        for floor in floors:
            floor_rooms = rooms_by_floor.get(floor, [])
            if not floor_rooms:
                continue
            best_room = None
            best_dist = float("inf")
            for room in floor_rooms:
                rx, rz = _room_centroid_xy(room)
                d = ((rx - cx) ** 2 + (rz - cz) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_room = str(room["room_id"])
            if best_room is not None:
                nearest_by_floor[floor] = (best_room, best_dist)

        for i in range(len(floors) - 1):
            fa, fb = floors[i], floors[i + 1]
            if fa not in nearest_by_floor or fb not in nearest_by_floor:
                continue
            ra, da = nearest_by_floor[fa]
            rb, db = nearest_by_floor[fb]
            add_edge(
                ra,
                rb,
                "vertical",
                max(0.1, (da + db) * 0.5),
                {
                    "connector_id": connector.get("connector_id"),
                    "connector_type": connector.get("connector_type", "stairs"),
                    "from_floor": fa,
                    "to_floor": fb,
                },
            )

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def shortest_room_route(graph: dict[str, Any], start_room_id: str, goal_room_id: str) -> list[str]:
    """Compute shortest room-id route using unweighted BFS over connectivity graph."""
    start = str(start_room_id)
    goal = str(goal_room_id)
    if start == goal:
        return [start]

    neighbors: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        a = str(edge.get("from_room_id"))
        b = str(edge.get("to_room_id"))
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)

    q: deque[str] = deque([start])
    parent: dict[str, str | None] = {start: None}

    while q:
        cur = q.popleft()
        for nxt in neighbors.get(cur, set()):
            if nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == goal:
                route = [goal]
                while route[-1] != start:
                    p = parent[route[-1]]
                    if p is None:
                        break
                    route.append(p)
                route.reverse()
                return route
            q.append(nxt)

    return []
