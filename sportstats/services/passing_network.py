from __future__ import annotations

from collections import defaultdict


DEMO_PASSES = [
    {"passer": "GK", "receiver": "LCB", "start_x": 8, "start_y": 50, "end_x": 24, "end_y": 28},
    {"passer": "GK", "receiver": "RCB", "start_x": 8, "start_y": 50, "end_x": 24, "end_y": 72},
    {"passer": "LCB", "receiver": "DM", "start_x": 24, "start_y": 28, "end_x": 42, "end_y": 50},
    {"passer": "RCB", "receiver": "DM", "start_x": 24, "start_y": 72, "end_x": 42, "end_y": 50},
    {"passer": "DM", "receiver": "LW", "start_x": 42, "start_y": 50, "end_x": 67, "end_y": 20},
    {"passer": "DM", "receiver": "RW", "start_x": 42, "start_y": 50, "end_x": 67, "end_y": 80},
    {"passer": "LW", "receiver": "ST", "start_x": 67, "start_y": 20, "end_x": 84, "end_y": 50},
    {"passer": "RW", "receiver": "ST", "start_x": 67, "start_y": 80, "end_x": 84, "end_y": 50},
    {"passer": "DM", "receiver": "AM", "start_x": 42, "start_y": 50, "end_x": 64, "end_y": 50},
    {"passer": "AM", "receiver": "ST", "start_x": 64, "start_y": 50, "end_x": 84, "end_y": 50},
]


def build_demo_network() -> dict:
    return build_network(DEMO_PASSES)


def build_network(events: list[dict]) -> dict:
    if not isinstance(events, list):
        raise ValueError("events must be a list")

    player_positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    player_touches: dict[str, int] = defaultdict(int)
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    for event in events:
        passer = _required_text(event, "passer")
        receiver = _required_text(event, "receiver")
        start = (_required_number(event, "start_x"), _required_number(event, "start_y"))
        end = (_required_number(event, "end_x"), _required_number(event, "end_y"))

        player_positions[passer].append(start)
        player_positions[receiver].append(end)
        player_touches[passer] += 1
        player_touches[receiver] += 1
        edge_counts[(passer, receiver)] += 1

    nodes = []
    for player, positions in player_positions.items():
        avg_x = sum(position[0] for position in positions) / len(positions)
        avg_y = sum(position[1] for position in positions) / len(positions)
        nodes.append(
            {
                "player": player,
                "x": round(avg_x, 1),
                "y": round(avg_y, 1),
                "touches": player_touches[player],
            }
        )

    node_lookup = {node["player"]: node for node in nodes}
    edges = []
    for (source, target), count in edge_counts.items():
        edges.append(
            {
                "source": source,
                "target": target,
                "source_x": node_lookup[source]["x"],
                "source_y": node_lookup[source]["y"],
                "target_x": node_lookup[target]["x"],
                "target_y": node_lookup[target]["y"],
                "count": count,
            }
        )
    return {
        "nodes": sorted(nodes, key=lambda node: node["player"]),
        "edges": sorted(edges, key=lambda edge: edge["count"], reverse=True),
        "total_passes": sum(edge["count"] for edge in edges),
    }


def _required_text(event: dict, key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_number(event: dict, key: str) -> float:
    try:
        value = float(event[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if value < 0 or value > 100:
        raise ValueError(f"{key} must be between 0 and 100")
    return value
