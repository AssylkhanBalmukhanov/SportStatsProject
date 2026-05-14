from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sportstats.vision.geometry import bbox_to_int
from sportstats.vision.tracking import Tracks


@dataclass
class TeamAssigner:
    player_team_cache: dict[int, int] = field(default_factory=dict)
    team_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    _team_centroids: np.ndarray | None = None

    def assign_team_colors(self, frame, player_tracks: dict[int, dict]) -> None:
        colors = []
        for player in player_tracks.values():
            bbox = player.get("bbox")
            if bbox:
                colors.append(self.get_player_color(frame, bbox))

        if len(colors) < 2:
            self._team_centroids = np.asarray([[255, 255, 255], [60, 160, 60]], dtype=float)
        else:
            self._team_centroids, _labels = kmeans(np.asarray(colors, dtype=float), k=2, iterations=30)

        self.team_colors = {
            1: _color_tuple(self._team_centroids[0]),
            2: _color_tuple(self._team_centroids[1]),
        }

    def assign_players_to_teams(self, frames: list, tracks: Tracks) -> None:
        if not frames or not tracks.get("players"):
            return

        first_frame_index = next((index for index, frame_tracks in enumerate(tracks["players"]) if frame_tracks), 0)
        self.assign_team_colors(frames[first_frame_index], tracks["players"][first_frame_index])

        for frame_index, player_tracks in enumerate(tracks["players"]):
            for player_id, player in player_tracks.items():
                team_id = self.get_player_team(frames[frame_index], player["bbox"], player_id)
                player["team"] = team_id
                player["team_color"] = self.team_colors.get(team_id, (0, 0, 255))

    def get_player_team(self, frame, player_bbox, player_id: int) -> int:
        if player_id in self.player_team_cache:
            return self.player_team_cache[player_id]

        if self._team_centroids is None:
            return 1

        player_color = np.asarray(self.get_player_color(frame, player_bbox), dtype=float)
        distances = np.linalg.norm(self._team_centroids - player_color, axis=1)
        team_id = int(np.argmin(distances)) + 1
        self.player_team_cache[player_id] = team_id
        return team_id

    def get_player_color(self, frame, player_bbox) -> np.ndarray:
        x1, y1, x2, y2 = bbox_to_int(player_bbox)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.asarray([255, 255, 255], dtype=float)

        top_half = crop[: max(crop.shape[0] // 2, 1), :]
        pixels = top_half.reshape(-1, 3).astype(float)
        if len(pixels) < 2:
            return pixels.mean(axis=0) if len(pixels) else np.asarray([255, 255, 255], dtype=float)

        sample = pixels[:: max(len(pixels) // 1500, 1)]
        centroids, labels = kmeans(sample, k=2, iterations=12)
        label_image = labels.reshape(top_half.shape[:2]) if len(sample) == len(pixels) else None

        if label_image is not None:
            corners = [
                int(label_image[0, 0]),
                int(label_image[0, -1]),
                int(label_image[-1, 0]),
                int(label_image[-1, -1]),
            ]
            background_cluster = max(set(corners), key=corners.count)
            player_cluster = 1 - background_cluster
            return centroids[player_cluster]

        distances_to_centroids = np.linalg.norm(sample[:, None, :] - centroids[None, :, :], axis=2)
        full_labels = np.argmin(distances_to_centroids, axis=1)
        player_cluster = int(np.argmin(np.bincount(full_labels, minlength=2)))
        return centroids[player_cluster]


def kmeans(points: np.ndarray, *, k: int, iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.zeros((k, 3), dtype=float), np.zeros(0, dtype=int)
    if len(points) < k:
        padded = np.vstack([points, np.repeat(points[-1][None, :], k - len(points), axis=0)])
        return padded.astype(float), np.zeros(len(points), dtype=int)

    centroids = _deterministic_initial_centroids(points, k)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return centroids, labels


def _deterministic_initial_centroids(points: np.ndarray, k: int) -> np.ndarray:
    brightness = points.mean(axis=1)
    order = np.argsort(brightness)
    indices = np.linspace(0, len(order) - 1, k).astype(int)
    return points[order[indices]].astype(float)


def _color_tuple(color: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(value)))) for value in color)  # type: ignore[return-value]
