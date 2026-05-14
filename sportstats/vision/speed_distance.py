from __future__ import annotations

from dataclasses import dataclass

from sportstats.vision.geometry import measure_distance
from sportstats.vision.tracking import Tracks


@dataclass
class SpeedAndDistanceEstimator:
    frame_rate: float = 24.0
    frame_window: int = 5

    def add_speed_and_distance_to_tracks(self, tracks: Tracks) -> None:
        total_distance: dict[int, float] = {}
        player_tracks = tracks.get("players", [])
        number_of_frames = len(player_tracks)
        if number_of_frames < 2:
            return

        for frame_index in range(0, number_of_frames - 1, self.frame_window):
            last_frame = min(frame_index + self.frame_window, number_of_frames - 1)
            elapsed_seconds = max((last_frame - frame_index) / self.frame_rate, 1e-6)

            for track_id, track_info in player_tracks[frame_index].items():
                if track_id not in player_tracks[last_frame]:
                    continue

                start_position = track_info.get("position_transformed")
                end_position = player_tracks[last_frame][track_id].get("position_transformed")
                if start_position is None or end_position is None:
                    continue

                distance_m = measure_distance(start_position, end_position)
                speed_kmh = (distance_m / elapsed_seconds) * 3.6
                total_distance[track_id] = total_distance.get(track_id, 0.0) + distance_m

                for batch_frame in range(frame_index, last_frame + 1):
                    if track_id not in player_tracks[batch_frame]:
                        continue
                    player_tracks[batch_frame][track_id]["speed_kmh"] = speed_kmh
                    player_tracks[batch_frame][track_id]["distance_m"] = total_distance[track_id]
