from __future__ import annotations

from dataclasses import dataclass

from sportstats.vision.geometry import get_center_of_bbox, measure_distance
from sportstats.vision.tracking import Tracks


@dataclass
class PlayerBallAssigner:
    max_player_ball_distance: float = 70.0

    def assign_ball_to_player(self, players: dict[int, dict], ball_bbox: list[float]) -> int:
        ball_position = get_center_of_bbox(ball_bbox)
        assigned_player = -1
        minimum_distance = self.max_player_ball_distance

        for player_id, player in players.items():
            bbox = player.get("bbox")
            if not bbox:
                continue
            left_foot = (float(bbox[0]), float(bbox[3]))
            right_foot = (float(bbox[2]), float(bbox[3]))
            distance = min(measure_distance(left_foot, ball_position), measure_distance(right_foot, ball_position))
            if distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player

    def assign_ball_control(self, tracks: Tracks) -> list[int]:
        team_ball_control: list[int] = []
        last_team = 0

        for frame_index, players in enumerate(tracks["players"]):
            ball = tracks["ball"][frame_index].get(1)
            assigned_player = -1
            if ball and ball.get("bbox"):
                assigned_player = self.assign_ball_to_player(players, ball["bbox"])

            if assigned_player != -1 and assigned_player in players:
                players[assigned_player]["has_ball"] = True
                last_team = int(players[assigned_player].get("team", last_team or 1))
                team_ball_control.append(last_team)
            else:
                team_ball_control.append(last_team)

        return team_ball_control
