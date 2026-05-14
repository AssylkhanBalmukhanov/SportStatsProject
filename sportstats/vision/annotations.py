from __future__ import annotations

import numpy as np

from sportstats.vision.geometry import get_bbox_width, get_center_of_bbox, get_foot_position
from sportstats.vision.tracking import Tracks


class FootballAnnotator:
    def __init__(self):
        import cv2

        self.cv2 = cv2

    def draw_annotations(self, frames: list, tracks: Tracks, team_ball_control: list[int] | None = None) -> list:
        output_frames = []
        for frame_index, frame in enumerate(frames):
            frame = frame.copy()
            for player_id, player in tracks["players"][frame_index].items():
                color = tuple(player.get("team_color", (45, 70, 220)))
                frame = self.draw_ellipse(frame, player["bbox"], color, player_id)
                if player.get("has_ball"):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for referee in tracks["referees"][frame_index].values():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 220, 255), None)

            for ball in tracks["ball"][frame_index].values():
                frame = self.draw_triangle(frame, ball["bbox"], (40, 220, 80))

            if team_ball_control:
                frame = self.draw_team_ball_control(frame, frame_index, team_ball_control)

            output_frames.append(frame)
        return output_frames

    def draw_camera_movement(self, frames: list, camera_movement_per_frame: list[tuple[float, float]]) -> list:
        output_frames = []
        for frame_index, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            self.cv2.rectangle(overlay, (0, 0), (500, 96), (255, 255, 255), -1)
            self.cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            movement_x, movement_y = camera_movement_per_frame[frame_index]
            self.cv2.putText(
                frame,
                f"Camera X: {movement_x:.2f}",
                (12, 34),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
            )
            self.cv2.putText(
                frame,
                f"Camera Y: {movement_y:.2f}",
                (12, 72),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
            )
            output_frames.append(frame)
        return output_frames

    def draw_speed_and_distance(self, frames: list, tracks: Tracks) -> list:
        output_frames = []
        for frame_index, frame in enumerate(frames):
            frame = frame.copy()
            for player in tracks["players"][frame_index].values():
                speed = player.get("speed_kmh")
                distance = player.get("distance_m")
                if speed is None or distance is None:
                    continue
                position = get_foot_position(player["bbox"])
                x = int(position[0] - 48)
                y = int(position[1] + 58)
                self.cv2.putText(frame, f"{speed:.1f} km/h", (x, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                self.cv2.putText(frame, f"{distance:.1f} m", (x, y + 22), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            output_frames.append(frame)
        return output_frames

    def draw_ellipse(self, frame, bbox, color: tuple[int, int, int], track_id: int | None):
        x_center, _y_center = get_center_of_bbox(bbox)
        y2 = int(bbox[3])
        width = max(int(get_bbox_width(bbox)), 18)
        self.cv2.ellipse(
            frame,
            (int(x_center), y2),
            (int(width), int(0.35 * width)),
            0.0,
            -45,
            235,
            color,
            2,
            lineType=self.cv2.LINE_4,
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1 = int(x_center - rectangle_width / 2)
            x2 = int(x_center + rectangle_width / 2)
            y1 = int(y2 - rectangle_height / 2 + 15)
            y2_rect = int(y2 + rectangle_height / 2 + 15)
            self.cv2.rectangle(frame, (x1, y1), (x2, y2_rect), color, self.cv2.FILLED)
            text_x = x1 + 12
            if track_id > 99:
                text_x -= 9
            self.cv2.putText(
                frame,
                str(track_id),
                (text_x, y1 + 15),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return frame

    def draw_triangle(self, frame, bbox, color: tuple[int, int, int]):
        x_center, _y_center = get_center_of_bbox(bbox)
        y = int(bbox[1])
        x = int(x_center)
        triangle_points = np.asarray([[x, y], [x - 10, y - 20], [x + 10, y - 20]], dtype=np.int32)
        self.cv2.drawContours(frame, [triangle_points], 0, color, self.cv2.FILLED)
        self.cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_index: int, team_ball_control: list[int]):
        height, width = frame.shape[:2]
        x1 = max(width - 560, 0)
        y1 = max(height - 150, 0)
        x2 = width - 24
        y2 = height - 24
        overlay = frame.copy()
        self.cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        self.cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

        control_until_now = np.asarray(team_ball_control[: frame_index + 1], dtype=int)
        team_1_frames = int((control_until_now == 1).sum())
        team_2_frames = int((control_until_now == 2).sum())
        total = max(team_1_frames + team_2_frames, 1)
        team_1 = team_1_frames / total * 100
        team_2 = team_2_frames / total * 100
        self.cv2.putText(frame, f"Team 1 ball control: {team_1:.1f}%", (x1 + 18, y1 + 48), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        self.cv2.putText(frame, f"Team 2 ball control: {team_2:.1f}%", (x1 + 18, y1 + 92), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return frame
