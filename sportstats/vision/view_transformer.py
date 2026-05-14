from __future__ import annotations

import numpy as np

from sportstats.vision.tracking import Tracks


DEFAULT_VERTICES_1080P = np.asarray(
    [
        [110, 1035],
        [265, 275],
        [910, 260],
        [1640, 915],
    ],
    dtype=np.float32,
)


class ViewTransformer:
    def __init__(
        self,
        *,
        frame_width: int = 1920,
        frame_height: int = 1080,
        pitch_width_m: float = 68.0,
        sampled_pitch_length_m: float = 23.32,
        pixel_vertices: np.ndarray | None = None,
    ):
        import cv2

        self.cv2 = cv2
        scale = np.asarray([frame_width / 1920.0, frame_height / 1080.0], dtype=np.float32)
        self.pixel_vertices = (pixel_vertices.astype(np.float32) if pixel_vertices is not None else DEFAULT_VERTICES_1080P)
        self.pixel_vertices = (self.pixel_vertices * scale).astype(np.float32)
        self.target_vertices = np.asarray(
            [
                [0.0, pitch_width_m],
                [0.0, 0.0],
                [sampled_pitch_length_m, 0.0],
                [sampled_pitch_length_m, pitch_width_m],
            ],
            dtype=np.float32,
        )
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float] | None:
        cv2 = self.cv2
        point_tuple = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, point_tuple, False) >= 0
        if not is_inside:
            return None

        reshaped = np.asarray(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.perspective_transformer).reshape(2)
        return (float(transformed[0]), float(transformed[1]))

    def add_transformed_positions_to_tracks(self, tracks: Tracks) -> None:
        for object_tracks in tracks.values():
            for frame_tracks in object_tracks:
                for track_info in frame_tracks.values():
                    position = track_info.get("position_adjusted") or track_info.get("position")
                    if position is None:
                        continue
                    transformed = self.transform_point(position)
                    track_info["position_transformed"] = transformed
