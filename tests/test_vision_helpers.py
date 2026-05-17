import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sportstats.vision.ball_assignment import PlayerBallAssigner
from sportstats.vision.camera_movement import _load_camera_stub
from sportstats.vision.geometry import get_center_of_bbox, get_foot_position, measure_distance
from sportstats.vision.team_assignment import kmeans
from sportstats.vision.tracking import FootballTracker, _load_tracks_stub


class VisionHelpersTest(unittest.TestCase):
    def test_geometry_helpers(self):
        bbox = [10, 20, 30, 80]

        self.assertEqual(get_center_of_bbox(bbox), (20.0, 50.0))
        self.assertEqual(get_foot_position(bbox), (20.0, 80.0))
        self.assertEqual(measure_distance((0, 0), (3, 4)), 5.0)

    def test_ball_interpolation_fills_missing_frames(self):
        ball_positions = [
            {1: {"bbox": [0, 0, 10, 10]}},
            {},
            {1: {"bbox": [20, 20, 30, 30]}},
        ]

        interpolated = FootballTracker.interpolate_ball_positions(ball_positions)

        self.assertEqual(interpolated[1][1]["bbox"], [10.0, 10.0, 20.0, 20.0])
        self.assertTrue(interpolated[1][1]["interpolated"])

    def test_player_ball_assignment_uses_nearest_foot(self):
        players = {
            7: {"bbox": [0, 0, 20, 100]},
            8: {"bbox": [200, 0, 220, 100]},
        }

        assigned = PlayerBallAssigner(max_player_ball_distance=50).assign_ball_to_player(players, [5, 88, 15, 98])

        self.assertEqual(assigned, 7)

    def test_kmeans_separates_two_colors(self):
        points = np.asarray([[0, 0, 0], [5, 5, 5], [250, 250, 250], [245, 245, 245]], dtype=float)

        centroids, labels = kmeans(points, k=2, iterations=10)

        self.assertEqual(len(centroids), 2)
        self.assertEqual(set(labels.tolist()), {0, 1})

    def test_track_stub_must_match_expected_frame_count(self):
        tracks = {"players": [{}], "referees": [{}], "ball": [{}]}
        with TemporaryDirectory() as directory:
            stub_path = Path(directory) / "tracks.pkl"
            with stub_path.open("wb") as handle:
                pickle.dump(tracks, handle)

            self.assertIsNotNone(_load_tracks_stub(stub_path, expected_frame_count=1))
            self.assertIsNone(_load_tracks_stub(stub_path, expected_frame_count=2))

    def test_camera_stub_must_match_expected_frame_count(self):
        with TemporaryDirectory() as directory:
            stub_path = Path(directory) / "camera.pkl"
            with stub_path.open("wb") as handle:
                pickle.dump([(0.0, 0.0)], handle)

            self.assertIsNotNone(_load_camera_stub(stub_path, expected_frame_count=1))
            self.assertIsNone(_load_camera_stub(stub_path, expected_frame_count=2))


if __name__ == "__main__":
    unittest.main()
