from io import BytesIO
import unittest

from sportstats import create_app
from sportstats.config import TestConfig


class AppTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

    def test_health_check(self):
        response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_predictor_api(self):
        response = self.client.post(
            "/api/predict",
            json={
                "home_name": "Home",
                "away_name": "Away",
                "home_elo": 1600,
                "away_elo": 1500,
                "simulations": 1000,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["home_team"], "Home")

    def test_vision_rejects_wrong_file_type(self):
        response = self.client.post(
            "/vision",
            data={"video": (BytesIO(b"not a video"), "notes.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Upload an MP4", response.data)


if __name__ == "__main__":
    unittest.main()
