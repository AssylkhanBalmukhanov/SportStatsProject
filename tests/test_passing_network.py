import unittest

from sportstats.services.passing_network import build_network


class PassingNetworkServiceTest(unittest.TestCase):
    def test_build_network_aggregates_edges_and_nodes(self):
        network = build_network(
            [
                {"passer": "A", "receiver": "B", "start_x": 10, "start_y": 40, "end_x": 30, "end_y": 50},
                {"passer": "A", "receiver": "B", "start_x": 12, "start_y": 42, "end_x": 32, "end_y": 52},
                {"passer": "B", "receiver": "C", "start_x": 30, "start_y": 50, "end_x": 60, "end_y": 45},
            ]
        )

        self.assertEqual(network["total_passes"], 3)
        self.assertEqual(network["edges"][0]["source"], "A")
        self.assertEqual(network["edges"][0]["target"], "B")
        self.assertEqual(network["edges"][0]["count"], 2)
        self.assertEqual(len(network["nodes"]), 3)

    def test_build_network_rejects_invalid_coordinates(self):
        with self.assertRaisesRegex(ValueError, "start_x"):
            build_network(
                [
                    {
                        "passer": "A",
                        "receiver": "B",
                        "start_x": -1,
                        "start_y": 40,
                        "end_x": 30,
                        "end_y": 50,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
