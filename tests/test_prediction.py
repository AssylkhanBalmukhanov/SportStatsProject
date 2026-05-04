import unittest

from sportstats.services.prediction import TeamProfile, predict_match


class PredictionServiceTest(unittest.TestCase):
    def test_predict_match_returns_probability_distribution(self):
        result = predict_match(
            TeamProfile("Home", elo=1550, attack_strength=1.1, defense_strength=0.95),
            TeamProfile("Away", elo=1490, attack_strength=1.0, defense_strength=1.05),
            simulations=2000,
        )

        probabilities = result["probabilities"]
        total = probabilities["home_win"] + probabilities["draw"] + probabilities["away_win"]

        self.assertGreaterEqual(total, 0.98)
        self.assertLessEqual(total, 1.02)
        self.assertGreater(result["expected_goals"]["home"], 0)
        self.assertGreater(result["expected_goals"]["away"], 0)
        self.assertEqual(len(result["most_likely_scores"]), 5)

    def test_stronger_home_team_has_higher_win_probability(self):
        result = predict_match(
            TeamProfile("Strong Home", elo=1700, attack_strength=1.3, defense_strength=0.8),
            TeamProfile("Weak Away", elo=1400, attack_strength=0.85, defense_strength=1.2),
            simulations=2000,
        )

        self.assertGreater(result["probabilities"]["home_win"], result["probabilities"]["away_win"])


if __name__ == "__main__":
    unittest.main()
