from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from random import Random


@dataclass(frozen=True)
class TeamProfile:
    name: str
    elo: float = 1500.0
    attack_strength: float = 1.0
    defense_strength: float = 1.0


def predict_match(
    home: TeamProfile,
    away: TeamProfile,
    *,
    simulations: int = 20000,
    seed: int = 42,
) -> dict:
    simulations = min(max(simulations, 1000), 100000)
    home_xg, away_xg = expected_goals(home, away)
    simulated = _run_monte_carlo(home_xg, away_xg, simulations, seed)
    scorelines = _scoreline_probabilities(home_xg, away_xg)

    return {
        "home_team": home.name,
        "away_team": away.name,
        "expected_goals": {
            "home": round(home_xg, 2),
            "away": round(away_xg, 2),
        },
        "probabilities": {
            "home_win": round(simulated["home_win"] / simulations, 4),
            "draw": round(simulated["draw"] / simulations, 4),
            "away_win": round(simulated["away_win"] / simulations, 4),
        },
        "most_likely_scores": scorelines[:5],
        "simulations": simulations,
    }


def expected_goals(home: TeamProfile, away: TeamProfile) -> tuple[float, float]:
    league_home_goals = 1.42
    league_away_goals = 1.16
    elo_adjustment = (home.elo - away.elo) / 800.0

    home_xg = league_home_goals * home.attack_strength / max(away.defense_strength, 0.2)
    away_xg = league_away_goals * away.attack_strength / max(home.defense_strength, 0.2)

    home_xg *= 1.0 + max(min(elo_adjustment, 0.35), -0.35)
    away_xg *= 1.0 - max(min(elo_adjustment, 0.35), -0.35)
    return max(home_xg, 0.05), max(away_xg, 0.05)


def _poisson_probability(mean: float, goals: int) -> float:
    return (mean**goals * exp(-mean)) / factorial(goals)


def _scoreline_probabilities(home_xg: float, away_xg: float, max_goals: int = 6) -> list[dict]:
    scorelines = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = _poisson_probability(home_xg, home_goals) * _poisson_probability(away_xg, away_goals)
            scorelines.append(
                {
                    "score": f"{home_goals}-{away_goals}",
                    "probability": round(probability, 4),
                }
            )
    return sorted(scorelines, key=lambda item: item["probability"], reverse=True)


def _run_monte_carlo(home_xg: float, away_xg: float, simulations: int, seed: int) -> dict:
    rng = Random(seed)
    totals = {"home_win": 0, "draw": 0, "away_win": 0}
    for _ in range(simulations):
        home_goals = _sample_poisson(home_xg, rng)
        away_goals = _sample_poisson(away_xg, rng)
        if home_goals > away_goals:
            totals["home_win"] += 1
        elif home_goals == away_goals:
            totals["draw"] += 1
        else:
            totals["away_win"] += 1
    return totals


def _sample_poisson(mean: float, rng: Random) -> int:
    threshold = exp(-mean)
    count = 0
    product = 1.0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1
