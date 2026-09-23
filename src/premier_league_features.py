from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "matches.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "premier_league"
    / "processed"
    / "model_features.csv"
)

INITIAL_ELO = 1500
ELO_K_FACTOR = 20
HOME_ADVANTAGE = 65
FORM_WINDOW = 5
SEASON_REGRESSION = 0.25


FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_win_rate",
    "away_win_rate",
    "win_rate_difference",
    "home_draw_rate",
    "away_draw_rate",
    "draw_rate_difference",
    "home_points_per_match",
    "away_points_per_match",
    "points_per_match_difference",
    "home_goals_scored",
    "away_goals_scored",
    "goals_scored_difference",
    "home_goals_conceded",
    "away_goals_conceded",
    "goals_conceded_difference",
    "home_goal_difference",
    "away_goal_difference",
    "recent_goal_difference_difference",
]


class PremierLeagueFeatureBuilder:
    def __init__(self):
        self.elo_ratings = defaultdict(lambda: INITIAL_ELO)
        self.recent_results = defaultdict(
            lambda: deque(maxlen=FORM_WINDOW)
        )
        self.current_season = None

    def regress_elo_at_new_season(self):
        """
        Move every existing Elo rating slightly toward the league average.

        This reduces the influence of older seasons while still preserving
        information about each team's previous strength.
        """
        for team in list(self.elo_ratings.keys()):
            current_rating = self.elo_ratings[team]

            self.elo_ratings[team] = (
                current_rating * (1 - SEASON_REGRESSION)
                + INITIAL_ELO * SEASON_REGRESSION
            )

    def start_season(self, season):
        if self.current_season is None:
            self.current_season = season
            return

        if season != self.current_season:
            self.regress_elo_at_new_season()
            self.current_season = season

    def get_form(self, team):
        results = list(self.recent_results[team])

        if not results:
            return {
                "win_rate": 1 / 3,
                "draw_rate": 1 / 3,
                "points_per_match": 1.0,
                "goals_scored": 1.2,
                "goals_conceded": 1.2,
                "goal_difference": 0.0,
            }

        matches = len(results)
        wins = sum(result["points"] == 3 for result in results)
        draws = sum(result["points"] == 1 for result in results)
        points = sum(result["points"] for result in results)
        goals_scored = sum(
            result["goals_scored"] for result in results
        )
        goals_conceded = sum(
            result["goals_conceded"] for result in results
        )

        return {
            "win_rate": wins / matches,
            "draw_rate": draws / matches,
            "points_per_match": points / matches,
            "goals_scored": goals_scored / matches,
            "goals_conceded": goals_conceded / matches,
            "goal_difference": (
                goals_scored - goals_conceded
            ) / matches,
        }

    def create_pre_match_features(self, home_team, away_team):
        """
        Create features before the match result is added.

        This is important because it prevents the match being predicted
        from leaking into its own feature values.
        """
        home_form = self.get_form(home_team)
        away_form = self.get_form(away_team)

        home_elo = self.elo_ratings[home_team]
        away_elo = self.elo_ratings[away_team]

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": (
                home_elo + HOME_ADVANTAGE - away_elo
            ),
            "home_win_rate": home_form["win_rate"],
            "away_win_rate": away_form["win_rate"],
            "win_rate_difference": (
                home_form["win_rate"] - away_form["win_rate"]
            ),
            "home_draw_rate": home_form["draw_rate"],
            "away_draw_rate": away_form["draw_rate"],
            "draw_rate_difference": (
                home_form["draw_rate"] - away_form["draw_rate"]
            ),
            "home_points_per_match": (
                home_form["points_per_match"]
            ),
            "away_points_per_match": (
                away_form["points_per_match"]
            ),
            "points_per_match_difference": (
                home_form["points_per_match"]
                - away_form["points_per_match"]
            ),
            "home_goals_scored": home_form["goals_scored"],
            "away_goals_scored": away_form["goals_scored"],
            "goals_scored_difference": (
                home_form["goals_scored"]
                - away_form["goals_scored"]
            ),
            "home_goals_conceded": (
                home_form["goals_conceded"]
            ),
            "away_goals_conceded": (
                away_form["goals_conceded"]
            ),
            "goals_conceded_difference": (
                away_form["goals_conceded"]
                - home_form["goals_conceded"]
            ),
            "home_goal_difference": (
                home_form["goal_difference"]
            ),
            "away_goal_difference": (
                away_form["goal_difference"]
            ),
            "recent_goal_difference_difference": (
                home_form["goal_difference"]
                - away_form["goal_difference"]
            ),
        }

    @staticmethod
    def expected_home_score(home_elo, away_elo):
        adjusted_home_elo = home_elo + HOME_ADVANTAGE

        return 1 / (
            1 + 10 ** ((away_elo - adjusted_home_elo) / 400)
        )

    @staticmethod
    def result_values(home_goals, away_goals):
        if home_goals > away_goals:
            return 1.0, 0.0
        if home_goals < away_goals:
            return 0.0, 1.0
        return 0.5, 0.5

    @staticmethod
    def match_target(home_goals, away_goals):
        if home_goals > away_goals:
            return "home_win"
        if home_goals < away_goals:
            return "away_win"
        return "draw"

    def update_elo(
        self,
        home_team,
        away_team,
        home_goals,
        away_goals,
    ):
        home_elo = self.elo_ratings[home_team]
        away_elo = self.elo_ratings[away_team]

        expected_home = self.expected_home_score(
            home_elo,
            away_elo,
        )
        expected_away = 1 - expected_home

        actual_home, actual_away = self.result_values(
            home_goals,
            away_goals,
        )

        goal_difference = abs(home_goals - away_goals)

        if goal_difference <= 1:
            goal_multiplier = 1.0
        elif goal_difference == 2:
            goal_multiplier = 1.5
        else:
            goal_multiplier = 1.75 + (
                goal_difference - 3
            ) * 0.125

        elo_change = (
            ELO_K_FACTOR
            * goal_multiplier
            * (actual_home - expected_home)
        )

        self.elo_ratings[home_team] = home_elo + elo_change
        self.elo_ratings[away_team] = away_elo - elo_change

    def update_recent_form(
        self,
        home_team,
        away_team,
        home_goals,
        away_goals,
    ):
        if home_goals > away_goals:
            home_points = 3
            away_points = 0
        elif home_goals < away_goals:
            home_points = 0
            away_points = 3
        else:
            home_points = 1
            away_points = 1

        self.recent_results[home_team].append(
            {
                "points": home_points,
                "goals_scored": home_goals,
                "goals_conceded": away_goals,
            }
        )

        self.recent_results[away_team].append(
            {
                "points": away_points,
                "goals_scored": away_goals,
                "goals_conceded": home_goals,
            }
        )

    def process_match(self, match):
        season = match["season"]
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        self.start_season(season)

        features = self.create_pre_match_features(
            home_team,
            away_team,
        )

        row = {
            "season": season,
            "date": match["date"],
            "home_team": home_team,
            "away_team": away_team,
            **features,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "target": self.match_target(
                home_goals,
                away_goals,
            ),
        }

        self.update_elo(
            home_team,
            away_team,
            home_goals,
            away_goals,
        )

        self.update_recent_form(
            home_team,
            away_team,
            home_goals,
            away_goals,
        )

        return row


def load_completed_matches():
    matches = pd.read_csv(INPUT_FILE)

    required_columns = {
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "status",
    }

    missing_columns = required_columns - set(matches.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns from matches.csv: "
            + ", ".join(sorted(missing_columns))
        )

    matches = matches[matches["status"] == "played"].copy()

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="raise",
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="raise",
    )

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="raise",
    )

    matches = matches.sort_values(
        ["date", "season"]
    ).reset_index(drop=True)

    return matches


def build_feature_dataset():
    matches = load_completed_matches()
    builder = PremierLeagueFeatureBuilder()

    feature_rows = []

    for _, match in matches.iterrows():
        feature_rows.append(
            builder.process_match(match)
        )

    feature_data = pd.DataFrame(feature_rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return feature_data


def main():
    feature_data = build_feature_dataset()

    print(
        f"Saved {len(feature_data)} feature rows to "
        f"{OUTPUT_FILE}"
    )

    print("\nMatches by season:")
    print(
        feature_data["season"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTarget distribution:")
    print(
        feature_data["target"]
        .value_counts()
        .to_string()
    )

    print("\nFeature columns:")
    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")


if __name__ == "__main__":
    main()