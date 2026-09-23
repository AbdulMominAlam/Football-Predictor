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
MAX_FORM_WINDOW = 10
SEASON_REGRESSION = 0.25


FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_difference",
    "elo_expected_home",
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
    "home_10_match_win_rate",
    "away_10_match_win_rate",
    "ten_match_win_rate_difference",
    "home_10_match_points_per_match",
    "away_10_match_points_per_match",
    "ten_match_points_difference",
    "home_10_match_goal_difference",
    "away_10_match_goal_difference",
    "ten_match_goal_difference_difference",
    "home_season_matches",
    "away_season_matches",
    "season_matches_difference",
    "home_season_points_per_match",
    "away_season_points_per_match",
    "season_points_difference",
    "home_season_goal_difference",
    "away_season_goal_difference",
    "season_goal_difference_difference",
    "home_home_points_per_match",
    "away_away_points_per_match",
    "venue_points_difference",
    "home_home_goal_difference",
    "away_away_goal_difference",
    "venue_goal_difference_difference",
    "home_rest_days",
    "away_rest_days",
    "rest_days_difference",
]


def empty_season_stats():
    return {
        "matches": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "home_matches": 0,
        "home_points": 0,
        "home_goals_for": 0,
        "home_goals_against": 0,
        "away_matches": 0,
        "away_points": 0,
        "away_goals_for": 0,
        "away_goals_against": 0,
    }


class PremierLeagueFeatureBuilder:
    def __init__(self):
        self.elo_ratings = defaultdict(lambda: INITIAL_ELO)

        self.recent_results = defaultdict(
            lambda: deque(maxlen=MAX_FORM_WINDOW)
        )

        self.season_stats = defaultdict(empty_season_stats)
        self.last_match_dates = {}
        self.current_season = None

    def regress_elo_at_new_season(self):
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
            self.season_stats = defaultdict(
                empty_season_stats
            )
            self.current_season = season

    def get_recent_form(self, team, window):
        results = list(self.recent_results[team])[-window:]

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

        wins = sum(
            result["points"] == 3
            for result in results
        )

        draws = sum(
            result["points"] == 1
            for result in results
        )

        points = sum(
            result["points"]
            for result in results
        )

        goals_scored = sum(
            result["goals_scored"]
            for result in results
        )

        goals_conceded = sum(
            result["goals_conceded"]
            for result in results
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

    def get_season_form(self, team):
        stats = self.season_stats[team]
        matches = stats["matches"]

        if matches == 0:
            return {
                "matches": 0,
                "points_per_match": 1.0,
                "goal_difference": 0.0,
            }

        return {
            "matches": matches,
            "points_per_match": (
                stats["points"] / matches
            ),
            "goal_difference": (
                stats["goals_for"]
                - stats["goals_against"]
            ) / matches,
        }

    def get_venue_form(self, team, venue):
        stats = self.season_stats[team]

        if venue == "home":
            matches = stats["home_matches"]
            points = stats["home_points"]
            goals_for = stats["home_goals_for"]
            goals_against = stats["home_goals_against"]
        else:
            matches = stats["away_matches"]
            points = stats["away_points"]
            goals_for = stats["away_goals_for"]
            goals_against = stats["away_goals_against"]

        if matches == 0:
            return {
                "points_per_match": 1.0,
                "goal_difference": 0.0,
            }

        return {
            "points_per_match": points / matches,
            "goal_difference": (
                goals_for - goals_against
            ) / matches,
        }

    def get_rest_days(self, team, match_date):
        previous_date = self.last_match_dates.get(team)

        if previous_date is None:
            return 7

        rest_days = (match_date - previous_date).days

        return max(0, min(rest_days, 30))

    @staticmethod
    def expected_home_score(home_elo, away_elo):
        adjusted_home_elo = home_elo + HOME_ADVANTAGE

        return 1 / (
            1
            + 10
            ** (
                (away_elo - adjusted_home_elo)
                / 400
            )
        )

    def create_pre_match_features(
        self,
        home_team,
        away_team,
        match_date,
    ):
        home_form_5 = self.get_recent_form(
            home_team,
            5,
        )
        away_form_5 = self.get_recent_form(
            away_team,
            5,
        )

        home_form_10 = self.get_recent_form(
            home_team,
            10,
        )
        away_form_10 = self.get_recent_form(
            away_team,
            10,
        )

        home_season = self.get_season_form(home_team)
        away_season = self.get_season_form(away_team)

        home_venue = self.get_venue_form(
            home_team,
            "home",
        )
        away_venue = self.get_venue_form(
            away_team,
            "away",
        )

        home_elo = self.elo_ratings[home_team]
        away_elo = self.elo_ratings[away_team]

        home_rest = self.get_rest_days(
            home_team,
            match_date,
        )
        away_rest = self.get_rest_days(
            away_team,
            match_date,
        )

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": (
                home_elo + HOME_ADVANTAGE - away_elo
            ),
            "elo_expected_home": (
                self.expected_home_score(
                    home_elo,
                    away_elo,
                )
            ),
            "home_win_rate": home_form_5["win_rate"],
            "away_win_rate": away_form_5["win_rate"],
            "win_rate_difference": (
                home_form_5["win_rate"]
                - away_form_5["win_rate"]
            ),
            "home_draw_rate": home_form_5["draw_rate"],
            "away_draw_rate": away_form_5["draw_rate"],
            "draw_rate_difference": (
                home_form_5["draw_rate"]
                - away_form_5["draw_rate"]
            ),
            "home_points_per_match": (
                home_form_5["points_per_match"]
            ),
            "away_points_per_match": (
                away_form_5["points_per_match"]
            ),
            "points_per_match_difference": (
                home_form_5["points_per_match"]
                - away_form_5["points_per_match"]
            ),
            "home_goals_scored": (
                home_form_5["goals_scored"]
            ),
            "away_goals_scored": (
                away_form_5["goals_scored"]
            ),
            "goals_scored_difference": (
                home_form_5["goals_scored"]
                - away_form_5["goals_scored"]
            ),
            "home_goals_conceded": (
                home_form_5["goals_conceded"]
            ),
            "away_goals_conceded": (
                away_form_5["goals_conceded"]
            ),
            "goals_conceded_difference": (
                away_form_5["goals_conceded"]
                - home_form_5["goals_conceded"]
            ),
            "home_goal_difference": (
                home_form_5["goal_difference"]
            ),
            "away_goal_difference": (
                away_form_5["goal_difference"]
            ),
            "recent_goal_difference_difference": (
                home_form_5["goal_difference"]
                - away_form_5["goal_difference"]
            ),
            "home_10_match_win_rate": (
                home_form_10["win_rate"]
            ),
            "away_10_match_win_rate": (
                away_form_10["win_rate"]
            ),
            "ten_match_win_rate_difference": (
                home_form_10["win_rate"]
                - away_form_10["win_rate"]
            ),
            "home_10_match_points_per_match": (
                home_form_10["points_per_match"]
            ),
            "away_10_match_points_per_match": (
                away_form_10["points_per_match"]
            ),
            "ten_match_points_difference": (
                home_form_10["points_per_match"]
                - away_form_10["points_per_match"]
            ),
            "home_10_match_goal_difference": (
                home_form_10["goal_difference"]
            ),
            "away_10_match_goal_difference": (
                away_form_10["goal_difference"]
            ),
            "ten_match_goal_difference_difference": (
                home_form_10["goal_difference"]
                - away_form_10["goal_difference"]
            ),
            "home_season_matches": home_season["matches"],
            "away_season_matches": away_season["matches"],
            "season_matches_difference": (
                home_season["matches"]
                - away_season["matches"]
            ),
            "home_season_points_per_match": (
                home_season["points_per_match"]
            ),
            "away_season_points_per_match": (
                away_season["points_per_match"]
            ),
            "season_points_difference": (
                home_season["points_per_match"]
                - away_season["points_per_match"]
            ),
            "home_season_goal_difference": (
                home_season["goal_difference"]
            ),
            "away_season_goal_difference": (
                away_season["goal_difference"]
            ),
            "season_goal_difference_difference": (
                home_season["goal_difference"]
                - away_season["goal_difference"]
            ),
            "home_home_points_per_match": (
                home_venue["points_per_match"]
            ),
            "away_away_points_per_match": (
                away_venue["points_per_match"]
            ),
            "venue_points_difference": (
                home_venue["points_per_match"]
                - away_venue["points_per_match"]
            ),
            "home_home_goal_difference": (
                home_venue["goal_difference"]
            ),
            "away_away_goal_difference": (
                away_venue["goal_difference"]
            ),
            "venue_goal_difference_difference": (
                home_venue["goal_difference"]
                - away_venue["goal_difference"]
            ),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_days_difference": (
                home_rest - away_rest
            ),
        }

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

    @staticmethod
    def match_points(home_goals, away_goals):
        if home_goals > away_goals:
            return 3, 0

        if home_goals < away_goals:
            return 0, 3

        return 1, 1

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

        actual_home, _ = self.result_values(
            home_goals,
            away_goals,
        )

        goal_difference = abs(
            home_goals - away_goals
        )

        if goal_difference <= 1:
            goal_multiplier = 1.0
        elif goal_difference == 2:
            goal_multiplier = 1.5
        else:
            goal_multiplier = (
                1.75
                + (goal_difference - 3) * 0.125
            )

        elo_change = (
            ELO_K_FACTOR
            * goal_multiplier
            * (actual_home - expected_home)
        )

        self.elo_ratings[home_team] = (
            home_elo + elo_change
        )
        self.elo_ratings[away_team] = (
            away_elo - elo_change
        )

    def update_recent_form(
        self,
        home_team,
        away_team,
        home_goals,
        away_goals,
    ):
        home_points, away_points = self.match_points(
            home_goals,
            away_goals,
        )

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

    def update_season_stats(
        self,
        home_team,
        away_team,
        home_goals,
        away_goals,
    ):
        home_points, away_points = self.match_points(
            home_goals,
            away_goals,
        )

        home_stats = self.season_stats[home_team]
        away_stats = self.season_stats[away_team]

        home_stats["matches"] += 1
        home_stats["points"] += home_points
        home_stats["goals_for"] += home_goals
        home_stats["goals_against"] += away_goals
        home_stats["home_matches"] += 1
        home_stats["home_points"] += home_points
        home_stats["home_goals_for"] += home_goals
        home_stats["home_goals_against"] += away_goals

        away_stats["matches"] += 1
        away_stats["points"] += away_points
        away_stats["goals_for"] += away_goals
        away_stats["goals_against"] += home_goals
        away_stats["away_matches"] += 1
        away_stats["away_points"] += away_points
        away_stats["away_goals_for"] += away_goals
        away_stats["away_goals_against"] += home_goals

    def process_match(self, match):
        season = match["season"]
        match_date = pd.Timestamp(match["date"])
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        self.start_season(season)

        features = self.create_pre_match_features(
            home_team,
            away_team,
            match_date,
        )

        row = {
            "season": season,
            "date": match_date.date().isoformat(),
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

        self.update_season_stats(
            home_team,
            away_team,
            home_goals,
            away_goals,
        )

        self.last_match_dates[home_team] = match_date
        self.last_match_dates[away_team] = match_date

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

    missing_columns = (
        required_columns - set(matches.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns from matches.csv: "
            + ", ".join(sorted(missing_columns))
        )

    matches = matches[
        matches["status"] == "played"
    ].copy()

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
        [
            "date",
            "home_team",
            "away_team",
        ]
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

    print(
        f"Feature count: {len(FEATURE_COLUMNS)}"
    )

    print(
        "Missing values: "
        f"{feature_data.isna().sum().sum()}"
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


if __name__ == "__main__":
    main()