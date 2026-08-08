import sys
import io
from pathlib import Path
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import (
    build_current_team_states,
    create_prediction_features,
    load_match_data,
    load_model,
    predict_match,
)

from world_cup_teams import WORLD_CUP_2026_TEAMS

from world_cup_simulator import (
    ensure_minimum_team_history,
    simulate_tournament,
)

SIMULATION_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "world_cup_1000_simulations.csv"
)

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

@st.cache_resource
def load_resources():
    model, feature_columns = load_model()
    match_data = load_match_data()

    (
        histories,
        elo_ratings,
        known_teams,
    ) = build_current_team_states(match_data)

    ensure_minimum_team_history(
        tournament_teams=WORLD_CUP_2026_TEAMS,
        histories=histories,
        elo_ratings=elo_ratings,
    )

    return (
        model,
        feature_columns,
        match_data,
        histories,
        elo_ratings,
        known_teams,
    )

@st.cache_data
def load_simulation_results():
    if not SIMULATION_RESULTS_FILE.exists():
        return None
    return pd.read_csv(SIMULATION_RESULTS_FILE)

with st.spinner("Loading model and historical match data..."):
    (
        model,
        feature_columns,
        match_data,
        histories,
        elo_ratings,
        known_teams,
    ) = load_resources()

simulation_results = load_simulation_results()

st.sidebar.title("⚽ World Cup Predictor")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Match Predictor",
        "Tournament Simulator",
        "Championship Odds",
        "About",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "FIFA World Cup 2026 prediction project "
    "using machine learning, Elo ratings, "
    "recent form, and Monte Carlo simulation."
)

if page == "Home":
    st.title("⚽ FIFA World Cup 2026 Predictor")

    st.write(
        """
        A machine learning-powered FIFA World Cup simulator
        using historical international football data,
        Elo ratings, recent team form, and a Random Forest model.
        """
    )

    st.success(
        "Prediction model and historical match data loaded successfully."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("World Cup Teams", len(WORLD_CUP_2026_TEAMS))

    with col2:
        st.metric("Historical Matches", f"{len(match_data):,}")

    with col3:
        st.metric("ML Model", "Random Forest")

    with col4:
        st.metric("Monte Carlo Runs", "1,000")

    st.divider()
    st.subheader("What You Can Do")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            ### Match Predictor
            Compare any two World Cup teams and view:

            - Team 1 win probability
            - Draw probability
            - Team 2 win probability
            - Predicted match outcome
            """
        )

    with feature_col2:
        st.markdown(
            """
            ### Tournament Simulator
            Simulate an entire World Cup including:

            - Group stage
            - 32 qualified teams
            - Round of 32
            - Round of 16
            - Quarterfinals
            - Semifinals
            - Final
            """
        )

    st.divider()
    st.subheader("2026 World Cup Teams")

    team_columns = st.columns(4)

    for index, team in enumerate(WORLD_CUP_2026_TEAMS):
        with team_columns[index % 4]:
            st.write(f"• {team}")

elif page == "Match Predictor":
    st.title("⚔️ Match Predictor")

    st.write(
        """
        Select two World Cup teams to estimate the probability
        of a win, draw, or loss using the trained Random Forest model.
        """
    )

    team_col1, team_col2 = st.columns(2)

    with team_col1:
        home_team = st.selectbox(
            "Team 1",
            WORLD_CUP_2026_TEAMS,
            index=WORLD_CUP_2026_TEAMS.index("Argentina"),
        )

    with team_col2:
        away_team = st.selectbox(
            "Team 2",
            WORLD_CUP_2026_TEAMS,
            index=WORLD_CUP_2026_TEAMS.index("France"),
        )

    if st.button("Predict Match", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select two different teams.")
        else:
            try:
                feature_values = create_prediction_features(
                    home_team=home_team,
                    away_team=away_team,
                    neutral=True,
                    histories=histories,
                    elo_ratings=elo_ratings,
                )

                predicted_result, probabilities = predict_match(
                    model=model,
                    feature_columns=feature_columns,
                    feature_values=feature_values,
                )

                away_probability = probabilities.get(0, 0)
                draw_probability = probabilities.get(1, 0)
                home_probability = probabilities.get(2, 0)

                st.divider()
                st.subheader(f"{home_team} vs {away_team}")

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    st.metric(
                        f"{home_team} Win",
                        f"{home_probability * 100:.2f}%",
                    )

                with result_col2:
                    st.metric(
                        "Draw",
                        f"{draw_probability * 100:.2f}%",
                    )

                with result_col3:
                    st.metric(
                        f"{away_team} Win",
                        f"{away_probability * 100:.2f}%",
                    )

                if predicted_result == 2:
                    st.success(f"Predicted outcome: {home_team} win")
                elif predicted_result == 0:
                    st.success(f"Predicted outcome: {away_team} win")
                else:
                    st.info("Predicted outcome: Draw")

            except Exception as error:
                st.error(f"Prediction error: {error}")

elif page == "Tournament Simulator":
    st.title("🏆 World Cup Tournament Simulator")

    st.write(
        """
        Run one complete simulation of the 48-team World Cup.

        The model simulates every group-stage match,
        determines the 32 qualified teams, and continues
        through the knockout rounds until a champion is produced.
        """
    )

    st.info(
        "Each simulation is probabilistic, so the result can "
        "change every time you run the tournament."
    )

    if st.button(
        "🏆 Simulate World Cup",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Simulating the complete World Cup..."):
                captured_output = io.StringIO()

                with redirect_stdout(captured_output):
                    champion, runner_up = simulate_tournament(
                        tournament_teams=WORLD_CUP_2026_TEAMS.copy(),
                        model=model,
                        feature_columns=feature_columns,
                        base_histories=histories,
                        base_elo_ratings=elo_ratings,
                        show_output=True,
                    )

                tournament_output = captured_output.getvalue()

            st.session_state["tournament_champion"] = champion
            st.session_state["tournament_runner_up"] = runner_up
            st.session_state["tournament_output"] = tournament_output

        except Exception as error:
            st.error(f"Tournament simulation error: {error}")

    if "tournament_champion" in st.session_state:
        st.divider()

        champion = st.session_state["tournament_champion"]
        runner_up = st.session_state["tournament_runner_up"]

        st.subheader("Tournament Result")

        champion_col, runner_col = st.columns(2)

        with champion_col:
            st.metric("🏆 World Cup Champion", champion)

        with runner_col:
            st.metric("🥈 Runner-up", runner_up)

        st.success(
            f"🏆 {champion} wins the simulated 2026 FIFA World Cup!"
        )

        with st.expander("View Full Tournament Results"):
            st.code(st.session_state["tournament_output"])

elif page == "Championship Odds":
    st.title("📊 1,000-Simulation Championship Odds")

    st.write(
        """
        The full World Cup was simulated 1,000 times.

        Championship probability represents the percentage
        of simulations in which each team won the tournament.
        """
    )

    if simulation_results is None:
        st.warning(
            "The 1,000-simulation results CSV could not be found."
        )
        st.write(f"Expected file: {SIMULATION_RESULTS_FILE}")
    else:
        results = simulation_results.copy()

        probability_columns = [
            "championship_probability",
            "runner_up_probability",
            "final_probability",
        ]

        for column in probability_columns:
            results[column] = results[column].round(2)

        results = results.sort_values(
            "championship_probability",
            ascending=False,
        ).reset_index(drop=True)

        first = results.iloc[0]
        second = results.iloc[1]
        third = results.iloc[2]

        st.subheader("Tournament Favorites")

        favorite1, favorite2, favorite3 = st.columns(3)

        with favorite1:
            st.metric(
                "🥇 Highest Odds",
                first["team"],
                f"{first['championship_probability']:.2f}%",
            )

        with favorite2:
            st.metric(
                "🥈 Second Highest",
                second["team"],
                f"{second['championship_probability']:.2f}%",
            )

        with favorite3:
            st.metric(
                "🥉 Third Highest",
                third["team"],
                f"{third['championship_probability']:.2f}%",
            )

        st.divider()
        st.subheader("Championship Probability")
        st.caption(
            "Top 15 teams by probability of winning the World Cup."
        )

        championship_chart = (
            results[
                [
                    "team",
                    "championship_probability",
                ]
            ]
            .head(15)
            .set_index("team")
        )

        st.bar_chart(championship_chart)

        st.divider()
        st.subheader("Probability of Reaching the Final")

        final_chart = (
            results[
                [
                    "team",
                    "final_probability",
                ]
            ]
            .sort_values(
                "final_probability",
                ascending=False,
            )
            .head(15)
            .set_index("team")
        )

        st.bar_chart(final_chart)

        st.divider()
        st.subheader("Full Monte Carlo Results")

        display_results = results[
            [
                "team",
                "championship_wins",
                "championship_probability",
                "runner_up_finishes",
                "runner_up_probability",
                "final_probability",
            ]
        ].copy()

        display_results.columns = [
            "Team",
            "Titles",
            "Win Probability (%)",
            "Runner-ups",
            "Runner-up Probability (%)",
            "Final Probability (%)",
        ]

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )

elif page == "About":
    st.title("ℹ️ About the Project")

    st.write(
        """
        This project predicts international football matches
        and simulates the 2026 FIFA World Cup using machine
        learning and historical match data.
        """
    )

    st.subheader("Machine Learning Model")

    st.write(
        """
        A Random Forest classifier is trained on historical
        international football matches.

        The model predicts three possible match outcomes:

        - Team 1 win
        - Draw
        - Team 2 win
        """
    )

    st.subheader("Prediction Features")

    st.write(
        """
        The model uses features including:

        - Elo ratings
        - Elo rating difference
        - Recent win rate
        - Recent draw rate
        - Average goals scored
        - Average goals conceded
        - Points per match
        - Recent goal difference
        - Neutral venue information
        """
    )

    st.subheader("Tournament Simulation")

    st.write(
        """
        The tournament simulator starts with the fixed
        48-team World Cup groups.

        Each match is simulated probabilistically using
        the model's predicted win, draw, and loss probabilities.

        Team Elo ratings and recent form are updated during
        the tournament so later matches reflect what happened
        earlier in the simulation.
        """
    )

    st.subheader("Monte Carlo Analysis")

    st.write(
        """
        The complete tournament was simulated 1,000 times
        to estimate championship probabilities.

        Instead of treating one tournament simulation as a
        definitive prediction, Monte Carlo analysis shows how
        frequently each team becomes champion across many
        possible tournament outcomes.
        """
    )

    st.subheader("Project Stack")

    stack_col1, stack_col2 = st.columns(2)

    with stack_col1:
        st.write(
            """
            **Machine Learning**
            - Python
            - pandas
            - scikit-learn
            - Random Forest
            """
        )

    with stack_col2:
        st.write(
            """
            **Application**
            - Streamlit
            - Historical match dataset
            - Elo rating system
            - Monte Carlo simulation
            """
        )