import sys
from pathlib import Path
import io
from contextlib import redirect_stdout

import streamlit as st


# ==========================================
# PROJECT PATH SETUP
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ==========================================
# PROJECT IMPORTS
# ==========================================

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


# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)


# ==========================================
# LOAD MODEL + DATA
# ==========================================

@st.cache_resource
def load_resources():
    """
    Load the trained model and historical match data.
    """

    model, feature_columns = load_model()

    match_data = load_match_data()

    (
        histories,
        elo_ratings,
        known_teams,
    ) = build_current_team_states(
        match_data
    )

    # Make sure every World Cup team has enough
    # historical form data for prediction.
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


# ==========================================
# LOAD RESOURCES
# ==========================================

with st.spinner(
    "Loading prediction model and historical match data..."
):
    (
        model,
        feature_columns,
        match_data,
        histories,
        elo_ratings,
        known_teams,
    ) = load_resources()


# ==========================================
# HEADER
# ==========================================

st.title("⚽ FIFA World Cup 2026 Predictor")

st.write(
    """
    Machine learning-powered FIFA World Cup simulator
    using historical international football data,
    Elo ratings, recent team form, and a
    Random Forest classification model.
    """
)

st.success(
    "Prediction model and historical match data loaded successfully."
)


# ==========================================
# PROJECT INFORMATION
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="World Cup Teams",
        value=len(WORLD_CUP_2026_TEAMS),
    )

with col2:
    st.metric(
        label="Historical Matches",
        value=f"{len(match_data):,}",
    )

with col3:
    st.metric(
        label="ML Model",
        value="Random Forest",
    )


st.divider()


# ==========================================
# MATCH PREDICTOR
# ==========================================

st.subheader("Match Predictor")

st.write(
    "Select two teams to predict the probability "
    "of a win, draw, or loss."
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


if st.button(
    "Predict Match",
    type="primary",
    use_container_width=True,
):

    if home_team == away_team:

        st.error(
            "Please select two different teams."
        )

    else:

        try:
            feature_values = create_prediction_features(
                home_team=home_team,
                away_team=away_team,
                neutral=True,
                histories=histories,
                elo_ratings=elo_ratings,
            )

            (
                predicted_result,
                probabilities,
            ) = predict_match(
                model=model,
                feature_columns=feature_columns,
                feature_values=feature_values,
            )

            away_probability = probabilities.get(0, 0)
            draw_probability = probabilities.get(1, 0)
            home_probability = probabilities.get(2, 0)

            st.markdown(
                f"### {home_team} vs {away_team}"
            )

            (
                result_col1,
                result_col2,
                result_col3,
            ) = st.columns(3)

            with result_col1:

                st.metric(
                    label=f"{home_team} Win",
                    value=(
                        f"{home_probability * 100:.2f}%"
                    ),
                )

            with result_col2:

                st.metric(
                    label="Draw",
                    value=(
                        f"{draw_probability * 100:.2f}%"
                    ),
                )

            with result_col3:

                st.metric(
                    label=f"{away_team} Win",
                    value=(
                        f"{away_probability * 100:.2f}%"
                    ),
                )

            if predicted_result == 2:

                st.success(
                    f"Predicted outcome: "
                    f"{home_team} win"
                )

            elif predicted_result == 0:

                st.success(
                    f"Predicted outcome: "
                    f"{away_team} win"
                )

            else:

                st.info(
                    "Predicted outcome: Draw"
                )

        except ValueError as error:

            st.error(str(error))


st.divider()


# ==========================================
# FULL WORLD CUP SIMULATOR
# ==========================================

st.subheader("World Cup Tournament Simulator")

st.write(
    """
    Simulate the complete 48-team FIFA World Cup.

    The model will simulate every group-stage match,
    determine the 32 qualified teams, and continue
    through the knockout rounds until a champion
    is produced.
    """
)


if st.button(
    "🏆 Simulate World Cup",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Simulating the complete World Cup..."
    ):

        # Capture the normal terminal output generated
        # by world_cup_simulator.py.
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):

            champion, runner_up = simulate_tournament(
                tournament_teams=(
                    WORLD_CUP_2026_TEAMS.copy()
                ),
                model=model,
                feature_columns=feature_columns,
                base_histories=histories,
                base_elo_ratings=elo_ratings,
                show_output=True,
            )

        tournament_output = (
            captured_output.getvalue()
        )

        # Save the results so Streamlit does not
        # immediately lose them after reruns.
        st.session_state[
            "tournament_champion"
        ] = champion

        st.session_state[
            "tournament_runner_up"
        ] = runner_up

        st.session_state[
            "tournament_output"
        ] = tournament_output


# ==========================================
# DISPLAY TOURNAMENT RESULT
# ==========================================

if "tournament_champion" in st.session_state:

    champion = st.session_state[
        "tournament_champion"
    ]

    runner_up = st.session_state[
        "tournament_runner_up"
    ]

    st.markdown("### Tournament Result")

    champion_col, runner_col = st.columns(2)

    with champion_col:

        st.metric(
            label="🏆 World Cup Champion",
            value=champion,
        )

    with runner_col:

        st.metric(
            label="🥈 Runner-up",
            value=runner_up,
        )

    st.success(
        f"🏆 {champion} wins the simulated "
        "2026 FIFA World Cup!"
    )

    # Full tournament output
    with st.expander(
        "View Full Tournament Results"
    ):

        st.code(
            st.session_state[
                "tournament_output"
            ],
            language=None,
        )


st.divider()


# ==========================================
# WORLD CUP TEAMS
# ==========================================

st.subheader("2026 World Cup Teams")

st.write(
    f"{len(WORLD_CUP_2026_TEAMS)} teams are included "
    "in the tournament simulation."
)

team_columns = st.columns(4)

for index, team in enumerate(
    WORLD_CUP_2026_TEAMS
):

    column = team_columns[
        index % 4
    ]

    with column:

        st.write(
            f"• {team}"
        )


st.divider()


# ==========================================
# PROJECT FEATURES
# ==========================================

st.subheader("Predictor Features")

st.write(
    """
    This application includes:

    - Individual match prediction
    - Complete 48-team World Cup simulation
    - Group-stage qualification
    - Round of 32
    - Round of 16
    - Quarterfinals
    - Semifinals
    - Final
    - Monte Carlo championship probability analysis
    """
)