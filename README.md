## ⚽ Football Predictor

A machine-learning football prediction platform supporting two competitions:

- FIFA World Cup 2026
- Premier League 2026–27

The project combines historical match data, Elo ratings, recent form, machine-learning classifiers, Poisson goal modeling, and Monte Carlo simulation. Its Streamlit application can predict individual matches, simulate complete competitions, and display long-term championship, qualification, and relegation probabilities.

## 🌐 Live Application

[Launch the Football Predictor](https://football-predictor-kr9foj8794qjm2vlplmf4h.streamlit.app/)

## Competition Modes

### FIFA World Cup 2026

The World Cup mode supports:

- Predictions between any two World Cup teams
- Home/team-one win, draw, and away/team-two win probabilities
- Complete 48-team World Cup simulations
- Group-stage standings
- Best third-place qualification
- Round of 32 through the final
- Championship probabilities from 1,000 simulations

### Premier League 2026–27

The Premier League mode supports:

- Home-win, draw, and away-win probabilities
- Expected goals and predicted scores
- Current league standings
- Simulation of all remaining fixtures
- A projected final league table
- Title, top-four, top-six, and relegation probabilities
- Results from 1,000 complete-season simulations

## Premier League 1,000-Simulation Results

The 40 completed 2026–27 matches are preserved in every simulation. The remaining 340 fixtures are simulated chronologically, with Elo ratings and recent form updating after every simulated result.

| Rank | Team | Titles | Title Probability | Top Four | Top Six | Relegation |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Arsenal | 483 | **48.3%** | 94.6% | 98.4% | 0.0% |
| 2 | Manchester City | 436 | **43.6%** | 94.3% | 97.3% | 0.0% |
| 3 | Liverpool | 15 | **1.5%** | 29.7% | 49.7% | 1.9% |
| 4 | Brighton & Hove Albion | 13 | **1.3%** | 27.7% | 48.3% | 1.4% |
| 5 | Leeds United | 13 | **1.3%** | 18.8% | 33.6% | 5.1% |
| 6 | Manchester United | 12 | **1.2%** | 29.1% | 48.7% | 1.8% |
| 7 | Brentford | 6 | **0.6%** | 15.0% | 32.1% | 4.4% |
| 8 | Nottingham Forest | 5 | **0.5%** | 10.0% | 21.0% | 8.6% |
| 9 | Chelsea | 4 | **0.4%** | 17.6% | 32.4% | 3.7% |
| 10 | Everton | 4 | **0.4%** | 9.3% | 18.9% | 9.0% |

Arsenal and Manchester City are the strongest title favorites in the model. The complete results for all 20 clubs are available in the Streamlit application and in `data/premier_league/processed/premier_league_1000_simulations.csv`.

These are probabilistic model estimates, not guaranteed outcomes.

## World Cup 1,000-Simulation Results

| Rank | Team | Championships | Win Probability | Final Probability |
|---:|---|---:|---:|---:|
| 1 | Argentina | 194 | **19.40%** | 28.50% |
| 2 | Spain | 135 | **13.50%** | 23.20% |
| 3 | Brazil | 71 | **7.10%** | 14.20% |
| 4 | France | 63 | **6.30%** | 10.20% |
| 5 | Colombia | 49 | **4.90%** | 9.00% |
| 6 | Portugal | 47 | **4.70%** | 8.30% |
| 7 | Ecuador | 47 | **4.70%** | 9.50% |
| 8 | Germany | 38 | **3.80%** | 8.60% |
| 9 | Japan | 34 | **3.40%** | 7.00% |
| 10 | England | 34 | **3.40%** | 7.90% |

## How It Works

```text
Historical Match Results
          ↓
Chronological Data Preparation
          ↓
Elo Ratings + Recent Form + Season Features
          ↓
Match Outcome Classifier
          ↓
Win / Draw / Loss Probabilities
          ↓
Poisson Expected-Goals Model
          ↓
Realistic Score Generation
          ↓
Competition Simulator
          ↓
Monte Carlo Probabilities
```

All match features are calculated before the corresponding result is added to team history. This prevents the result being predicted from leaking into its own features.

## Premier League Modeling

### Data

The Premier League pipeline uses match results from 2015–16 through the completed portion of 2026–27.

- 12 seasons represented
- 4,560 total fixtures
- 4,220 completed matches used by the final production models
- 40 completed 2026–27 matches
- 340 remaining 2026–27 fixtures
- No missing values in the engineered model dataset

The raw fixture data is sourced from the [OpenFootball football.json project](https://github.com/openfootball/football.json).

### Outcome Classifier

The production match-outcome model uses Logistic Regression to estimate:

- Home-win probability
- Draw probability
- Away-win probability

Several approaches were compared with rolling chronological validation:

- Logistic Regression
- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- Soft-voting ensembles
- Poisson outcome probabilities

### Poisson Goal Model

A separate Poisson regression model estimates expected home and away goals. It is used to generate plausible scorelines that agree with the sampled match outcome.

For example, after sampling a draw, the score model chooses between plausible draw scores such as 0–0, 1–1, or 2–2. This provides realistic goals scored, goals conceded, and goal difference for the league table.

### Premier League Performance

The data is split chronologically rather than randomly. The final held-out evaluation trains on 2015–16 through 2024–25 and tests on the unseen 2025–26 season.

```text
Held-out accuracy: 49.47%
Most-common-outcome baseline: 42.63%
Test log loss: 1.0283
Test matches: 380
```

Accuracy measures whether the model's single most likely outcome matched the real home win, draw, or away win. Monte Carlo simulation uses all three probabilities rather than only the most likely class.

## Premier League Features

The feature-engineering pipeline creates candidate features representing:

- Home and away Elo ratings
- Elo difference and expected home score
- Five-match form
- Ten-match form
- Recent win and draw rates
- Recent points per match
- Recent goals scored and conceded
- Recent goal difference
- Current-season points per match
- Current-season goal difference
- Home-only and away-only performance
- Rest days between matches
- Home advantage

At the beginning of a new season, existing Elo ratings are partially moved toward the league average to account for transfers, managerial changes, and other changes between seasons.

## World Cup Modeling

The World Cup mode uses a Random Forest classifier trained on historical international match results. Its features include Elo ratings, recent form, scoring performance, and neutral-venue information.

```text
Accuracy: 56.63%

                 Precision    Recall    F1
Away Win            0.56       0.62    0.59
Draw                0.29       0.26    0.27
Home Win            0.69       0.68    0.69
```

The World Cup model does not use results from the 2026 tournament itself when producing predictions or simulations.

## Simulation Logic

### Premier League

Each full-season simulation:

1. Preserves every completed 2026–27 result.
2. Processes the remaining fixtures chronologically.
3. Creates pre-match features from the latest simulated state.
4. Samples home win, draw, or away win from classifier probabilities.
5. Generates a compatible score with the Poisson model.
6. Updates the league table, Elo ratings, and recent form.
7. Ranks clubs by points, goal difference, and goals scored.

The Monte Carlo pipeline records:

- Championships
- Runner-up finishes
- Top-four finishes
- Top-six finishes
- Relegations
- Average finishing position
- Average points
- Best and worst simulated finish

### World Cup

The World Cup simulator handles:

- 12 groups of four
- Group-stage standings
- Best third-place qualification
- Round of 32
- Round of 16
- Quarterfinals
- Semifinals
- Final

## Streamlit Application

The sidebar competition selector switches between the World Cup and Premier League modes.

### Premier League Pages

- **Home:** Current standings and season progress
- **Match Predictor:** Outcome probabilities, expected goals, and predicted score
- **Season Simulator:** One complete simulation of the remaining season
- **Season Odds:** Title, top-four, top-six, and relegation analysis
- **About:** Model methodology, evaluation, and limitations

### World Cup Pages

- **Home:** Project and tournament overview
- **Match Predictor:** Predictions between World Cup teams
- **Tournament Simulator:** Complete 48-team tournament simulation
- **Championship Odds:** Results from 1,000 simulations
- **About:** World Cup model methodology

## Project Structure

```text
Football-Predictor/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── world_cup_1000_simulations.csv
│   └── premier_league/
│       ├── raw/
│       │   ├── 2015-16.json
│       │   ├── ...
│       │   └── 2026-27.json
│       └── processed/
│           ├── matches.csv
│           ├── model_features.csv
│           └── premier_league_1000_simulations.csv
│
├── models/
│   ├── random_forest_model.joblib
│   └── premier_league_model.joblib
│
└── src/
    ├── train_model.py
    ├── predict.py
    ├── world_cup_teams.py
    ├── world_cup_groups.py
    ├── world_cup_simulator.py
    ├── tournament_statistics.py
    ├── premier_league_teams.py
    ├── prepare_premier_league_data.py
    ├── premier_league_table.py
    ├── premier_league_features.py
    ├── compare_premier_league_models.py
    ├── tune_premier_league_model.py
    ├── compare_poisson_model.py
    ├── train_premier_league_model.py
    ├── premier_league_predict.py
    ├── premier_league_simulator.py
    └── premier_league_statistics.py
```

## Technology Stack

### Machine Learning and Data

- Python
- pandas
- NumPy
- scikit-learn
- Logistic Regression
- Random Forest
- Extra Trees
- Poisson Regression
- joblib

### Simulation and Application

- Elo rating system
- Monte Carlo simulation
- Streamlit
- Historical football data

### Deployment

- GitHub
- Streamlit Community Cloud

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/AbdulMominAlam/Football-Predictor.git
cd Football-Predictor
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

On macOS or Linux:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Rebuilding the Premier League Pipeline

Prepare the raw fixture data:

```bash
python src/prepare_premier_league_data.py
```

Generate leakage-safe features:

```bash
python src/premier_league_features.py
```

Compare candidate models:

```bash
python src/compare_premier_league_models.py
python src/tune_premier_league_model.py
python src/compare_poisson_model.py
```

Train the production models:

```bash
python src/train_premier_league_model.py
```

Run one season simulation:

```bash
python src/premier_league_simulator.py
```

Run the 1,000-season Monte Carlo experiment:

```bash
python src/premier_league_statistics.py --simulations 1000 --jobs -1
```

## Limitations

The models do not directly include:

- Injuries and suspensions
- Starting lineups
- Player-level form
- Transfers and squad changes
- Managerial changes
- Tactical matchups
- Travel and fatigue beyond rest-day estimates
- Weather and pitch conditions
- Betting-market information
- Match-specific circumstances

The Premier League forecasts use completed results available through September 14, 2026. Predictions should be interpreted as probabilistic estimates rather than guarantees.

## Future Improvements

- Player-level and squad-strength features
- Injury and suspension data
- Expected-goals data from completed matches
- Automated fixture and result updates
- Automated model retraining
- Probability calibration monitoring
- Manager and transfer information
- Head-to-head tiebreaker logic
- Additional domestic leagues and tournaments

## Author

**Abdul Momin Alam**

GitHub: [@AbdulMominAlam](https://github.com/AbdulMominAlam)

---

Built as a machine-learning and football analytics project.
