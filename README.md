# ⚽ FIFA World Cup 2026 Predictor

A machine learning-powered FIFA World Cup simulator that predicts individual match outcomes and simulates the complete 48-team tournament.

Built using **historical international football data, Elo ratings, recent team form, a Random Forest classifier, and Monte Carlo simulation**.

The project includes an interactive Streamlit application for predicting matches, simulating complete tournaments, and exploring championship probabilities.

## 🌐 Live Demo

**Try the deployed application here:**

[Launch FIFA World Cup 2026 Predictor](https://football-predictor-kr9foj8794qjm2vlplmf4h.streamlit.app/)

---

## 🏆 1,000 World Cup Simulations

To estimate each team's chances of winning the tournament, the complete 2026 World Cup was simulated **1,000 times**.

| Rank | Team | Championships | Win Probability | Final Probability |
|------|------|--------------:|----------------:|------------------:|
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

According to the simulation, **Argentina is the strongest favorite**, winning 194 of the 1,000 simulated tournaments for an estimated championship probability of **19.4%**.

The complete results for all 48 teams are available in the Streamlit application.

---

## Features

### Match Predictor

Select any two World Cup teams and calculate:

- Team 1 win probability
- Draw probability
- Team 2 win probability
- Most likely match outcome

### Full World Cup Simulator

Simulates the complete **48-team FIFA World Cup format**, including:

- 12 groups of 4 teams
- Group-stage matches and standings
- Best third-place qualification
- Round of 32
- Round of 16
- Quarterfinals
- Semifinals
- Final
- World Cup champion

Team form and Elo ratings are updated as the simulated tournament progresses.

### Monte Carlo Simulation

The complete tournament can be simulated repeatedly to estimate long-term tournament probabilities.

The simulation records:

- Championship wins
- Championship probability
- Runner-up finishes
- Runner-up probability
- Probability of reaching the final

The project currently includes results from **1,000 complete World Cup simulations**.

---

## How It Works

The prediction pipeline:

```text
Historical International Matches
              ↓
       Feature Engineering
              ↓
   Elo Ratings + Recent Form
              ↓
    Random Forest Classifier
              ↓
 Win / Draw / Loss Probabilities
              ↓
        Match Simulation
              ↓
   World Cup Tournament Logic
              ↓
     Monte Carlo Simulation
              ↓
 Championship Probabilities
```

For each match, the model estimates the probability of a **home/team 1 win, draw, or away/team 2 win**.

These probabilities are then used by the tournament simulator to generate match results and progress teams through the competition.

---

## Machine Learning Model

The project uses a **Random Forest classifier** trained on historical international football results.

A chronological train/test split is used so that the model is trained on earlier matches and evaluated on later matches rather than randomly mixing past and future results.

### Model Performance

```text
Accuracy: 56.63%

                 Precision    Recall    F1
Away Win            0.56       0.62    0.59
Draw                0.29       0.26    0.27
Home Win            0.69       0.68    0.69
```

Draws are the most difficult outcome for the model to predict, while home wins are predicted more reliably.

---

## Feature Engineering

Predictions are based on features representing team strength and recent performance, including:

- Elo rating
- Elo rating difference
- Recent win rate
- Recent draw rate
- Points per match
- Average goals scored
- Average goals conceded
- Recent goal difference
- Neutral venue indicator

During model training, **Elo rating difference** was the most influential feature.

### Elo Ratings

Elo ratings provide a continuously updated measure of team strength.

After each simulated match, the ratings can change based on the result, allowing the tournament simulation to maintain an evolving representation of team strength.

### Recent Form

The model also considers recent team performance rather than relying only on long-term historical strength.

This includes results, points, goals scored, and goals conceded across recent matches.

---

## Tournament Simulation

The simulator follows the 48-team World Cup structure.

```text
48 Teams
   ↓
12 Groups of 4
   ↓
Group Stage
   ↓
32 Qualified Teams
   ↓
Round of 32
   ↓
Round of 16
   ↓
Quarterfinals
   ↓
Semifinals
   ↓
Final
   ↓
World Cup Champion
```

Group standings are calculated from simulated match results before qualifying teams advance into the knockout stage.

---

## Streamlit Application

The interactive web application contains five sections:

### Home
Overview of the project, model, tournament, and participating teams.

### Match Predictor
Select two teams and view their predicted win/draw/loss probabilities.

### Tournament Simulator
Run a complete World Cup simulation and inspect group standings and knockout results.

### Championship Odds
Explore the results of the **1,000 Monte Carlo simulations** through tables and visualizations.

### About
Learn about the model, feature engineering, Elo ratings, and simulation methodology.

You can use the deployed version here:

[Open the Live Application](https://football-predictor-kr9foj8794qjm2vlplmf4h.streamlit.app/)

---

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
│   └── processed/
│       └── world_cup_1000_simulations.csv
│
├── models/
│   └── random_forest_model.joblib
│
└── src/
    ├── train_model.py
    ├── predict.py
    ├── world_cup_teams.py
    ├── world_cup_groups.py
    ├── world_cup_simulator.py
    └── tournament_statistics.py
```

### Main Files

| File | Purpose |
|------|---------|
| `app.py` | Runs the Streamlit web application |
| `src/train_model.py` | Trains and evaluates the Random Forest model |
| `src/predict.py` | Creates prediction features and predicts match outcomes |
| `src/world_cup_teams.py` | Stores the 48 tournament teams |
| `src/world_cup_groups.py` | Defines the 12 tournament groups |
| `src/world_cup_simulator.py` | Handles group-stage and knockout tournament simulation |
| `src/tournament_statistics.py` | Runs repeated Monte Carlo simulations and calculates tournament probabilities |
| `world_cup_1000_simulations.csv` | Stores results from the 1,000-tournament experiment |

---

## Technology Stack

### Machine Learning & Data

- Python
- pandas
- NumPy
- scikit-learn
- Random Forest
- joblib

### Simulation & Application

- Elo rating system
- Monte Carlo simulation
- Streamlit
- Historical international football data

### Deployment

- GitHub
- Streamlit Community Cloud

---

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

On macOS/Linux:

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

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

### 5. Run Monte Carlo simulations

The tournament statistics script can also be run directly:

```bash
python src/tournament_statistics.py
```

---

## Limitations

Football matches are highly unpredictable, and several real-world factors are not currently represented in the model.

These include:

- Injuries and suspensions
- Starting lineups
- Player-level form
- Managerial changes
- Tactical matchups
- Travel and fatigue
- Weather and pitch conditions
- Match-specific circumstances

Historical results can also favor teams with strong past performance even when their current squads have changed significantly.

The predictions should therefore be interpreted as **probabilistic estimates rather than guaranteed results**.

---

## Future Improvements

Potential improvements include:

- Player-level statistics
- Injury and suspension data
- Live team rankings
- Squad strength metrics
- Host-country and venue effects
- Hyperparameter tuning
- Comparison with models such as XGBoost and neural networks
- Parallelized Monte Carlo simulations
- Automated model retraining with new match results
- Support for additional international tournaments

---

## Author

**Abdul Momin Alam**

GitHub: [@AbdulMominAlam](https://github.com/AbdulMominAlam)

---

*Built as a machine learning and football analytics project.*