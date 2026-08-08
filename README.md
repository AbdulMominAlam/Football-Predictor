# ⚽ FIFA World Cup 2026 Predictor

A machine learning-powered FIFA World Cup simulator that predicts individual match outcomes and simulates the complete 48-team tournament.

Built using historical international football data, Elo ratings, recent team form, a Random Forest classifier, and Monte Carlo simulation.

The project also includes an interactive Streamlit application for exploring match predictions, tournament simulations, and championship probabilities.

---

## 🏆 1,000 World Cup Simulations

The complete 2026 World Cup was simulated **1,000 times** to estimate each team's probability of becoming world champion.

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

According to the model, **Argentina enters the tournament as the strongest favorite**, winning 194 of the 1,000 simulated tournaments.

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

Team form and Elo ratings are updated throughout the simulated tournament.

### Monte Carlo Simulation

The complete tournament can be simulated hundreds or thousands of times.

The project records:

- Championship wins
- Championship probability
- Runner-up finishes
- Runner-up probability
- Probability of reaching the final

Results from the 1,000-run experiment are saved and displayed directly inside the Streamlit application.

---

## How the Model Works

The prediction pipeline is:

```text
Historical Match Data
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
World Cup Tournament Simulation
        ↓
Monte Carlo Championship Probabilities
```

### Model

The project uses a **Random Forest classifier** trained on historical international football matches.

Training/testing uses a chronological split rather than randomly mixing past and future matches.

The trained model achieved approximately:

```text
Accuracy: 56.63%

                 Precision    Recall    F1
Away Win            0.56       0.62    0.59
Draw                0.29       0.26    0.27
Home Win            0.69       0.68    0.69
```

Predicting draws remains the most difficult class, while home wins are predicted considerably more reliably.

---

## Feature Engineering

The model uses team-strength and recent-form features including:

- Elo rating
- Elo rating difference
- Recent win rate
- Recent draw rate
- Points per match
- Average goals scored
- Average goals conceded
- Recent goal difference
- Neutral venue indicator

The most influential feature during training was **Elo rating difference**.

---

## Streamlit Application

The project includes an interactive Streamlit dashboard with five sections:

**Home**  
Project overview and participating teams.

**Match Predictor**  
Predict a match between any two tournament teams.

**Tournament Simulator**  
Run a complete World Cup and inspect group standings and knockout results.

**Championship Odds**  
Explore results from the 1,000 Monte Carlo simulations through tables and charts.

**About**  
Explanation of the model, features, and simulation methodology.

Run the application locally with:

```bash
streamlit run app.py
```

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

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web application |
| `train_model.py` | Trains and evaluates the Random Forest model |
| `predict.py` | Creates prediction features and predicts matches |
| `world_cup_teams.py` | Stores the 48 tournament teams |
| `world_cup_groups.py` | Defines the tournament groups |
| `world_cup_simulator.py` | Simulates the complete World Cup |
| `tournament_statistics.py` | Runs Monte Carlo tournament simulations |

---

## Technology Stack

**Machine Learning**

- Python
- pandas
- NumPy
- scikit-learn
- Random Forest
- joblib

**Application & Analysis**

- Streamlit
- Elo rating system
- Monte Carlo simulation
- Historical international football data

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/AbdulMominAlam/Football-Predictor.git
cd Football-Predictor
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Or run the Monte Carlo simulation directly:

```bash
python src/tournament_statistics.py
```

---

## Limitations

Football is highly unpredictable, and the model does not currently account for every factor that can affect a match.

Examples include:

- Injuries and suspensions
- Starting lineups
- Player-level form
- Managerial or tactical changes
- Travel and fatigue
- Weather conditions
- Match-specific circumstances

The predictions should therefore be interpreted as **probabilistic estimates rather than guaranteed results**.

---

## Future Improvements

Potential improvements include:

- Player-level statistics
- Injury and suspension data
- Live team rankings
- Hyperparameter tuning and model comparison
- Parallelized Monte Carlo simulations
- Automated model retraining
- Additional international tournaments

---

## Author

**Abdul Momin Alam**

GitHub: [@AbdulMominAlam](https://github.com/AbdulMominAlam)

---

*Built as a machine learning and football analytics project.*