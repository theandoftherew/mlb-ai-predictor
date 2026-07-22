# ⚾ MLB AI Predictor

A Monte Carlo MLB game predictor. Pulls live schedules & lineups from the MLB API,
then simulates each game thousands of times to project win probabilities,
moneylines, total runs, and player stat lines.

## Model
Player true-talent rates (K / BB / hit-on-contact) joined by MLBAM id, blended via
log5 matchup math, with:
- **Platoon splits** (L/R batter-vs-pitcher)
- **Park factors** and **weather** (temp / wind)
- **Recency weighting** (recent form counts more)
- **Per-team bullpen quality**
- Empirical-Bayes regression so small samples don't lie

Validated with a leakage-free walk-forward backtest (`backtest.py`): ~59% game-winner
accuracy, unbiased run totals, calibrated win probabilities — beats naive baselines.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — the Streamlit app + simulation engine
- `backtest.py` — walk-forward accuracy backtest
- `fetch_data.py` — downloads Statcast data (regenerates the full dataset)
- `mlb_batting_data.csv` — slim plate-appearance dataset the app runs on
