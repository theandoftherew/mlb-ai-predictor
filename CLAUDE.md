# MLB AI Predictor: project context

This file is auto-loaded by Claude Code at session start. It is the standing
context for this repo. Keep it current; anything worth surviving a `/clear`
belongs here, not in chat scrollback.

**Repo:** `/Users/boys/mlb-ai-predictor/`
**GitHub:** `theandoftherew/mlb-ai-predictor` (public)
**Live:** Streamlit Community Cloud (`*.streamlit.app`)
**Local port:** 8501
**Current version:** `v1.4-mlblend`
**Status:** LIVE and deployed

---

## Part 1: Shared philosophy and methodology (READ FIRST)

This is the most important section. It is *how* we build, and it applies to
every sport in the suite.

1. **Backtest-driven, no leakage.** Everything is validated with **walk-forward**
   backtesting: predict each game using only data from *before* it. A model's
   real quality is a *number* from the backtest, never a hunch.

2. **A/B every change on the SAME games.** To judge a feature, run the backtest
   **with and without it on identical games plus the same random seed**, and
   compare. Do NOT compare two separate backtest runs. Sim randomness makes
   accuracy swing plus or minus 3% even with zero real change.

3. **Brier score beats accuracy.** Accuracy (which side crosses 50%) is coarse
   and noisy. **Brier score** (uses the full probability) is the reliable
   metric. When they disagree, trust Brier.

4. **Keep only what improves.** Many plausible ideas get *rejected* by the
   backtest. That is the process working, not a failure. We document rejected
   ideas so they are not re-tried.

5. **Calibration is the goal.** Honest probabilities (when the model says 60%,
   those teams win about 60% of the time) are what make the betting and edge
   tools valid. Each app shows a calibration chart.

6. **Honesty over hype.** Report the real numbers, flag limitations (small data,
   near-ceiling targets), never oversell. Single-game outcomes are near the
   prediction ceiling in every sport.

7. **Version every model change** (`MODEL_VERSION`) so the live track record can
   compare versions on real, realized results.

8. **Double-counting is the recurring trap.** Several rejected ideas failed
   because they re-counted signal already in the model (team defense is already
   in pitcher rates; naive times-through-order; leverage bullpen). The fix is to
   *center* the adjustment so it does not shift the mean.

**Tech stack:** Python plus Streamlit (dark "sportsbook" theme, green `#00e676`
accent), deployed free on Streamlit Community Cloud via GitHub (GitHub Desktop
to share.streamlit.io). Data lives as CSVs committed to the repo. Betting tools
support American *and* decimal odds and evaluate both sides of every market. A
daily/weekly GitHub Action grows a live track record.

---

## Part 2: The MLB model

A **Monte Carlo game simulator**. Simulates each game thousands of times, plate
appearance by plate appearance, using each player's true-talent rates.

Validated stack (each A/B-tested, walk-forward):

- Batter and pitcher rates (K, BB, hit-on-contact) **joined by MLBAM player ID**.
  This is the core fix: names are unreliable, and in Statcast `player_name` is
  the *pitcher*.
- **Platoon** L/R splits, **park factors**, **weather** (temp/wind)
- **Recency weighting** (exponential, roughly 150-day half-life)
- **Per-team bullpen** quality, **pitcher stamina** (batters-faced, so aces go
  about 6 IP)
- **Times-through-order** penalty (centered)
- **Inherited-runner ERA** (charge inherited runs to the pitcher who allowed them)
- **Home-field advantage** (offense x1.03/0.97), fixed calibration
- **ML blend:** a logistic model on game features, blended 50/50 with the sim's
  win%. Validated to beat either alone.
- Empirical-Bayes shrinkage everywhere so small samples regress to league mean

### Validated performance (300-game walk-forward)

About **58% winner accuracy**. Brier about **0.244** (beats naive). Calibrated
probabilities. Run totals unbiased. Run-total MAE about 3.6, near the
irreducible single-game floor.

### App features

Live schedule and lineups (MLB API), editable lineups (roster dropdowns plus
up/down reorder), park and weather, win% / moneyline / total plus run-distribution
chart, hitter and pitcher projections, **player props** (1+ hit, 2+ hits, HR,
pitcher K and hits-allowed over/under), **edge finder** (ML, total O/U, run line,
both sides, American and decimal), full-slate predictions, live **Track Record**
dashboard (accuracy curve plus calibration), About/accuracy panel, disclaimer.
A GitHub Action every 6h logs predictions and grades finished games into
`predictions.csv`.

### Rejected ideas (do not retry without a new angle)

- Team defense (double-counts pitcher stats)
- Leverage-based reliever quality (small-sample noise)
- Naive/uncentered times-through-order (double-count)
- Gradient boosting / trees for the ML component (overfit at about 1,200 games)

### Key files

- `app.py`: sim engine plus UI. Importable; UI lives under `if __name__=="__main__"`
- `backtest.py`: walk-forward harness
- `fetch_data.py`: Statcast via pybaseball
- `track_record.py`: log and grade
- `ml_dataset.py` / `ml_train.py` / `ml_build.py` plus `ml_model.pkl`: the blend
- `predictions.csv`: track record
- `mlb_batting_data.csv`: 23 MB slim file (PA rows plus 15 cols). The full 689 MB
  version is backed up locally as `mlb_batting_data_full.csv` and is gitignored.

---

## Environment and commands

MLB uses the **global** Python (pandas 2.3.3 / numpy 2.0.2) and runs on port
**8501**. This is deliberately isolated from the NFL project, which has its own
venv on port 8502. They never conflict.

```bash
# Run the app locally
cd /Users/boys/mlb-ai-predictor && python3 -m streamlit run app.py   # :8501

# Refresh data (about 2h, pybaseball is slow), then retrain the ML blend
python3 fetch_data.py            # then re-slim to mlb_batting_data.csv
python3 ml_dataset.py && python3 ml_build.py
```

**Deploy loop:** edit code, commit, Push origin in GitHub Desktop, Streamlit
Cloud auto-redeploys in 1 to 2 minutes. If the daily Action pushed data, Pull
origin first.

## Gotchas

- Statcast `player_name` is the *pitcher*. The `batter` and `pitcher` columns are
  MLBAM ids and are the reliable join key.
- The slim CSV is PA-rows only, 15 columns.
- Accuracy across separate backtest runs is noise. Always A/B on the same games
  and read Brier.
- The Streamlit Cloud filesystem is ephemeral, so the track record must live in
  git (the GitHub Action commits it back). Never write it at runtime.

## Roadmap / next up

- Keep data fresh; it is a snapshot
- Confirmed lineups and injuries
- Retrain the ML as data grows (`python3 ml_build.py`). The ML edge should grow
  with more seasons.
