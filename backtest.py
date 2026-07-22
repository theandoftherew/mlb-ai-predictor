"""
Backtesting harness for the MLB predictor.

WHY THIS MATTERS: it turns "does the model feel right?" into hard numbers.

It runs a rigorous WALK-FORWARD test with NO data leakage: for each historical
game on date D, the player profiles and league baselines are built ONLY from
games played BEFORE D. We then simulate that game and compare the prediction to
what actually happened. Everything (lineups, starters, final scores) is
reconstructed from the Statcast CSV itself — no extra downloads.

Run:  python3 backtest.py --sims 300 --max-games 80
"""
import warnings; warnings.filterwarnings("ignore")

import argparse
import math
import random
import numpy as np
import pandas as pd

import app  # the engine — importing runs NO UI and hits NO network

random.seed(0)
np.random.seed(0)

# Statcast team abbreviations -> full names used in app.PARK_FACTORS.
ABBR_TO_TEAM = {
    'ATH': 'Athletics', 'ATL': 'Atlanta Braves', 'AZ': 'Arizona Diamondbacks',
    'BAL': 'Baltimore Orioles', 'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
    'CWS': 'Chicago White Sox', 'DET': 'Detroit Tigers', 'HOU': 'Houston Astros',
    'KC': 'Kansas City Royals', 'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins',
    'NYM': 'New York Mets', 'NYY': 'New York Yankees', 'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SEA': 'Seattle Mariners',
    'SF': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals',
}
NEUTRAL_WEATHER = {"temp": 70, "wind_mph": 5, "wind_dir": "Neutral"}


def reconstruct_game(g):
    """From one game's PA rows, derive lineups, starters, and the final score.
    Returns None if the game is malformed (too few batters, etc.)."""
    g = g.sort_values("at_bat_number")
    away_bat = g[g["inning_topbot"] == "Top"]   # away team bats in the top
    home_bat = g[g["inning_topbot"] == "Bot"]   # home team bats in the bottom
    if away_bat.empty or home_bat.empty:
        return None

    def lineup(side_df):
        recs = []
        for bid in side_df["batter"].dropna().drop_duplicates().tolist()[:9]:
            stand = side_df[side_df["batter"] == bid]["stand"].mode()
            recs.append({"name": f"B{int(bid)}", "id": int(bid),
                         "bats": stand.iloc[0] if len(stand) else "R"})
        return recs

    def starter(side_df):
        first = side_df[side_df["inning"] == 1]
        if first.empty:
            first = side_df
        pid = first.sort_values("at_bat_number")["pitcher"].iloc[0]
        thr = side_df[side_df["pitcher"] == pid]["p_throws"].mode()
        return {"name": f"P{int(pid)}", "id": int(pid),
                "throws": thr.iloc[0] if len(thr) else "R"}

    away_lineup, home_lineup = lineup(away_bat), lineup(home_bat)
    if len(away_lineup) < 9 or len(home_lineup) < 9:
        return None

    return {
        "away_lineup": away_lineup, "home_lineup": home_lineup,
        "home_sp": starter(away_bat),   # home pitches while away bats
        "away_sp": starter(home_bat),   # away pitches while home bats
        "home_abbr": g["home_team"].iloc[0],
        "away_abbr": g["away_team"].iloc[0],
        "final_away": int(g["post_away_score"].max()),
        "final_home": int(g["post_home_score"].max()),
    }


def run_backtest(sims, max_games, start, end, half_life=150):
    print("Loading data...")
    _, pa = app._load_data()
    if pa is None:
        print("No data. Run fetch_data.py first."); return
    pa["game_date"] = pa["game_date"].astype(str)

    # Candidate test games: completed games in the chosen window.
    test = pa[(pa["game_date"] >= start) & (pa["game_date"] <= end)]
    game_ids = test["game_pk"].dropna().unique().tolist()
    random.shuffle(game_ids)
    game_ids = game_ids[:max_games]

    # Map each test game to its date, then process date-by-date so we build the
    # (expensive) training baselines once per date instead of once per game.
    gid_date = test.groupby("game_pk")["game_date"].min()
    by_date = {}
    for gid in game_ids:
        by_date.setdefault(gid_date[gid], []).append(gid)

    rows = []
    done = 0
    for D in sorted(by_date):
        train = pa[pa["game_date"] < D]
        if len(train) < 20000:       # not enough history yet — skip early dates
            continue
        train = app.add_recency_weights(train, D, half_life)   # weight recent form higher (as of game date)
        league = app._compute_league_baselines(train)
        bullpens = app.compute_bullpen_profiles(train, league)   # per-team relief quality as of D
        for gid in by_date[D]:
            g = pa[pa["game_pk"] == gid]
            info = reconstruct_game(g)
            if info is None:
                continue
            park = app.PARK_FACTORS.get(ABBR_TO_TEAM.get(info["home_abbr"], ""), app.NEUTRAL_PARK)
            _, _, betting = app.simulate_games(
                info["away_lineup"], info["home_lineup"],
                info["away_sp"], info["home_sp"],
                park, NEUTRAL_WEATHER, league, train, simulations=sims,
                away_bullpen=bullpens.get(info["away_abbr"]),
                home_bullpen=bullpens.get(info["home_abbr"]))

            actual_total = info["final_away"] + info["final_home"]
            home_won = 1 if info["final_home"] > info["final_away"] else 0
            rows.append({
                "date": D,
                "pred_home_wp": betting["home_win_pct"],
                "home_won": home_won,
                "pred_total": betting["avg_total_runs"],
                "actual_total": actual_total,
            })
            done += 1
            if done % 10 == 0:
                print(f"  ...{done} games simulated")

    if not rows:
        print("No games could be evaluated in that window."); return None
    return report(pd.DataFrame(rows), sims)


def report(df, sims):
    n = len(df)
    p = df["pred_home_wp"].clip(1e-6, 1 - 1e-6).values
    y = df["home_won"].values

    # --- Win prediction ---
    acc = float(((p > 0.5) == (y == 1)).mean())
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base_rate = float(y.mean())
    brier_homefield = float(np.mean((0.54 - y) ** 2))   # always guess 54% home

    # --- Total runs ---
    pt, at = df["pred_total"].values, df["actual_total"].values
    mae = float(np.mean(np.abs(pt - at)))
    rmse = float(np.sqrt(np.mean((pt - at) ** 2)))
    bias = float(np.mean(pt - at))
    naive = float(np.mean(np.abs(np.mean(at) - at)))   # always guess the mean

    print("\n" + "=" * 60)
    print(f"  BACKTEST REPORT  ({n} games, {sims} sims each)")
    print("=" * 60)
    print("\n  ── WIN PREDICTION ──")
    print(f"   Accuracy (picked winner) : {acc*100:5.1f}%   (home actually won {base_rate*100:.1f}%)")
    print(f"   Brier score              : {brier:.4f}   (lower better; coinflip=0.2500)")
    print(f"      vs always-54%-home     : {brier_homefield:.4f}")
    print(f"   Log loss                 : {logloss:.4f}   (lower better; coinflip=0.6931)")
    print("\n  ── TOTAL RUNS ──")
    print(f"   MAE  (avg runs off)      : {mae:.2f}   (naive 'always mean' = {naive:.2f})")
    print(f"   RMSE                     : {rmse:.2f}")
    print(f"   Bias (pred - actual)     : {bias:+.2f}   ({'runs high' if bias>0 else 'runs low'})")

    # --- Calibration: are our win probabilities honest? ---
    print("\n  ── WIN-PROB CALIBRATION ──")
    print("   pred bucket   n   predicted   actual")
    bins = [0, .35, .45, .55, .65, 1.01]
    labels = ["<35%", "35-45%", "45-55%", "55-65%", ">65%"]
    dfc = df.copy(); dfc["bucket"] = pd.cut(dfc["pred_home_wp"], bins=bins, labels=labels)
    for lab in labels:
        b = dfc[dfc["bucket"] == lab]
        if len(b):
            print(f"   {lab:>7}    {len(b):3d}    {b['pred_home_wp'].mean()*100:5.1f}%   {b['home_won'].mean()*100:5.1f}%")
    print("=" * 60)
    print("  Read: Accuracy > 54% and Brier < 0.25 means the model beats naive")
    print("  baselines. Bias near 0 means run totals are calibrated.")
    print("=" * 60)
    return {"n": n, "acc": acc, "brier": brier, "logloss": logloss,
            "mae": mae, "rmse": rmse, "bias": bias}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=300, help="Monte Carlo sims per game")
    ap.add_argument("--max-games", type=int, default=80, help="how many games to test")
    ap.add_argument("--start", default="2026-07-01", help="test window start (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-07-20", help="test window end (YYYY-MM-DD)")
    ap.add_argument("--half-life", type=int, default=150, help="recency half-life in days")
    args = ap.parse_args()
    run_backtest(args.sims, args.max_games, args.start, args.end, args.half_life)
