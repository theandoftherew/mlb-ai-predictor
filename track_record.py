"""
Ongoing model-performance log — the model's LIVE track record.

Every prediction is written to `predictions.csv` (which lives in the repo, so it's
durable + auditable). Finished games are graded against real results. The Streamlit
app reads this file to visualize accuracy over time.

Each row is tagged with MODEL_VERSION so different weightings can be compared on
their real, realized results — the basis for tuning.

Commands:
  python3 track_record.py seed  --start 2026-05-01 --end 2026-07-20   # backfill from history (settled)
  python3 track_record.py predict --date 2026-07-21                   # log today's slate (open)
  python3 track_record.py grade                                       # settle finished open games
  python3 track_record.py daily                                       # what the GitHub Action runs
"""
import warnings; warnings.filterwarnings("ignore")

import os
import argparse
from datetime import date
import numpy as np
import pandas as pd
import requests

import app
import backtest   # reuse reconstruct_game + ABBR_TO_TEAM (importing runs no UI)

LOG_FILE = "predictions.csv"
COLUMNS = ["pred_date", "game_pk", "away_team", "home_team", "model_version",
           "pred_home_win_pct", "pred_total", "pred_margin", "status",
           "actual_away", "actual_home", "home_won", "correct", "brier",
           "total_abs_err", "margin_abs_err"]


# --------------------------------------------------------------------------
def _load_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=COLUMNS)


def _save_log(df):
    df.to_csv(LOG_FILE, index=False)


def _grade(pred_home_wp, pred_total, pred_margin, a_away, a_home):
    """Score one prediction against the actual final (away, home) runs."""
    home_won = 1 if a_home > a_away else 0
    return {
        "actual_away": a_away, "actual_home": a_home, "home_won": home_won,
        "correct": int((pred_home_wp > 0.5) == (home_won == 1)),
        "brier": round((pred_home_wp - home_won) ** 2, 4),
        "total_abs_err": round(abs(pred_total - (a_away + a_home)), 2),
        "margin_abs_err": round(abs(pred_margin - (a_away - a_home)), 2),
    }


def _predict_one(away_lineup, home_lineup, away_sp, home_sp, home_abbr, away_abbr,
                 pa, league, bullpens, sims):
    park = app.PARK_FACTORS.get(backtest.ABBR_TO_TEAM.get(home_abbr, ""), app.NEUTRAL_PARK)
    _, _, bet = app.simulate_games(
        away_lineup, home_lineup, away_sp, home_sp, park,
        {"temp": 70, "wind_mph": 5, "wind_dir": "Neutral"},
        league, pa, simulations=sims,
        away_bullpen=bullpens.get(away_abbr), home_bullpen=bullpens.get(home_abbr))
    return bet


def get_final_score(game_pk):
    """Return (away_runs, home_runs) if the game is Final, else None."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePk={game_pk}"
        g = requests.get(url, timeout=15).json()["dates"][0]["games"][0]
        if "final" in g["status"]["abstractGameState"].lower():
            return int(g["teams"]["away"]["score"]), int(g["teams"]["home"]["score"])
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
def seed(start, end, sims, max_games):
    """Backfill the log from history: walk-forward predictions graded with the
    real final scores already in the Statcast data (no leakage)."""
    print(f"Seeding {LOG_FILE} from {start}..{end} (walk-forward)...")
    _, pa = app._load_data()
    pa["game_date"] = pa["game_date"].astype(str)
    test = pa[(pa["game_date"] >= start) & (pa["game_date"] <= end)]
    gids = test["game_pk"].dropna().unique().tolist()[:max_games]
    gid_date = test.groupby("game_pk")["game_date"].min()

    by_date = {}
    for gid in gids:
        by_date.setdefault(gid_date[gid], []).append(gid)

    rows = []
    for D in sorted(by_date):
        train = pa[pa["game_date"] < D]
        if len(train) < 20000:
            continue
        train = app.add_recency_weights(train, D)
        league = app._compute_league_baselines(train)
        bullpens = app.compute_bullpen_profiles(train, league)
        for gid in by_date[D]:
            info = backtest.reconstruct_game(pa[pa["game_pk"] == gid])
            if info is None:
                continue
            bet = _predict_one(info["away_lineup"], info["home_lineup"], info["away_sp"],
                               info["home_sp"], info["home_abbr"], info["away_abbr"],
                               train, league, bullpens, sims)
            g = _grade(bet["home_win_pct"], bet["avg_total_runs"],
                       bet["away_exp_runs"] - bet["home_exp_runs"],
                       info["final_away"], info["final_home"])
            rows.append({
                "pred_date": D, "game_pk": int(gid),
                "away_team": info["away_abbr"], "home_team": info["home_abbr"],
                "model_version": app.MODEL_VERSION,
                "pred_home_win_pct": round(bet["home_win_pct"], 4),
                "pred_total": round(bet["avg_total_runs"], 2),
                "pred_margin": round(bet["away_exp_runs"] - bet["home_exp_runs"], 2),
                "status": "settled", **g,
            })
        print(f"  {D}: {len(by_date[D])} games")
    _save_log(pd.DataFrame(rows, columns=COLUMNS))
    print(f"✅ Seeded {len(rows)} graded games -> {LOG_FILE}")


def predict(day):
    """Log (as 'open') every game on the slate for `day`."""
    _, pa = app._load_data()
    pa_w = app.add_recency_weights(pa, day)
    league = app._compute_league_baselines(pa_w)
    bullpens = app.compute_bullpen_profiles(pa_w, league)

    games = app.get_games_for_date(day)
    log = _load_log()
    existing = set(zip(log["game_pk"], log["model_version"])) if len(log) else set()
    new = []
    for g in games:
        if (g["id"], app.MODEL_VERSION) in existing:
            continue
        cards = app.get_game_lineups(g)
        bet = _predict_one(cards["away_lineup"], cards["home_lineup"],
                           cards["away_pitcher"], cards["home_pitcher"],
                           app.TEAM_NAME_TO_ABBR.get(g["home_team_name"]),
                           app.TEAM_NAME_TO_ABBR.get(g["away_team_name"]),
                           pa_w, league, bullpens, 800)
        new.append({
            "pred_date": day, "game_pk": int(g["id"]),
            "away_team": app.TEAM_NAME_TO_ABBR.get(g["away_team_name"]),
            "home_team": app.TEAM_NAME_TO_ABBR.get(g["home_team_name"]),
            "model_version": app.MODEL_VERSION,
            "pred_home_win_pct": round(bet["home_win_pct"], 4),
            "pred_total": round(bet["avg_total_runs"], 2),
            "pred_margin": round(bet["away_exp_runs"] - bet["home_exp_runs"], 2),
            "status": "open", "actual_away": np.nan, "actual_home": np.nan,
            "home_won": np.nan, "correct": np.nan, "brier": np.nan,
            "total_abs_err": np.nan, "margin_abs_err": np.nan,
        })
    if new:
        _save_log(pd.concat([log, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True))
    print(f"✅ Logged {len(new)} new predictions for {day}")


def grade():
    """Settle any 'open' predictions whose games have finished."""
    log = _load_log()
    if len(log) == 0:
        print("No log yet."); return
    graded = 0
    for idx in log.index[log["status"] == "open"]:
        res = get_final_score(int(log.at[idx, "game_pk"]))
        if res is None:
            continue
        g = _grade(log.at[idx, "pred_home_win_pct"], log.at[idx, "pred_total"],
                   log.at[idx, "pred_margin"], res[0], res[1])
        for k, v in g.items():
            log.at[idx, k] = v
        log.at[idx, "status"] = "settled"
        graded += 1
    _save_log(log)
    print(f"✅ Graded {graded} finished games")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("seed")
    s.add_argument("--start", default="2026-05-01")
    s.add_argument("--end", default="2026-07-20")
    s.add_argument("--sims", type=int, default=400)
    s.add_argument("--max-games", type=int, default=400)
    p = sub.add_parser("predict"); p.add_argument("--date", default=date.today().isoformat())
    sub.add_parser("grade")
    sub.add_parser("daily")
    a = ap.parse_args()

    if a.cmd == "seed":
        seed(a.start, a.end, a.sims, a.max_games)
    elif a.cmd == "predict":
        predict(a.date)
    elif a.cmd == "grade":
        grade()
    elif a.cmd == "daily":
        grade()                      # settle yesterday's games first
        predict(date.today().isoformat())
    else:
        ap.print_help()
