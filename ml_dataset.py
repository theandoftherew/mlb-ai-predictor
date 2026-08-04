"""
ML Phase — Stage 1: build the training dataset.

For each historical game, compute a feature row from data available BEFORE that
game (no leakage), plus the SIMULATION's own outputs (stacking features), plus the
actual result. Writes ml_dataset.csv for ml_train.py.

Run:  python3 ml_dataset.py --start 2025-05-01 --end 2026-07-20 --max-games 1200 --sims 300
"""
import warnings; warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd

import app
import backtest

FEATURES = [
    "home_off_k", "home_off_bb", "home_off_chit", "home_off_hr",
    "away_off_k", "away_off_bb", "away_off_chit", "away_off_hr",
    "home_sp_k", "home_sp_bb", "home_sp_chit", "home_sp_stam",
    "away_sp_k", "away_sp_bb", "away_sp_chit", "away_sp_stam",
    "home_pen_k", "home_pen_bb", "home_pen_chit",
    "away_pen_k", "away_pen_bb", "away_pen_chit",
    "park_runs", "park_hr",
    "sim_home_wp", "sim_total",   # the simulation's own predictions (stacking)
]


def _team_offense(lineup, pa, league, vs_throws):
    """Mean batter rates for a lineup vs the opposing starter's hand."""
    ks, bbs, chits, hrs = [], [], [], []
    for rec in lineup:
        prof = app.build_batter_profile(pa, rec["id"], rec["bats"], league)
        r = prof["split"][vs_throws]
        ks.append(r["k"]); bbs.append(r["bb"]); chits.append(r["chit"]); hrs.append(prof["hr"])
    return np.mean(ks), np.mean(bbs), np.mean(chits), np.mean(hrs)


def _pitcher_feats(pid, throws, pa, league, stamina):
    prof = app.build_pitcher_profile(pa, pid, throws, league)
    k = (prof["split"]["L"]["k"] + prof["split"]["R"]["k"]) / 2
    bb = (prof["split"]["L"]["bb"] + prof["split"]["R"]["bb"]) / 2
    chit = (prof["split"]["L"]["chit"] + prof["split"]["R"]["chit"]) / 2
    return k, bb, chit, stamina.get(pid, app.DEFAULT_STAMINA)


def _pen_feats(pen):
    if not pen:
        return np.nan, np.nan, np.nan
    k = (pen["split"]["L"]["k"] + pen["split"]["R"]["k"]) / 2
    bb = (pen["split"]["L"]["bb"] + pen["split"]["R"]["bb"]) / 2
    chit = (pen["split"]["L"]["chit"] + pen["split"]["R"]["chit"]) / 2
    return k, bb, chit


def build(start, end, max_games, sims):
    print("Loading data...")
    _, pa = app._load_data()
    pa["game_date"] = pa["game_date"].astype(str)
    test = pa[(pa["game_date"] >= start) & (pa["game_date"] <= end)]
    import random
    gids = test["game_pk"].dropna().unique().tolist()
    random.seed(0); random.shuffle(gids); gids = gids[:max_games]
    gid_date = test.groupby("game_pk")["game_date"].min()
    by_date = {}
    for g in gids:
        by_date.setdefault(gid_date[g], []).append(g)

    rows, done = [], 0
    for D in sorted(by_date):
        train = pa[pa["game_date"] < D]
        if len(train) < 30000:
            continue
        train = app.add_recency_weights(train, D)
        league = app._compute_league_baselines(train)
        bullpens = app.compute_bullpen_profiles(train, league)
        stamina = app.compute_pitcher_stamina(train)
        for gid in by_date[D]:
            info = backtest.reconstruct_game(pa[pa["game_pk"] == gid])
            if info is None:
                continue
            hsp, asp = info["home_sp"], info["away_sp"]
            park = app.PARK_FACTORS.get(backtest.ABBR_TO_TEAM.get(info["home_abbr"], ""), app.NEUTRAL_PARK)
            hp = bullpens.get(info["home_abbr"]); ap = bullpens.get(info["away_abbr"])
            _, _, bet = app.simulate_games(
                info["away_lineup"], info["home_lineup"], asp, hsp, park,
                {"temp": 70, "wind_mph": 5, "wind_dir": "Neutral"},
                league, train, simulations=sims,
                away_bullpen=ap, home_bullpen=hp, stamina=stamina)

            ho = _team_offense(info["home_lineup"], train, league, asp["throws"])
            ao = _team_offense(info["away_lineup"], train, league, hsp["throws"])
            hpf = _pitcher_feats(hsp["id"], hsp["throws"], train, league, stamina)
            apf = _pitcher_feats(asp["id"], asp["throws"], train, league, stamina)
            hpen = _pen_feats(hp); apen = _pen_feats(ap)
            row = dict(zip(FEATURES, [
                *ho, *ao, *hpf, *apf, *hpen, *apen,
                park["runs"], park["hr"], bet["home_win_pct"], bet["avg_total_runs"],
            ]))
            row["home_won"] = 1 if info["final_home"] > info["final_away"] else 0
            row["actual_total"] = info["final_home"] + info["final_away"]
            row["date"] = D
            rows.append(row)
            done += 1
            if done % 50 == 0:
                print(f"  ...{done} games")
    df = pd.DataFrame(rows)
    df.to_csv("ml_dataset.csv", index=False)
    print(f"\n✅ Built {len(df)} rows x {len(FEATURES)} features -> ml_dataset.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default="2026-07-20")
    ap.add_argument("--max-games", type=int, default=1200)
    ap.add_argument("--sims", type=int, default=300)
    a = ap.parse_args()
    build(a.start, a.end, a.max_games, a.sims)
