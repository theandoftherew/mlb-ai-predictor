"""
ML Phase — Stage 2: train models and test whether ML beats the simulation.

Uses a TIME-ORDERED holdout (train on earlier games, test on later) so there's no
leakage. Compares, on the same held-out games:
  * SIM ALONE     — the simulation's win% (our current model)
  * ML (raw)      — logistic reg / gradient boosting on features, WITHOUT the sim
  * ML (stacked)  — same but WITH the sim's output as a feature (correct the sim)

The honest question: does any ML variant beat the sim's Brier/accuracy?

Run:  python3 ml_train.py
"""
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

import ml_dataset as md

SIM_COLS = ["sim_home_wp", "sim_total"]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def acc(p, y):
    return float(np.mean((p > 0.5) == (y == 1)) * 100)


def report(name, p, y):
    print(f"  {name:22s}  acc {acc(p,y):5.1f}%   Brier {brier(p,y):.4f}   logloss {logloss(p,y):.4f}")


def main():
    df = pd.read_csv("ml_dataset.csv").sort_values("date").reset_index(drop=True)
    df = df.dropna()
    n = len(df)
    cut = int(n * 0.70)                      # earlier 70% train, later 30% test (no leakage)
    train, test = df.iloc[:cut], df.iloc[cut:]
    y_tr, y_te = train["home_won"].values, test["home_won"].values
    print(f"Dataset: {n} games ({cut} train / {n-cut} test). Test home-win rate: {y_te.mean()*100:.1f}%\n")

    raw_feats = [f for f in md.FEATURES if f not in SIM_COLS]
    all_feats = md.FEATURES

    print("=== WIN PREDICTION (held-out games) ===")
    # 1) Baseline: the simulation alone
    report("SIM alone (baseline)", test["sim_home_wp"].values, y_te)

    # 2) ML from raw features only (no sim)
    for label, feats in [("ML raw", raw_feats), ("ML stacked (+sim)", all_feats)]:
        Xtr, Xte = train[feats].values, test[feats].values
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, C=0.3)).fit(Xtr, y_tr)
        report(f"{label} · logistic", lr.predict_proba(Xte)[:, 1], y_te)
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                            learning_rate=0.05, l2_regularization=1.0).fit(Xtr, y_tr)
        report(f"{label} · gbm", gb.predict_proba(Xte)[:, 1], y_te)

    print("\nRead: lower Brier/logloss = better. If no ML row clearly beats 'SIM alone',")
    print("the simulation is already at/near the ceiling for this dataset size.")


if __name__ == "__main__":
    main()
