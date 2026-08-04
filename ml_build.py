"""
ML Phase — Stage 3: train the final logistic model on the full dataset and save it.

The app loads ml_model.pkl and blends its win% 50/50 with the simulation. Uses the
SAME feature list (app.ML_RAW_FEATURES) the app computes at prediction time.

Run:  python3 ml_build.py   (re-run whenever ml_dataset.csv is rebuilt)
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import app

df = pd.read_csv("ml_dataset.csv").dropna()
X = df[app.ML_RAW_FEATURES].values
y = df["home_won"].values
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, C=0.3)).fit(X, y)
with open("ml_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"✅ Trained logistic on {len(df)} games ({len(app.ML_RAW_FEATURES)} features) -> ml_model.pkl")
