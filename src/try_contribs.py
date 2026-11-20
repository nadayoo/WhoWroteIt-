import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))        # the saved model needs wwi_models, features, wwi_normalize

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

bundle = joblib.load(ROOT / "models" / "best.joblib")
pipe, thr = bundle["pipeline"], bundle["threshold"]
print("model:", bundle["name"], "| threshold:", round(thr, 4))
print("pipeline steps:", [name for name, _ in pipe.steps])

feat, clf = pipe.named_steps["feat"], pipe.named_steps["clf"]
cols = list(feat.cols_)
print("number of features:", len(cols), "| first 5:", cols[:5])

val = pd.read_parquet(ROOT / "data" / "processed_aig" / "val.parquet")
code = val["code_raw"].iloc[0]
X = feat.transform([code])
p = float(pipe.predict_proba([code])[0, 1])
print("P(AI):", round(p, 4))

contrib = clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
print("contribution vector length:", len(contrib), "(expect features + 1)")
print("sigmoid(sum of contributions):", round(float(1 / (1 + np.exp(-contrib.sum()))), 4), "(should equal P(AI))")

top = pd.Series(contrib[:-1], index=cols).sort_values(key=abs, ascending=False).head(5)
print("\ntop 5 features for this file (positive = pushes toward AI):")
print(top.round(3))