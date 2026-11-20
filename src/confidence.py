"""confidence.py - calibrated confidence for the detector (everything is fitted on VALIDATION only).

    python src/confidence.py

Step 2 (calibration): the raw score is not a real probability (the model was trained with extra weight on the
AI class, so its scores run high). Platt scaling, a 2-parameter logistic curve, fixes that.
Step 3 (bands): files whose calibrated probability sits close to the verdict threshold are "inconclusive";
the rest get a low / medium / high confidence band. Cut-offs come from validation.

The VERDICT never changes: it still uses the saved threshold in models/best_meta.json.
Only the displayed confidence is new.

Writes:
    models/calibration.json                  Platt parameters (plain JSON, no pickle)
    models/confidence_cuts.json              threshold on the calibrated scale, inconclusive zone, band cut-offs
    results/phase45/calibration_summary.csv  ECE, Brier, AUC before and after
    results/phase45/confidence_bands_val.csv, confidence_bands_test.csv

Used by evidence.py:   from confidence import confidence_info
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase45"
CAL_PATH = ROOT / "models" / "calibration.json"
CUTS_PATH = ROOT / "models" / "confidence_cuts.json"
META_PATH = ROOT / "models" / "best_meta.json"
PREDS = ROOT / "results" / "preds_feat_xgb.parquet"
EPS = 1e-6
ZONE_COVERAGE = 0.15      # the 15% of validation files closest to the threshold are called "inconclusive"
BANDS = ["low", "medium", "high", "inconclusive"]


# ------------------------------------------------------------------ calibration (Step 2)
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(scores, y) -> dict:
    """Logistic regression on logit(score). C is huge so there is practically no regularisation."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logit(scores).reshape(-1, 1), np.asarray(y))
    return {"coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def apply_platt(cal: dict, scores):
    z = cal["coef"] * logit(scores) + cal["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: weighted average gap between claimed probability and real AI share per bin."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total, err = len(y), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return float(err)


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_claimed": float(p[m].mean()) if m.any() else np.nan,
                     "real_ai_share": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ bands (Step 3)
def verdict_and_conf(p_cal, t_cal):
    """verdict 1 = AI. conf = calibrated probability of the verdict itself (P(AI) for AI, P(human) for human)."""
    p_cal = np.asarray(p_cal, dtype=float)
    verdict = (p_cal >= t_cal).astype(int)
    conf = np.where(verdict == 1, p_cal, 1.0 - p_cal)
    return verdict, conf


def fit_bands(p_val, t_cal, coverage: float = ZONE_COVERAGE) -> dict:
    """Cut-offs from VALIDATION: inconclusive zone = the `coverage` share of files closest to the threshold;
    low / medium / high = thirds of the verdict confidence among the remaining files."""
    p_val = np.asarray(p_val, dtype=float)
    dist = np.abs(p_val - t_cal)
    delta = float(np.quantile(dist, coverage))
    _, conf = verdict_and_conf(p_val, t_cal)
    answered = dist >= delta
    q1, q2 = np.quantile(conf[answered], [1 / 3, 2 / 3])
    return {"t_cal": float(t_cal), "delta": delta, "low_max": float(q1), "high_min": float(q2),
            "zone_coverage": coverage}


def label_bands(p_cal, cuts: dict):
    p_cal = np.asarray(p_cal, dtype=float)
    verdict, conf = verdict_and_conf(p_cal, cuts["t_cal"])
    inconclusive = np.abs(p_cal - cuts["t_cal"]) < cuts["delta"]
    band = np.where(inconclusive, "inconclusive",
                    np.where(conf < cuts["low_max"], "low",
                             np.where(conf < cuts["high_min"], "medium", "high")))
    return verdict, conf, band


def band_table(y, verdict, band) -> pd.DataFrame:
    y, verdict, band = np.asarray(y), np.asarray(verdict), np.asarray(band)
    rows = []

    def row(name, m):
        ai = m & (verdict == 1)
        rows.append({"band": name, "files": int(m.sum()), "share": float(m.mean()),
                     "accuracy": float((verdict[m] == y[m]).mean()) if m.any() else np.nan,
                     "ai_verdicts": int(ai.sum()),
                     "ai_precision": float(y[ai].mean()) if ai.any() else np.nan})

    for b in BANDS:
        row(b, band == b)
    row("answered (not inconclusive)", band != "inconclusive")
    row("all files", np.ones(len(y), dtype=bool))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ used by evidence.py
def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def confidence_info(p_raw: float, cal: dict | None = None, cuts: dict | None = None) -> dict:
    """raw P(AI) from the saved model -> calibrated probability, verdict, verdict confidence and band."""
    cal = cal or load_json(CAL_PATH)
    cuts = cuts or load_json(CUTS_PATH)
    p = float(apply_platt(cal, [p_raw])[0])
    verdict, conf, band = label_bands(np.array([p]), cuts)
    return {"p_ai_raw": float(p_raw), "p_ai_calibrated": p,
            "verdict": "ai" if verdict[0] == 1 else "human",
            "verdict_confidence": float(conf[0]), "band": str(band[0])}


# ------------------------------------------------------------------ run
def main():
    preds = pd.read_parquet(PREDS)
    val, test = preds[preds.split == "val"], preds[preds.split == "test"]

    # ---- Step 2: calibration
    cal = fit_platt(val.y_score.values, val.y_true.values)
    CAL_PATH.parent.mkdir(exist_ok=True)
    CAL_PATH.write_text(json.dumps(cal, indent=2))
    print("calibration (fitted on validation only):", cal, "\n")

    rows = []
    for name, d in [("validation (used to fit)", val), ("test", test)]:
        raw, c = d.y_score.values, apply_platt(cal, d.y_score.values)
        rows.append({"set": name, "n": len(d),
                     "ECE raw": ece(d.y_true, raw), "ECE calibrated": ece(d.y_true, c),
                     "Brier raw": brier_score_loss(d.y_true, raw), "Brier calibrated": brier_score_loss(d.y_true, c),
                     "AUC raw": roc_auc_score(d.y_true, raw), "AUC calibrated": roc_auc_score(d.y_true, c)})
    table = pd.DataFrame(rows).round(4)
    print(table.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "calibration_summary.csv", index=False)

    # ---- Step 3: inconclusive zone and bands (cut-offs from validation only)
    thr = load_json(META_PATH)["threshold"]
    p_val = apply_platt(cal, val.y_score.values)
    p_test = apply_platt(cal, test.y_score.values)
    t_cal = float(apply_platt(cal, [thr])[0])
    print(f"\nsaved verdict threshold {thr:.4f} (raw) = calibrated probability {t_cal:.4f}")

    cuts = fit_bands(p_val, t_cal)
    CUTS_PATH.write_text(json.dumps(cuts, indent=2))
    print("confidence cut-offs (from validation):", {k: round(v, 4) for k, v in cuts.items()})
    print(f"  inconclusive = calibrated P(AI) within +-{cuts['delta']:.3f} of {t_cal:.3f}, "
          f"i.e. between {t_cal - cuts['delta']:.3f} and {t_cal + cuts['delta']:.3f}")

    for name, d, p in [("validation", val, p_val), ("test", test, p_test)]:
        verdict, conf, band = label_bands(p, cuts)
        # the verdict must be identical to the saved model's verdict
        same = bool((verdict == (d.y_score.values >= thr).astype(int)).all())
        t = band_table(d.y_true.values, verdict, band)
        t.to_csv(OUT / f"confidence_bands_{'val' if name == 'validation' else 'test'}.csv", index=False)
        print(f"\n--- {name}: verdict identical to the saved model: {same}")
        print(t.round(3).to_string(index=False))

    print("\nreliability on the test set after calibration (claimed vs real AI share):")
    print(reliability_table(test.y_true, p_test).round(3).to_string(index=False))


if __name__ == "__main__":
    main()