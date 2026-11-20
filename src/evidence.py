"""evidence.py - build the evidence object for ONE code file (Phase 4+5, evidence builder).

    from evidence import build_evidence
    ev = build_evidence(code_string)

Returns the same shape as tests/mock_evidence.json, minus the two parts that belong to the retrieval layer
(`problem` and `retrieved`, which explain() adds from retrieve.py):

    detector : p_ai, threshold, verdict ("ai"/"human"), confidence (low / medium / high / inconclusive), model
    features : F1..F5   the 5 features that pushed the detector hardest for THIS file
    ngrams   : N1..N3   the 3 strongest token n-grams present in the file (second model, supporting evidence)
    caveats  : C1, ...  fixed texts triggered by rules
    _debug   : raw numbers for evaluation (keys starting with "_" must NOT be sent to the LLM)

How the pieces are made
    verdict     saved model models/best.joblib and its saved threshold (never changed here)
    confidence  Platt-calibrated probability and the band cut-offs from confidence.py (fitted on validation only)
    features    XGBoost per-feature contributions (TreeSHAP: positive = pushes toward AI, negative = toward human)
    reference   class means come from the TRAIN split (results/phase2/feature_ranking_all.csv)
    ngrams      weights of the saved token n-gram logistic regression (models/tfidf_token_lr.joblib)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # the saved models need wwi_models / features / wwi_normalize

import joblib                                        # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from confidence import CUTS_PATH, confidence_info, load_json   # noqa: E402

MODELS = ROOT / "models"
DICTIONARY = ROOT / "data" / "feature_dictionary.csv"
RANKING = ROOT / "results" / "phase2" / "feature_ranking_all.csv"

N_FEATURES = 5
N_NGRAMS = 3
SPACING_FEATURES = ("op_spaced_ratio", "comma_space_ratio")
MIN_LINES = 5

C_STANDING = ("The detector misses about 4 in 10 AI files and about 3 in 10 of its AI verdicts are false alarms; "
              "a verdict is never proof of authorship.")
C_UNPARSABLE = "The code is not valid Python 3, so syntax-tree (AST) evidence is unavailable."
C_SPACING = "Consistent operator or comma spacing does not prove AI authorship; many human authors also space consistently."
C_CLOSE = "The score is close to the decision threshold."
C_SHORT = "The file is very short, so there is little style information to judge."

_CACHE: dict = {}


# ------------------------------------------------------------------ resources
def _resources() -> dict:
    if not _CACHE:
        best = joblib.load(MODELS / "best.joblib")
        pipe = best["pipeline"]
        tok = joblib.load(MODELS / "tfidf_token_lr.joblib")["pipeline"]
        dic = pd.read_csv(DICTIONARY).set_index("feature")
        rank = pd.read_csv(RANKING).set_index("feature")
        _CACHE.update({
            "model_name": best["name"],
            "feat": pipe.named_steps["feat"],
            "clf": pipe.named_steps["clf"],
            "cols": list(pipe.named_steps["feat"].cols_),
            "tok": tok,
            "ngram_names": np.asarray(tok.named_steps["vec"].get_feature_names_out()),
            "meaning": dic["meaning"].to_dict(),
            "group": dic["group"].to_dict(),
            "mean_human": rank["mean_human"].to_dict(),
            "mean_ai": rank["mean_ai"].to_dict(),
        })
    return _CACHE


def _contributions(clf, X: np.ndarray) -> np.ndarray:
    """XGBoost TreeSHAP values for one row: one number per feature, last element = bias. Positive = toward AI."""
    import xgboost as xgb
    return clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]


def _num(v):
    v = float(v)
    if np.isnan(v):
        return None
    return int(v) if v.is_integer() else round(v, 3)


# ------------------------------------------------------------------ the three evidence blocks
def _feature_items(x_row: np.ndarray, contrib: np.ndarray, ast_ok: bool):
    r = _resources()
    cols = r["cols"]
    items, debug = [], []
    for j in np.argsort(-np.abs(contrib[:-1])):
        name = cols[j]
        v = x_row[j]
        if abs(contrib[j]) < 1e-9:
            break
        if np.isnan(v) or name == "ast_ok":
            continue
        if not ast_ok and r["group"].get(name) == "ast":      # AST columns are missing for unparsable code
            continue
        if name not in r["mean_human"]:
            continue
        mh, ma = r["mean_human"][name], r["mean_ai"][name]
        points = "AI" if contrib[j] > 0 else "human"
        closer_to = "human" if abs(v - mh) < abs(v - ma) else "AI"
        items.append({
            "id": f"F{len(items) + 1}", "name": name, "value": _num(v),
            "human_mean": _num(mh), "ai_mean": _num(ma),
            "points_toward": points, "meaning": r["meaning"].get(name, name),
        })
        debug.append({"name": name, "contribution": round(float(contrib[j]), 4),
                      "value_closer_to": closer_to, "agrees_with_class_means": closer_to == points})
        if len(items) == N_FEATURES:
            break
    return items, debug


def _ngram_items(code: str):
    r = _resources()
    tok = r["tok"]
    Xs = tok.named_steps["vec"].transform(tok.named_steps["doc"].transform([code]))
    if Xs.nnz == 0:
        return []
    weights = Xs.data * tok.named_steps["clf"].coef_[0][Xs.indices]
    items = []
    for o in np.argsort(-np.abs(weights))[:N_NGRAMS]:
        items.append({"id": f"N{len(items) + 1}", "ngram": str(r["ngram_names"][Xs.indices[o]]),
                      "points_toward": "AI" if weights[o] > 0 else "human"})
    return items


def _caveats(ast_ok: bool, features: list, band: str, n_lines: int):
    texts = [C_STANDING]
    if not ast_ok:
        texts.append(C_UNPARSABLE)
    if any(f["name"] in SPACING_FEATURES and f["points_toward"] == "AI" for f in features):
        texts.append(C_SPACING)
    if band == "inconclusive":
        texts.append(C_CLOSE)
    if n_lines < MIN_LINES:
        texts.append(C_SHORT)
    return [{"id": f"C{i + 1}", "text": t} for i, t in enumerate(texts)]


# ------------------------------------------------------------------ main entry point
def build_evidence(code: str) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("build_evidence needs a non-empty code string")
    r = _resources()
    X = r["feat"].transform([code])
    x_row = X[0]
    p_raw = float(r["clf"].predict_proba(X)[0, 1])
    info = confidence_info(p_raw)

    cols = r["cols"]
    ast_ok = bool(x_row[cols.index("ast_ok")] == 1) if "ast_ok" in cols else True
    features, feat_debug = _feature_items(x_row, _contributions(r["clf"], X), ast_ok)
    n_lines = len(code.strip().splitlines())

    t_cal = load_json(CUTS_PATH)["t_cal"]               # threshold on the same (calibrated) scale as p_ai
    return {
        "detector": {"p_ai": round(info["p_ai_calibrated"], 3), "threshold": round(t_cal, 3),
                     "verdict": info["verdict"], "confidence": info["band"], "model": r["model_name"]},
        "features": features,
        "ngrams": _ngram_items(code),
        "caveats": _caveats(ast_ok, features, info["band"], n_lines),
        "_debug": {"p_ai_raw": round(p_raw, 4), "verdict_confidence": round(info["verdict_confidence"], 4),
                   "ast_ok": ast_ok, "n_lines": n_lines, "feature_checks": feat_debug},
    }
