"""
train_models.py - Phase 3 (Nour): train, tune on VALIDATION, write prediction files, save models + evidence.

    python src/train_models.py                 # everything
    python src/train_models.py --only tfidf_token_lr,feat_rf
    python src/train_models.py --quick         # 1 setting per model, for a fast smoke test

Rules followed (plan section 3):
  * Phase 1 split files are used as they are (data/processed_aig/{train,val,test}.parquet). Never re-split.
  * Everything is fitted on TRAIN only. Hyper-parameters and the decision threshold are chosen on VAL only.
  * TEST is only *predicted* (so evaluate.py can score it). No test metric is ever computed or printed here.
  * Positive class = AI (label 1). class_weight balanced / scale_pos_weight. Seed 42.
"""
from __future__ import annotations

import argparse, itertools, json, shutil, sys, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wwi_models import SEED, model_zoo, make_pipeline          # noqa: E402

warnings.filterwarnings("ignore")
np.random.seed(SEED)

DATA = ROOT / "data" / "processed_aig"
MODELS = ROOT / "models"
RES = ROOT / "results"
OUT3 = RES / "phase3"


# ----------------------------------------------------------------------------- helpers
def best_threshold(y, s) -> float:
    """Threshold that maximises F1 on VALIDATION. Applied unchanged to test."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def metrics(y, s, thr) -> dict:
    yp = (s >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp), f1=f1_score(y, yp), auc=roc_auc_score(y, s), threshold=thr)


def write_preds(name, pipe, thr, val, test):
    rows = []
    for split, df in (("val", val), ("test", test)):
        score = pipe.predict_proba(df["code_raw"])[:, 1]
        rows.append(pd.DataFrame({"sample_id": df["sample_id"].values, "split": split,
                                  "y_true": df["label"].values, "y_score": score,
                                  "y_pred": (score >= thr).astype(int)}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RES / f"preds_{name}.parquet", index=False)


def tune(name, kind, fixed, grid, tr, va, cv, pos_w, exclude=None):
    """Try every grid point, fit on train, score on val AUC (ties -> F1). Returns best (params, pipe, val_score)."""
    keys = list(grid)
    best = None
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = {**fixed, **dict(zip(keys, vals))}
        kw = dict(cv=cv, pos_weight=pos_w)
        if exclude is not None:
            kw["exclude"] = exclude
        pipe = make_pipeline(kind, params, **kw)
        t0 = time.time()
        pipe.fit(tr["code_raw"], tr["label"])
        s = pipe.predict_proba(va["code_raw"])[:, 1]
        auc = roc_auc_score(va["label"], s)
        f1 = f1_score(va["label"], (s >= best_threshold(va["label"], s)).astype(int))
        print(f"    {name:16s} {params}  val AUC={auc:.4f}  F1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if best is None or (auc, f1) > (best[2], best[3]):
            best = (params, pipe, auc, f1)
    return best


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of model names")
    ap.add_argument("--quick", action="store_true", help="first grid value only (smoke test)")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()

    for d in (MODELS, RES, OUT3):
        d.mkdir(parents=True, exist_ok=True)

    tr = pd.read_parquet(DATA / "train.parquet")
    va = pd.read_parquet(DATA / "val.parquet")
    te = pd.read_parquet(DATA / "test.parquet")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}   AI share train {tr.label.mean():.3f}")
    assert not set(tr.problem_id) & set(va.problem_id) and not set(tr.problem_id) & set(te.problem_id)

    pos_w = float((tr.label == 0).sum() / (tr.label == 1).sum())
    cv = list(GroupKFold(n_splits=5).split(np.zeros(len(tr)), tr.label, groups=tr.problem_id))

    zoo = model_zoo()
    if a.only:
        zoo = {k: v for k, v in zoo.items() if k in a.only.split(",")}
    if a.quick:
        zoo = {k: (kind, fx, {g: v[:1] for g, v in grid.items()}) for k, (kind, fx, grid) in zoo.items()}

    fitted, table, best_params = {}, [], {}
    for name, (kind, fixed, grid) in zoo.items():
        print(f"\n=== {name} ===")
        params, pipe, auc, f1 = tune(name, kind, fixed, grid, tr, va, cv, pos_w)
        sv = pipe.predict_proba(va["code_raw"])[:, 1]
        thr = best_threshold(va["label"], sv)
        m = metrics(va["label"], sv, thr)
        write_preds(name, pipe, thr, va, te)
        joblib.dump({"pipeline": pipe, "threshold": thr, "name": name}, MODELS / f"{name}.joblib", compress=3)
        fitted[name] = (pipe, thr)
        best_params[name] = params
        table.append({"model": name, **m, "params": json.dumps(params)})
        print(f"  -> best {params} | val acc {m['accuracy']:.3f} F1 {m['f1']:.3f} AUC {m['auc']:.3f} thr {thr:.3f}")

    # ---- ablation: add the flagged trailing_ws_line_ratio back into the best feature model (plan section 5.1)
    if not a.no_ablation and "feat_xgb" in best_params:
        print("\n=== ablation: feat_xgb WITH trailing_ws_line_ratio ===")
        pipe = make_pipeline("feat_xgb", best_params["feat_xgb"], pos_weight=pos_w, exclude=())
        pipe.fit(tr["code_raw"], tr["label"])
        sv = pipe.predict_proba(va["code_raw"])[:, 1]
        thr = best_threshold(va["label"], sv)
        m = metrics(va["label"], sv, thr)
        write_preds("feat_xgb_with_trailing_ws", pipe, thr, va, te)
        joblib.dump({"pipeline": pipe, "threshold": thr, "name": "feat_xgb_with_trailing_ws"},
                    MODELS / "feat_xgb_with_trailing_ws.joblib", compress=3)
        table.append({"model": "feat_xgb_with_trailing_ws", **m, "params": json.dumps(best_params["feat_xgb"])})
        print(f"  -> val acc {m['accuracy']:.3f} F1 {m['f1']:.3f} AUC {m['auc']:.3f}")

    # ---- validation comparison table + pick the best (mean of val F1 and val AUC; ablation not eligible)
    tab = pd.DataFrame(table)
    tab["rank_score"] = (tab.f1 + tab.auc) / 2
    tab = tab.sort_values("rank_score", ascending=False).reset_index(drop=True)
    tab.to_csv(OUT3 / "val_comparison.csv", index=False)
    json.dump(best_params, open(OUT3 / "best_params.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print("\n================ VALIDATION COMPARISON (test not used) ================")
    print(tab[["model", "accuracy", "precision", "recall", "f1", "auc", "threshold"]].round(3).to_string(index=False))

    elig = tab[tab.model != "feat_xgb_with_trailing_ws"]
    best_name = elig.iloc[0].model
    shutil.copy(MODELS / f"{best_name}.joblib", MODELS / "best.joblib")
    json.dump({"best_model": best_name, "threshold": float(elig.iloc[0].threshold),
               "params": best_params[best_name], "chosen_by": "mean(val F1, val AUC)"},
              open(MODELS / "best_meta.json", "w"), indent=2)
    print(f"\nBEST MODEL (validation): {best_name}  -> models/best.joblib")

    # ---- evidence for Phase 4
    evidence(fitted, tr)
    print("\nDone. Prediction files are in results/preds_*.parquet, models in models/, evidence in results/phase3/")


def evidence(fitted, tr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "tfidf_token_lr" in fitted:                       # top 15 n-grams each way (linear model, word-level)
        pipe, _ = fitted["tfidf_token_lr"]
        names = np.array(pipe.named_steps["vec"].get_feature_names_out())
        coef = pipe.named_steps["clf"].coef_[0]
        order = np.argsort(coef)
        ai = pd.DataFrame({"ngram": names[order[::-1][:15]], "coef": coef[order[::-1][:15]], "points_to": "AI"})
        hu = pd.DataFrame({"ngram": names[order[:15]], "coef": coef[order[:15]], "points_to": "human"})
        pd.concat([ai, hu]).to_csv(OUT3 / "top_ngrams.csv", index=False)
        print("\nTop 15 n-grams -> AI:\n", ai.to_string(index=False))
        print("\nTop 15 n-grams -> human:\n", hu.to_string(index=False))
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].barh(ai.ngram[::-1], ai.coef[::-1], color="#c55a11"); ax[0].set_title("n-grams pointing to AI")
        ax[1].barh(hu.ngram[::-1], hu.coef[::-1], color="#2e75b6"); ax[1].set_title("n-grams pointing to human")
        plt.tight_layout(); plt.savefig(OUT3 / "top_ngrams.png", dpi=150); plt.close()

    for tree in ("feat_rf", "feat_xgb"):                 # top 15 feature importances (tree models)
        if tree in fitted:
            pipe, _ = fitted[tree]
            cols = list(pipe.named_steps["feat"].cols_)
            imp = pipe.named_steps["imp"] if "imp" in pipe.named_steps else None
            if imp is not None and imp.indicator_ is not None:
                cols += [f"{cols[i]}__missing" for i in imp.indicator_.features_]
            vals = pipe.named_steps["clf"].feature_importances_
            if len(vals) != len(cols):
                continue
            top = pd.Series(vals, index=cols).sort_values(ascending=False).head(15)
            top.rename("importance").to_csv(OUT3 / f"top15_importances_{tree}.csv")
            print(f"\nTop 15 feature importances ({tree}):\n", top.round(4).to_string())
            plt.figure(figsize=(7, 5)); top[::-1].plot.barh(color="#2e75b6")
            plt.title(f"Top 15 feature importances - {tree}"); plt.tight_layout()
            plt.savefig(OUT3 / f"importances_{tree}.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
