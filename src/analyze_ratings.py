"""
analyze_ratings.py - means per variant, rater agreement and failure candidates.

    python src/analyze_ratings.py                       # reads rating/rater_*.csv + rating_key.csv

Scores are 1-5 for faithfulness (every claim supported), usefulness, clarity, hedging.
Agreement = mean pairwise quadratic-weighted Cohen kappa over all items, per criterion and overall.
"""
from __future__ import annotations

import argparse, itertools, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain as E                                          # noqa: E402

DIR = E.OUT / "rating"
CRIT = ["faithfulness", "usefulness", "clarity", "hedging"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", default="nada,nour,yassin")
    a = ap.parse_args()
    key = pd.read_csv(DIR / "rating_key.csv")
    R = {n: pd.read_csv(DIR / f"rater_{n}.csv").set_index("item_id") for n in a.raters.split(",")}
    for n, d in R.items():
        miss = d[CRIT].isna().any(axis=1).sum()
        if miss:
            sys.exit(f"{n} still has {miss} items with empty scores")
        assert d[CRIT].apply(lambda c: c.between(1, 5)).all().all(), f"{n}: scores must be 1-5"

    allr = pd.concat({n: d[CRIT] for n, d in R.items()}, names=["rater", "item_id"]).reset_index()
    allr = allr.merge(key[["item_id", "variant", "sample_id", "outcome"]], on="item_id")
    means = allr.groupby("variant")[CRIT].mean().round(2)
    print("=== mean rating (1-5), all raters ===\n", means.to_string())

    kap = {}
    for c in CRIT:
        ks = [cohen_kappa_score(R[x].loc[key.item_id, c].astype(int), R[y].loc[key.item_id, c].astype(int),
                                weights="quadratic", labels=[1, 2, 3, 4, 5])
              for x, y in itertools.combinations(R, 2)]
        kap[c] = float(np.nanmean(ks))
    kap["mean_all"] = float(np.mean(list(kap.values())))
    print("\n=== mean pairwise weighted kappa ===\n", {k: round(v, 2) for k, v in kap.items()})
    means.assign(kappa_mean=round(kap["mean_all"], 2)).to_csv(E.OUT / "human_rating_table.csv")
    pd.Series(kap).round(3).to_csv(E.OUT / "rater_agreement.csv", header=["weighted_kappa"])

    worst = (allr[allr.variant != "plain"].groupby(["item_id", "sample_id", "variant", "outcome"])["faithfulness"]
             .mean().sort_values().head(6).reset_index())
    print("\n=== failure-case candidates (lowest mean faithfulness, grounded variants) ===\n", worst.to_string(index=False))
    worst.to_csv(E.OUT / "failure_candidates.csv", index=False)


if __name__ == "__main__":
    main()
