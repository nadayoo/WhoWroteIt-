"""
phase2_analysis.py  -  Nada's remaining Phase 2 work (trimmed):
  A. sanity-check the feature files already built by features.py
  B. human-vs-AI table (top 10 features) + 4 plots
  C. length-only baseline (closes the Phase 1 length-bias decision)

Run from the repo root:
    python phase2_analysis.py --data data/processed_aig --feats data/features_aig --out results/phase2

Rules followed: everything is computed on TRAIN (the analysis never looks at test);
the length baseline is scored on VALIDATION only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

EXPECTED = {"train": 4670, "val": 1005, "test": 1009}   # from the Phase 1 report
COL = {0: "#2E75B6", 1: "#C55A11"}                       # human = blue, AI = orange
NAME = {0: "Human", 1: "AI"}


# ------------------------------------------------------------------ A. sanity checks
def check_features(splits: dict, feats: dict) -> pd.DataFrame:
    rows, cols0 = [], None
    for s in ("train", "val", "test"):
        d, f = splits[s], feats[s]
        cols = list(f.columns)
        cols0 = cols0 or cols
        rows.append({
            "split": s, "rows_split": len(d), "rows_features": len(f),
            "expected_rows": EXPECTED[s],
            "ids_match_in_order": bool((d["sample_id"].to_numpy() == f["sample_id"].to_numpy()).all()),
            "same_columns_as_train": cols == cols0,
            "n_feature_cols": len(cols) - 1,
            "all_nan_cols": int(f.drop(columns="sample_id").isna().all().sum()),
            "ast_ok_rate": round(float(f["ast_ok"].mean()), 3),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. comparison
def rank_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Per-feature AUC for 'AI vs human' (0.5 = no signal). Computed on the frame given (train)."""
    y = df["label"].to_numpy()
    out = []
    for c in feat_cols:
        x = df[c].astype(float)
        x = x.fillna(x.median())
        if x.nunique() < 2:
            continue
        auc = roc_auc_score(y, x)
        out.append({"feature": c,
                    "mean_human": x[y == 0].mean(), "mean_ai": x[y == 1].mean(),
                    "auc": auc, "strength": abs(auc - 0.5)})
    return pd.DataFrame(out).sort_values("strength", ascending=False).reset_index(drop=True)


def _savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_plots(df: pd.DataFrame, ranked: pd.DataFrame, out: Path) -> list[str]:
    paths = []
    # 1. length distribution
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.logspace(np.log10(max(df.n_chars_f.min(), 1)), np.log10(df.n_chars_f.max()), 40)
    for lab in (0, 1):
        ax.hist(df.loc[df.label == lab, "n_chars_f"], bins=bins, alpha=.6, color=COL[lab], label=NAME[lab])
    ax.set_xscale("log"); ax.set_xlabel("characters per sample (log scale)"); ax.set_ylabel("samples")
    ax.set_title("Code length: human vs AI (train)"); ax.legend()
    p = out / "plot1_length.png"; _savefig(fig, p); paths.append(str(p))

    # 2. comments: share of samples with at least one comment, and mean comment ratio
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))
    has = df.assign(h=(df.n_comments > 0)).groupby("label")["h"].mean()
    axs[0].bar([NAME[i] for i in has.index], has.values, color=[COL[i] for i in has.index])
    axs[0].set_ylabel("share with >= 1 comment"); axs[0].set_ylim(0, 1)
    m = df.groupby("label")["comment_line_ratio"].mean()
    axs[1].bar([NAME[i] for i in m.index], m.values, color=[COL[i] for i in m.index])
    axs[1].set_ylabel("mean comment-line ratio")
    fig.suptitle("Comments: human vs AI (train)")
    p = out / "plot2_comments.png"; _savefig(fig, p); paths.append(str(p))

    # 3. AST depth
    fig, ax = plt.subplots(figsize=(6.5, 4))
    d = df[df.ast_ok == 1]
    top = int(d.ast_max_depth.quantile(.99))
    for lab in (0, 1):
        ax.hist(d.loc[d.label == lab, "ast_max_depth"].clip(upper=top), bins=range(0, top + 2),
                alpha=.6, color=COL[lab], label=NAME[lab], density=True)
    ax.set_xlabel("maximum AST depth"); ax.set_ylabel("share of samples")
    ax.set_title("AST depth: human vs AI (train)"); ax.legend()
    p = out / "plot3_ast_depth.png"; _savefig(fig, p); paths.append(str(p))

    # 4. strongest feature, split by LLM (human shown as a group)
    skip = {"n_lines_f", "n_chars_f"}
    feat = next((f for f in ranked.feature if f not in skip), "ast_max_depth")
    if "llm_name" in df.columns:
        order = [g for g in ["human", "codestral", "codellama", "gemini"] if g in set(df.llm_name)]
        data = [df.loc[df.llm_name == g, feat].dropna() for g in order]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        bp = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
        for patch, g in zip(bp["boxes"], order):
            patch.set_facecolor(COL[0] if g == "human" else COL[1]); patch.set_alpha(.6)
        ax.set_ylabel(feat); ax.set_title(f"{feat} by author (train)")
        p = out / "plot4_feature_by_llm.png"; _savefig(fig, p); paths.append(str(p))
    return paths, feat


# ------------------------------------------------------------------ C. length-only baseline
def length_baseline(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    def X(d):  # length is the ONLY information: log-chars and log-lines
        return np.column_stack([np.log1p(d["n_chars_f"]), np.log1p(d["n_lines_f"])])
    sc = StandardScaler().fit(X(tr))
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(sc.transform(X(tr)), tr["label"])
    s = clf.predict_proba(sc.transform(X(va)))[:, 1]
    pred = (s >= 0.5).astype(int); y = va["label"].to_numpy()
    res = {
        "length_only": {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
                        "recall": recall_score(y, pred), "f1": f1_score(y, pred), "auc": roc_auc_score(y, s)},
        "always_human": {"accuracy": float((y == 0).mean()), "f1": 0.0, "auc": 0.5},
    }
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_aig")
    ap.add_argument("--feats", default="data/features_aig")
    ap.add_argument("--out", default="results/phase2")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    splits = {s: pd.read_parquet(Path(a.data) / f"{s}.parquet") for s in ("train", "val", "test")}
    feats = {s: pd.read_parquet(Path(a.feats) / f"features_{s}.parquet") for s in ("train", "val", "test")}

    print("\n=== A. feature file checks ===")
    chk = check_features(splits, feats); print(chk.to_string(index=False))
    chk.to_csv(out / "feature_checks.csv", index=False)

    meta_cols = [c for c in ["sample_id", "label", "llm_name", "scenario", "problem_id"] if c in splits["train"].columns]
    tr = splits["train"][meta_cols].merge(feats["train"], on="sample_id")
    va = splits["val"][meta_cols].merge(feats["val"], on="sample_id")
    feat_cols = [c for c in feats["train"].columns if c != "sample_id"]

    print("\n=== B. top features, human vs AI (train) ===")
    ranked = rank_features(tr, feat_cols)
    top10 = ranked.head(10).round(3); print(top10.to_string(index=False))
    ranked.to_csv(out / "feature_ranking_all.csv", index=False)
    top10.to_csv(out / "top10_features.csv", index=False)
    paths, feat4 = make_plots(tr, ranked, out)
    print("plots:", *paths, sep="\n  ")

    print("\n=== C. length-only baseline (validation) ===")
    res = length_baseline(tr, va)
    print(json.dumps(res, indent=2))
    (out / "length_baseline.json").write_text(json.dumps(res, indent=2))
    la = res["length_only"]
    print(f"\nSentence for the report: a model that sees only the length scores "
          f"accuracy {la['accuracy']:.3f}, F1 {la['f1']:.3f}, AUC {la['auc']:.3f} on validation "
          f"(always-human: accuracy {res['always_human']['accuracy']:.3f}).")


if __name__ == "__main__":
    main()