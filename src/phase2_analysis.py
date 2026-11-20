"""
features.py already turned each code sample into 91 numbers (the "features").
This script does three jobs with those numbers:

  A. CHECK  : make sure the feature files are correct (right size, right order, nothing empty).
  B. COMPARE: find which numbers differ most between human and AI code, and draw 4 plots.
  C. LENGTH : test "can a model tell human from AI just by how long the code is?"
              (if yes, our detector might be cheating on length instead of learning style).

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

# ---------------------------------------------------------------- constants (fixed facts used below)
EXPECTED = {"train": 4670, "val": 1005, "test": 1009}   
COL = {0: "#2E75B6", 1: "#C55A11"}                       # plot colours: label 0 (human) = blue, label 1 (AI) = orange
NAME = {0: "Human", 1: "AI"}                             # turns the label number into a readable word for plots


# ================================================================ PART A: CHECK THE FEATURE FILES
def check_features(splits: dict, feats: dict) -> pd.DataFrame:
    """Checks each split's feature file against its original split file. Returns a summary table."""
    rows, cols0 = [], None                               # rows = our result list; cols0 = remembers train's column names
    for s in ("train", "val", "test"):                   # repeat the checks for each of the three splits
        d, f = splits[s], feats[s]                       # d = the original split table, f = its feature table
        cols = list(f.columns)                           # the column names of this feature table
        cols0 = cols0 or cols                            # first time (train): save its columns as the reference
        rows.append({                                    # add one summary row for this split
            "split": s,                                  # which split this row describes
            "rows_split": len(d),                        # how many rows the original split has
            "rows_features": len(f),                     # how many rows the feature file has (should be equal)
            "expected_rows": EXPECTED[s],                # how many Phase 1 said there should be
            # are the sample IDs identical AND in the same order? (if not, features would be glued to the wrong samples)
            "ids_match_in_order": bool((d["sample_id"].to_numpy() == f["sample_id"].to_numpy()).all()),
            "same_columns_as_train": cols == cols0,      # does this split have exactly the same columns as train?
            "n_feature_cols": len(cols) - 1,             # number of feature columns (minus 1 for the sample_id column)
            # count columns where EVERY value is missing (such a column is useless)
            "all_nan_cols": int(f.drop(columns="sample_id").isna().all().sum()),
            # share of samples where the syntax tree built successfully (ast_ok is 1 or 0, so the mean is a percentage)
            "ast_ok_rate": round(float(f["ast_ok"].mean()), 3),
        })
    return pd.DataFrame(rows)                            # turn the list of summary rows into one tidy table


# ================================================================ PART B: HUMAN vs AI COMPARISON
def rank_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """For each feature, ask: how well does THIS number alone separate AI from human? (uses AUC)"""
    y = df["label"].to_numpy()                           # the true answers: 0 = human, 1 = AI
    out = []                                             # will collect one result row per feature
    for c in feat_cols:                                  # go through the features one at a time
        x = df[c].astype(float)                          # this feature's values, as plain numbers
        x = x.fillna(x.median())                         # fill missing values with the middle value so AUC can run
        if x.nunique() < 2:                              # if every sample has the same value...
            continue                                     # ...the feature can't separate anything, so skip it
        auc = roc_auc_score(y, x)                        # AUC: chance a random AI sample scores higher than a random human one
        out.append({"feature": c,                        # which feature
                    "mean_human": x[y == 0].mean(),      # its average value among human samples
                    "mean_ai": x[y == 1].mean(),         # its average value among AI samples
                    "auc": auc,                          # 0.5 = useless, >0.5 = higher in AI, <0.5 = higher in human
                    "strength": abs(auc - 0.5)})         # distance from 0.5 = how strong the signal is, in either direction
    # sort strongest first, and renumber the rows from 0
    return pd.DataFrame(out).sort_values("strength", ascending=False).reset_index(drop=True)


def _savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)   # tidy spacing, save as image (150 dpi = sharp), free memory


def make_plots(df: pd.DataFrame, ranked: pd.DataFrame, out: Path) -> list[str]:
    paths = []                                           # will list the image files we create

    # ---- PLOT 1: how long is the code? (human vs AI)
    fig, ax = plt.subplots(figsize=(6.5, 4))             # new blank figure, 6.5 x 4 inches; ax = the drawing area
    # histogram bin edges, spaced on a log scale (short and long code both visible). 40 bins from shortest to longest.
    bins = np.logspace(np.log10(max(df.n_chars_f.min(), 1)), np.log10(df.n_chars_f.max()), 40)
    for lab in (0, 1):                                   # draw human first, then AI, on the same chart
        ax.hist(df.loc[df.label == lab, "n_chars_f"],    # the character counts of just this class
                bins=bins, alpha=.6,                     # same bins for both; alpha .6 = see-through so overlaps show
                color=COL[lab], label=NAME[lab])         # blue/orange, and a legend name
    ax.set_xscale("log")                                 # log scale on the x axis
    ax.set_xlabel("characters per sample (log scale)"); ax.set_ylabel("samples")   # axis titles
    ax.set_title("Code length: human vs AI (train)"); ax.legend()                  # chart title and legend box
    p = out / "plot1_length.png"; _savefig(fig, p); paths.append(str(p))           # save it and remember the filename

    # ---- PLOT 2: comments (two small charts side by side)
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))        # one row, two charts
    # per class: what share of samples have at least one comment? (True/False averaged = a percentage)
    has = df.assign(h=(df.n_comments > 0)).groupby("label")["h"].mean()
    axs[0].bar([NAME[i] for i in has.index], has.values, color=[COL[i] for i in has.index])  # left chart: two bars
    axs[0].set_ylabel("share with >= 1 comment"); axs[0].set_ylim(0, 1)   # label it; y axis runs 0% to 100%
    m = df.groupby("label")["comment_line_ratio"].mean() # per class: average share of lines that are comments
    axs[1].bar([NAME[i] for i in m.index], m.values, color=[COL[i] for i in m.index])        # right chart: two bars
    axs[1].set_ylabel("mean comment-line ratio")         # label the right chart
    fig.suptitle("Comments: human vs AI (train)")        # one title above both charts
    p = out / "plot2_comments.png"; _savefig(fig, p); paths.append(str(p))

    # ---- PLOT 3: how deeply nested is the program's syntax tree?
    fig, ax = plt.subplots(figsize=(6.5, 4))
    d = df[df.ast_ok == 1]                               # only samples whose syntax tree built (others have no depth)
    top = int(d.ast_max_depth.quantile(.99))             # the depth that 99% of samples stay under (hides extreme outliers)
    for lab in (0, 1):
        ax.hist(d.loc[d.label == lab, "ast_max_depth"].clip(upper=top),   # this class's depths, capped at 'top'
                bins=range(0, top + 2),                  # one bar per whole-number depth
                alpha=.6, color=COL[lab], label=NAME[lab],
                density=True)                            # density: bars show SHARE of the class, so size difference doesn't matter
    ax.set_xlabel("maximum AST depth"); ax.set_ylabel("share of samples")
    ax.set_title("AST depth: human vs AI (train)"); ax.legend()
    p = out / "plot3_ast_depth.png"; _savefig(fig, p); paths.append(str(p))

    # ---- PLOT 4: the strongest feature, shown separately for each author (human + 3 LLMs)
    skip = {"n_lines_f", "n_chars_f"}                    # don't pick the plain length features (plot 1 already covers length)
    # take the first feature in the ranking that isn't a length feature (fallback: ast_max_depth)
    feat = next((f for f in ranked.feature if f not in skip), "ast_max_depth")
    if "llm_name" in df.columns:                         # only possible if the table says which LLM wrote each sample
        order = [g for g in ["human", "codestral", "codellama", "gemini"] if g in set(df.llm_name)]  # left-to-right order
        data = [df.loc[df.llm_name == g, feat].dropna() for g in order]   # the feature's values for each author, blanks removed
        fig, ax = plt.subplots(figsize=(6.5, 4))
        # box plot: box = middle 50% of values, line inside = median, whiskers = typical range. showfliers=False hides outlier dots.
        bp = ax.boxplot(data, labels=order, showfliers=False, patch_artist=True)
        for patch, g in zip(bp["boxes"], order):         # colour each box
            patch.set_facecolor(COL[0] if g == "human" else COL[1]); patch.set_alpha(.6)   # human blue, LLMs orange
        ax.set_ylabel(feat); ax.set_title(f"{feat} by author (train)")
        p = out / "plot4_feature_by_llm.png"; _savefig(fig, p); paths.append(str(p))
    return paths, feat                                   # give back the image names and which feature plot 4 used


# ================================================================ PART C: CAN LENGTH ALONE TELL HUMAN FROM AI?
def length_baseline(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    def X(d):                                            # Each sample is described by just two numbers: its character count and its line count.
        return np.column_stack([np.log1p(d["n_chars_f"]), np.log1p(d["n_lines_f"])])   # log of chars and of lines
    sc = StandardScaler().fit(X(tr))                     # learn the average and spread of the lengths FROM TRAIN ONLY
    # a simple yes/no model. "balanced" = count the rarer AI class more, so it isn't ignored. random_state=42 = repeatable.
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(sc.transform(X(tr)), tr["label"])            # TRAIN the model on the training set's lengths and answers
    s = clf.predict_proba(sc.transform(X(va)))[:, 1]     # for each VALIDATION sample: probability it is AI
    pred = (s >= 0.5).astype(int)                        # turn probability into a yes/no: 50% or more = "AI"
    y = va["label"].to_numpy()                           # the true answers for validation
    res = {
        "length_only": {                                 # how the length-only model scored
            "accuracy": accuracy_score(y, pred),         # share of all answers that were right
            "precision": precision_score(y, pred, zero_division=0),   # of the ones called AI, how many really were
            "recall": recall_score(y, pred),             # of the real AI samples, how many it found
            "f1": f1_score(y, pred),                     # one number balancing precision and recall
            "auc": roc_auc_score(y, s)},                 # how well its scores rank AI above human
        # reference point: a lazy model that always says "human" (accuracy = share of human samples, F1 = 0, AUC = 0.5)
        "always_human": {"accuracy": float((y == 0).mean()), "f1": 0.0, "auc": 0.5},
    }
    return res


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