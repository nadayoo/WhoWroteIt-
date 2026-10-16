"""
build_dataset.py  -  Phase 1, Person 3 (Yassin): normalize -> filter -> dedup -> merge -> balance -> split.

Inputs  (from Person 1 and Person 2):
    aigcodeset_clean.parquet, codenet_subset.parquet, problems.parquet
Outputs (in --out, default data/processed/):
    merged_all.parquet   every kept row with flags: split, in_detector, rag_eligible
    train.parquet / val.parquet / test.parquet   detector rows only (split by problem_id)
    problems.parquet     copied through for the RAG step
    problem_split.csv    problem_id -> split
    split_report.json    all decisions + counts (copy into the report)
    build_log.csv        row count after every step

Run:
    python build_dataset.py --aig aigcodeset_clean.parquet --codenet codenet_subset.parquet \
                            --problems problems.parquet --out data/processed
    (add --minhash for the optional MinHash near-duplicate pass; needs `pip install datasketch`)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wwi_normalize import (code_hash, count_lines, has_real_code, loose_hash,
                           looks_like_py2, normalize_code, parses_ok, token_stream)

# ----------------------------------------------------------------------------- config
SEED = 42
MIN_CHARS, MAX_CHARS = 50, 5000          # same window as Person 2, now applied to EVERY source
MIN_LINES, MAX_LINES = 3, 200
DROP_PY2 = True                          # Python-2 code is ~always human in AIGCodeSet -> shortcut
MAX_HUMAN_PER_AI = 3.0                   # class-imbalance rule: cap human:AI per split ...
                                         # ... by downsampling CodeNet rows only (never AIGCodeSet)
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_SPLIT_TRIES = 500                      # random problem partitions tried; best-balanced one wins

SCHEMA_COLS = ["sample_id", "source", "submission_id", "problem_id", "language",
               "code_raw", "code_norm", "label", "llm_name", "scenario", "status",
               "n_lines", "n_chars", "code_hash", "parses_ok", "role",
               "split", "in_detector", "rag_eligible"]


# ----------------------------------------------------------------------------- steps
def load_inputs(aig_path, codenet_path):
    aig = pd.read_parquet(aig_path)
    aig["role"] = "core"
    common = ["sample_id", "source", "submission_id", "problem_id", "language", "code_raw",
              "label", "llm_name", "scenario", "status", "role"]
    if codenet_path is None:                                        # --aig-only mode
        return aig[common].copy()
    cn = pd.read_parquet(codenet_path)
    if "role" not in cn:
        cn["role"] = "train_extra"
    return pd.concat([aig[common], cn[common]], ignore_index=True)  # AIGCodeSet first = priority in dedup


def normalize_and_filter(df, log):
    df = df.copy()
    df["code_norm"] = df["code_raw"].map(normalize_code)
    df["parses_ok"] = df["code_norm"].map(parses_ok)
    log["merged_rows"] = len(df)

    # empty after normalization
    df = df[df["code_norm"].str.strip() != ""]
    log["after_drop_empty"] = len(df)

    # Python 2
    if DROP_PY2:
        py2 = np.array([looks_like_py2(c, p) for c, p in zip(df["code_norm"], df["parses_ok"])])
        log["py2_removed_human"] = int((py2 & (df["label"] == 0).to_numpy()).sum())
        log["py2_removed_ai"] = int((py2 & (df["label"] == 1).to_numpy()).sum())
        df = df[~py2]
    log["after_py2"] = len(df)

    # recompute length columns on the NORMALIZED code, then one window for all sources
    df["n_chars"] = df["code_norm"].str.len()
    df["n_lines"] = df["code_norm"].map(count_lines)
    ok = (df["n_chars"].between(MIN_CHARS, MAX_CHARS) & df["n_lines"].between(MIN_LINES, MAX_LINES)
          & df["code_norm"].map(has_real_code))
    log["length_removed_human"] = int((~ok & (df["label"] == 0)).sum())
    log["length_removed_ai"] = int((~ok & (df["label"] == 1)).sum())
    df = df[ok]
    log["after_length_filter"] = len(df)
    return df.reset_index(drop=True)


def dedup(df, log, use_minhash=False):
    """Priority when two rows collide: human beats AI (same rule as Person 1), then
    AIGCodeSet beats CodeNet, then first occurrence."""
    df = df.copy()
    df["_src_rank"] = (df["source"] != "aigcodeset").astype(int)
    df = df.sort_values(["label", "_src_rank"], kind="stable").reset_index(drop=True)

    df["code_hash"] = df["code_norm"].map(code_hash)
    n0 = len(df)
    df = df.drop_duplicates("code_hash", keep="first")
    log["dedup_exact_removed"] = n0 - len(df)

    df["_loose"] = df["code_norm"].map(loose_hash)
    n0, ai0 = len(df), int((df["label"] == 1).sum())
    df = df.drop_duplicates("_loose", keep="first")
    log["dedup_loose_removed"] = n0 - len(df)
    log["dedup_loose_removed_ai"] = ai0 - int((df["label"] == 1).sum())

    if use_minhash:
        df = minhash_dedup(df, log)

    return df.drop(columns=["_src_rank", "_loose"]).reset_index(drop=True)


def minhash_dedup(df, log, threshold=0.9, num_perm=128):
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep, n0 = [], len(df)
    for i, code in enumerate(df["code_norm"]):
        toks = token_stream(code, placeholders=False)
        shingles = {" ".join(toks[j:j + 3]) for j in range(max(len(toks) - 2, 1))}
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        if lsh.query(m):
            keep.append(False)
        else:
            lsh.insert(str(i), m)
            keep.append(True)
    df = df[np.array(keep)]
    log["dedup_minhash_removed"] = n0 - len(df)
    return df


def make_problem_split(df, rng):
    """Group split by problem_id. Only problems that have BOTH human and AI rows may go to
    val/test (otherwise the metrics are meaningless). Everything else (CodeNet Pool B) -> train.
    Tries many random partitions and keeps the one whose label / LLM / scenario mix is most
    similar across splits (= stratification at the problem level)."""
    has_ai = set(df.loc[df.label == 1, "problem_id"])
    has_h = set(df.loc[df.label == 0, "problem_id"])
    eligible = np.array(sorted(has_ai & has_h))
    others = sorted(set(df.problem_id) - set(eligible))

    ai = df[df.label == 1]
    ai_total = len(ai)
    # per-problem counts so each try is cheap
    per_p = pd.DataFrame({"n_ai": ai.groupby("problem_id").size(),
                          "n_h": df[df.label == 0].groupby("problem_id").size()}).fillna(0)
    llm = pd.crosstab(ai["problem_id"], ai["llm_name"]).reindex(per_p.index).fillna(0)
    scen = pd.crosstab(ai["problem_id"], ai["scenario"]).reindex(per_p.index).fillna(0)
    feats = pd.concat([per_p, llm.add_prefix("llm_"), scen.add_prefix("sc_")], axis=1).loc[eligible]
    M = feats.to_numpy(float)
    names = list(feats.columns)
    i_ai = names.index("n_ai")
    ai_cols = [i for i, n in enumerate(names) if n.startswith(("llm_", "sc_"))]
    tgt = np.array([SPLIT_FRACTIONS["train"], SPLIT_FRACTIONS["val"], SPLIT_FRACTIONS["test"]])
    n = len(eligible)
    cut1, cut2 = int(round(n * tgt[0])), int(round(n * (tgt[0] + tgt[1])))

    best, best_score = None, 1e9
    for _ in range(N_SPLIT_TRIES):
        perm = rng.permutation(n)
        parts = [perm[:cut1], perm[cut1:cut2], perm[cut2:]]
        sums = np.stack([M[p].sum(0) for p in parts])               # 3 x features
        share_ai = sums[:, i_ai] / max(sums[:, i_ai].sum(), 1)
        score = np.abs(share_ai - tgt).sum()                         # AI rows ~ 70/15/15
        h_share = sums[:, names.index("n_h")] / max(sums[:, names.index("n_h")].sum(), 1)
        score += np.abs(h_share - tgt).sum()                         # human rows ~ 70/15/15
        # each LLM / scenario should have the same 70/15/15 shape
        for c in ai_cols:
            col = sums[:, c]
            if col.sum() > 0:
                score += np.abs(col / col.sum() - tgt).sum() * 0.5
        if score < best_score:
            best, best_score = parts, score

    mapping = {p: "train" for p in others}
    for split, part in zip(["train", "val", "test"], best):
        for p in eligible[part]:
            mapping[p] = split
    return mapping, best_score, ai_total


def balance(df, rng, log):
    """Chosen strategy (record in report): cap human:AI at MAX_HUMAN_PER_AI inside every split
    by randomly dropping CodeNet human rows (never AIGCodeSet rows), THEN use class weights
    computed on the train split for whatever imbalance is left."""
    df["in_detector"] = df["role"] != "rag_only"          # rag_only rows are for RAG, never trained on
    for split in ["train", "val", "test"]:
        m = (df["split"] == split) & df["in_detector"]
        n_ai = int((m & (df.label == 1)).sum())
        n_h = int((m & (df.label == 0)).sum())
        allowed = int(MAX_HUMAN_PER_AI * n_ai)
        if n_h > allowed:
            cand = df.index[m & (df.label == 0) & (df.source == "codenet")]
            n_drop = min(n_h - allowed, len(cand))
            drop = rng.choice(cand, size=n_drop, replace=False)
            df.loc[drop, "in_detector"] = False
            log[f"balance_dropped_codenet_human_{split}"] = int(n_drop)
    return df


def class_weights(train):
    n = len(train)
    c = train["label"].value_counts()
    return {int(k): round(n / (2 * v), 4) for k, v in c.items()}


def run_checks(df):
    det = df[df.in_detector]
    # 1. no problem in two splits
    assert (det.groupby("problem_id")["split"].nunique() == 1).all(), "problem in >1 split"
    assert (df.groupby("problem_id")["split"].nunique() == 1).all(), "problem in >1 split (all rows)"
    # 2. no identical code in two splits
    assert (det.groupby("code_hash")["split"].nunique() == 1).all(), "same code in >1 split"
    assert df["code_hash"].is_unique, "duplicate code_hash left"
    # 3. val/test contain both classes and every LLM
    for s in ["val", "test"]:
        d = det[det.split == s]
        assert set(d.label) == {0, 1}, f"{s} is missing a class"
        assert d[d.label == 1].llm_name.nunique() == det[det.label == 1].llm_name.nunique(), f"{s} misses an LLM"
    # 4. rag_only never trained on
    assert not det.role.eq("rag_only").any()
    # 5. normalization shortcuts are gone
    assert not det.code_norm.str.match(r"^[ \t]").any(), "leading whitespace left"
    assert det.code_norm.str.endswith("\n").all() and not det.code_norm.str.endswith("\n\n").any()
    print("ALL SPLIT CHECKS PASSED")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aig", default="aigcodeset_clean.parquet")
    ap.add_argument("--codenet", default="codenet_subset.parquet")
    ap.add_argument("--problems", default="problems.parquet")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--aig-only", action="store_true",
                    help="use ONLY AIGCodeSet (no CodeNet yet). Rerun without it once Person 2 delivers.")
    ap.add_argument("--minhash", action="store_true", help="extra MinHash near-duplicate pass")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    log = {}

    df = load_inputs(args.aig, None if args.aig_only else args.codenet)
    df = normalize_and_filter(df, log)
    df = dedup(df, log, use_minhash=args.minhash)
    log["after_dedup"] = len(df)

    mapping, split_score, _ = make_problem_split(df, rng)
    df["split"] = df["problem_id"].map(mapping)
    df = balance(df, rng, log)
    df["rag_eligible"] = (df["label"] == 1) | (df["status"] == "Accepted")
    df = df[SCHEMA_COLS].sort_values(["problem_id", "sample_id"]).reset_index(drop=True)

    run_checks(df)

    det = df[df.in_detector]
    for s in ["train", "val", "test"]:
        det[det.split == s].reset_index(drop=True).to_parquet(out / f"{s}.parquet", index=False)
    df.to_parquet(out / "merged_all.parquet", index=False)
    pd.Series(mapping, name="split").rename_axis("problem_id").to_csv(out / "problem_split.csv")
    if not args.aig_only:
        pd.read_parquet(args.problems).to_parquet(out / "problems.parquet", index=False)

    log["final_all_rows"] = len(df)
    log["final_detector_rows"] = len(det)
    pd.Series(log).to_csv(out / "build_log.csv", header=["count"])

    report = {
        "seed": SEED,
        "mode": "AIGCodeSet only" if args.aig_only else "AIGCodeSet + CodeNet",
        "normalization": "line endings + BOM/NUL, rstrip each line, strip file, exactly one final newline; comments KEPT",
        "length_window": {"chars": [MIN_CHARS, MAX_CHARS], "lines": [MIN_LINES, MAX_LINES], "applied_to": "all sources"},
        "python2_dropped": DROP_PY2,
        "dedup": "exact hash on code_norm, then token-level hash ignoring comments/whitespace"
                 + (", then MinHash(3-gram shingles, J>=0.9)" if args.minhash else ""),
        "dedup_priority": "human over AI, AIGCodeSet over CodeNet",
        "imbalance_strategy": f"downsample CodeNet human rows so human:AI <= {MAX_HUMAN_PER_AI} in every split, "
                              "then class weights on train",
        "class_weights_train": class_weights(det[det.split == "train"]),
        "split": {"by": "problem_id", "fractions": SPLIT_FRACTIONS, "tries": N_SPLIT_TRIES,
                  "balance_score": round(float(split_score), 4),
                  "note": "val/test only contain problems that have both human and AI rows; CodeNet-only problems are train-only"},
        "counts": {s: {
            "rows": int((det.split == s).sum()),
            "problems": int(det[det.split == s].problem_id.nunique()),
            "human": int(((det.split == s) & (det.label == 0)).sum()),
            "ai": int(((det.split == s) & (det.label == 1)).sum()),
            "ai_by_llm": det[(det.split == s) & (det.label == 1)].llm_name.value_counts().to_dict(),
            "ai_by_scenario": det[(det.split == s) & (det.label == 1)].scenario.value_counts().to_dict(),
        } for s in ["train", "val", "test"]},
        "rag_only_rows": int((df.role == "rag_only").sum()),
        "unparsable_share_detector": round(float((~det.parses_ok).mean()), 4),
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(pd.Series(log).to_string())
    print(json.dumps(report["counts"], indent=1, default=str))
    print("class weights (train):", report["class_weights_train"])


if __name__ == "__main__":
    main()
