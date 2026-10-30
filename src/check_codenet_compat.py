"""
check_codenet_compat.py - run this the moment Nour's REAL files arrive.

    python check_codenet_compat.py --aig aigcodeset_clean.parquet \
           --codenet codenet_subset.parquet --problems problems.parquet

Prints PASS / WARN / FAIL per check. If there is no FAIL, build_dataset.py will accept the files.
"""
import argparse
import sys

import pandas as pd

from wwi_normalize import code_hash, normalize_code

ap = argparse.ArgumentParser()
ap.add_argument("--aig", default="aigcodeset_clean.parquet")
ap.add_argument("--codenet", default="codenet_subset.parquet")
ap.add_argument("--problems", default="problems.parquet")
a = ap.parse_args()

fails = 0


def check(ok, name, detail="", warn=False):
    global fails
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    if not ok and not warn:
        fails += 1
    print(f"[{tag}] {name}" + (f"  ->  {detail}" if detail else ""))


aig = pd.read_parquet(a.aig)
cn = pd.read_parquet(a.codenet)
pr = pd.read_parquet(a.problems)

need = ["sample_id", "source", "submission_id", "problem_id", "language", "code_raw",
        "label", "llm_name", "scenario", "status", "role"]
missing = [c for c in need if c not in cn.columns]
check(not missing, "codenet has every column build_dataset.py needs", f"missing: {missing}")
check(all(c in pr.columns for c in ["problem_id", "title", "description"]),
      "problems has problem_id, title, description", f"columns: {list(pr.columns)}")
if missing:
    sys.exit("Fix the missing columns first.")

check(len(cn) >= 5000, "codenet_subset size is realistic (expected ~20k)", f"{len(cn)} rows",
      warn=len(cn) >= 500)
check(cn.sample_id.is_unique, "sample_id unique")
check((cn.label == 0).all(), "all CodeNet rows are label 0 (human)")
check((cn.source == "codenet").all(), "source == 'codenet'", str(list(cn.source.unique())))
check(set(cn.role) <= {"train_extra", "rag_only"}, "role values", str(cn.role.value_counts().to_dict()))
check((cn.language == "python").all(), "language == 'python' (matches AIGCodeSet)", str(list(cn.language.unique())))
check(cn[["code_raw", "problem_id", "status"]].notna().all().all(), "no nulls in code/problem/status")
check((cn.code_raw.str.strip() != "").all(), "no empty code")
check((cn[cn.role == "rag_only"].status == "Accepted").all(), "rag_only rows are all Accepted")
print("      status mix:", cn.status.value_counts().to_dict())

# overlap with Nada's data (Nour was supposed to remove it)
overlap_ids = set(cn.submission_id) & set(aig.submission_id[aig.submission_id != "unrelated"])
check(len(overlap_ids) == 0, "no submission_id overlap with AIGCodeSet", f"{len(overlap_ids)} overlapping")
h_aig = set(aig.code_raw.map(normalize_code).map(code_hash))
h_cn = cn.code_raw.map(normalize_code).map(code_hash)
n_ov = int(h_cn.isin(h_aig).sum())
check(n_ov == 0, "no identical code with AIGCodeSet (after normalization)", f"{n_ov} identical",
      warn=n_ov < 0.02 * len(cn))

# problems table
miss_p = set(cn.problem_id) - set(pr.problem_id)
check(not miss_p, "every CodeNet problem_id has a description", f"{len(miss_p)} without one", warn=len(miss_p) < 20)
check(pr.problem_id.is_unique, "problems.problem_id unique")
check((pr.description.fillna("").str.len() > 30).mean() > 0.95, "descriptions are real text (not empty/mock)")
check(not pr.description.str.contains("<html|<p>|<div", case=False, regex=True).any(),
      "descriptions have no leftover HTML tags")

# same problem_id format as AIGCodeSet (needed for grouped split)
check(cn.problem_id.str.match(r"^p\d{5}$").all() and aig.problem_id.str.match(r"^p\d{5}$").all(),
      "problem_id format pN{5} in both files", f"examples: {cn.problem_id.iloc[0]} / {aig.problem_id.iloc[0]}")

print("\nRESULT:", "READY - run build_dataset.py" if fails == 0 else f"{fails} FAIL(s) - send this output to Nour")
sys.exit(1 if fails else 0)
