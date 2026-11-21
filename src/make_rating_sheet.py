"""
make_rating_sheet.py - blind human rating sheets (all three of you rate the SAME items).

    python src/make_rating_sheet.py                       # 30 samples x all variants, shuffled, variant hidden
    python src/make_rating_sheet.py --variants evidence,plain   # fewer items per rater (60 instead of 90)

Picks 10 correct-AI, 10 correct-human, 5 false alarms, 5 misses (seed 42) from the 'evidence' run.
Writes results/phase45/rating/rater_<name>.csv (fill the 4 score columns, 1-5) and rating_key.csv (KEEP SECRET
until everyone has finished: it says which variant wrote each explanation).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain as E                                          # noqa: E402

RUNS = E.OUT / "eval_runs"
OUT = E.OUT / "rating"
QUOTA = {"correct_ai": 10, "correct_human": 10, "false_alarm": 5, "miss": 5}
SEED = 42


def load(v):
    f = RUNS / f"{v}.jsonl"
    return {r["sample_id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if r} if f.exists() else {}


def render_explanation(rec: dict) -> str:
    if "text" in rec:                                      # plain variant
        return rec["text"].strip()
    e = rec["explanation"]
    lines = [e["summary"]] + [f"- {k['text']}" for k in e["key_points"]]
    if e.get("comparison"):
        lines.append(f"Comparison: {e['comparison']['text']}")
    lines += [f"Caveat: {c['text']}" for c in e["caveats"]]
    return "\n".join(lines)


def render_evidence(ev: dict) -> str:
    d = ev["detector"]
    L = [f"Detector: verdict {d['verdict']}, confidence {d['confidence']}, P(AI) {d['p_ai']} (threshold {d['threshold']})"]
    L += [f"{f['id']} {f['name']} = {f['value']} (human mean {f['human_mean']}, AI mean {f['ai_mean']}) -> {f['points_toward']}"
          for f in ev.get("features", [])]
    L += [f"{n['id']} n-gram '{n['ngram']}' -> {n['points_toward']}" for n in ev.get("ngrams", [])]
    L += [f"{c['id']} {c['text']}" for c in ev.get("caveats", [])]
    if ev.get("problem"):
        L.append(f"P problem '{ev['problem'].get('title', '')}' ({ev['problem'].get('source')})")
    for side in ("human", "ai"):
        for it in (ev.get("retrieved") or {}).get(side, []):
            code = "\n    ".join(str(it["code"]).split("\n")[:8])
            L.append(f"{it['id']} ({side} example, similarity {it['similarity']}, same_problem {it['same_problem']}):\n    {code}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="evidence,no_retrieval,plain")
    ap.add_argument("--raters", default="nada,nour,yassin")
    a = ap.parse_args()
    variants = a.variants.split(",")
    runs = {v: load(v) for v in variants}
    base = runs[variants[0]]
    if not base:
        sys.exit(f"no results for '{variants[0]}': run run_eval.py first")

    chosen = []
    for oc, k in QUOTA.items():
        ids = sorted(s for s, r in base.items() if r["outcome"] == oc)
        take = pd.Series(ids).sample(min(k, len(ids)), random_state=SEED).tolist() if ids else []
        if len(take) < k:
            print(f"WARNING: only {len(take)} '{oc}' samples available (wanted {k})")
        chosen += take
    code = pd.read_parquet(E.ROOT / "data" / "processed_aig" / "test.parquet").set_index("sample_id")["code_raw"]

    items = []
    for s in chosen:
        for v in variants:
            r = runs[v].get(s)
            if r:
                items.append({"sample_id": s, "variant": v, "label": r["label"], "outcome": r["outcome"],
                              "detector_verdict": f"{r['verdict']} ({r['confidence']})",
                              "submitted_code": "\n".join(str(code[s]).split("\n")[:E.EXAMPLE_LINES]),
                              "evidence": render_evidence(r["evidence"]), "explanation": render_explanation(r)})
    df = pd.DataFrame(items).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "item_id", [f"item{i + 1:03d}" for i in range(len(df))])
    OUT.mkdir(parents=True, exist_ok=True)
    df[["item_id", "sample_id", "variant", "label", "outcome"]].to_csv(OUT / "rating_key.csv", index=False)
    sheet = df[["item_id", "detector_verdict", "submitted_code", "evidence", "explanation"]].copy()
    for c in ("faithfulness", "usefulness", "clarity", "hedging"):
        sheet[c] = ""
    for name in a.raters.split(","):
        sheet.to_csv(OUT / f"rater_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{len(chosen)} samples x {len(variants)} variants = {len(df)} items per rater -> {OUT}")
    print(df.groupby(["outcome", "variant"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
