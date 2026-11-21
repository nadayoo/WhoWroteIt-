"""
run_eval.py - Phase 5 explanation evaluation on the fixed 120-sample set (Nada).

    python src/run_eval.py --limit 6                 # small smoke test first
    python src/run_eval.py                           # all 120 samples, all three variants (resumable)
    python src/run_eval.py --variants evidence       # only one variant

Variants
    evidence      evidence-only prompt + retrieved examples (the full system)
    no_retrieval  same prompt, retrieved examples removed            (retrieval ablation)
    plain         loose "explain why" prompt, same information, NO rules, no citations, no checks   (grounding ablation)

Writes results/phase45/eval_runs/<variant>.jsonl (one line per sample; reruns skip finished samples)
and results/phase45/faithfulness_table.csv.  Needs GEMINI_API_KEY in .env; pace with GEMINI_RPM (default 10).
The detector verdict never depends on retrieval or the LLM.
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import explain as E                                                  # noqa: E402

ROOT = E.ROOT
EVAL_SET = E.OUT / "eval_set.csv"
TEST = ROOT / "data" / "processed_aig" / "test.parquet"
RUNS = E.OUT / "eval_runs"
PLAIN_PROMPT = ROOT / "prompts" / "plain_v0.txt"
VARIANTS = ["evidence", "no_retrieval", "plain"]


def outcome(verdict: str, label: int) -> str:
    return {("ai", 1): "correct_ai", ("human", 0): "correct_human",
            ("ai", 0): "false_alarm", ("human", 1): "miss"}[(verdict, int(label))]


def fail_letters(failures) -> set:
    return {m.group(1) for f in failures for m in [re.match(r"\((\w)\)", f)] if m}


def plain_metrics(text: str, clean: dict) -> dict:
    """Same number / feature-name tests as the evidence-only checks, applied to free text."""
    ev_nums = {round(float(x), 6) for x in E.NUM_RE.findall(json.dumps(E._no_code(clean)))}
    bad_nums = sorted({x for x in E.NUM_RE.findall(text) if round(float(x), 6) not in ev_nums})
    shown = {f["name"] for f in clean.get("features", [])}
    bad_feats = sorted(n for n in E._feature_names() if n not in shown and re.search(rf"\b{re.escape(n)}\b", text))
    return {"unsupported_numbers": bad_nums, "unsupported_features": bad_feats, "words": len(text.split())}


def done_ids(v: str) -> set:
    f = RUNS / f"{v}.jsonl"
    return {json.loads(l)["sample_id"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()} if f.exists() else set()


def append(v: str, rec: dict):
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / f"{v}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def summarize() -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        f = RUNS / f"{v}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(recs)
        if v == "plain":
            rows.append({"variant": v, "n": n,
                         "numbers_match_evidence": 1 - sum(bool(r["unsupported_numbers"]) for r in recs) / n,
                         "unsupported_feature_mentions": sum(bool(r["unsupported_features"]) for r in recs) / n})
        else:
            first = [fail_letters(r["attempts"][0]["failures"]) if r["attempts"] else set() for r in recs]
            rows.append({"variant": v, "n": n,
                         "first_try_pass": sum(not f for f in first) / n,
                         "valid_citations": sum("b" not in f and "a" not in f for f in first) / n,
                         "numbers_match_evidence": sum("c" not in f for f in first) / n,
                         "unsupported_feature_mentions": sum("d" in f for f in first) / n,
                         "verdict_contradictions": sum("e" in f for f in first) / n,
                         "fallback_used": sum(r["source"] == "fallback" for r in recs) / n})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(E.OUT / "faithfulness_table.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only N random samples (smoke test)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--mode", default="deploy", choices=["deploy", "strict"])
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v in VARIANTS]

    ev_set = pd.read_csv(EVAL_SET)
    if a.limit:
        ev_set = ev_set.sample(a.limit, random_state=42)
    code = pd.read_parquet(TEST)[["sample_id", "code_raw"]]
    df = ev_set.merge(code, on="sample_id", how="left")
    assert df.code_raw.notna().all(), "some eval samples are missing from test.parquet"

    from evidence import build_evidence
    done = {v: done_ids(v) for v in variants}
    plain_tpl = PLAIN_PROMPT.read_text(encoding="utf-8")
    t0, n = time.time(), len(df)
    for i, r in enumerate(df.itertuples(), 1):
        pend = [v for v in variants if r.sample_id not in done[v]]
        if not pend:
            continue
        ev = build_evidence(r.code_raw)
        ev, warn = E.attach_retrieval(ev, r.code_raw, r.sample_id, a.mode)
        has_ret = bool((ev.get("retrieved") or {}).get("human") or (ev.get("retrieved") or {}).get("ai"))
        out = outcome(ev["detector"]["verdict"], r.label)
        base = {"sample_id": r.sample_id, "label": int(r.label), "llm_name": r.llm_name, "scenario": r.scenario,
                "verdict": ev["detector"]["verdict"], "confidence": ev["detector"]["confidence"],
                "p_ai": ev["detector"]["p_ai"], "outcome": out, "retrieval_warnings": warn,
                "retrieved_ids": [x["sample_id"] for s in ("human", "ai") for x in (ev.get("retrieved") or {}).get(s, [])]}
        print(f"[{i}/{n}] {r.sample_id} {r.llm_name} -> {out} ({ev['detector']['confidence']})", flush=True)
        for v in pend:
            if v == "no_retrieval" and not has_ret:
                print("    no_retrieval skipped: retrieval gave no examples (same as 'evidence')")
                continue
            if v == "plain":
                clean = E.clean_for_llm({**ev, "submitted": {"id": "S", "code": r.code_raw}})
                text = E.cached_call(plain_tpl.replace("{evidence_json}", json.dumps(clean, indent=1)), json_mode=False)
                rec = {**base, "text": text, "evidence": clean, **plain_metrics(text, clean)}
            else:
                e2 = {k: val for k, val in ev.items() if k != "retrieved"} if v == "no_retrieval" else ev
                res = E.explain_from_evidence(e2, use_llm=True, code=r.code_raw)
                rec = {**base, "source": res["source"], "attempts": res["attempts"],
                       "explanation": res["explanation"], "evidence": res["evidence"]}
            append(v, rec)
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min")
    t = summarize()
    print("\n=== faithfulness (automatic) ===\n", t.round(3).to_string(index=False))
    print("\nsaved:", E.OUT / "faithfulness_table.csv")


if __name__ == "__main__":
    main()
