"""
pick_failures.py - candidate failure cases for the report (no human rating needed).

    python src/pick_failures.py              # shows 2 candidates per type, writes results/phase45/failure_cases.csv
    python src/pick_failures.py --per-type 3

Types (from the plan): a false alarm driven by consistent operator/comma spacing, a miss on Gemini,
a miss in a fix scenario. Pick three, read the printed explanation, and keep or edit the draft cause.
Reads results/phase45/eval_runs/evidence.jsonl (the full system).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain as E                                          # noqa: E402

CONF_ORDER = {"high": 0, "medium": 1, "low": 2, "inconclusive": 3}
SPACING = ("op_spaced_ratio", "comma_space_ratio")


def feats(r, toward=None, names=None):
    return [f for f in r["evidence"].get("features", [])
            if (toward is None or f["points_toward"] == toward) and (names is None or f["name"] in names)]


def fmt(fs):
    return ", ".join(f"{f['name']} = {f['value']} (human mean {f['human_mean']}, AI mean {f['ai_mean']})" for f in fs)


def draft_cause(kind, r):
    if kind == "false_alarm_spacing":
        return (f"Human code was called AI because {fmt(feats(r, 'AI', SPACING))} look AI-like; "
                f"consistent spacing does not prove AI, since many humans space consistently.")
    fs = feats(r, "human")[:2]
    why = f"{fmt(fs)} sit nearer the human mean" if fs else "its top features pointed toward human"
    return f"AI code ({r['llm_name']}, {r['scenario']}) was missed: {why}, so the detector said human."


def text_of(r):
    e = r["explanation"]
    return "\n".join([e["summary"]] + [f"  - {k['text']}" for k in e["key_points"]]
                     + [f"  caveat: {c['text']}" for c in e["caveats"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=2)
    a = ap.parse_args()
    recs = [json.loads(l) for l in (E.OUT / "eval_runs" / "evidence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rank = lambda r: (CONF_ORDER.get(r["confidence"], 9), -r["p_ai"] if r["outcome"] == "false_alarm" else r["p_ai"])
    groups = {
        "false_alarm_spacing": [r for r in recs if r["outcome"] == "false_alarm" and feats(r, "AI", SPACING)],
        "miss_gemini": [r for r in recs if r["outcome"] == "miss" and r["llm_name"] == "gemini"],
        "miss_fix_scenario": [r for r in recs if r["outcome"] == "miss" and r["scenario"].startswith("fix")
                              and r["llm_name"] != "gemini"],
    }
    rows = []
    for kind, g in groups.items():
        g = sorted(g, key=rank)[:a.per_type]
        print(f"\n########## {kind}: {len(groups[kind])} available, showing {len(g)} ##########")
        for r in g:
            print(f"\n--- {r['sample_id']} | {r['llm_name']} | {r['scenario']} | verdict {r['verdict']} "
                  f"({r['confidence']}) | P(AI) {r['p_ai']} | explanation source: {r['source']}")
            print(text_of(r))
            print("DRAFT CAUSE:", draft_cause(kind, r))
            rows.append({"type": kind, "sample_id": r["sample_id"], "llm_name": r["llm_name"], "scenario": r["scenario"],
                         "verdict": r["verdict"], "confidence": r["confidence"], "p_ai": r["p_ai"],
                         "explanation": text_of(r), "draft_cause": draft_cause(kind, r)})
    pd.DataFrame(rows).to_csv(E.OUT / "failure_cases.csv", index=False, encoding="utf-8-sig")
    fb = [r["sample_id"] for r in recs if r["source"] == "fallback"]
    print(f"\nsaved {E.OUT / 'failure_cases.csv'}   (samples that used the fallback template: {fb})")


if __name__ == "__main__":
    main()
