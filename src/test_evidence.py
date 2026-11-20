"""test_evidence.py - run build_evidence() on 5 files (2 human, 2 AI, 1 unparsable) and check the format.
 
    python src/test_evidence.py
 
Checks:
  * the evidence has the same keys as Nada's tests/mock_evidence.json (detector, features, ngrams, caveats)
  * ids are F1.., N1.., C1.. in order
  * p_ai_raw equals load_and_predict(code) from predict.py (same model, same number)
  * unparsable code: the unparsable caveat is present and no AST feature is cited
Writes results/phase45/evidence_examples.json (use it for the screenshot).
"""
from __future__ import annotations
 
import json
import sys
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
 
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
 
from evidence import C_UNPARSABLE, build_evidence     # noqa: E402
from paths import to_binary_label                     # noqa: E402
from predict import load_and_predict                  # noqa: E402
 
OUT = ROOT / "results" / "phase45"
BROKEN = "def total(xs:\n    s = 0\n    for x in xs\n        s += x\n    print(s)\n"
 
 
def check_schema(ev: dict, mock: dict) -> list[str]:
    bad = []
    for key in ("detector", "features", "ngrams", "caveats"):
        if key not in ev:
            bad.append(f"missing key: {key}")
    if bad:
        return bad
    if set(ev["detector"]) != set(mock["detector"]):
        bad.append(f"detector keys differ: {sorted(set(ev['detector']) ^ set(mock['detector']))}")
    for block, prefix in (("features", "F"), ("ngrams", "N"), ("caveats", "C")):
        for i, item in enumerate(ev[block], start=1):
            if set(item) != set(mock[block][0]):
                bad.append(f"{block}[{i}] keys differ: {sorted(set(item) ^ set(mock[block][0]))}")
            if item["id"] != f"{prefix}{i}":
                bad.append(f"{block}[{i}] has id {item['id']}, expected {prefix}{i}")
    if ev["detector"]["confidence"] not in ("low", "medium", "high", "inconclusive"):
        bad.append(f"unexpected confidence value: {ev['detector']['confidence']}")
    return bad
 
 
def main():
    mock = json.loads((ROOT / "tests" / "mock_evidence.json").read_text(encoding="utf-8"))[0]
    test = pd.read_parquet(ROOT / "data" / "processed_aig" / "test.parquet")
    y = to_binary_label(test["label"])
    rng = np.random.default_rng(42)
    picks = [("human", i) for i in rng.choice(np.flatnonzero(y == 0), 2, replace=False)] + \
            [("ai", i) for i in rng.choice(np.flatnonzero(y == 1), 2, replace=False)]
    cases = [(f"{a}_{k + 1}", a, test["code_raw"].iloc[i]) for k, (a, i) in enumerate(picks)]
    cases.append(("unparsable", "human", BROKEN))
 
    results, failures = [], 0
    for name, author, code in cases:
        ev = build_evidence(code)
        problems = check_schema(ev, mock)
        if abs(ev["_debug"]["p_ai_raw"] - load_and_predict(code)) > 1e-3:
            problems.append("p_ai_raw differs from load_and_predict")
        if name == "unparsable":
            if not any(c["text"] == C_UNPARSABLE for c in ev["caveats"]):
                problems.append("unparsable caveat missing")
            if any(f["name"].startswith("ast_") for f in ev["features"]):
                problems.append("an AST feature was cited for unparsable code")
 
        d = ev["detector"]
        print("=" * 72)
        print(f"{name}  (true author: {author})")
        print(f"  verdict: {d['verdict']} | confidence: {d['confidence']} | p_ai {d['p_ai']} (threshold {d['threshold']})")
        for f in ev["features"]:
            print(f"  {f['id']} {f['name']:<24} value {f['value']!s:<8} human {f['human_mean']!s:<8} "
                  f"AI {f['ai_mean']!s:<8} -> {f['points_toward']}")
        for n in ev["ngrams"]:
            print(f"  {n['id']} n-gram '{n['ngram']}' -> {n['points_toward']}")
        for c in ev["caveats"]:
            print(f"  {c['id']} {c['text'][:90]}")
        print("  schema + checks:", "OK" if not problems else problems)
        failures += bool(problems)
        results.append({"case": name, "true_author": author, **ev})
 
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_examples.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + ("ALL 5 CASES PASSED" if not failures else f"{failures} CASE(S) HAVE PROBLEMS"))
    print("saved results/phase45/evidence_examples.json")
 
 
if __name__ == "__main__":
    main()
 
