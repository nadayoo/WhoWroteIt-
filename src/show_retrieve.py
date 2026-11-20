"""show_retrieve.py - prints retrieve() for one test file (for the 'retrieve() output' screenshot).
    python src/show_retrieve.py                 # first test sample, deploy mode
    python src/show_retrieve.py aigcodeset_000103 strict
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import retrieve                                  # noqa: E402

te = pd.read_parquet(ROOT / "data" / "processed_aig" / "test.parquet")
sid = sys.argv[1] if len(sys.argv) > 1 else te.sample_id.iloc[0]
mode = sys.argv[2] if len(sys.argv) > 2 else "deploy"
row = te[te.sample_id == sid].iloc[0]
r = retrieve(row.code_raw, mode=mode, exclude_ids={sid})
print(f"query {sid}  true label {'AI' if row.label else 'human'}  true problem {row.problem_id}  mode {mode}")
p = r["problem"]
print(f"problem found: {p['problem_id']} ({p['source']}, via {p['method']})  title: {p['title']}  top5: {p['candidates']}")
print(f"{'id':3} {'sample_id':20} {'sim':>6}  {'same problem':12} {'llm':10} scenario")
for it in r["human"] + r["ai"]:
    print(f"{it['id']:3} {it['sample_id']:20} {it['similarity']:6.3f}  {str(it['same_problem']):12} {it['llm_name']:10} {it['scenario']}")
print("excluded (exact copy / near-duplicate):", r["n_excluded_exact"], "/", r["n_excluded_near_dup"], " warnings:", r["warnings"])
print("\n--- query code (first lines) ---\n" + "\n".join(str(row.code_raw).split("\n")[:8]))
print("\n--- H1 code ---\n" + r["human"][0]["code"][:400] if r["human"] else "")
print("\n--- A1 code ---\n" + r["ai"][0]["code"][:400] if r["ai"] else "")
