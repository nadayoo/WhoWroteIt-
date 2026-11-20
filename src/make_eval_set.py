"""
make_eval_set.py - the fixed evaluation set for Phase 5 (Nada). Run ONCE, then never change it.

    python src/make_eval_set.py

Picks 120 samples from the TEST split with seed 42: 60 AI (20 per LLM, spread across the three scenarios)
and 60 human. Writes results/phase45/eval_set.csv (no code in it: look the code up by sample_id).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_PER_LLM, N_HUMAN = 20, 60


def spread(g: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """n rows from one LLM, taken round-robin across its scenarios so all scenarios are represented."""
    pools = [d.sample(frac=1, random_state=int(rng.integers(1_000_000_000)))
             for _, d in g.groupby("scenario")]
    out, i = [], 0
    while len(out) < n and any(len(p) for p in pools):
        k = i % len(pools)
        if len(pools[k]):
            out.append(pools[k].iloc[0])
            pools[k] = pools[k].iloc[1:]
        i += 1
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(ROOT / "data" / "processed_aig" / "test.parquet"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase45" / "eval_set.csv"))
    a = ap.parse_args()

    te = pd.read_parquet(a.test)
    assert (te["label"].isin([0, 1])).all()
    rng = np.random.default_rng(SEED)
    ai, hu = te[te.label == 1], te[te.label == 0]

    parts = []
    for llm, g in ai.groupby("llm_name"):
        assert len(g) >= N_PER_LLM, f"only {len(g)} test samples for {llm}"
        parts.append(spread(g, N_PER_LLM, rng))
    parts.append(hu.sample(N_HUMAN, random_state=SEED))
    ev = pd.concat(parts, ignore_index=True)
    cols = [c for c in ["sample_id", "label", "llm_name", "scenario", "problem_id", "split"] if c in ev.columns]
    ev = ev[cols].sort_values(["label", "llm_name", "sample_id"]).reset_index(drop=True)

    assert ev.sample_id.is_unique and len(ev) == N_PER_LLM * ai.llm_name.nunique() + N_HUMAN
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(a.out, index=False)
    print(f"{len(ev)} samples -> {a.out}")
    print(ev.groupby(["label", "llm_name"]).size().to_string())
    print("\nAI by scenario:\n", ev[ev.label == 1].groupby(["llm_name", "scenario"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
