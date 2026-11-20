"""
retrieve.py - Phase 4+5, Nour: the retrieval layer used by explain().

    from retrieve import retrieve
    r = retrieve(code)                                   # deploy-like mode, problem inferred
    r = retrieve(code, problem_id="p02534", mode="strict")

Returns a dict:
    {"problem": {problem_id, title, excerpt, source: "given"|"inferred", method, candidates},
     "human":   [ {id: "H1", sample_id, similarity, same_problem, llm_name, scenario, problem_id, code}, ... ],
     "ai":      [ {id: "A1", ...}, ... ],
     "mode", "n_excluded_exact", "n_excluded_near_dup", "warnings": [...]}

Rules (plan section 3):
  * strict mode  : corpus = train split only. Use for anything that touches a detection number.
  * deploy mode  : corpus = all splits. Used for retrieval/explanation quality and the demo.
  * ALWAYS excluded: the sample ids in exclude_ids, any sample with the same code_hash as the query, and any
    near-duplicate (cosine similarity >= near_dup_sim, 0.98 by default).
  * If a problem has fewer than k examples of a label, the rest are filled with the nearest examples overall and
    marked same_problem = False.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wwi_normalize import code_hash, normalize_code          # noqa: E402

DEFAULT_DIR = ROOT / "data" / "rag_aig" / "v2"


class RagStore:
    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_DIR)
        p = self.path
        self.cfg = json.loads((p / "config.json").read_text())
        self.meta = pd.read_parquet(p / "meta.parquet")
        self.emb = np.load(p / "emb.npy")
        self.pmeta = pd.read_parquet(p / "problems_chunks.parquet")
        self.pemb = np.load(p / "problems_emb.npy")
        self.pinfo = pd.read_parquet(p / "problems_info.parquet").set_index("problem_id")
        from rag_prep import Embedder
        if self.cfg["backend"] == "tfidf":
            import joblib
            st = joblib.load(p / "embedder.joblib")
            self.embedder = Embedder("tfidf", dim=st["dim"])
            self.embedder._vec, self.embedder._svd = st["vec"], st["svd"]
        else:
            self.embedder = Embedder("st", self.cfg["model"])
        self.near_dup = float(self.cfg.get("near_dup_sim", 0.98))
        m = self.meta
        self.ids, self.lab = m.sample_id.to_numpy(), m.label.to_numpy()
        self.prob, self.hash = m.problem_id.to_numpy(), m.code_hash.to_numpy()
        self.modes = {"strict": m.in_strict.to_numpy(), "deploy": m.in_deploy.to_numpy(),
                      "knn_train": m.in_knn_train.to_numpy()}
        self.has_statements = len(self.pmeta) > 0
        self.row_of = {s: i for i, s in enumerate(self.ids)}

    # ---- query side
    def embed(self, code: str) -> np.ndarray:
        return self.embedder.transform([normalize_code(code)])[0]

    def candidates(self, q, qhash, mode, exclude_ids=None):
        mask = self.modes[mode].copy()
        if exclude_ids:
            mask &= ~np.isin(self.ids, list(exclude_ids))
        exact = mask & (self.hash == qhash) if qhash else np.zeros_like(mask)
        mask &= ~exact
        idx = np.flatnonzero(mask)
        sims = self.emb[idx] @ q
        dup = sims >= self.near_dup
        return idx[~dup], sims[~dup], int(exact.sum()), int(dup.sum())

    def rank_problems(self, q, idx, sims, method="vote", n_neighbors=10):
        """Which problem does this code solve?  vote = weighted vote of the nearest code neighbours;
        statement = best match against the problem statements."""
        if method == "statement" and self.has_statements:
            s = pd.Series(self.pemb @ q, index=self.pmeta.problem_id.values).groupby(level=0).max()
            s = s.sort_values(ascending=False)
            return list(zip(s.index, s.values))
        score = {}
        for j in np.argsort(-sims)[:n_neighbors]:
            p = self.prob[idx[j]]
            score[p] = score.get(p, 0.0) + float(sims[j])
        return sorted(score.items(), key=lambda t: -t[1])

    def _item(self, tag, row, sim, same):
        r = self.meta.iloc[row]
        return {"id": tag, "sample_id": r.sample_id, "similarity": round(float(sim), 4), "same_problem": bool(same),
                "label": int(r.label), "llm_name": r.llm_name, "scenario": r.scenario,
                "problem_id": r.problem_id, "split": r.split, "code_hash": r.code_hash, "code": r.code_head}

    def retrieve_vec(self, q, qhash=None, problem_id=None, k_human=3, k_ai=3, mode="deploy",
                     exclude_ids=None, finder="vote"):
        if mode not in ("strict", "deploy"):
            raise ValueError("mode must be 'strict' or 'deploy'")
        idx, sims, n_exact, n_dup = self.candidates(q, qhash, mode, exclude_ids)
        warnings = []
        if len(idx) == 0:
            return {"problem": None, "human": [], "ai": [], "mode": mode, "n_excluded_exact": n_exact,
                    "n_excluded_near_dup": n_dup, "warnings": ["empty corpus after exclusions"]}
        if problem_id is not None:
            ranked, source, method = [(problem_id, 1.0)], "given", "given"
        else:
            ranked, source, method = self.rank_problems(q, idx, sims, finder), "inferred", finder
            if finder == "statement" and not self.has_statements:
                method = "vote"; warnings.append("no problem statements in the index -> used the neighbour vote")
        pid = ranked[0][0]
        info = self.pinfo.loc[pid] if pid in self.pinfo.index else None
        out = {"problem": {"problem_id": pid, "title": None if info is None else info.title,
                           "excerpt": None if info is None else info.excerpt, "source": source, "method": method,
                           "candidates": [p for p, _ in ranked[:5]]},
               "human": [], "ai": [], "mode": mode, "n_excluded_exact": n_exact,
               "n_excluded_near_dup": n_dup, "warnings": warnings}
        for label, k, key, letter in ((0, k_human, "human", "H"), (1, k_ai, "ai", "A")):
            ok = self.lab[idx] == label
            same = ok & (self.prob[idx] == pid)
            order = np.argsort(-sims)
            chosen = [j for j in order if same[j]][:k]
            if len(chosen) < k:
                rest = [j for j in order if ok[j] and j not in set(chosen)][: k - len(chosen)]
                chosen += rest
                if rest:
                    warnings.append(f"only {int(same.sum())} {key} example(s) for {pid}; filled from other problems")
            out[key] = [self._item(f"{letter}{n+1}", idx[j], sims[j], same[j]) for n, j in enumerate(chosen)]
        return out


_STORE = {}


def get_store(path=None) -> RagStore:
    key = str(Path(path or DEFAULT_DIR))
    if key not in _STORE:
        _STORE[key] = RagStore(path)
    return _STORE[key]


def retrieve(code: str, problem_id=None, k_human=3, k_ai=3, mode="deploy", exclude_ids=None,
             finder="vote", store=None):
    s = store or get_store()
    q = s.embed(code)
    return s.retrieve_vec(q, code_hash(normalize_code(code)), problem_id, k_human, k_ai, mode, exclude_ids, finder)


if __name__ == "__main__":
    code = sys.stdin.read() if not sys.stdin.isatty() else "n = int(input())\nprint(n * 2)\n"
    r = retrieve(code)
    print("problem:", r["problem"])
    for key in ("human", "ai"):
        for it in r[key]:
            print(it["id"], it["sample_id"], it["similarity"], "same_problem" if it["same_problem"] else "OTHER",
                  it["llm_name"], it["scenario"])
    print("warnings:", r["warnings"])
