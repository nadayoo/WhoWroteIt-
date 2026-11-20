"""
build_rag.py - Phase 4+5, Nour, Part A (audit + build the two retrieval corpora).

    python src/build_rag.py --audit-only                  # step 1: audit the existing data/rag_aig/ files
    python src/build_rag.py                               # audit + build data/rag_aig/v2 (strict + deploy-like)
    python src/build_rag.py --backend tfidf               # offline test version (no model download)
    python src/build_rag.py --with-codenet --out data/rag_aig/v2_codenet     # optional ablation corpus
    python src/build_rag.py --model <other-st-model> --out data/rag_aig/v2_alt   # optional alternative embedder

What is built (all in the output folder):
    meta.parquet / emb.npy              one row + one vector per code file (whole file = one chunk), ALL splits
    problems_chunks.parquet / problems_emb.npy   problem-statement chunks (for the statement-based problem finder)
    problems_info.parquet               problem_id, title, short excerpt
    index_strict.faiss / index_deploy.faiss      exact cosine (inner product on L2-normalised vectors) indexes
    config.json                          embedding model, dimension, counts

Corpora (plan section 3):
    strict  = rag_eligible AND split == train      -> used whenever retrieval touches a detection number
    deploy  = rag_eligible, all splits (+CodeNet rag_only with --with-codenet) -> leave-one-out at query time
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wwi_normalize import code_hash, normalize_code          # noqa: E402
from rag_prep import DEFAULT_ST_MODEL, Embedder, chunk_problem  # noqa: E402

PROC = ROOT / "data" / "processed_aig"
RES = ROOT / "results" / "phase45"
HEAD_LINES, HEAD_CHARS = 25, 1500
NEAR_DUP_SIM = 0.98


def head(code: str) -> str:
    return "\n".join(str(code).rstrip("\n").split("\n")[:HEAD_LINES])[:HEAD_CHARS]


def find_problems():
    """Prefer a problems.parquet that has descriptions (Yassin's updated one), else any problems.parquet."""
    cands = [ROOT / "data" / "problems.parquet", ROOT / "data" / "interim" / "problems.parquet"]
    best = None
    for p in cands:
        if p.exists():
            df = pd.read_parquet(p)
            if "description" in df and df["description"].fillna("").str.len().gt(0).any():
                return p, df
            best = best or (p, df)
    return best or (None, pd.DataFrame(columns=["problem_id", "title", "description"]))


# ----------------------------------------------------------------------------- step 1: audit
def audit(old_dir=ROOT / "data" / "rag_aig") -> str:
    out = []
    P = out.append
    P("=== AUDIT of data/rag_aig/ (the Phase 1 index) ===")
    cfg = json.loads((old_dir / "config.json").read_text())
    P(f"config.json : {cfg}")
    ch = pd.read_parquet(old_dir / "chunks.parquet")
    emb = np.load(old_dir / "embeddings.npy")
    P(f"chunks.parquet columns: {ch.columns.tolist()}")
    P(f"rows: chunks={len(ch)}  embeddings={emb.shape}")
    try:
        import faiss
        n = faiss.read_index(str(old_dir / "index.faiss")).ntotal
        P(f"index.faiss rows: {n}")
        same = len(ch) == emb.shape[0] == n
    except Exception as e:                                         # faiss missing
        P(f"index.faiss could not be read here ({type(e).__name__})")
        same = len(ch) == emb.shape[0]
    P(f"chunks == embeddings == index rows: {same}")
    P("\nby chunk_type / split / label:\n" + ch.groupby(["chunk_type", "split", "label"]).size().to_string())
    P("\nAI chunks by LLM:\n" + ch[ch.label == 1].llm_name.value_counts().to_string())
    P("AI chunks by scenario:\n" + ch[ch.label == 1].scenario.value_counts().to_string())
    mp = PROC / "merged_all.parquet"
    if mp.exists():
        m = pd.read_parquet(mp, columns=["sample_id", "source"])
        src = ch[ch.chunk_type == "code"].merge(m, on="sample_id", how="left").source.value_counts(dropna=False)
        P("\nby source:\n" + src.to_string())
    bad = int((ch.split != "train").sum())
    P(f"\nvalidation/test chunks in the index: {bad}  -> {'OK for strict mode' if bad == 0 else 'NOT strict!'}")
    P(f"problem-statement chunks in the index: {(ch.chunk_type == 'problem').sum()}"
      "  (0 means the old index cannot find a problem from its statement -> rebuilt here)")
    txt = "\n".join(out)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rag_audit.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return txt


# ----------------------------------------------------------------------------- step 2: build
def build(out: Path, backend: str, model: str, with_codenet: bool):
    m = pd.read_parquet(PROC / "merged_all.parquet")
    meta = m[["sample_id", "problem_id", "label", "llm_name", "scenario", "split", "code_hash",
              "rag_eligible", "source"]].copy()
    meta["code_head"] = m["code_raw"].map(head)
    texts = m["code_norm"].tolist()

    if with_codenet:
        cn = pd.read_parquet(ROOT / "data" / "interim" / "codenet_subset.parquet")
        cn = cn[cn.role == "rag_only"].reset_index(drop=True)
        norm = cn["code_raw"].map(normalize_code)
        extra = cn[["sample_id", "problem_id", "label", "llm_name", "scenario", "source"]].copy()
        extra["split"] = "codenet"
        extra["code_hash"] = norm.map(code_hash)
        extra["rag_eligible"] = True
        extra["code_head"] = cn["code_raw"].map(head)
        meta = pd.concat([meta, extra], ignore_index=True)
        texts += norm.tolist()
        print(f"+ {len(extra)} CodeNet rag_only human rows (ablation corpus)")

    meta["in_strict"] = (meta.split == "train") & meta.rag_eligible
    meta["in_deploy"] = meta.rag_eligible
    meta["in_knn_train"] = (meta.split == "train") & (meta.source == "aigcodeset")   # all train rows, for the kNN check
    assert set(meta.loc[meta.in_strict, "split"]) == {"train"}, "val/test leaked into the strict corpus"

    pf, prob = find_problems()
    n_desc = int(prob["description"].fillna("").str.len().gt(0).sum()) if len(prob) else 0
    print(f"problems file: {pf}  ({len(prob)} problems, {n_desc} with a description)")
    prows = []
    for r in prob.itertuples():
        d = str(r.description) if isinstance(r.description, str) else ""
        if d.strip():
            for j, c in enumerate(chunk_problem(str(r.title), d)):
                prows.append(dict(problem_id=r.problem_id, title=r.title, chunk=j, text=c))
    pmeta = pd.DataFrame(prows, columns=["problem_id", "title", "chunk", "text"])
    pinfo = pd.DataFrame({"problem_id": prob.problem_id.values, "title": prob.title.values,
                          "excerpt": [str(d)[:300] if isinstance(d, str) else "" for d in prob.description.values]}) \
        if len(prob) else pd.DataFrame(columns=["problem_id", "title", "excerpt"])

    emb_model = Embedder(backend, model)
    emb = emb_model.fit_transform(texts)
    pemb = emb_model.transform(pmeta.text.tolist()) if len(pmeta) else np.zeros((0, emb.shape[1]), "float32")
    print(f"embedded {emb.shape[0]} code files, {pemb.shape[0]} statement chunks, dim {emb.shape[1]}")

    out.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(out / "meta.parquet", index=False)
    np.save(out / "emb.npy", emb.astype("float32"))
    pmeta.to_parquet(out / "problems_chunks.parquet", index=False)
    np.save(out / "problems_emb.npy", pemb.astype("float32"))
    pinfo.to_parquet(out / "problems_info.parquet", index=False)
    try:
        import faiss
        for name in ("strict", "deploy"):
            rows = np.flatnonzero(meta[f"in_{name}"].to_numpy())
            ix = faiss.IndexFlatIP(emb.shape[1]); ix.add(emb[rows].astype("float32"))
            faiss.write_index(ix, str(out / f"index_{name}.faiss"))
            np.save(out / f"rows_{name}.npy", rows)
            print(f"  index_{name}.faiss: {ix.ntotal} vectors")
    except ImportError:
        print("faiss not installed -> numpy exact search will be used (same results)")
    cfg = dict(backend=backend, model=model, dim=int(emb.shape[1]), n_code=int(len(meta)),
               n_strict=int(meta.in_strict.sum()), n_deploy=int(meta.in_deploy.sum()),
               n_problem_chunks=int(len(pmeta)), problems_with_description=n_desc,
               near_dup_sim=NEAR_DUP_SIM, with_codenet=with_codenet)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if backend == "tfidf":
        import joblib
        joblib.dump({"vec": emb_model._vec, "svd": emb_model._svd, "dim": emb_model.dim}, out / "embedder.joblib")
    print(json.dumps(cfg, indent=2))
    print("saved ->", out)
    if n_desc == 0:
        print("\nWARNING: no problem has a description, so the statement-based problem finder is disabled.\n"
              "         Pull Yassin's updated problems.parquet (data/problems.parquet) and rebuild.")
    sizes = {p.name: p.stat().st_size / 1e6 for p in out.iterdir()}
    big = {k: round(v, 1) for k, v in sizes.items() if v > 90}
    if big:
        print("FILES OVER 90 MB (do not commit):", big)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "rag_aig" / "v2"))
    ap.add_argument("--backend", choices=["st", "tfidf"], default="st")
    ap.add_argument("--model", default=DEFAULT_ST_MODEL)
    ap.add_argument("--with-codenet", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    a = ap.parse_args()
    if not a.no_audit:
        audit()
    if not a.audit_only:
        build(Path(a.out), a.backend, a.model, a.with_codenet)


if __name__ == "__main__":
    main()
