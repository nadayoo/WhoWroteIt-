"""
rag_prep.py  -  RAG corpus preparation (Person 3; the index is used in Phase 4).

    python rag_prep.py --data data/processed --out data/rag                  # fair-evaluation corpus
    python rag_prep.py --data data/processed --out data/rag_full --scope all # final demo corpus

What goes in (only `rag_eligible` rows = ALL AI rows + Accepted human rows; rag_only rows included):
    * code chunks      : whole file if short, else top-level-statement groups (fallback: line windows)
    * problem chunks   : paragraph groups of the problem statement, title attached
Metadata on every chunk: chunk_id, chunk_type (code|problem), sample_id, problem_id, language,
    label, llm_name, scenario, status, split, title, text.
    -> the interface filters on label to say "closest human solution" / "closest AI solution".

--scope train (DEFAULT): only problems whose split == train. Val/test problems are excluded from the
                         corpus so retrieval evaluation is fair (no test leakage).
--scope all            : everything (use only for the deployed demo, never for evaluation).

Embeddings: --backend st     sentence-transformers (needs internet once to download the model)
            --backend tfidf  offline fallback (TF-IDF + SVD), good enough to test the pipeline
Index     : FAISS (inner product on L2-normalised vectors) if installed, else numpy.
Reload    : r = Retriever.load("data/rag"); r.search(code, k=3, label=0)
"""
from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from wwi_normalize import normalize_code, token_stream

warnings.filterwarnings("ignore", category=SyntaxWarning)

MAX_CODE_CHARS = 1500
WINDOW_LINES, WINDOW_OVERLAP = 40, 5
MAX_PROBLEM_CHARS = 1000
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------------ chunking
def _line_windows(lines, size=WINDOW_LINES, overlap=WINDOW_OVERLAP):
    step = size - overlap
    return ["\n".join(lines[i:i + size]) for i in range(0, max(len(lines) - overlap, 1), step)]


def chunk_code(code: str, max_chars: int = MAX_CODE_CHARS) -> list[str]:
    code = code.rstrip("\n")
    if len(code) <= max_chars:
        return [code]
    lines = code.split("\n")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return _line_windows(lines)
    chunks, cur = [], []
    for node in tree.body:
        seg = lines[node.lineno - 1: node.end_lineno]
        seg_text = "\n".join(seg)
        if len(seg_text) > max_chars:                       # one huge function/class
            if cur:
                chunks.append("\n".join(cur)); cur = []
            chunks.extend(_line_windows(seg))
            continue
        if cur and len("\n".join(cur)) + len(seg_text) + 1 > max_chars:
            chunks.append("\n".join(cur)); cur = []
        cur.extend(seg)
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if c.strip()]


def chunk_problem(title: str, description: str, max_chars: int = MAX_PROBLEM_CHARS) -> list[str]:
    paras = [p.strip() for p in str(description).split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur); cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
        while len(cur) > max_chars * 1.5:                    # very long paragraph
            chunks.append(cur[:max_chars]); cur = cur[max_chars:]
    if cur:
        chunks.append(cur)
    return [f"{title}\n\n{c}" for c in chunks]


def build_chunks(df: pd.DataFrame, problems: pd.DataFrame, problem_split: dict) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        for j, c in enumerate(chunk_code(r.code_norm)):
            rows.append(dict(chunk_id=f"{r.sample_id}#{j}", chunk_type="code", sample_id=r.sample_id,
                             problem_id=r.problem_id, language=r.language, label=int(r.label),
                             llm_name=r.llm_name, scenario=r.scenario, status=r.status,
                             split=r.split, title="", text=c))
    for r in problems.itertuples():
        sp = problem_split.get(r.problem_id)
        if sp is None:                                       # problem has no kept samples
            continue
        for j, c in enumerate(chunk_problem(r.title, r.description)):
            rows.append(dict(chunk_id=f"{r.problem_id}#p{j}", chunk_type="problem", sample_id="",
                             problem_id=r.problem_id, language="python", label=-1, llm_name="",
                             scenario="", status="", split=sp, title=r.title, text=c))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ embeddings
def _l2(x):
    x = np.asarray(x, dtype="float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


class Embedder:
    def __init__(self, backend="st", model=DEFAULT_ST_MODEL, dim=256):
        self.backend, self.model_name, self.dim = backend, model, dim
        self._st = self._vec = self._svd = None

    @staticmethod
    def _tok(text):
        return " ".join(token_stream(text, placeholders=False)) if "\n" in text or "(" in text else text

    def fit_transform(self, texts):
        if self.backend == "st":
            return self.transform(texts)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]\w+|\S", lowercase=True, sublinear_tf=True,
                                    ngram_range=(1, 2), min_df=2, max_features=50_000)
        X = self._vec.fit_transform(texts)
        self._svd = TruncatedSVD(min(self.dim, X.shape[1] - 1), random_state=42)
        return _l2(self._svd.fit_transform(X))

    def transform(self, texts):
        if self.backend == "st":
            if self._st is None:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(self.model_name)
            return _l2(self._st.encode(list(texts), batch_size=64, show_progress_bar=True))
        return _l2(self._svd.transform(self._vec.transform(texts)))


# ------------------------------------------------------------------ index + retriever
class Retriever:
    def __init__(self, meta, emb, embedder, index=None):
        self.meta, self.emb, self.embedder, self.index = meta.reset_index(drop=True), emb, embedder, index

    @classmethod
    def build(cls, meta, emb, embedder):
        index = None
        try:
            import faiss
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
        except ImportError:
            pass
        return cls(meta, emb, embedder, index)

    def save(self, out: Path):
        import joblib
        out.mkdir(parents=True, exist_ok=True)
        self.meta.to_parquet(out / "chunks.parquet", index=False)
        np.save(out / "embeddings.npy", self.emb)
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(out / "index.faiss"))
        cfg = {"backend": self.embedder.backend, "model": self.embedder.model_name, "n_chunks": len(self.meta),
               "dim": int(self.emb.shape[1]), "faiss": self.index is not None}
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        if self.embedder.backend == "tfidf":
            joblib.dump({"vec": self.embedder._vec, "svd": self.embedder._svd,
                         "dim": self.embedder.dim}, out / "embedder.joblib")

    @classmethod
    def load(cls, path):
        import joblib
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        meta = pd.read_parquet(path / "chunks.parquet")
        emb = np.load(path / "embeddings.npy")
        if cfg["backend"] == "tfidf":
            st = joblib.load(path / "embedder.joblib")
            embedder = Embedder("tfidf", dim=st["dim"])
            embedder._vec, embedder._svd = st["vec"], st["svd"]
        else:
            embedder = Embedder("st", cfg["model"])
        index = None
        if (path / "index.faiss").exists():
            import faiss
            index = faiss.read_index(str(path / "index.faiss"))
        return cls(meta, emb, embedder, index)

    def search(self, query_code: str, k: int = 3, label: int | None = None,
               chunk_type: str = "code", problem_id: str | None = None,
               exclude_sample_id: str | None = None) -> pd.DataFrame:
        """Nearest chunks. label=0 -> closest human, label=1 -> closest AI, chunk_type='problem' -> statement."""
        q = self.embedder.transform([normalize_code(query_code)])
        mask = np.array(self.meta["chunk_type"] == chunk_type)
        if label is not None:
            mask &= (self.meta["label"] == label).to_numpy()
        if problem_id is not None:
            mask &= (self.meta["problem_id"] == problem_id).to_numpy()
        if exclude_sample_id:
            mask &= (self.meta["sample_id"] != exclude_sample_id).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return self.meta.iloc[0:0].assign(score=[])
        scores = self.emb[idx] @ q[0]                      # exact, masked search (corpus is small)
        top = np.argsort(-scores)[:k]
        return self.meta.iloc[idx[top]].assign(score=scores[top]).reset_index(drop=True)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/rag")
    ap.add_argument("--scope", choices=["train", "all"], default="train")
    ap.add_argument("--backend", choices=["st", "tfidf"], default="st")
    ap.add_argument("--model", default=DEFAULT_ST_MODEL)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    df = pd.read_parquet(data / "merged_all.parquet")
    pp = data / "problems.parquet"          # absent in --aig-only mode -> code chunks only
    problems = pd.read_parquet(pp) if pp.exists() else pd.DataFrame(columns=["problem_id", "title", "description"])
    psplit = pd.read_csv(data / "problem_split.csv").set_index("problem_id")["split"].to_dict()

    df = df[df["rag_eligible"]]
    if args.scope == "train":
        df = df[df["split"] == "train"]
        psplit = {p: s for p, s in psplit.items() if s == "train"}
    problems = problems[problems.problem_id.isin(psplit)]

    meta = build_chunks(df, problems, psplit)
    print(f"scope={args.scope}: {len(meta)} chunks "
          f"({(meta.chunk_type == 'code').sum()} code, {(meta.chunk_type == 'problem').sum()} problem)")

    embedder = Embedder(args.backend, args.model)
    emb = embedder.fit_transform(meta["text"].tolist())
    r = Retriever.build(meta, emb, embedder)
    r.save(out)

    # leakage guard: no val/test problem may be in the evaluation corpus
    if args.scope == "train":
        assert set(meta["split"]) == {"train"}, "val/test problem leaked into the RAG corpus"
        held = {p for p, s in pd.read_csv(data / "problem_split.csv").values if s != "train"}
        assert not set(meta["problem_id"]) & held
        print("RAG LEAKAGE CHECK PASSED (no val/test problems in the corpus)")
    print("saved ->", out)


if __name__ == "__main__":
    main()
