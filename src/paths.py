"""Shared paths and small helpers for the evaluation notebooks.

Every notebook does:  from paths import *
Nothing here depends on where the repo is cloned.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --- data already in the repo (tracked) ---
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed_aig"
FEATURES = DATA / "features_aig"

# --- big downloads (ignored by git, local only) ---
EXTERNAL = DATA / "external"
CODENET_DIR = EXTERNAL / "codenet"
AIG_EMB_DIR = EXTERNAL / "aigcodeset"
CODEBERT_DIR = EXTERNAL / "codebert"

# --- outputs ---
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FINAL = RESULTS / "final"

SEED = 42
LABEL = "label"  # 1 = AI, 0 = human


def find_dir(base, name):
    """First directory called `name` anywhere under `base` (or None)."""
    base = Path(base)
    if not base.exists():
        return None
    for p in base.rglob(name):
        if p.is_dir():
            return p
    return None


def find_file(base, name):
    """First file called `name` anywhere under `base` (or None)."""
    base = Path(base)
    if not base.exists():
        return None
    for p in base.rglob(name):
        if p.is_file():
            return p
    return None


def find_embeddings_csv():
    """The biggest .csv in data/external/aigcodeset (the 265 MB embeddings file)."""
    if not AIG_EMB_DIR.exists():
        return None
    files = sorted(AIG_EMB_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    return files[0] if files else None


def detect_id_col(df):
    """Name of the sample-ID column in a Phase 1 split file."""
    for c in ("sample_id", "id", "sid"):
        if c in df.columns:
            return c
    raise KeyError(f"No sample ID column found. Columns are: {list(df.columns)}")


def code_col(df):
    """Name of the column holding the code text."""
    for c in ("code_raw", "code", "code_norm"):
        if c in df.columns:
            return c
    raise KeyError(f"No code column found. Columns are: {list(df.columns)}")


def to_binary_label(series):
    """Turn a label column into 0/1 (1 = AI). Works for 0/1 numbers or 'human'/'ai' text."""
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "1.0", "ai", "true"]).astype(int).to_numpy()
