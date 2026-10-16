# Person 3 (Yassin): normalization, dedup, merge, split, features, RAG prep

Put these files in the repo root next to `aigcodeset_clean.parquet`, `codenet_subset.parquet`, `problems.parquet`.

    pip install -r requirements.txt
    python wwi_normalize.py                                  # smoke test of the shared module
    python build_dataset.py --aig aigcodeset_clean.parquet --codenet codenet_subset.parquet \
           --problems problems.parquet --out data/processed [--minhash]
    python features.py  --data data/processed --out data/features          # Phase 2 input
    python rag_prep.py  --data data/processed --out data/rag  --backend st # fair-eval corpus
    python rag_prep.py  --data data/processed --out data/rag_full --scope all --backend st  # demo corpus

Files: wwi_normalize.py (importable module), build_dataset.py, features.py, rag_prep.py.
Everything that must go in the report is written to data/processed/split_report.json and build_log.csv.

## While waiting for Person 2 (CodeNet)
    python build_dataset.py --aig aigcodeset_clean.parquet --aig-only --out data/processed_aig
    python features.py --data data/processed_aig --out data/features_aig
    python rag_prep.py --data data/processed_aig --out data/rag_aig --backend tfidf
When her real files arrive:
    python check_codenet_compat.py --aig aigcodeset_clean.parquet --codenet codenet_subset.parquet --problems problems.parquet
    (no FAIL -> rerun build_dataset.py WITHOUT --aig-only)
