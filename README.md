# WhoWroteIt-
WhoWroteIt?: Detecting and Explaining LLM-Written Code with Evidence-Grounded Reasoning

## Phase 3 - detector models 

Run from the repo root:

    python src/train_models.py          # trains + tunes on validation, writes everything below
    python src/predict.py --check       # fresh-session reload test of models/best.joblib

| Output | Where |
|---|---|
| Prediction files (val + test rows: sample_id, split, y_true, y_score, y_pred) | `results/preds_<model>.parquet` |
| Saved pipelines (raw code -> P(AI)) | `models/<model>.joblib`, best one = `models/best.joblib` (+ `best_meta.json`) |
| Validation comparison table, chosen hyper-parameters | `results/phase3/val_comparison.csv`, `best_params.json` |
| Top n-grams (AI vs human), top-15 feature importances | `results/phase3/top_ngrams.*`, `importances_*.png` |

Use in the demo: `from predict import load_and_predict; load_and_predict(code_string)` returns P(AI).
Models are fitted on train only; hyper-parameters and the decision threshold are chosen on validation only.
## Phase 4+5 - retrieval layer (Nour)

    python src/build_rag.py             # audit data/rag_aig/ + build the strict and deploy-like indexes -> data/rag_aig/v2/
    python src/eval_retrieval.py        # ONE command: correctness tests, problem finder, coverage,
                                        # retrieval-as-detector, usefulness sheet, leakage re-check, tables + plots

| What | Where |
|---|---|
| `retrieve(code, problem_id=None, k_human=3, k_ai=3, mode="deploy", exclude_ids=None)` | `src/retrieve.py` |
| Index (vectors, metadata, FAISS indexes, config) | `data/rag_aig/v2/` (small; if a file is over 100 MB it goes on the shared drive) |
| All retrieval tables and plots | `results/phase45/` (`retrieval_results_table.csv`, `*.png`, `leakage_checks.txt`) |

Modes: `strict` = train split only (use for anything that touches a detection number);
`deploy` = all splits, always excluding the query, its exact copies and near-duplicates (cosine >= 0.98).
