
## Phase 3 - detector models (Nour)

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
