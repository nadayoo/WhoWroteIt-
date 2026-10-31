"""
predict.py - what the demo (Phase 4) will call.

    from predict import load_and_predict
    load_and_predict("print(1)")            # -> probability the code is AI-written (0..1)

    python src/predict.py --check           # fresh-session reload test: reloads models/best.joblib and compares
                                            # with the validation predictions saved by train_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # the pickle needs wwi_models / features / wwi_normalize

import joblib                                        # noqa: E402
import numpy as np                                   # noqa: E402

_CACHE = {}


def _load(path=None):
    path = Path(path or ROOT / "models" / "best.joblib")
    if path not in _CACHE:
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def load_and_predict(code: str, model_path=None, return_label: bool = False):
    """raw Python source string -> P(AI). With return_label=True -> (P(AI), 0/1 using the validation threshold)."""
    bundle = _load(model_path)
    p = float(bundle["pipeline"].predict_proba([code])[0, 1])
    return (p, int(p >= bundle["threshold"])) if return_label else p


if __name__ == "__main__":
    import pandas as pd
    if "--check" in sys.argv:
        bundle = _load()
        val = pd.read_parquet(ROOT / "data" / "processed_aig" / "val.parquet")
        saved = pd.read_parquet(ROOT / "results" / f"preds_{bundle['name']}.parquet")
        saved = saved[saved.split == "val"].set_index("sample_id").loc[val.sample_id]
        now = bundle["pipeline"].predict_proba(val["code_raw"])[:, 1]
        print("model:", bundle["name"], "| val rows:", len(val))
        print("max |difference| vs saved predictions:", float(np.abs(now - saved.y_score.values).max()))
        print("RELOAD OK" if np.allclose(now, saved.y_score.values, atol=1e-8) else "MISMATCH!")
        ex = val.iloc[0]
        print("\nexample sample", ex.sample_id, "true label", ex.label, "-> P(AI) =", round(load_and_predict(ex.code_raw), 4))
    else:
        print(load_and_predict(sys.stdin.read()))
