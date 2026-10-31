"""
wwi_models.py - building blocks for Phase 3 (Nour).

Every pipeline built here goes from a RAW CODE STRING to a probability of AI, so the same
saved file can be used by evaluate.py and by the demo later:

    raw code -> normalize_code -> (n-grams | style+AST features) -> classifier

Nothing here is fitted on validation or test data.  Seed 42, class_weight balanced everywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from features import extract_features, ngram_doc          # Nada's code (src/features.py)
from wwi_normalize import normalize_code                  # code_raw -> code_norm

SEED = 42
# trailing_ws_line_ratio is flagged in the plan (may be a dataset artifact) -> excluded from main models
FLAGGED = ("trailing_ws_line_ratio",)


# ----------------------------------------------------------------------------- transformers
class NormCode(BaseEstimator, TransformerMixin):
    """raw code string -> code_norm (used by the character n-gram models)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_code(str(c)) for c in X]


class NormTokens(BaseEstimator, TransformerMixin):
    """raw code string -> 'ID = ID ( STR ) <NL> ...' (identifiers/numbers/strings replaced by placeholders)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [ngram_doc(normalize_code(str(c))) for c in X]


class StyleAstFeatures(BaseEstimator, TransformerMixin):
    """raw code string -> numeric matrix of Nada's 90 style + AST features (same function as Phase 2)."""
    def __init__(self, exclude=FLAGGED):
        self.exclude = exclude

    def _rows(self, X):
        rows = []
        for c in X:
            c = str(c)
            rows.append(extract_features(normalize_code(c), c))
        return pd.DataFrame(rows)

    def fit(self, X, y=None):
        df = self._rows(X)
        self.cols_ = [c for c in df.columns if c not in set(self.exclude)]
        return self

    def transform(self, X):
        df = self._rows(X).reindex(columns=self.cols_)
        return df.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


# ----------------------------------------------------------------------------- pipelines
def _tok_vec():
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 3),
                           min_df=5, max_features=100_000, sublinear_tf=True, dtype=np.float32)


def _char_vec():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, min_df=5,
                           max_features=200_000, sublinear_tf=True, dtype=np.float32)


def _lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)


def _svm(C, cv):
    # LinearSVC has no probabilities -> calibrate (Platt) so y_score is a real P(AI).
    # cv = precomputed GroupKFold(problem_id) splits, so one problem never sits in two folds.
    return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", dual="auto", random_state=SEED),
                                  method="sigmoid", cv=cv)


def _dense_prep(exclude, scale=True):
    steps = [("feat", StyleAstFeatures(exclude=exclude)),
             ("imp", SimpleImputer(strategy="median", add_indicator=True))]   # unparsable code -> NaN AST cols
    if scale:
        steps.append(("sc", StandardScaler()))
    return steps


def make_pipeline(kind: str, params: dict | None = None, cv=None, pos_weight: float = 2.0,
                  exclude=FLAGGED) -> Pipeline:
    p = dict(params or {})
    if kind == "tok_lr":
        return Pipeline([("doc", NormTokens()), ("vec", _tok_vec()), ("clf", _lr(p.get("C", 1)))])
    if kind == "tok_svm":
        return Pipeline([("doc", NormTokens()), ("vec", _tok_vec()), ("clf", _svm(p.get("C", 0.1), cv))])
    if kind == "char_lr":
        return Pipeline([("doc", NormCode()), ("vec", _char_vec()), ("clf", _lr(p.get("C", 1)))])
    if kind == "char_svm":
        return Pipeline([("doc", NormCode()), ("vec", _char_vec()), ("clf", _svm(p.get("C", 0.1), cv))])
    if kind == "feat_lr":
        return Pipeline(_dense_prep(exclude) + [("clf", _lr(p.get("C", 1)))])
    if kind == "feat_rf":
        clf = RandomForestClassifier(n_estimators=p.get("n_estimators", 500), max_depth=p.get("max_depth"),
                                     min_samples_leaf=p.get("min_samples_leaf", 3),
                                     max_features=p.get("max_features", "sqrt"),
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
        return Pipeline(_dense_prep(exclude, scale=False) + [("clf", clf)])
    if kind == "feat_xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth", 4),
                            learning_rate=p.get("learning_rate", 0.05), subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=pos_weight, eval_metric="logloss", tree_method="hist",
                            n_jobs=-1, random_state=SEED)
        return Pipeline([("feat", StyleAstFeatures(exclude=exclude)), ("clf", clf)])   # XGBoost handles NaN itself
    if kind == "combined_lr":
        # n-gram block + dense block in ONE model. The dense block is scaled to unit variance and has ~90
        # columns, the TF-IDF rows have L2 norm 1, so the dense block gets a small weight w (tuned on val).
        union = FeatureUnion([("tok", Pipeline([("doc", NormTokens()), ("vec", _tok_vec())])),
                              ("dense", Pipeline(_dense_prep(exclude)))],
                             transformer_weights={"tok": 1.0, "dense": p.get("w", 0.1)})
        return Pipeline([("union", union), ("clf", _lr(p.get("C", 1)))])
    raise ValueError(kind)


# name -> (kind, fixed params, grid).  Small grids on purpose ("a handful of settings per model").
def model_zoo():
    return {
        "tfidf_token_lr":  ("tok_lr",   {}, {"C": [0.3, 1, 3, 10, 30]}),
        "tfidf_token_svm": ("tok_svm",  {}, {"C": [0.03, 0.1, 0.3, 1]}),
        "tfidf_char_lr":   ("char_lr",  {}, {"C": [0.3, 1, 3, 10]}),
        "tfidf_char_svm":  ("char_svm", {}, {"C": [0.03, 0.1, 0.3]}),
        "feat_lr":         ("feat_lr",  {}, {"C": [0.03, 0.1, 0.3, 1, 3]}),
        "feat_rf":         ("feat_rf",  {}, {"min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.3]}),
        "feat_xgb":        ("feat_xgb", {}, {"max_depth": [3, 5], "learning_rate": [0.05, 0.1],
                                             "n_estimators": [300, 600]}),
        "combined_lr":     ("combined_lr", {}, {"C": [1, 3, 10], "w": [0.03, 0.1, 0.3]}),
    }
