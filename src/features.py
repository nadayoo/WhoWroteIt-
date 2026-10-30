"""
features.py  -  data-mining features for WhoWroteIt (Person 3; used in Phase 2).

    python features.py --data data/processed --out data/features

Writes (keyed by sample_id, kept separate from the raw data so features can be added/dropped later):
    features_{train,val,test}.parquet   AST + stylometric + comment features
    ngram_vectorizer.joblib             TF-IDF token n-gram vectorizer (FIT ON TRAIN ONLY)
    ngrams_{train,val,test}.npz         sparse n-gram matrices (row order = the split parquet)

Deliberately NOT used as features: "ends with newline" and "starts with whitespace"
(formatting artifacts of how the dataset was produced, not coding style).
"""
from __future__ import annotations

import argparse
import ast
import io
import math
import re
import tokenize
import warnings
from collections import Counter
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from wwi_normalize import is_identifier_token, token_stream
warnings.filterwarnings("ignore", category=SyntaxWarning)   # ast.parse on messy student code

NODE_TYPES = [
    "FunctionDef", "ClassDef", "Return", "Assign", "AugAssign", "AnnAssign", "For", "While", "If",
    "With", "Try", "ExceptHandler", "Raise", "Assert", "Import", "ImportFrom", "Expr", "Call",
    "Attribute", "Subscript", "BinOp", "BoolOp", "Compare", "UnaryOp", "IfExp", "Lambda", "ListComp",
    "DictComp", "SetComp", "GeneratorExp", "List", "Dict", "Set", "Tuple", "Constant", "JoinedStr",
    "Name", "Slice", "Starred", "Break", "Continue", "Pass", "Global", "Nonlocal", "Yield",
]
_BLOCKS = (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)


# ------------------------------------------------------------------ AST features
def _max_depth(tree) -> int:
    best, stack = 0, [(tree, 1)]
    while stack:
        n, d = stack.pop()
        best = max(best, d)
        stack.extend((c, d + 1) for c in ast.iter_child_nodes(n))
    return best


def _max_nesting(node, depth=0) -> int:
    best = depth
    for child in ast.iter_child_nodes(node):
        best = max(best, _max_nesting(child, depth + (1 if isinstance(child, _BLOCKS) else 0)))
    return best


def ast_features(code: str) -> dict:
    nan = {"ast_ok": 0}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return nan
    nodes = list(ast.walk(tree))
    counts = Counter(type(n).__name__ for n in nodes)
    total = len(nodes)
    f = {"ast_ok": 1, "ast_total_nodes": total, "ast_max_depth": _max_depth(tree)}
    try:
        f["ast_max_block_nesting"] = _max_nesting(tree)
    except RecursionError:
        f["ast_max_block_nesting"] = np.nan
    for t in NODE_TYPES:
        f[f"ast_frac_{t}"] = counts.get(t, 0) / total
    # cyclomatic complexity of the whole file (1 + decision points)
    cc = 1 + sum(counts.get(t, 0) for t in ("If", "For", "While", "ExceptHandler", "IfExp"))
    cc += sum(1 for n in nodes if isinstance(n, ast.comprehension))
    cc += sum(len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp))
    f["cyclomatic"] = cc
    funcs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    f["n_functions_ast"] = len(funcs)
    f["avg_function_len"] = (np.mean([(n.end_lineno - n.lineno + 1) for n in funcs]) if funcs else 0.0)
    f["avg_args_per_function"] = (np.mean([len(n.args.args) for n in funcs]) if funcs else 0.0)
    f["n_classes"] = counts.get("ClassDef", 0)
    f["has_docstring_ast"] = int(any(ast.get_docstring(n) for n in nodes
                                     if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module))))
    f["has_type_hints"] = int(any(n.returns is not None or any(a.annotation for a in n.args.args)
                                  for n in funcs))
    return f


# ------------------------------------------------------------------ stylometric features
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
_CAMEL = re.compile(r"^[a-z]+[A-Z][A-Za-z0-9]*$")
_PASCAL = re.compile(r"^[A-Z][a-z0-9]+([A-Z][a-z0-9]*)+$")
_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SPACED_OP = re.compile(r"\s(==|!=|<=|>=|\+=|-=|\*=|/=|//|\+|-|\*|/|%|=)\s")
_TIGHT_OP = re.compile(r"[\w)\]](==|!=|<=|>=|\+=|-=|\*=|/=|//|\+|-|\*|/|%|=)[\w(\[\-]")


def _comments(code: str):
    """returns (comment_only_lines, inline_comments, comment_chars)"""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        toks = None
    lines = code.split("\n")
    only = inline = chars = 0
    if toks is not None:
        for t in toks:
            if t.type == tokenize.COMMENT:
                chars += len(t.string)
                if lines[t.start[0] - 1][:t.start[1]].strip() == "":
                    only += 1
                else:
                    inline += 1
    else:                                             # crude fallback
        for l in lines:
            s = l.strip()
            if s.startswith("#"):
                only += 1
                chars += len(s)
            elif " #" in l:
                inline += 1
                chars += len(l.split(" #", 1)[1]) + 1
    return only, inline, chars


def _identifiers(code: str) -> list[str]:
    try:
        toks = tokenize.generate_tokens(io.StringIO(code).readline)
        return [t.string for t in toks if t.type == tokenize.NAME and is_identifier_token(t.string)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [w for w in re.findall(r"[A-Za-z_]\w*", code) if is_identifier_token(w)]


def style_features(code_norm: str, code_raw: str | None = None) -> dict:
    lines = code_norm.rstrip("\n").split("\n")
    n = len(lines)
    nonblank = [l for l in lines if l.strip()]
    nb = max(len(nonblank), 1)
    lens = np.array([len(l) for l in lines], float)
    f = {
        "n_lines_f": n,
        "n_chars_f": len(code_norm),
        "line_len_mean": lens.mean(),
        "line_len_max": lens.max(),
        "line_len_std": lens.std(),
        "long_line_ratio": float((lens > 79).mean()),
        "blank_line_ratio": 1 - len(nonblank) / n,
    }
    # blank-line runs
    run = best = 0
    for l in lines:
        run = run + 1 if not l.strip() else 0
        best = max(best, run)
    f["max_blank_run"] = best

    # comments
    only, inline, cchars = _comments(code_norm)
    f["comment_line_ratio"] = only / nb
    f["inline_comment_ratio"] = inline / nb
    f["comment_char_ratio"] = cchars / max(len(code_norm), 1)
    f["n_comments"] = only + inline
    f["n_triple_quotes"] = len(re.findall(r'"""|\'\'\'', code_norm)) // 2

    # indentation / whitespace habits
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.startswith(" ") and l.strip()]
    f["indent_unit"] = reduce(math.gcd, indents) if indents else 0
    indented = [l for l in lines if l[:1] in (" ", "\t") and l.strip()]
    f["tab_indent_ratio"] = (sum(l.startswith("\t") for l in indented) / len(indented)) if indented else 0.0
    spaced, tight = len(_SPACED_OP.findall(code_norm)), len(_TIGHT_OP.findall(code_norm))
    f["op_spaced_ratio"] = spaced / (spaced + tight) if spaced + tight else np.nan
    with_sp, no_sp = code_norm.count(", "), len(re.findall(r",\S", code_norm))
    f["comma_space_ratio"] = with_sp / (with_sp + no_sp) if with_sp + no_sp else np.nan
    raw_lines = (code_raw if code_raw is not None else code_norm).split("\n")
    f["trailing_ws_line_ratio"] = sum(1 for l in raw_lines if l != l.rstrip()) / max(len(raw_lines), 1)
    f["semicolon_count"] = code_norm.count(";")

    # identifier naming
    ids = _identifiers(code_norm)
    if ids:
        uniq = set(ids)
        u = len(uniq)
        f["id_avg_len"] = float(np.mean([len(i) for i in ids]))
        f["id_single_letter_ratio"] = sum(len(i) == 1 for i in ids) / len(ids)
        f["id_snake_ratio"] = sum(bool(_SNAKE.match(i)) for i in uniq) / u
        f["id_camel_ratio"] = sum(bool(_CAMEL.match(i)) for i in uniq) / u
        f["id_pascal_ratio"] = sum(bool(_PASCAL.match(i)) for i in uniq) / u
        f["id_upper_ratio"] = sum(bool(_UPPER.match(i)) and len(i) > 1 for i in uniq) / u
        f["id_digit_suffix_ratio"] = sum(bool(re.search(r"\d$", i)) for i in uniq) / u
        f["id_unique_ratio"] = u / len(ids)
    else:
        for k in ["id_avg_len", "id_single_letter_ratio", "id_snake_ratio", "id_camel_ratio",
                  "id_pascal_ratio", "id_upper_ratio", "id_digit_suffix_ratio", "id_unique_ratio"]:
            f[k] = np.nan

    # misc habits
    f["uses_main_guard"] = int("__name__" in code_norm and "__main__" in code_norm)
    f["n_def"] = len(re.findall(r"(?m)^\s*def\s", code_norm))
    f["n_import"] = len(re.findall(r"(?m)^\s*(?:import|from)\s", code_norm))
    f["n_fstring"] = len(re.findall(r"\bf[\"']", code_norm))
    f["n_lambda"] = code_norm.count("lambda ")
    f["n_print"] = len(re.findall(r"\bprint\s*\(", code_norm))
    f["n_input"] = len(re.findall(r"\binput\s*\(", code_norm))
    f["uses_sys_stdin"] = int("sys.stdin" in code_norm)
    return f


def extract_features(code_norm: str, code_raw: str | None = None) -> dict:
    return {**style_features(code_norm, code_raw), **ast_features(code_norm)}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_features(cn, cr) for cn, cr in zip(df["code_norm"], df["code_raw"])]
    feats = pd.DataFrame(rows)
    feats.insert(0, "sample_id", df["sample_id"].to_numpy())
    return feats


# ------------------------------------------------------------------ token n-grams
def ngram_doc(code_norm: str) -> str:
    return " ".join(token_stream(code_norm, placeholders=True))


def fit_ngrams(train_df: pd.DataFrame, ngram_range=(1, 3), min_df=5, max_features=100_000):
    """Fit on TRAIN ONLY (fitting on val/test would leak vocabulary/idf)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                          ngram_range=ngram_range, min_df=min_df, max_features=max_features,
                          sublinear_tf=True, dtype=np.float32)
    vec.fit([ngram_doc(c) for c in train_df["code_norm"]])
    return vec


def transform_ngrams(vec, df: pd.DataFrame):
    return vec.transform([ngram_doc(c) for c in df["code_norm"]])


# ------------------------------------------------------------------ CLI
def main():
    import joblib
    import scipy.sparse as sp
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--no-ngrams", action="store_true")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {s: pd.read_parquet(data / f"{s}.parquet") for s in ("train", "val", "test")}
    for s, d in splits.items():
        feats = build_features(d)
        feats.to_parquet(out / f"features_{s}.parquet", index=False)
        print(s, feats.shape, "AST ok:", round(float(feats.ast_ok.mean()), 3))
    if not args.no_ngrams:
        vec = fit_ngrams(splits["train"])
        joblib.dump(vec, out / "ngram_vectorizer.joblib")
        for s, d in splits.items():
            sp.save_npz(out / f"ngrams_{s}.npz", transform_ngrams(vec, d))
        print("n-gram vocabulary:", len(vec.vocabulary_))


if __name__ == "__main__":
    main()
