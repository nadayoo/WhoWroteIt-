"""
wwi_normalize.py  -  shared preprocessing functions for WhoWroteIt (Phase 1, Person 3).

Import this from every notebook/script:   from wwi_normalize import normalize_code, ...

Design decisions (put these in the report):
  * code_raw  = untouched original (kept for display and RAG examples).
  * code_norm = what the model sees. Line endings fixed, trailing spaces stripped,
                the WHOLE FILE stripped (removes CodeLlama's stray leading indent) and
                ending in exactly one newline (removes the "ends with newline" shortcut).
  * Comments are KEPT in code_norm (they are a strong human/AI signal) and are extracted as
    features in features.py. strip_comments() is provided for an ablation run
    ("does the detector still work without comments?").
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import keyword
import re
import tokenize
import warnings

# --------------------------------------------------------------------------- text cleaning
def clean_text(s: str) -> str:
    """Same low-level cleaning as AIGCodeSet / CodeNet notebooks: line endings, BOM, NUL."""
    return (str(s).replace("\r\n", "\n").replace("\r", "\n")
                  .replace("\ufeff", "").replace("\x00", ""))


def normalize_code(s: str) -> str:
    """code_raw -> code_norm.  Does NOT touch comments, indentation inside the file or identifiers."""
    s = clean_text(s)
    s = "\n".join(line.rstrip() for line in s.split("\n"))   # strip trailing spaces
    s = s.strip()                                           # leading indent of 1st line, blank edges
    return s + "\n" if s else ""


def count_lines(s: str) -> int:
    """Number of lines, ignoring the single final newline added by normalize_code."""
    return s.rstrip("\n").count("\n") + 1 if s else 0


# --------------------------------------------------------------------------- parsing helpers
def parses_ok(s: str) -> bool:
    """True if the code is valid Python 3 syntax (needed for AST features)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(s)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


_PY2_PATTERN = re.compile(
    r"(?m)^\s*print\s+(?!\(|=|\s*$)[^\s=]"      # print "x"   print x
    r"|\braw_input\s*\(|\bxrange\s*\(|\bunicode\s*\("
    r"|^\s*exec\s+[\"'\w]"
    r"|except\s+[\w.]+\s*,\s*\w+\s*:"           # except Foo, e:
    r"|\.has_key\s*\(|<>|\bprint\s*>>"
)


def looks_like_py2(s: str, parsed: bool | None = None) -> bool:
    """Python-2-style code. Only flagged when it ALSO fails to parse as Python 3,
    which keeps false positives (e.g. the word 'print' inside a docstring) very low."""
    if parsed is None:
        parsed = parses_ok(s)
    return (not parsed) and bool(_PY2_PATTERN.search(s))


def has_real_code(s: str) -> bool:
    """False when the file is only blank lines / comments."""
    return any(l.strip() and not l.strip().startswith("#") for l in s.split("\n"))


# --------------------------------------------------------------------------- comments
def _strip_comments_scan(code: str) -> str:
    """Fallback comment remover (when tokenize fails). Tracks single-line quotes only."""
    out = []
    for line in code.split("\n"):
        quote, i, cut = None, 0, None
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "#":
                cut = i
                break
            i += 1
        if cut is None:
            out.append(line)
        else:
            kept = line[:cut].rstrip()
            if kept:
                out.append(kept)
    return "\n".join(out)


def strip_comments(code: str) -> str:
    """Remove '# ...' comments (comment-only lines disappear, inline ones are cut)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _strip_comments_scan(code)
    cuts = {t.start[0]: t.start[1] for t in toks if t.type == tokenize.COMMENT}
    out = []
    for i, line in enumerate(code.split("\n"), 1):
        if i in cuts:
            kept = line[:cuts[i]].rstrip()
            if kept:
                out.append(kept)
        else:
            out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- tokens
_KEYWORDS = set(keyword.kwlist)
_FALLBACK_TOK = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|\S")


def token_stream(code: str, placeholders: bool = True) -> list[str]:
    """Token list used for n-grams and near-duplicate hashing.
    placeholders=True : names->ID, numbers->NUM, strings->STR, comments->COMMENT
                        (keywords, operators and layout tokens are kept: this is 'coding style').
    placeholders=False: raw token strings, comments dropped (used for hashing)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out = []
        for t in _FALLBACK_TOK.findall(_strip_comments_scan(code) if not placeholders else code):
            if not placeholders:
                out.append(t)
            elif t in _KEYWORDS:
                out.append(t)
            elif t[0].isalpha() or t[0] == "_":
                out.append("ID")
            elif t[0].isdigit():
                out.append("NUM")
            else:
                out.append(t)
        return out

    out = []
    for t in toks:
        tt, s = t.type, t.string
        if tt == tokenize.INDENT:
            out.append("<INDENT>")
        elif tt == tokenize.DEDENT:
            out.append("<DEDENT>")
        elif tt in (tokenize.NEWLINE, tokenize.NL):
            if placeholders:
                out.append("<NL>")
        elif tt == tokenize.COMMENT:
            if placeholders:
                out.append("COMMENT")
        elif tt in (tokenize.ENDMARKER, tokenize.ENCODING):
            continue
        elif not placeholders:
            out.append(s)
        elif tt == tokenize.NAME:
            out.append(s if s in _KEYWORDS else "ID")
        elif tt == tokenize.NUMBER:
            out.append("NUM")
        elif tt == tokenize.STRING:
            out.append("STR")
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------- hashing
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def code_hash(code_norm: str) -> str:
    """Exact hash of the normalized code (the `code_hash` column of the schema)."""
    return md5(code_norm)


def loose_hash(code_norm: str) -> str:
    """Near-duplicate hash: ignores comments, whitespace and layout differences."""
    toks = [t for t in token_stream(code_norm, placeholders=False)
            if t not in ("<INDENT>", "<DEDENT>")]
    return md5(" ".join(toks))


# --------------------------------------------------------------------------- misc
BUILTIN_NAMES = set(dir(builtins))


def is_identifier_token(tok: str) -> bool:
    return tok.isidentifier() and tok not in _KEYWORDS and tok not in BUILTIN_NAMES


if __name__ == "__main__":
    demo = "  a = int(input())   \nb = 2  # two\r\n\r\n\r\nprint(a+b)\n\n"
    n = normalize_code(demo)
    print(repr(n))
    print(repr(strip_comments(n)))
    print(parses_ok(n), looks_like_py2(n), code_hash(n)[:8], loose_hash(n)[:8])
