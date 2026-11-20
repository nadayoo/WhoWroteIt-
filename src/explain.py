"""
explain.py - Phase 4+5, explanation pipeline (Nada).

    from explain import explain
    out = explain(code_string)                      # real: build_evidence + (retrieve if available) + LLM + checks

    python src/explain.py --mock --offline          # test the checks + fallback on tests/mock_evidence.json (no API)
    python src/explain.py --mock                    # same, but calls Gemini
    python src/explain.py --file some_code.py       # one real file

The LLM only REWORDS evidence. It never decides the verdict. After every call five automatic checks run
(parse, cited IDs exist, quoted numbers come from the evidence, named features are in the evidence,
verdict and confidence unchanged). One retry, then a plain template written from the evidence (no LLM).
The key lives in .env (GEMINI_API_KEY=...). The model name can be set with GEMINI_MODEL in .env.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

PROMPT_PATH = ROOT / "prompts" / "explain_v1.txt"
DICTIONARY = ROOT / "data" / "feature_dictionary.csv"
MOCK_PATH = ROOT / "tests" / "mock_evidence.json"
OUT = ROOT / "results" / "phase45"
CACHE = OUT / "cache"
LOG = OUT / "llm_log.jsonl"
MAX_WORDS = 200                       # the prompt asks for ~150; this is the hard limit for the check
EXAMPLE_LINES = 25                    # retrieved code is truncated to this many lines
NOT_FOR_LLM = {"case", "true_author"}  # mock-only fields that would leak the answer


def get_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")   # set GEMINI_MODEL in .env to the name AI Studio shows


# ------------------------------------------------------------------ evidence -> prompt
def clean_for_llm(ev: dict) -> dict:
    """Drop debug / mock-only keys and truncate retrieved code. This is all the LLM ever sees."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not k.startswith("_") and k not in NOT_FOR_LLM}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    c = strip(copy.deepcopy(ev))
    for side in ("human", "ai"):
        for it in (c.get("retrieved") or {}).get(side, []):
            it["code"] = "\n".join(str(it.get("code", "")).split("\n")[:EXAMPLE_LINES])
    if c.get("submitted"):
        c["submitted"]["code"] = "\n".join(str(c["submitted"].get("code", "")).split("\n")[:EXAMPLE_LINES])
    return c


def render_prompt(clean: dict) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{evidence_json}", json.dumps(clean, indent=1))


def evidence_ids(clean: dict) -> set:
    ids = {x["id"] for k in ("features", "ngrams", "caveats") for x in clean.get(k, []) if "id" in x}
    if clean.get("problem"):
        ids.add("P")
    if clean.get("submitted"):
        ids.add("S")
    for side in ("human", "ai"):
        ids |= {x["id"] for x in (clean.get("retrieved") or {}).get(side, [])}
    return ids


# ------------------------------------------------------------------ LLM call, cache, log
def call_llm(prompt: str) -> str:
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found. Put it in .env (GEMINI_API_KEY=...) in the repo root.")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))  # 180 s per request
    last = None
    for attempt in range(5):
        try:
            print(f"      calling {get_model()} (try {attempt + 1}/5) ...", flush=True)
            r = client.models.generate_content(
                model=get_model(), contents=prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
            return r.text
        except Exception as e:                                   # rate limits / overload -> wait and retry
            last = e
            if not any(s in str(e) for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "504", "timed out", "Timeout")):
                raise
            wait = min(2 ** attempt * 3, 60)
            print(f"      {type(e).__name__}: {str(e)[:120]} -> waiting {wait}s", flush=True)
            time.sleep(wait)
    raise last


def cached_call(prompt: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((get_model() + "\n" + prompt).encode("utf-8")).hexdigest()
    f = CACHE / f"{key}.json"
    hit = f.exists()
    t0 = time.time()
    text = json.loads(f.read_text(encoding="utf-8"))["response"] if hit else call_llm(prompt)
    print(f"      {'cache hit' if hit else 'answer received'} ({time.time() - t0:.1f}s)", flush=True)
    if not hit:
        f.write_text(json.dumps({"response": text}), encoding="utf-8")
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "model": get_model(), "key": key[:12],
                             "cached": hit, "prompt": prompt, "response": text}) + "\n")
    return text


# ------------------------------------------------------------------ automatic checks
NUM_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")     # digits glued to letters (F1, x2) are not numbers


def _texts(out: dict) -> list:
    t = [out.get("summary") or ""]
    t += [k.get("text", "") for k in out.get("key_points") or []]
    t += [(out.get("comparison") or {}).get("text", "")]
    t += [c.get("text", "") for c in out.get("caveats") or []]
    return t


def _cites(out: dict) -> list:
    c = list(out.get("summary_cites") or [])
    for k in out.get("key_points") or []:
        c += k.get("cites") or []
    c += (out.get("comparison") or {}).get("cites") or []
    for k in out.get("caveats") or []:
        c += k.get("cites") or []
    return c


def _feature_names() -> list:
    import csv
    with open(DICTIONARY, encoding="utf-8") as fh:
        return [r["feature"] for r in csv.DictReader(fh)]


def parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    return json.loads(t)


def run_checks(text: str, clean: dict):
    """returns (parsed_or_None, [failure messages])"""
    try:
        out = parse_json(text)
        assert isinstance(out, dict)
    except Exception:
        return None, ["(a) the answer is not valid JSON"]
    fails = []
    if not (out.get("summary") and isinstance(out.get("key_points"), list) and out.get("key_points")
            and isinstance(out.get("caveats"), list) and out.get("caveats")):
        fails.append("(a) missing summary, key_points or at least one caveat")
        return out, fails
    ids = evidence_ids(clean)
    bad = sorted({c for c in _cites(out) if c not in ids})
    if bad:
        fails.append(f"(b) cited IDs that do not exist: {bad}")
    if any(not (k.get("cites")) for k in out["key_points"] + out["caveats"]):
        fails.append("(b) every key point and caveat must cite at least one ID")
    comp, sub = out.get("comparison"), clean.get("submitted")
    has_ret = bool((clean.get("retrieved") or {}).get("human") or (clean.get("retrieved") or {}).get("ai"))
    if comp and not (sub and has_ret):
        fails.append("(b) without submitted code S and retrieved examples the comparison must be null")
    elif comp and "S" not in (comp.get("cites") or []):
        fails.append("(b) the comparison must cite S (the submitted code)")
    ev_nums = {round(float(x), 6) for x in NUM_RE.findall(json.dumps(_no_code(clean)))}
    wrong = sorted({x for t in _texts(out) for x in NUM_RE.findall(t) if round(float(x), 6) not in ev_nums})
    if wrong:
        fails.append(f"(c) numbers not found in the evidence: {wrong}")
    shown = {f["name"] for f in clean.get("features", [])}
    blob = " ".join(_texts(out))
    extra = sorted(n for n in _feature_names() if n not in shown and re.search(rf"\b{re.escape(n)}\b", blob))
    if extra:
        fails.append(f"(d) features named that are not in the evidence: {extra}")
    d = clean["detector"]
    if out.get("verdict") != d["verdict"] or out.get("confidence") != d["confidence"]:
        fails.append(f"(e) verdict/confidence must be copied exactly: verdict={d['verdict']}, confidence={d['confidence']}")
    if len(blob.split()) > MAX_WORDS:
        fails.append(f"(f) too long: {len(blob.split())} words (limit {MAX_WORDS})")
    return out, fails


def _no_code(clean: dict) -> dict:
    c = copy.deepcopy(clean)
    for side in ("human", "ai"):
        for it in (c.get("retrieved") or {}).get(side, []):
            it.pop("code", None)
    if c.get("submitted"):
        c["submitted"].pop("code", None)
    return c


# ------------------------------------------------------------------ fallback template (no LLM)
def fallback(clean: dict) -> dict:
    d = clean["detector"]
    if d["confidence"] == "inconclusive":
        summary = (f"The score is too close to the decision threshold to be reliable "
                   f"(verdict: {d['verdict']}, confidence: inconclusive).")
    else:
        summary = f"The detector's verdict is {d['verdict']} with {d['confidence']} confidence (P(AI) = {d['p_ai']})."
    kps = [{"text": f"{f['name']} = {f['value']} (human mean {f['human_mean']}, AI mean {f['ai_mean']}) "
                    f"is consistent with {f['points_toward']} authorship.", "cites": [f["id"]]}
           for f in clean.get("features", [])[:3]]
    ret = clean.get("retrieved") or {}
    h, a = (ret.get("human") or [None])[0], (ret.get("ai") or [None])[0]
    comp = None
    if h and a:
        comp = {"text": f"The nearest human example (similarity {h['similarity']}) and the nearest AI example "
                        f"(similarity {a['similarity']}) are shown for comparison.", "cites": [h["id"], a["id"]]}
    return {"verdict": d["verdict"], "confidence": d["confidence"], "summary": summary, "summary_cites": [],
            "key_points": kps, "comparison": comp,
            "caveats": [{"text": c["text"], "cites": [c["id"]]} for c in clean.get("caveats", [])]}


# ------------------------------------------------------------------ main entry points
def explain_from_evidence(ev: dict, use_llm: bool = True, code: str | None = None) -> dict:
    if code and not ev.get("submitted"):
        ev = {**ev, "submitted": {"id": "S", "code": code}}     # the LLM may only compare what it can see
    clean = clean_for_llm(ev)
    prompt = render_prompt(clean)
    warnings, attempts, out, source = [], [], None, "fallback"
    if not clean.get("retrieved") or not (clean["retrieved"].get("human") or clean["retrieved"].get("ai")):
        warnings.append("no retrieved examples were used")
    if use_llm:
        p = prompt
        for attempt in (1, 2):
            text = cached_call(p)
            parsed, fails = run_checks(text, clean)
            attempts.append({"attempt": attempt, "failures": fails})
            if not fails:
                out, source = parsed, "llm"
                break
            p = (prompt + "\n\nYOUR PREVIOUS ANSWER FAILED THESE CHECKS:\n- " + "\n- ".join(fails)
                 + "\nFix them. Use only numbers exactly as written in the evidence. Reply with JSON only.")
    if out is None:
        out = fallback(clean)
        warnings.append("fallback template used" if use_llm else "offline mode: fallback template")
    return {"verdict": clean["detector"]["verdict"], "confidence": clean["detector"]["confidence"],
            "evidence": clean, "retrieved": clean.get("retrieved"), "explanation": out,
            "source": source, "attempts": attempts, "warnings": warnings}


def explain(code: str, problem_text: str | None = None, use_llm: bool = True) -> dict:
    from evidence import build_evidence
    ev = build_evidence(code)
    warn = []
    try:                                                    # Phase 4 retrieval, if it exists yet
        from retrieve import retrieve
        r = retrieve(code, problem_id=None, k_human=3, k_ai=3, mode="deploy")
        ev["problem"], ev["retrieved"] = r.get("problem"), r.get("retrieved")
    except ImportError:
        warn.append("retrieve.py not available yet")
    except Exception as e:
        warn.append(f"retrieve() failed: {e}")
    if problem_text and not ev.get("problem"):
        ev["problem"] = {"id": "P", "problem_id": None, "title": "", "excerpt": problem_text[:400], "source": "given"}
    res = explain_from_evidence(ev, use_llm=use_llm, code=code)
    res["warnings"] = warn + res["warnings"]
    return res


def _show(r: dict, name: str = ""):
    e = r["explanation"]
    print(f"\n=== {name} | verdict {r['verdict']} | confidence {r['confidence']} | source {r['source']} ===")
    print(e["summary"])
    for k in e["key_points"]:
        print(f"  - {k['text']}  {k['cites']}")
    if e.get("comparison"):
        print(f"  compare: {e['comparison']['text']}  {e['comparison']['cites']}")
    for c in e["caveats"]:
        print(f"  caveat: {c['text']}  {c['cites']}")
    if r["attempts"]:
        print("  checks:", r["attempts"])
    if r["warnings"]:
        print("  warnings:", r["warnings"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="run on tests/mock_evidence.json")
    ap.add_argument("--file", help="explain one real Python file")
    ap.add_argument("--offline", action="store_true", help="no LLM: test the fallback template")
    a = ap.parse_args()
    if a.mock:
        for case in json.loads(MOCK_PATH.read_text(encoding="utf-8")):
            print(f"\n>>> running {case['case']} ...", flush=True)
            _show(explain_from_evidence(case, use_llm=not a.offline), case["case"])
    elif a.file:
        _show(explain(Path(a.file).read_text(encoding="utf-8"), use_llm=not a.offline), a.file)
    else:
        ap.print_help()
