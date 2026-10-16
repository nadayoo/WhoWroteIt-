"""Builds a FAKE mini Project_CodeNet (same folder layout / columns as the real one)
so you can test codenet_subsetting.py before touching the 14M-row dataset.
It plants traps: Python 2 files, overlap with AIGCodeSet (by ID and by code), CRLF,
tiny / huge / comment-only files, syntax errors, whitespace-only copies of AIGCodeSet's code."""
import os, random
from pathlib import Path
import pandas as pd

AIG_FILE = os.environ.get("AIG_PATH", "aigcodeset_clean.parquet")
ROOT = Path(os.environ.get("CODENET_ROOT", "mock/Project_CodeNet"))
random.seed(1)

aig = pd.read_parquet(AIG_FILE)
pool_a = sorted(aig.problem_id.unique())
pool_b = [f"p{n:05d}" for n in range(4100, 4160)]          # 60 extra AtCoder-like problems
aizu   = [f"p{n:05d}" for n in range(100, 110)]            # AIZU problems (should be ignored for Pool B)
all_p  = pool_a + pool_b + aizu

(ROOT / "metadata").mkdir(parents=True, exist_ok=True)
(ROOT / "problem_descriptions").mkdir(exist_ok=True)

pd.DataFrame({
    "id": all_p,
    "name": [f"Problem {p}" for p in all_p],
    "dataset": ["AtCoder"] * (len(pool_a) + len(pool_b)) + ["AIZU"] * len(aizu),
    "time_limit": 2000, "memory_limit": 1048576, "rating": None, "tags": None, "complexity": None,
}).to_csv(ROOT / "metadata" / "problem_list.csv", index=False)

def make_code(seed, style):
    r = random.Random(seed)
    n = r.randint(3, 25)
    lines = ["import sys", f"# solution {seed}"]
    for i in range(n):
        lines.append(f"x{i} = {r.randint(0, 10**6)} + {r.randint(0, 99)}  # step {i}")
    lines.append("print(x0)")
    code = "\n".join(lines)
    if style == "crlf":   code = code.replace("\n", "\r\n")
    if style == "tiny":   code = "print(1)"
    if style == "huge":   code = code + "\n" + "\n".join(f"y{i} = {i}" for i in range(400))
    if style == "comments": code = "# only\n# comments\n# here\n" + "# x\n" * 20
    if style == "py2":    code = code + "\nprint 'hello'"
    if style == "syntax": code = code + "\nif x0 ==:\n    pass"
    return code

aig_h = aig[aig.label == 0].reset_index(drop=True)
statuses = ["Accepted", "Wrong Answer", "Runtime Error"]
sid = 100_000_000
for pid in all_p:
    rows = []
    (ROOT / "data" / pid / "Python").mkdir(parents=True, exist_ok=True)
    for st in statuses:
        for k in range(45):
            sid += 1
            s = f"s{sid}"
            style = random.choices(
                ["normal", "crlf", "tiny", "huge", "comments", "py2", "syntax", "dup", "aig_id", "aig_code", "aig_ws"],
                weights=[60, 6, 4, 3, 3, 6, 3, 4, 3, 4, 4])[0]
            orig = "Python (3.8.2)" if random.random() < .5 else "PyPy3 (7.3.0)"
            code = make_code(sid, "normal" if style in ("normal", "dup", "aig_id", "aig_code", "aig_ws") else style)
            if style == "py2":
                orig = "Python (2.7.6)"
            if style == "syntax" and st != "Runtime Error":
                pass                                   # syntax error in non-RE -> must be dropped
            if style == "dup":
                code = make_code(sid % 7, "normal")    # many identical codes -> internal duplicates
            if style == "aig_code" and len(aig_h):   # exactly the same code as an AIGCodeSet human sample
                code = aig_h.code_raw.iloc[sid % len(aig_h)]
            if style == "aig_ws" and len(aig_h):     # same code + trailing spaces (loose-hash trap)
                code = "\n".join(l + "  " for l in aig_h.code_raw.iloc[(sid * 3) % len(aig_h)].split("\n")) + "\n\n"
            if style == "aig_id":                     # reuse a real AIGCodeSet submission id
                s = aig_h.submission_id.iloc[sid % len(aig_h)]
            (ROOT / "data" / pid / "Python" / f"{s}.py").write_bytes(code.encode("utf-8"))
            rows.append(dict(submission_id=s, problem_id=pid, user_id="u1", date=1, language="Python",
                             original_language=orig, filename_ext="py", status=st, cpu_time=1, memory=1,
                             code_size=len(code.encode("utf-8")), accuracy=None))
    # a few non-python / bad-status rows that must be ignored
    for j in range(5):
        sid += 1
        rows.append(dict(submission_id=f"s{sid}", problem_id=pid, user_id="u", date=1, language="C++",
                         original_language="C++14", filename_ext="cpp", status="Accepted", cpu_time=1,
                         memory=1, code_size=300, accuracy=None))
        sid += 1
        rows.append(dict(submission_id=f"s{sid}", problem_id=pid, user_id="u", date=1, language="Python",
                         original_language="Python (3.8.2)", filename_ext="py", status="Time Limit Exceeded",
                         cpu_time=1, memory=1, code_size=300, accuracy=None))
    pd.DataFrame(rows).to_csv(ROOT / "metadata" / f"{pid}.csv", index=False)
    # description (skip one Pool B problem to test the missing-description path)
    if pid != pool_b[3]:
        (ROOT / "problem_descriptions" / f"{pid}.html").write_text(
            f"<html><body><h1>Problem {pid}</h1><h2>Problem Statement</h2>"
            f"<p>Given <var>N</var> integers, with <var>N</var> &le; 10<sup>5</sup> and <var>a</var><sub>i</sub> &ge; 1.</p>"
            f"<h2>Input</h2><pre>3\n1 2 3\n</pre><script>x=1</script></body></html>", encoding="utf-8")
print("mock CodeNet written to", ROOT)
