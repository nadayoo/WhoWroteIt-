"""
eval_retrieval.py - Phase 5, Nour: retrieval evaluation + final run.   ONE command regenerates everything:

    python src/eval_retrieval.py                 # uses the index in data/rag_aig/v2
    python src/eval_retrieval.py --rebuild       # audit + rebuild the index first, then evaluate
    python src/eval_retrieval.py --rag-dir data/rag_aig/v2_codenet --out results/phase45/ablation_codenet
    python src/eval_retrieval.py --compare results/phase45/ablation_codenet   # print two result tables side by side
    python src/eval_retrieval.py --usefulness-summary                         # after the 3 raters filled the sheet

Outputs (in results/phase45/ unless --out is given):
    rag_audit.txt, eval_set.csv (shared, made once with seed 42), retrieval_correctness.csv, problem_finder.csv,
    coverage.csv, retrieval_as_detector.csv, retrieval_results_table.csv, leakage_checks.txt,
    usefulness_sheet.md/.csv, plots (problem_finder_hitrate.png, same_problem_coverage.png, retrieval_vs_detector_roc.png)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import RagStore                               # noqa: E402
from wwi_normalize import code_hash, normalize_code          # noqa: E402

PROC = ROOT / "data" / "processed_aig"
SEED = 42
K = 3


# ----------------------------------------------------------------------------- evaluation set
def get_eval_set(m: pd.DataFrame, shared_dir: Path) -> pd.DataFrame:
    """The fixed 120-sample set (60 AI: 20 per LLM spread over scenarios, 60 human), seed 42, made once."""
    p = shared_dir / "eval_set.csv"
    if p.exists():
        es = pd.read_csv(p)
        print(f"using the existing {p.name} ({len(es)} samples)")
        return es
    t = m[m.split == "test"]
    parts = []
    for llm in ("codestral", "codellama", "gemini"):
        d = t[t.llm_name == llm].sample(frac=1, random_state=SEED)
        d = d.assign(_r=d.groupby("scenario").cumcount()).sort_values("_r", kind="stable").head(20)
        parts.append(d)
    parts.append(t[t.label == 0].sample(60, random_state=SEED))
    es = pd.concat(parts)[["sample_id", "label", "llm_name", "scenario", "problem_id", "split"]]
    shared_dir.mkdir(parents=True, exist_ok=True)
    es.to_csv(p, index=False)
    print(f"created {p} ({len(es)} samples, seed {SEED})")
    return es


# ----------------------------------------------------------------------------- helpers
def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def binmetrics(y, s, thr):
    yp = (s >= thr).astype(int)
    return {"acc": accuracy_score(y, yp), "f1": f1_score(y, yp), "auc": roc_auc_score(y, s)}


def run(store, row, mode, problem_id=None, exclude_self=True, finder="vote"):
    sid = store.ids[row]
    return store.retrieve_vec(store.emb[row], store.hash[row], problem_id, K, K, mode,
                              {sid} if exclude_self else None, finder)


# ----------------------------------------------------------------------------- 8. correctness tests
def correctness(store, es, m, public_n=10):
    rows = [store.row_of[s] for s in es.sample_id]
    res = []

    def add(name, ok, total, note=""):
        res.append({"test": name, "passed": int(ok), "total": int(total), "ok": ok == total, "note": note})

    bad_lab = bad_dup = bad_split = short = n = 0
    for mode in ("deploy", "strict"):
        for r in rows:
            out = run(store, r, mode, problem_id=store.prob[r])
            n += 1
            items = out["human"] + out["ai"]
            bad_lab += any(i["label"] != 0 for i in out["human"]) or any(i["label"] != 1 for i in out["ai"])
            bad_dup += any(i["code_hash"] == store.hash[r] or i["similarity"] >= store.near_dup for i in items)
            bad_split += (mode == "strict") and any(i["split"] != "train" for i in items)
            short += len(out["human"]) < K or len(out["ai"]) < K
    add("human results are label human, AI results are label AI", n - bad_lab, n, "120 queries x 2 modes")
    add("no exact or near duplicate of the query is returned", n - bad_dup, n, f"near-dup limit {store.near_dup}")
    add("strict mode returns only train-split examples", n - bad_split, n)
    add("every query gets 3 human + 3 AI examples (fill rule)", n - short, n)

    # the query itself must disappear even when the caller does NOT pass exclude_ids
    elig = [r for r in rows if store.modes["deploy"][r]]
    self_found = sum(any(i["sample_id"] == store.ids[r] for i in
                         (lambda o: o["human"] + o["ai"])(run(store, r, "deploy", store.prob[r], exclude_self=False)))
                     for r in elig)
    add("query never appears in its own results (no exclude_ids given)", len(elig) - self_found, len(elig),
        f"{len(elig)} queries that are in the corpus")

    # strict corpus content
    st = store.meta[store.meta.in_strict]
    add("strict corpus has no validation/test ids", int((st.split == "train").sum()), len(st))
    held = set(m[m.split != "train"].problem_id)
    add("strict corpus has no validation/test problems", int((~st.problem_id.isin(held)).sum()), len(st))

    # the full path (raw code string -> embedding -> results) matches the stored vectors
    from retrieve import retrieve
    pick = rows[:public_n]
    cos = [float(store.embed(m.set_index("sample_id").code_raw[store.ids[r]]) @ store.emb[r]) for r in pick]
    add("fresh embedding of raw code matches the stored vector (cos > 0.999)", sum(c > 0.999 for c in cos), len(cos))
    full_ok = 0
    for r in pick:
        o = retrieve(m.set_index("sample_id").code_raw[store.ids[r]], mode="deploy", store=store)
        full_ok += len(o["human"]) == K and len(o["ai"]) == K and store.ids[r] not in {i["sample_id"] for i in o["human"] + o["ai"]}
    add("retrieve(code) works end to end on raw code", full_ok, len(pick))

    # FAISS agrees with the exact numpy search
    try:
        import faiss
        ix = faiss.read_index(str(store.path / "index_deploy.faiss"))
        rws = np.load(store.path / "rows_deploy.npy")
        agree = 0
        for r in rows[:20]:
            _, I = ix.search(store.emb[r][None, :].astype("float32"), 2)
            exact = rws[np.argsort(-(store.emb[rws] @ store.emb[r]))[:2]]
            agree += set(rws[I[0]]) == set(exact)
        add("FAISS index gives the same neighbours as exact search", agree, 20)
    except Exception as e:
        add("FAISS index gives the same neighbours as exact search", 0, 0, f"skipped ({type(e).__name__})")
    return pd.DataFrame(res)


# ----------------------------------------------------------------------------- 9. problem finder
def problem_finder(store, es):
    rows = [store.row_of[s] for s in es.sample_id]
    out = {}
    answerable = 0
    for method in ("statement", "vote"):
        h1 = h5 = h1a = h5a = nA = 0
        for r in rows:
            idx, sims, _, _ = store.candidates(store.emb[r], store.hash[r], "deploy", {store.ids[r]})
            truth = store.prob[r]
            can = bool((store.prob[idx] == truth).any())
            if method == "vote":
                answerable += can
            if method == "statement" and not store.has_statements:
                continue
            top = [p for p, _ in store.rank_problems(store.emb[r], idx, sims, method)][:5]
            a, b = truth in top[:1], truth in top[:5]
            h1 += a; h5 += b; nA += can; h1a += a and can; h5a += b and can
        n = len(rows)
        if method == "statement" and not store.has_statements:
            out[method] = dict(hit1=np.nan, hit5=np.nan, hit1_answerable=np.nan, hit5_answerable=np.nan)
        else:
            out[method] = dict(hit1=h1 / n, hit5=h5 / n, hit1_answerable=h1a / max(nA, 1), hit5_answerable=h5a / max(nA, 1))
    out["_n"], out["_answerable"] = len(rows), answerable
    return out


# ----------------------------------------------------------------------------- 10. coverage
def coverage(store, es):
    rows = [store.row_of[s] for s in es.sample_id]
    table = []
    for label, mode, given in (("deploy, problem given", "deploy", True), ("deploy, problem inferred", "deploy", False),
                               ("strict, problem given", "strict", True)):
        full = part = none = 0
        for r in rows:
            o = run(store, r, mode, store.prob[r] if given else None)
            true = store.prob[r]
            same = [i["problem_id"] == true for i in o["human"] + o["ai"]]
            nh = sum(i["problem_id"] == true for i in o["human"]); na = sum(i["problem_id"] == true for i in o["ai"])
            if nh >= K and na >= K: full += 1
            elif any(same): part += 1
            else: none += 1
        n = len(rows)
        table.append({"setting": label, "full_3h_3ai_same_problem": full / n, "some_same_problem": part / n,
                      "fallback_only": none / n})
    return pd.DataFrame(table)


# ----------------------------------------------------------------------------- 11. retrieval as a detector (strict)
def knn_scores(store, q_rows, k):
    tr = np.flatnonzero(store.modes["knn_train"])
    sims = store.emb[q_rows] @ store.emb[tr].T
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    s = np.clip(np.take_along_axis(sims, top, 1), 1e-6, None)
    lab = store.lab[tr][top]
    return (s * lab).sum(1) / s.sum(1)


def saved_detector():
    name = "feat_xgb"
    mp = ROOT / "models" / "best_meta.json"
    if mp.exists():
        name = json.loads(mp.read_text()).get("best_model", name)
    p = ROOT / "results" / f"preds_{name}.parquet"
    if not p.exists():
        return name, None
    return name, pd.read_parquet(p)


def retrieval_detector(store):
    m = store.meta
    va = np.flatnonzero((m.split == "val") & (m.source == "aigcodeset"))
    te = np.flatnonzero((m.split == "test") & (m.source == "aigcodeset"))
    rows, curves = [], {}
    for k in (5, 15):
        sv, st_ = knn_scores(store, va, k), knn_scores(store, te, k)
        thr = best_threshold(store.lab[va], sv)                   # threshold chosen on validation only
        v, t = binmetrics(store.lab[va], sv, thr), binmetrics(store.lab[te], st_, thr)
        rows.append({"model": f"retrieval kNN (k={k})", "val_f1": v["f1"], "val_auc": v["auc"],
                     "test_f1": t["f1"], "test_auc": t["auc"], "test_acc": t["acc"], "threshold": thr})
        curves[f"retrieval kNN (k={k})"] = (store.lab[te], st_)
    name, pr = saved_detector()
    if pr is not None:
        v, t = pr[pr.split == "val"], pr[pr.split == "test"]
        rows.append({"model": f"saved detector ({name})",
                     "val_f1": f1_score(v.y_true, v.y_pred), "val_auc": roc_auc_score(v.y_true, v.y_score),
                     "test_f1": f1_score(t.y_true, t.y_pred), "test_auc": roc_auc_score(t.y_true, t.y_score),
                     "test_acc": accuracy_score(t.y_true, t.y_pred), "threshold": np.nan})
        curves[f"saved detector ({name})"] = (t.y_true.values, t.y_score.values)
    else:
        print(f"NOTE: results/preds_{name}.parquet not found - run train_models.py first for the comparison row")
    return pd.DataFrame(rows), curves


# ----------------------------------------------------------------------------- 12. usefulness sheet
def usefulness_sheet(store, es, m, out: Path, n=15):
    rng = np.random.RandomState(SEED)
    ai = es[es.label == 1].sample(8, random_state=SEED)
    hu = es[es.label == 0].sample(n - 8, random_state=SEED)
    qs = pd.concat([ai, hu]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    code = m.set_index("sample_id").code_raw
    md, sheet = ["# Usefulness check: are the retrieved examples comparable solutions to the same task?\n",
                 "Rate each side 0 = not comparable, 1 = partly, 2 = clearly comparable. Fill `usefulness_sheet.csv`.\n"], []
    for qi, q in qs.iterrows():
        r = store.row_of[q.sample_id]
        o = run(store, r, "deploy", None)                       # as in real use: problem inferred
        found = o["problem"]["problem_id"]
        qcode = "\n".join(str(code[q.sample_id]).split("\n")[:20])
        verdict = "OK" if found == q.problem_id else "WRONG"
        md.append(f"\n## Query {qi+1}  (true: {'AI' if q.label else 'human'}, problem {q.problem_id}, "
                  f"found {found} {verdict})\n```python\n{qcode}\n```")
        for key in ("human", "ai"):
            for it in o[key]:
                md.append(f"\n**{it['id']}** {it['sample_id']} sim={it['similarity']} "
                          f"{'same problem' if it['same_problem'] else 'OTHER problem'} {it['llm_name']}/{it['scenario']}\n"
                          f"```python\n{it['code'][:600]}\n```")
        sheet.append({"query": qi + 1, "sample_id": q.sample_id, "true_problem": q.problem_id, "found_problem": found,
                      "problem_correct": int(found == q.problem_id), "r1_human": "", "r1_ai": "", "r2_human": "",
                      "r2_ai": "", "r3_human": "", "r3_ai": "", "notes": ""})
    out.mkdir(parents=True, exist_ok=True)
    (out / "usefulness_sheet.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(sheet).to_csv(out / "usefulness_sheet.csv", index=False)
    print(f"wrote {out/'usefulness_sheet.md'} and usefulness_sheet.csv (15 queries; 3 raters fill the r*_ columns)")


def usefulness_summary(out: Path):
    d = pd.read_csv(out / "usefulness_sheet.csv")
    res = {}
    for side in ("human", "ai"):
        cols = [f"r{i}_{side}" for i in (1, 2, 3)]
        v = d[cols].apply(pd.to_numeric, errors="coerce")
        if v.isna().all().all():
            print("the sheet is still empty - fill r1/r2/r3 columns first"); return
        res[side] = {"mean_rating_0_to_2": float(np.nanmean(v.values)),
                     "share_rated_comparable(2)": float(np.nanmean(v.values == 2)),
                     "share_rated_not_comparable(0)": float(np.nanmean(v.values == 0))}
    res["problem_finder_correct_on_these_15"] = float(d.problem_correct.mean())
    (out / "retrieval_usefulness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------- leakage re-check
def leakage(m, store):
    L = []
    sp = {s: m[m.split == s] for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        L.append(f"shared problem_ids {a}/{b}: {len(set(sp[a].problem_id) & set(sp[b].problem_id))}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        L.append(f"shared code_hash {a}/{b}: {len(set(sp[a].code_hash) & set(sp[b].code_hash))}")
    st = store.meta[store.meta.in_strict]
    L.append(f"validation/test ids in the strict corpus: {int((st.split != 'train').sum())}")
    L.append(f"validation/test problems in the strict corpus: {len(set(st.problem_id) & (set(sp['val'].problem_id) | set(sp['test'].problem_id)))}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-dir", default=str(ROOT / "data" / "rag_aig" / "v2"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase45"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--backend", default="st")
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--usefulness-summary", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.usefulness_summary:
        usefulness_summary(out); return
    if a.rebuild:
        import build_rag
        build_rag.audit()
        build_rag.build(Path(a.rag_dir), a.backend, a.model or build_rag.DEFAULT_ST_MODEL, "codenet" in a.rag_dir)

    m = pd.read_parquet(PROC / "merged_all.parquet")
    store = RagStore(a.rag_dir)
    print(f"\nindex: {a.rag_dir}  backend={store.cfg['backend']} model={store.cfg['model']} dim={store.cfg['dim']}"
          f"  strict={store.cfg['n_strict']} deploy={store.cfg['n_deploy']}")
    es = get_eval_set(m, ROOT / "results" / "phase45")
    es = es[es.sample_id.isin(store.row_of)].reset_index(drop=True)

    print("\n=== 8. Correctness tests ===")
    ct = correctness(store, es, m)
    print(ct.to_string(index=False)); ct.to_csv(out / "retrieval_correctness.csv", index=False)

    print("\n=== 9. Problem finder (deploy mode, leave-one-out, problem id hidden) ===")
    pf = problem_finder(store, es)
    pft = pd.DataFrame({k: v for k, v in pf.items() if not k.startswith("_")}).T
    print(pft.round(3).to_string())
    print(f"queries whose problem has at least one other example in the corpus: {pf['_answerable']} of {pf['_n']}")
    pft.to_csv(out / "problem_finder.csv")

    print("\n=== 10. Same-problem coverage ===")
    cv = coverage(store, es)
    print(cv.round(3).to_string(index=False)); cv.to_csv(out / "coverage.csv", index=False)

    print("\n=== 11. Retrieval as a detector (strict mode, train-only neighbours) ===")
    det, curves = retrieval_detector(store)
    print(det.round(4).to_string(index=False)); det.to_csv(out / "retrieval_as_detector.csv", index=False)

    print("\n=== 12. Usefulness sheet ===")
    usefulness_sheet(store, es, m, out)

    print("\n=== 13. Leakage and retrieval checks ===")
    lk = leakage(m, store)
    n_ok = int(ct.passed.sum()); n_tot = int(ct.total.sum())
    lk += f"\nretrieval correctness checks passed: {n_ok} of {n_tot}"
    print(lk); (out / "leakage_checks.txt").write_text(lk)

    # ---- the table for the report (section 9 of the plan)
    f = lambda x: "n/a" if pd.isna(x) else f"{x:.3f}"
    k5, k15 = det.iloc[0], det.iloc[1]
    sv = det[det.model.str.startswith("saved")]
    rowsT = [("Problem finder hit rate at 1 / at 5 (statement-based)", f"{f(pf['statement']['hit1'])} / {f(pf['statement']['hit5'])}"),
             ("Problem finder hit rate at 1 / at 5 (code-neighbor vote)", f"{f(pf['vote']['hit1'])} / {f(pf['vote']['hit5'])}"),
             ("Same, only queries whose problem is in the corpus (vote)", f"{f(pf['vote']['hit1_answerable'])} / {f(pf['vote']['hit5_answerable'])}"),
             ("Queries with k human and k AI examples from the same problem (given problem)", f(cv.iloc[0].full_3h_3ai_same_problem)),
             ("Same, problem inferred by the finder", f(cv.iloc[1].full_3h_3ai_same_problem)),
             ("Correctness tests passed", f"{int(ct.ok.sum())} of {len(ct)} tests ({n_ok} of {n_tot} checks)"),
             ("Retrieval as a detector (k = 5): test F1 / AUC", f"{k5.test_f1:.3f} / {k5.test_auc:.3f}"),
             ("Retrieval as a detector (k = 15): test F1 / AUC", f"{k15.test_f1:.3f} / {k15.test_auc:.3f}"),
             ("Saved detector, for comparison: test F1 / AUC", "n/a (run train_models.py)" if sv.empty else f"{sv.iloc[0].test_f1:.3f} / {sv.iloc[0].test_auc:.3f}")]
    tab = pd.DataFrame(rowsT, columns=["Measure", "Value"])
    tab.to_csv(out / "retrieval_results_table.csv", index=False)
    print("\n=== RETRIEVAL RESULTS TABLE (for the report) ===")
    print(tab.to_string(index=False))

    if not a.no_plots:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4)); w = 0.35; x = np.arange(2)
        ax.bar(x - w/2, [pf["statement"]["hit1"], pf["statement"]["hit5"]], w, label="statement-based")
        ax.bar(x + w/2, [pf["vote"]["hit1"], pf["vote"]["hit5"]], w, label="code-neighbour vote")
        ax.set_xticks(x); ax.set_xticklabels(["hit@1", "hit@5"]); ax.set_ylim(0, 1); ax.legend()
        ax.set_title("Problem finder (problem id hidden)"); plt.tight_layout()
        plt.savefig(out / "problem_finder_hitrate.png", dpi=150); plt.close()
        cv.set_index("setting")[["full_3h_3ai_same_problem", "some_same_problem", "fallback_only"]].plot.bar(
            stacked=True, figsize=(7, 4), rot=15); plt.ylabel("share of queries"); plt.tight_layout()
        plt.savefig(out / "same_problem_coverage.png", dpi=150); plt.close()
        plt.figure(figsize=(5.5, 5))
        for name, (y, s) in curves.items():
            fpr, tpr, _ = roc_curve(y, s); plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title("Retrieval as a detector vs saved detector (test)"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out / "retrieval_vs_detector_roc.png", dpi=150); plt.close()
        print("plots saved in", out)

    if a.compare:
        o = pd.read_csv(Path(a.compare) / "retrieval_results_table.csv")
        print("\n=== COMPARISON (this run vs", a.compare, ") ===")
        print(tab.merge(o, on="Measure", suffixes=(" [this run]", " [other]")).to_string(index=False))


if __name__ == "__main__":
    main()
