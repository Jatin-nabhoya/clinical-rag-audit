"""
Clinical RAG Hallucination Audit — Report Generator.
Reads results/eval_hallucination_audit/summary.json and (optionally)
ragas_scores.csv, then prints a full audit report and saves
results/reports/hallucination_analysis.json.

Usage:
    python scripts/generate_report.py               # after analyze_hallucinations.py
    python scripts/generate_report.py --with-ragas  # after ragas_scorer.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR    = ROOT / "results" / "eval_hallucination_audit"
REPORTS_DIR = ROOT / "results" / "reports"
MODELS      = ["llama3_8b", "mistral_7b", "phi3_mini"]


def load_summary() -> dict:
    p = EVAL_DIR / "summary.json"
    if not p.exists():
        print("[ERROR] summary.json not found. Run analyze_hallucinations.py first.")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def load_ragas() -> dict | None:
    p = EVAL_DIR / "ragas_scores.csv"
    if not p.exists():
        return None
    try:
        import csv
        rows = []
        with open(p) as f:
            rows = list(csv.DictReader(f))
        # Aggregate per model per tier
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(list))
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        for r in rows:
            model = r.get("model", "")
            tier  = r.get("tier", "ALL")
            for m in metrics:
                if r.get(m) not in (None, "", "nan"):
                    try:
                        agg[model][tier].append(float(r[m]))
                    except ValueError:
                        pass
        # Build summary: {model: {tier: {metric: avg}}}
        result = {}
        for model in MODELS:
            result[model] = {}
            for tier in ["answerable", "partial", "ambiguous", "ALL"]:
                subset_rows = [r for r in rows if r.get("model") == model
                               and (tier == "ALL" or r.get("tier") == tier)]
                if not subset_rows:
                    continue
                result[model][tier] = {}
                for m in metrics:
                    vals = []
                    for r in subset_rows:
                        try:
                            vals.append(float(r[m]))
                        except (ValueError, KeyError):
                            pass
                    result[model][tier][m] = round(sum(vals)/len(vals), 4) if vals else None
        return result
    except Exception as e:
        print(f"  [WARNING] Could not load RAGAS results: {e}")
        return None


def rank(models, summary, tier, metric, higher_better=True):
    vals = [(m, summary.get(m, {}).get(tier, {}).get(metric, None)) for m in models]
    vals = [(m, v) for m, v in vals if v is not None]
    vals.sort(key=lambda x: x[1], reverse=higher_better)
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ragas", action="store_true")
    args = parser.parse_args()

    summary = load_summary()
    ragas   = load_ragas() if args.with_ragas else None

    print("=" * 70)
    print("  CLINICAL RAG HALLUCINATION AUDIT — PHASE 5 REPORT")
    print("=" * 70)

    # ── 1. Overall refusal rates ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  1. OVERALL REFUSAL RATES")
    print(f"{'─'*70}")
    print(f"  {'Model':<12} {'All':>8} {'Answerable':>12} {'Partial':>10} {'Ambiguous':>10} {'Unanswer':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for model in MODELS:
        s = summary.get(model, {})
        row = [
            f"{s.get('ALL',       {}).get('refusal_rate', 'N/A'):>8}",
            f"{s.get('answerable',{}).get('refusal_rate', 'N/A'):>12}",
            f"{s.get('partial',   {}).get('refusal_rate', 'N/A'):>10}",
            f"{s.get('ambiguous', {}).get('refusal_rate', 'N/A'):>10}",
            f"{s.get('unanswerable',{}).get('refusal_rate','N/A'):>10}",
        ]
        print(f"  {model:<12}" + "".join(row))

    print(f"\n  NOTE: High refusal on 'unanswerable' = GOOD (model correctly refuses).")
    print(f"        Low refusal on 'answerable'    = GOOD (model answers when it should).")

    # ── 2. Answer quality (ROUGE-L + keyword recall) ──────────────────────
    print(f"\n{'─'*70}")
    print("  2. ANSWER QUALITY — ROUGE-L & KEYWORD RECALL (non-refusals only)")
    print(f"{'─'*70}")
    print(f"  {'Model':<12} {'ROUGE-L(all)':>13} {'ROUGE-L(ans)':>13} {'KW-Recall(all)':>15}")
    print(f"  {'─'*12} {'─'*13} {'─'*13} {'─'*15}")
    for model in MODELS:
        s = summary.get(model, {})
        rl_all = s.get("ALL",       {}).get("rouge_l_avg",    "N/A")
        rl_ans = s.get("answerable",{}).get("rouge_l_avg",    "N/A")
        kr_all = s.get("ALL",       {}).get("keyword_recall", "N/A")
        print(f"  {model:<12} {str(rl_all):>13} {str(rl_ans):>13} {str(kr_all):>15}")

    # ── 3. RAGAS metrics (if available) ──────────────────────────────────
    if ragas:
        print(f"\n{'─'*70}")
        print("  3. RAGAS METRICS")
        print(f"{'─'*70}")
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        print(f"  {'Model':<12}" + "".join(f"  {m[:10]:>12}" for m in metrics))
        print(f"  {'─'*12}" + "  " + "  ".join(["─"*12]*4))
        for model in MODELS:
            vals = [str(ragas.get(model, {}).get("ALL", {}).get(m, "N/A")) for m in metrics]
            print(f"  {model:<12}" + "".join(f"  {v:>12}" for v in vals))

    # ── 4. Rankings ───────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  4. MODEL RANKINGS")
    print(f"{'─'*70}")

    ranked_rouge  = rank(MODELS, summary, "ALL", "rouge_l_avg",    higher_better=True)
    ranked_refans  = rank(MODELS, summary, "answerable", "refusal_rate", higher_better=False)
    ranked_refunans = rank(MODELS, summary, "unanswerable", "refusal_rate", higher_better=True)

    medals = ["🥇", "🥈", "🥉"]

    print(f"\n  Best answer quality (ROUGE-L, all tiers):")
    for i, (m, v) in enumerate(ranked_rouge):
        print(f"    {medals[i] if i < 3 else ' '} {m:<10} {v}")

    print(f"\n  Fewest wrong refusals on ANSWERABLE questions:")
    for i, (m, v) in enumerate(ranked_refans):
        print(f"    {medals[i] if i < 3 else ' '} {m:<10} refusal_rate={v}")

    print(f"\n  Most correct refusals on UNANSWERABLE questions:")
    for i, (m, v) in enumerate(ranked_refunans):
        print(f"    {medals[i] if i < 3 else ' '} {m:<10} refusal_rate={v}")

    # ── 5. Hallucination failure mode breakdown ───────────────────────────
    print(f"\n{'─'*70}")
    print("  5. HALLUCINATION FAILURE MODE BREAKDOWN")
    print(f"{'─'*70}")
    print("""
  Tier          | Hallucination type  | What it means
  ─────────────────────────────────────────────────────────────────────
  answerable    | factual_drift       | Model answered but changed facts
  partial       | gap_filling         | Model invented missing information
  ambiguous     | false_certainty     | Model picked one answer as definitive
  unanswerable  | fabrication         | Model answered instead of refusing
""")
    for model in MODELS:
        s = summary.get(model, {})
        print(f"  {model.upper()}:")
        ans_rr  = s.get("answerable",   {}).get("refusal_rate", "?")
        par_rr  = s.get("partial",      {}).get("refusal_rate", "?")
        amb_rr  = s.get("ambiguous",    {}).get("refusal_rate", "?")
        una_rr  = s.get("unanswerable", {}).get("refusal_rate", "?")
        ans_rl  = s.get("answerable",   {}).get("rouge_l_avg",  "?")
        par_rl  = s.get("partial",      {}).get("rouge_l_avg",  "?")
        # factual_drift risk: answered but low ROUGE-L
        print(f"    factual_drift  risk : ROUGE-L={ans_rl} on answerable Qs")
        print(f"    gap_filling    risk : {par_rr} refusal rate on partial Qs (low = model filled gaps)")
        print(f"    false_certainty risk: {amb_rr} refusal rate on ambiguous Qs")
        print(f"    fabrication    risk : {1 - float(una_rr) if una_rr != '?' else '?'} answer rate on unanswerable Qs")
        print()

    # ── Save report ───────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "local_analysis": summary,
        "ragas":          ragas,
        "rankings": {
            "rouge_l":              ranked_rouge,
            "answerable_refusal":   ranked_refans,
            "unanswerable_refusal": ranked_refunans,
        }
    }
    out = REPORTS_DIR / "hallucination_analysis.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'═'*70}")
    print(f"  Full report saved → {out}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
