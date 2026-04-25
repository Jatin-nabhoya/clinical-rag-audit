"""
Phase 8 — Final analysis script.
Generates all tables, statistical summaries, and example extracts
used in the final report. Uses only real project data.

Outputs (reports/phase8/tables/):
  headline_results.csv        — per-model overall taxonomy distribution
  per_tier_results.csv        — correct rate per model × tier
  context_overlap.csv         — faithfulness proxy with 95% CIs
  answer_length_stats.csv     — answer length distribution
  hallucination_examples.csv  — one real example per taxonomy category

Usage:
    cd project_root
    python reports/phase8/generate_analysis.py
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT     = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT / "results" / "eval_hallucination_audit"
OUT_DIR  = Path(__file__).resolve().parent / "tables"
OUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

MODELS       = ["llama3_8b", "mistral_7b", "phi3_mini"]
MODEL_LABELS = {"llama3_8b": "Llama-3-8B", "mistral_7b": "Mistral-7B", "phi3_mini": "Phi-3-mini"}
TIERS        = ["answerable", "partial", "ambiguous", "unanswerable"]
LABELS       = ["correct_refusal","grounded","over_refusal","fabrication",
                "gap_filling","factual_drift","false_certainty"]


def bci(vals, n=1000, ci=95):
    if not vals:
        return 0.0, 0.0, 0.0
    a     = np.array(vals, dtype=float)
    boots = [np.mean(np.random.choice(a, len(a), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(boots, [(100-ci)/2, 100-(100-ci)/2])
    return round(float(a.mean()),4), round(float(lo),4), round(float(hi),4)


def load_taxonomy():
    with open(EVAL_DIR / "taxonomy.csv") as f:
        return list(csv.DictReader(f))


def load_summary():
    with open(EVAL_DIR / "scoring_summary.json") as f:
        return json.load(f)


def load_generations(model):
    path = EVAL_DIR / model / "generations.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f]


# ── Table 1: Headline results ─────────────────────────────────────────────────
def table_headline(summary):
    rows = []
    for model in MODELS:
        s   = summary[model]["ALL"]
        pct = s["label_pct"]
        rows.append({
            "model":           MODEL_LABELS[model],
            "correct_pct":     round((pct.get("correct_refusal",0) + pct.get("grounded",0))*100, 1),
            "over_refusal_pct":round(pct.get("over_refusal",0)*100, 1),
            "fabrication_pct": round(pct.get("fabrication",0)*100, 1),
            "gap_filling_pct": round(pct.get("gap_filling",0)*100, 1),
            "factual_drift_pct":round(pct.get("factual_drift",0)*100, 1),
            "false_certainty_pct":round(pct.get("false_certainty",0)*100, 1),
            "n":               s["n"],
        })
    out = OUT_DIR / "headline_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── Table 2: Per-tier correct rate ────────────────────────────────────────────
def table_per_tier(summary):
    rows = []
    for tier in TIERS:
        row = {"tier": tier}
        for model in MODELS:
            d   = summary[model].get(tier, {})
            pct = d.get("label_pct", {})
            cr  = round((pct.get("correct_refusal",0)+pct.get("grounded",0))*100, 1)
            row[MODEL_LABELS[model]] = cr
            row[f"{MODEL_LABELS[model]}_n"] = d.get("n", 0)
        rows.append(row)
    out = OUT_DIR / "per_tier_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── Table 3: Context overlap with CIs ────────────────────────────────────────
def table_context_overlap(taxonomy_rows):
    rows = []
    for model in MODELS:
        vals = [float(r["context_overlap"]) for r in taxonomy_rows
                if r["model"]==model and float(r["context_overlap"])>0]
        mean, lo, hi = bci(vals)
        rows.append({
            "model":  MODEL_LABELS[model],
            "mean":   mean,
            "ci_lo":  lo,
            "ci_hi":  hi,
            "n":      len(vals),
            "interpretation": "Most grounded" if mean >= 0.5 else ("Moderate" if mean >= 0.3 else "Least grounded — uses parametric knowledge"),
        })
    out = OUT_DIR / "context_overlap.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── Table 4: Answer length statistics ────────────────────────────────────────
def table_answer_length():
    rows = []
    for model in MODELS:
        gens    = load_generations(model)
        lengths = [len(g["answer"].split()) for g in gens]
        a       = np.array(lengths)
        rows.append({
            "model":   MODEL_LABELS[model],
            "median":  int(np.median(a)),
            "mean":    round(float(a.mean()), 1),
            "p25":     int(np.percentile(a, 25)),
            "p75":     int(np.percentile(a, 75)),
            "min":     int(a.min()),
            "max":     int(a.max()),
            "note":    "Short, consistent (mostly refusals)" if np.median(a) < 55 else
                       ("Long, variable (answers + hedging)" if np.median(a) > 65 else "Moderate"),
        })
    out = OUT_DIR / "answer_length_stats.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── Table 5: Real hallucination examples ─────────────────────────────────────
def table_examples(taxonomy_rows):
    target_labels = ["fabrication","gap_filling","factual_drift",
                     "false_certainty","over_refusal","grounded","correct_refusal"]
    rows = []
    for label in target_labels:
        candidates = [r for r in taxonomy_rows if r["label"] == label]
        if not candidates:
            continue
        # Pick clearest example (shortest question, non-trivial answer)
        ex = sorted(candidates, key=lambda r: len(r["question"]))[0]
        rows.append({
            "label":        label,
            "model":        MODEL_LABELS.get(ex["model"], ex["model"]),
            "tier":         ex["tier"],
            "question":     ex["question"][:120],
            "model_answer": ex["answer_snippet"][:300],
            "gold_answer":  ex["gold_snippet"][:200],
            "rouge_l":      ex["rouge_l"],
        })
    out = OUT_DIR / "hallucination_examples.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── Print report-ready summary ────────────────────────────────────────────────
def print_report_numbers(headline, per_tier, overlap, lengths):
    print(f"\n{'═'*70}")
    print("  NUMBERS FOR THE REPORT")
    print(f"{'═'*70}")

    print("\n  FINDING 1: Model correctness ranking")
    for r in sorted(headline, key=lambda x: -x["correct_pct"]):
        print(f"    {r['model']:<14}: {r['correct_pct']:.1f}% correct overall")

    print("\n  FINDING 2: Over-refusal dominates failure modes")
    for r in headline:
        print(f"    {r['model']:<14}: {r['over_refusal_pct']:.1f}% over-refusal  "
              f"{r['fabrication_pct']:.1f}% fabrication")

    print("\n  FINDING 3: Phi-3 ignores context (faithfulness proxy)")
    for r in overlap:
        print(f"    {r['model']:<14}: context overlap = {r['mean']:.3f} "
              f"[{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]")

    print("\n  PER-TIER CORRECT RATE")
    print(f"  {'Tier':<14} {'Llama-3-8B':>12} {'Mistral-7B':>12} {'Phi-3-mini':>12}")
    for r in per_tier:
        la = str(r.get('Llama-3-8B','')) + "%"
        mi = str(r.get('Mistral-7B',''))  + "%"
        ph = str(r.get('Phi-3-mini',''))  + "%"
        print(f"  {r['tier']:<14} {la:>12} {mi:>12} {ph:>12}")

    print("\n  ANSWER LENGTH (behavioral fingerprint)")
    for r in lengths:
        print(f"    {r['model']:<14}: median={r['median']} words  — {r['note']}")

    print(f"\n  NOTE: No-RAG ablation not run (GPU time constraint).")
    print(f"  This is an acknowledged limitation — document it explicitly.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Phase 8 — Generating Analysis Tables")
    print("=" * 70)

    summary      = load_summary()
    taxonomy_rows = load_taxonomy()

    print()
    headline  = table_headline(summary)
    per_tier  = table_per_tier(summary)
    overlap   = table_context_overlap(taxonomy_rows)
    lengths   = table_answer_length()
    examples  = table_examples(taxonomy_rows)

    print_report_numbers(headline, per_tier, overlap, lengths)

    print(f"\n{'═'*70}")
    print(f"  Tables saved to: {OUT_DIR}")
    print(f"  Next: review final_report.md — replace [PLACEHOLDER] values if any remain")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
