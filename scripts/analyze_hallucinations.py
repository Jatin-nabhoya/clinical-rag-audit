"""
Clinical RAG Hallucination Audit — Hallucination Analysis Script.
Mac-compatible, no GPU needed. Reads generations from
results/eval_hallucination_audit/*/generations.jsonl.

Metrics computed per model per tier:
  - refusal_rate    : % of questions where model refused to answer
  - rouge_l         : ROUGE-L F1 vs gold answer (non-refusals only)
  - keyword_recall  : % of gold answer key terms present in model answer
  - answer_length   : avg word count of model answers

Saves:
  results/eval_hallucination_audit/metrics.csv   — per-question rows
  results/eval_hallucination_audit/summary.json  — aggregated per-model-per-tier

Usage:
    python scripts/analyze_hallucinations.py
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import EVAL_QUESTIONS  # noqa: E402

EVAL_DIR       = ROOT / "results" / "eval_hallucination_audit"
REFUSAL_PHRASE = "does not contain enough information"

MODEL_DIRS = {
    "llama3_8b":  EVAL_DIR / "llama3_8b",
    "mistral_7b": EVAL_DIR / "mistral_7b",
    "phi3_mini":  EVAL_DIR / "phi3_mini",
}

# ── Metrics ───────────────────────────────────────────────────────────────────

def is_refusal(answer: str) -> bool:
    a = answer.lower().strip()
    return REFUSAL_PHRASE in a or a.startswith("the provided context does not")


def lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 score."""
    h = re.findall(r"\w+", hypothesis.lower())
    r = re.findall(r"\w+", reference.lower())
    if not h or not r:
        return 0.0
    lcs = lcs_length(h, r)
    precision = lcs / len(h)
    recall    = lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_recall(answer: str, gold: str) -> float:
    """Fraction of gold key terms (≥5 chars) present in model answer."""
    stopwords = {"which", "these", "those", "their", "about", "after", "before",
                 "there", "where", "would", "could", "should", "other", "since"}
    gold_terms = {w for w in re.findall(r"[a-z]{5,}", gold.lower()) if w not in stopwords}
    if not gold_terms:
        return 1.0
    ans_lower = answer.lower()
    found = sum(1 for t in gold_terms if t in ans_lower)
    return found / len(gold_terms)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_eval_questions() -> dict:
    """Returns {question_text: record}"""
    qs = {}
    with open(EVAL_QUESTIONS) as f:
        for line in f:
            r = json.loads(line)
            qs[r["question"]] = r
    return qs


def load_model_results(model_dir_name: str) -> dict:
    """Returns {question_text: result_dict} from generations.jsonl"""
    path = MODEL_DIRS[model_dir_name] / "generations.jsonl"
    with open(path) as f:
        return {r["question"]: r for r in (json.loads(l) for l in f)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Clinical RAG Hallucination Audit — Analysis")
    print("=" * 70)

    eval_qs    = load_eval_questions()
    models     = list(MODEL_DIRS.keys())
    model_data = {m: load_model_results(m) for m in models}

    print(f"\n  Eval questions loaded : {len(eval_qs)}")
    for m in models:
        print(f"  [{m}] generations : {len(model_data[m])}")

    # ── Per-question analysis ─────────────────────────────────────────────
    rows = []
    missing = 0

    for q_text, eq in eval_qs.items():
        for model in models:
            result = model_data[model].get(q_text)
            if result is None:
                missing += 1
                continue

            answer     = result["answer"]
            gold       = eq["gold_answer"]
            tier       = eq["tier"]
            sub_tier   = eq["sub_tier"]
            domain     = eq["domain"]
            refused    = is_refusal(answer)

            rl  = rouge_l(answer, gold)    if not refused else 0.0
            kr  = keyword_recall(answer, gold) if not refused else 0.0
            wc  = len(answer.split())

            rows.append({
                "question_id":   eq["question_id"],
                "question":      q_text[:80],
                "tier":          tier,
                "sub_tier":      sub_tier,
                "domain":        domain,
                "model":         model,
                "refused":       refused,
                "rouge_l":       round(rl, 4),
                "keyword_recall":round(kr, 4),
                "answer_length": wc,
                "gold_answer":   gold[:120],
                "model_answer":  answer[:120],
            })

    if missing:
        print(f"\n  WARNING: {missing} question/model combos not found in generations")

    # ── Save per-question CSV ─────────────────────────────────────────────
    import csv
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EVAL_DIR / "metrics.csv"
    fieldnames = ["question_id", "question", "tier", "sub_tier", "domain",
                  "model", "refused", "rouge_l", "keyword_recall", "answer_length",
                  "gold_answer", "model_answer"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved {len(rows)} rows → {out_csv}")

    # ── Aggregate per model × tier ────────────────────────────────────────
    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {}
    for model in models:
        summary[model] = {}
        model_rows = [r for r in rows if r["model"] == model]
        tiers = ["answerable", "partial", "ambiguous", "unanswerable", "ALL"]
        for tier in tiers:
            subset = model_rows if tier == "ALL" else [r for r in model_rows if r["tier"] == tier]
            if not subset:
                continue
            refused = [r for r in subset if r["refused"]]
            answered = [r for r in subset if not r["refused"]]
            summary[model][tier] = {
                "n":               len(subset),
                "refusal_rate":    round(len(refused) / len(subset), 4),
                "rouge_l_avg":     avg([r["rouge_l"] for r in answered]),
                "keyword_recall":  avg([r["keyword_recall"] for r in answered]),
                "answer_len_avg":  avg([r["answer_length"] for r in subset]),
                "answered_n":      len(answered),
                "refused_n":       len(refused),
            }

    out_json = EVAL_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary      → {out_json}")

    # ── Print comparison table ────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  HALLUCINATION ANALYSIS — CROSS-MODEL COMPARISON")
    print(f"{'═'*70}")

    TIERS_ORDER = ["answerable", "partial", "ambiguous", "unanswerable", "ALL"]
    col_w = 14

    # Header
    header = f"  {'Tier':<16}" + "".join(f"{'  '+m:>{col_w}}" for m in models)
    print(f"\n  {'─'*66}")
    print("  REFUSAL RATE  (lower is better for answerable, higher for unanswerable)")
    print(f"  {'─'*66}")
    print(header)
    for tier in TIERS_ORDER:
        line = f"  {tier:<16}"
        for model in models:
            val = summary.get(model, {}).get(tier, {}).get("refusal_rate", "-")
            line += f"{str(val) if val != '-' else '-':>{col_w}}"
        print(line)

    print(f"\n  {'─'*66}")
    print("  ROUGE-L (answered questions only — higher is better)")
    print(f"  {'─'*66}")
    print(header)
    for tier in TIERS_ORDER:
        line = f"  {tier:<16}"
        for model in models:
            val = summary.get(model, {}).get(tier, {}).get("rouge_l_avg", "-")
            line += f"{str(val) if val != '-' else '-':>{col_w}}"
        print(line)

    print(f"\n  {'─'*66}")
    print("  KEYWORD RECALL (answered questions only — higher is better)")
    print(f"  {'─'*66}")
    print(header)
    for tier in TIERS_ORDER:
        line = f"  {tier:<16}"
        for model in models:
            val = summary.get(model, {}).get(tier, {}).get("keyword_recall", "-")
            line += f"{str(val) if val != '-' else '-':>{col_w}}"
        print(line)

    print(f"\n  {'─'*66}")
    print("  ANSWERED / TOTAL per tier per model")
    print(f"  {'─'*66}")
    print(header)
    for tier in TIERS_ORDER:
        line = f"  {tier:<16}"
        for model in models:
            s = summary.get(model, {}).get(tier, {})
            val = f"{s.get('answered_n','-')}/{s.get('n','-')}" if s else "-"
            line += f"{val:>{col_w}}"
        print(line)

    # ── Key findings ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  KEY FINDINGS")
    print(f"{'═'*70}")

    # Best model by overall ROUGE-L
    best_rouge = max(models, key=lambda m: summary.get(m, {}).get("ALL", {}).get("rouge_l_avg", 0))
    # Lowest refusal on answerable
    best_ans = min(models, key=lambda m: summary.get(m, {}).get("answerable", {}).get("refusal_rate", 1))
    # Highest refusal on unanswerable (safest)
    safest = max(models, key=lambda m: summary.get(m, {}).get("unanswerable", {}).get("refusal_rate", 0))

    print(f"\n  Best ROUGE-L overall  : {best_rouge}")
    print(f"  Fewest refusals on answerable Qs : {best_ans}")
    print(f"  Most correct refusals (unanswerable) : {safest}")

    all_rates = {m: summary.get(m, {}).get("ALL", {}).get("refusal_rate", 0) for m in models}
    print(f"\n  Overall refusal rates : " + " | ".join(f"{m}={v}" for m, v in all_rates.items()))

    print(f"\n{'═'*70}")
    print(f"  Done.")
    print(f"  metrics.csv → {EVAL_DIR / 'metrics.csv'}")
    print(f"  summary.json → {EVAL_DIR / 'summary.json'}")
    print(f"  Next: python scripts/generate_report.py")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
