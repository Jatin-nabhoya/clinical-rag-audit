"""
Phase 5 — RAGAS evaluation scorer.
Runs on Kaggle T4 (GPU). Requires: pip install ragas datasets

Metrics:
  faithfulness      — are all claims in the answer supported by the context?
  answer_relevancy  — how relevant is the answer to the question?
  context_precision — are the retrieved chunks relevant to the ground truth?
  context_recall    — can the ground truth be inferred from the retrieved chunks?

Usage (Kaggle):
    python src/evaluation/ragas_scorer.py
    python src/evaluation/ragas_scorer.py --model mistral
    python src/evaluation/ragas_scorer.py --model all
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVAL_QUESTIONS = ROOT / "data" / "processed" / "eval_questions.jsonl"
EVAL_DIR       = ROOT / "results" / "eval_hallucination_audit"

MODEL_DIR_NAMES = {
    "llama3":  "llama3_8b",
    "mistral": "mistral_7b",
    "phi3":    "phi3_mini",
}


def load_eval_questions() -> dict:
    qs = {}
    with open(EVAL_QUESTIONS) as f:
        for line in f:
            r = json.loads(line)
            qs[r["question"]] = r
    return qs


def load_model_results(model: str) -> dict:
    dir_name = MODEL_DIR_NAMES[model]
    path = EVAL_DIR / dir_name / "generations.jsonl"
    with open(path) as f:
        return {r["question"]: r for r in (json.loads(l) for l in f)}


REFUSAL_PHRASE = "does not contain enough information"

def is_refusal(answer: str) -> bool:
    a = answer.lower().strip()
    return REFUSAL_PHRASE in a or a.startswith("the provided context does not")


def build_ragas_dataset(model: str, eval_qs: dict, model_results: dict):
    """Build lists needed by RAGAS: questions, answers, contexts, ground_truths."""
    questions, answers, contexts, ground_truths, meta = [], [], [], [], []

    for q_text, eq in eval_qs.items():
        result = model_results.get(q_text)
        if result is None:
            continue
        answer = result["answer"]
        if is_refusal(answer):
            continue   # skip refusals — RAGAS metrics are undefined for refusals

        chunk_texts = [c["text"] for c in result.get("retrieved_chunks", [])]
        if not chunk_texts:
            continue

        questions.append(q_text)
        answers.append(answer)
        contexts.append(chunk_texts)
        ground_truths.append(eq["gold_answer"])
        meta.append({
            "question_id": eq["question_id"],
            "tier":        eq["tier"],
            "sub_tier":    eq["sub_tier"],
            "domain":      eq["domain"],
            "model":       model,
        })

    return questions, answers, contexts, ground_truths, meta


def run_ragas(model: str, eval_qs: dict):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    print(f"\n  Building dataset for {model}…")
    questions, answers, contexts, ground_truths, meta = build_ragas_dataset(
        model, eval_qs, load_model_results(model)
    )
    print(f"  Rows for RAGAS: {len(questions)} (refusals excluded)")

    dataset = Dataset.from_dict({
        "question":      questions,
        "answer":        answers,
        "contexts":      contexts,
        "ground_truth":  ground_truths,
    })

    print(f"  Running RAGAS metrics…")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # Attach metadata and save per-row results
    df = result.to_pandas()
    for col in ["question_id", "tier", "sub_tier", "domain", "model"]:
        df[col] = [m[col] for m in meta]

    model_dir = EVAL_DIR / MODEL_DIR_NAMES.get(model, model)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_csv = model_dir / "ragas_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

    # Print per-tier summary
    print(f"\n  {'─'*60}")
    print(f"  RAGAS scores — {model.upper()}")
    print(f"  {'─'*60}")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"  {'Tier':<16}" + "".join(f"  {m[:8]:>10}" for m in metrics))
    print(f"  {'─'*16}" + "  " + "  ".join(["─"*10]*4))

    for tier in ["answerable", "partial", "ambiguous", "ALL"]:
        sub = df if tier == "ALL" else df[df["tier"] == tier]
        if sub.empty:
            continue
        vals = []
        for m in metrics:
            v = sub[m].mean() if m in sub.columns else float("nan")
            vals.append(f"{v:.4f}")
        print(f"  {tier:<16}" + "".join(f"  {v:>10}" for v in vals))

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama3", "mistral", "phi3", "all"], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 5 — RAGAS Evaluation")
    print("=" * 70)

    eval_qs = load_eval_questions()
    print(f"  Eval questions: {len(eval_qs)}")

    targets = ["llama3", "mistral", "phi3"] if args.model == "all" else [args.model]
    all_dfs = []

    for model in targets:
        df = run_ragas(model, eval_qs)
        all_dfs.append(df)

    # Combined file
    if len(all_dfs) > 1:
        import pandas as pd
        combined = pd.concat(all_dfs, ignore_index=True)
        out = EVAL_DIR / "ragas_scores.csv"
        combined.to_csv(out, index=False)
        print(f"\n  Combined RAGAS results → {out}")

        # Cross-model summary
        print(f"\n{'═'*70}")
        print("  CROSS-MODEL RAGAS SUMMARY")
        print(f"{'═'*70}")
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        print(f"\n  {'Model':<10}" + "".join(f"  {m[:8]:>10}" for m in metrics))
        print(f"  {'─'*10}" + "  " + "  ".join(["─"*10]*4))
        for model in targets:
            sub = combined[combined["model"] == model]
            vals = [f"{sub[m].mean():.4f}" if m in sub.columns else "N/A" for m in metrics]
            print(f"  {model:<10}" + "".join(f"  {v:>10}" for v in vals))

    print(f"\n{'═'*70}")
    print("  Done. Next: python scripts/hallucination_report.py")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
