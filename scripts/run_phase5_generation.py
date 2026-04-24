"""
Clinical RAG Hallucination Audit — Evaluation Generation Script.
Runs 110 eval questions through Llama-3-8B, Mistral-7B, and Phi-3-mini
sequentially on a Kaggle T4 GPU. Saves per-model generations + combined CSV.

Usage (Kaggle):
    python scripts/run_phase5_generation.py
    python scripts/run_phase5_generation.py --model mistral_7b
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import EVAL_QUESTIONS                          # noqa: E402
from src.retrieval.retriever import Retriever             # noqa: E402
from src.generation import LLMWrapper, RAGPipeline, MODEL_IDS  # noqa: E402

EVAL_DIR     = ROOT / "results" / "eval_hallucination_audit"
VECTOR_STORE = str(ROOT / "data" / "vector_store" / "medical")

MODEL_DIR_NAMES = {
    "llama3":  "llama3_8b",
    "mistral": "mistral_7b",
    "phi3":    "phi3_mini",
}


def load_questions() -> list[dict]:
    with open(EVAL_QUESTIONS) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_model(model_key: str, questions: list[dict], retriever: Retriever) -> list[dict]:
    model_dir = EVAL_DIR / MODEL_DIR_NAMES[model_key]
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}")
    print(f"  Model: {MODEL_DIR_NAMES[model_key]}  ({len(questions)} questions)")
    print(f"{'─'*70}")

    llm = LLMWrapper(model_key)
    rag = RAGPipeline(llm, retriever, k=5)
    results = []

    for i, q in enumerate(questions, 1):
        if i % 10 == 0 or i == 1:
            print(f"  [{i:3d}/{len(questions)}] {q['question'][:70]}")

        result = rag.answer(q["question"], use_rag=True)
        results.append({
            "question_id":       q["question_id"],
            "model":             MODEL_DIR_NAMES[model_key],
            "question":          q["question"],
            "tier":              q["tier"],
            "sub_tier":          q["sub_tier"],
            "domain":            q["domain"],
            "use_rag":           True,
            "k":                 result["k"],
            "retrieved_chunks":  result["retrieved_chunks"],
            "context":           result["context"],
            "answer":            result["answer"],
            "gold_answer":       q["gold_answer"],
            "expected_behavior": q["expected_behavior"],
        })

    # Per-model generations.jsonl
    out = model_dir / "generations.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(results)} generations → {out}")

    llm.unload()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=list(MODEL_IDS.keys()) + ["all"],
                        default="all",
                        help="llama3 | mistral | phi3 | all")
    args = parser.parse_args()

    print("=" * 70)
    print("  Clinical RAG Hallucination Audit — Evaluation Generation")
    print("=" * 70)

    questions = load_questions()
    print(f"\n  Eval questions   : {len(questions)}")
    print(f"  Vector store     : {VECTOR_STORE}")
    print(f"  Output directory : {EVAL_DIR}")
    retriever = Retriever(vector_store_dir=VECTOR_STORE)

    targets = list(MODEL_IDS.keys()) if args.model == "all" else [args.model]
    all_results = []

    for model_key in targets:
        results = run_model(model_key, questions, retriever)
        all_results.extend(results)

    # Combined CSV in eval root
    import csv
    out_csv = EVAL_DIR / "combined_results.csv"
    fieldnames = ["question_id", "model", "question", "tier", "sub_tier",
                  "domain", "answer", "gold_answer", "expected_behavior"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\n  Combined CSV → {out_csv}")

    print("=" * 70)
    print(f"  Done — {len(targets)} model(s), {len(questions)} questions each.")
    print(f"  Pull results then run: python scripts/analyze_hallucinations.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
