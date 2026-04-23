"""
Phase 4 smoke test — run one question through all 3 LLMs with RAG.
Loads models sequentially to fit on a single GPU (T4 / 16 GB).

Usage:
    python scripts/test_rag_generation.py               # all 3 models
    python scripts/test_rag_generation.py --model mistral
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.retriever import Retriever          # noqa: E402
from src.generation import LLMWrapper, RAGPipeline, MODEL_IDS  # noqa: E402

TEST_QUESTION = "What are the common symptoms and first-line treatments for Type 2 diabetes?"

VECTOR_STORE = str(ROOT / "data" / "vector_store" / "medical")


def run_model(model_key: str, retriever: Retriever):
    print(f"\n{'─'*70}")
    print(f"  Model: {model_key.upper()}")
    print(f"{'─'*70}")

    llm = LLMWrapper(model_key)
    rag = RAGPipeline(llm, retriever, k=5)

    result = rag.answer(TEST_QUESTION)

    print(f"\nQuestion : {result['question']}")
    print(f"\nTop retrieved chunk:")
    if result["retrieved_chunks"]:
        top = result["retrieved_chunks"][0]
        print(f"  score={top['score']:.4f} | {top['metadata'].get('source')} | "
              f"{top['metadata'].get('domain')}")
        print(f"  {top['text'][:200].replace(chr(10), ' ')}...")

    print(f"\nAnswer:\n{result['answer']}\n")

    llm.unload()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_IDS.keys()) + ["all"],
        default="all",
        help="Which model to test (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 4 — RAG Generation Smoke Test")
    print("=" * 70)

    print(f"\nLoading retriever from: {VECTOR_STORE}")
    retriever = Retriever(vector_store_dir=VECTOR_STORE)

    targets = list(MODEL_IDS.keys()) if args.model == "all" else [args.model]

    for model_key in targets:
        run_model(model_key, retriever)

    print("=" * 70)
    print(f"  Done — {len(targets)} model(s) tested successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()