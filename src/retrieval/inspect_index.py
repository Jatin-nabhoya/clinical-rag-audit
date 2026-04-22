"""
Phase 3 — Index Inspection & Sanity Check
src/retrieval/inspect_index.py

Verifies FAISS indexes are working correctly.
Tests queries across 4 evaluation tiers and prints retrieval results.

Run:
  python src/retrieval/inspect_index.py              # checks both indexes
  python src/retrieval/inspect_index.py --index general
  python src/retrieval/inspect_index.py --index medical
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.retriever import Retriever  # noqa: E402

# ── Test Queries ───────────────────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "tier":   "answerable",
        "query":  "What are the first-line medications for hypertension?",
        "expect": ">0.40. Cardiology chunks. Should mention antihypertensives.",
    },
    {
        "tier":   "answerable",
        "query":  "What medications are used to treat tuberculosis?",
        "expect": ">0.40. Infectious disease chunks. Should mention antibiotics.",
    },
    {
        "tier":   "partial",
        "query":  "How does kidney disease affect blood pressure management?",
        "expect": "0.15–0.45. Related but indirect match across nephrology/cardiology.",
    },
    {
        "tier":   "ambiguous",
        "query":  "What are the risks of beta blockers?",
        "expect": "Any score valid. Multiple interpretations possible.",
    },
    {
        "tier":   "unanswerable",
        "query":  "What is the market size of AI in healthcare in 2024?",
        "expect": "<0.25. Off-topic. High score here = hallucination risk.",
    },
]

TOP_K = 5

INDEX_CONFIGS = {
    "general": str(ROOT / "data" / "vector_store" / "general"),
    "medical": str(ROOT / "data" / "vector_store" / "medical"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def score_label(score: float, tier: str) -> str:
    if tier == "answerable":
        return "GOOD" if score > 0.40 else "LOW — check corpus coverage"
    elif tier == "partial":
        return "OK"   if 0.15 < score < 0.55 else "UNEXPECTED"
    elif tier == "unanswerable":
        return "GOOD" if score < 0.25 else "HIGH — hallucination risk"
    return "OK"


def inspect_index(index_key: str, store_dir: str):
    print(f"\n{'='*62}")
    print(f"  Index: [{index_key.upper()}]  →  {store_dir}")
    print(f"{'='*62}")

    try:
        r = Retriever(vector_store_dir=store_dir)
    except FileNotFoundError as e:
        print(f"\n  Could not load index: {e}")
        return

    cfg = json.loads((Path(store_dir) / "embed_config.json").read_text())
    print(f"\n  Model   : {cfg['embedding_model']}")
    print(f"  Dim     : {cfg['embedding_dim']}  |  Vectors: {cfg['num_chunks']:,}")

    for item in TEST_QUERIES:
        tier  = item["tier"]
        query = item["query"]

        print(f"\n  {'─'*58}")
        print(f"  Tier  : [{tier.upper()}]")
        print(f"  Query : {query}")
        print(f"  Expect: {item['expect']}")
        print()

        results = r.retrieve(query, k=TOP_K)

        for res in results:
            source  = res["metadata"].get("source",  "?")
            domain  = res["metadata"].get("domain",  "?")
            title   = res["metadata"].get("title",   "")[:55]
            preview = res["text"][:160].replace("\n", " ")
            label   = f"  ← {score_label(res['score'], tier)}" if res["rank"] == 1 else ""

            print(f"    Rank {res['rank']}  score={res['score']:.4f}  "
                  f"{source}  {domain}{label}")
            print(f"    Title  : {title}")
            print(f"    Preview: {preview}...")
            print()


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'='*62}")
    print("  Score Reference")
    print(f"{'='*62}")
    print("""
  Tier            Top-1 score    What a bad result means
  ──────────────────────────────────────────────────────
  answerable      > 0.40         Corpus missing this topic
  partial         0.15 – 0.45   OK if chunks are related
  ambiguous       any            Multiple valid matches fine
  unanswerable    < 0.25         High score = hallucination risk

  Comparing general vs. medical:
    Medical should score higher on clinical terminology.
    Notable score difference = strong ablation finding for paper.
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        choices=list(INDEX_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which index to inspect (default: all)",
    )
    args = parser.parse_args()

    targets = (
        list(INDEX_CONFIGS.items())
        if args.index == "all"
        else [(args.index, INDEX_CONFIGS[args.index])]
    )

    for index_key, store_dir in targets:
        inspect_index(index_key, store_dir)

    print_summary()
    print("  Inspection complete.\n")


if __name__ == "__main__":
    main()
