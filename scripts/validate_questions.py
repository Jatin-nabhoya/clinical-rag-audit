"""
Phase 5 — Evaluation question validator.
Run after every batch of ~20 questions to catch problems early.

Checks:
  1. Schema — all required fields present and values in allowed sets
  2. Chunk IDs — gold_sources exist in chunks_clean.jsonl
  3. Retrieval — for answerable/partial, ≥1 gold source appears in top-10
  4. Unanswerable sanity — top-1 retrieval score < threshold
  5. Distribution — tier/sub_tier counts vs. targets

Usage:
    python scripts/validate_questions.py
    python scripts/validate_questions.py --strict   # exit 1 if any check fails
    python scripts/validate_questions.py --no-retrieval  # skip slow retrieval checks
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import CHUNKS_CLEAN, EVAL_QUESTIONS  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

ALLOWED = {
    "tier":               {"answerable", "partial", "ambiguous", "unanswerable"},
    "sub_tier":           {
        "direct_lookup", "single_chunk_reasoning", "multi_chunk_synthesis",
        "missing_specificity", "missing_subgroup", "missing_recent_update",
        "underspecified", "conflicting_sources",
        "out_of_domain", "in_domain_absent",
    },
    "expected_behavior":  {"cite_and_answer", "acknowledge_gap", "present_options", "refuse"},
    "difficulty":         {1, 2, 3},
}

REQUIRED_FIELDS = [
    "question_id", "question", "tier", "sub_tier", "hallucination_target",
    "gold_answer", "gold_sources", "expected_behavior", "domain",
    "annotated_on", "difficulty",
]

TARGET_COUNTS = {
    ("answerable",   "direct_lookup"):          10,
    ("answerable",   "single_chunk_reasoning"): 15,
    ("answerable",   "multi_chunk_synthesis"):   5,
    ("partial",      "missing_specificity"):    16,  # q_094 moved from unanswerable → partial
    ("partial",      "missing_subgroup"):       10,
    ("partial",      "missing_recent_update"):   5,
    ("ambiguous",    "underspecified"):         15,
    ("ambiguous",    "conflicting_sources"):     5,
    ("unanswerable", "out_of_domain"):          10,
    ("unanswerable", "in_domain_absent"):       19,  # q_094 moved out to partial
}

# PubMedBERT scores cluster 0.87–0.94 for all medical queries.
# Empirical midpoint between answerable avg (0.924) and unanswerable avg (0.909).
UNANSWERABLE_SCORE_THRESHOLD = 0.916
RETRIEVAL_TOP_K = 10


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[ERROR] {path} not found. Run annotate_questions.py first.")
        sys.exit(1)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_chunk_ids(path: Path) -> set[str]:
    with open(path) as f:
        return {json.loads(line)["chunk_id"] for line in f if line.strip()}


# ── Checks ────────────────────────────────────────────────────────────────────

def check_schema(questions: list[dict]) -> list[str]:
    errors = []
    for q in questions:
        qid = q.get("question_id", "?")
        for field in REQUIRED_FIELDS:
            if field not in q:
                errors.append(f"{qid}: missing field '{field}'")
        for field, allowed in ALLOWED.items():
            if field in q and q[field] not in allowed:
                errors.append(f"{qid}: '{field}' = {q[field]!r} not in {sorted(allowed)}")
        tier = q.get("tier", "")
        sub  = q.get("sub_tier", "")
        valid_subs = {
            "answerable":   {"direct_lookup", "single_chunk_reasoning", "multi_chunk_synthesis"},
            "partial":      {"missing_specificity", "missing_subgroup", "missing_recent_update"},
            "ambiguous":    {"underspecified", "conflicting_sources"},
            "unanswerable": {"out_of_domain", "in_domain_absent"},
        }
        if tier in valid_subs and sub and sub not in valid_subs[tier]:
            errors.append(f"{qid}: sub_tier '{sub}' is not valid for tier '{tier}'")
    return errors


def check_chunk_ids(questions: list[dict], valid_ids: set[str]) -> list[str]:
    errors = []
    for q in questions:
        for cid in q.get("gold_sources", []):
            if cid not in valid_ids:
                errors.append(f"{q['question_id']}: chunk_id '{cid}' not found in chunks_clean.jsonl")
    return errors


def check_retrieval(questions: list[dict], valid_ids: set[str]) -> list[str]:
    try:
        from src.retrieval.retriever import Retriever  # noqa: E402
    except ImportError:
        return ["[SKIP] Could not import Retriever — skipping retrieval checks."]

    retrieval_qs = [q for q in questions if q["tier"] in ("answerable", "partial")
                    and q.get("gold_sources")]
    unanswerable_qs = [q for q in questions if q["tier"] == "unanswerable"]

    if not retrieval_qs and not unanswerable_qs:
        return []

    print("  Loading retriever (medical index)…", end=" ", flush=True)
    retriever = Retriever(str(ROOT / "data" / "vector_store" / "medical"))
    print("ready.")

    errors = []

    for q in retrieval_qs:
        results = retriever.retrieve(q["question"], k=RETRIEVAL_TOP_K)
        retrieved_ids = {r["metadata"].get("chunk_id", "") for r in results}
        gold = set(q["gold_sources"])
        if not gold & retrieved_ids:
            errors.append(
                f"{q['question_id']} [{q['tier']}]: gold source(s) not in top-{RETRIEVAL_TOP_K} "
                f"— question may need rewording or different gold source"
            )

    for q in unanswerable_qs:
        results = retriever.retrieve(q["question"], k=1)
        if results and results[0]["score"] >= UNANSWERABLE_SCORE_THRESHOLD:
            errors.append(
                f"{q['question_id']} [unanswerable]: top-1 score={results[0]['score']:.3f} "
                f"≥ {UNANSWERABLE_SCORE_THRESHOLD} — corpus may actually cover this question"
            )

    return errors


def check_distribution(questions: list[dict]) -> list[str]:
    warnings = []
    counts = Counter((q["tier"], q["sub_tier"]) for q in questions)
    total = len(questions)
    for (tier, sub), target in TARGET_COUNTS.items():
        actual = counts.get((tier, sub), 0)
        if abs(actual - target) > 5:
            warnings.append(
                f"  {tier}/{sub}: {actual}/{target} "
                f"({'over' if actual > target else 'under'} by {abs(actual - target)})"
            )
    return warnings


# ── Report ────────────────────────────────────────────────────────────────────

def print_distribution(questions: list[dict]):
    counts = Counter((q["tier"], q["sub_tier"]) for q in questions)
    total = len(questions)
    print(f"\n  {'Tier':<14} {'Sub-tier':<28} {'Done':>4} / {'Target':>6}  {'Bar'}")
    print(f"  {'─'*14}  {'─'*28}  {'─'*4}   {'─'*6}  {'─'*20}")
    for (tier, sub), target in TARGET_COUNTS.items():
        done = counts.get((tier, sub), 0)
        bar = "█" * done + "░" * max(0, target - done)
        flag = " ✓" if done >= target else ""
        print(f"  {tier:<14}  {sub:<28}  {done:>4} / {target:>4}  {bar}{flag}")
    print(f"\n  Total: {total} / 110")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(EVAL_QUESTIONS))
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any check fails")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Skip retrieval checks (faster, no GPU needed)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 5 — Question Validation Report")
    print("=" * 70)

    questions = load_questions(Path(args.file))
    print(f"\n  Loaded {len(questions)} questions from {args.file}")

    valid_ids = load_chunk_ids(CHUNKS_CLEAN)
    print(f"  Loaded {len(valid_ids)} chunk IDs from chunks_clean.jsonl")

    all_errors = []
    all_warnings = []

    # Check 1 — schema
    print("\n  [1/4] Schema check…", end=" ")
    errs = check_schema(questions)
    print(f"{'OK' if not errs else f'{len(errs)} error(s)'}")
    all_errors.extend(errs)

    # Check 2 — chunk IDs
    print("  [2/4] Chunk ID check…", end=" ")
    errs = check_chunk_ids(questions, valid_ids)
    print(f"{'OK' if not errs else f'{len(errs)} error(s)'}")
    all_errors.extend(errs)

    # Check 3 — retrieval
    if args.no_retrieval:
        print("  [3/4] Retrieval check… SKIPPED (--no-retrieval)")
    else:
        print("  [3/4] Retrieval check…")
        errs = check_retrieval(questions, valid_ids)
        if errs and errs[0].startswith("[SKIP]"):
            print(f"    {errs[0]}")
        else:
            print(f"    {'OK' if not errs else f'{len(errs)} warning(s)'}")
            all_warnings.extend(errs)

    # Check 4 — distribution
    print("  [4/4] Distribution check…", end=" ")
    warns = check_distribution(questions)
    print(f"{'OK' if not warns else f'{len(warns)} warning(s)'}")
    all_warnings.extend(warns)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  DISTRIBUTION")
    print(f"{'─'*70}")
    print_distribution(questions)

    if all_errors:
        print(f"\n{'─'*70}")
        print(f"  ERRORS ({len(all_errors)}) — must fix before running RAGAS:")
        print(f"{'─'*70}")
        for e in all_errors:
            print(f"  ✗ {e}")

    if all_warnings:
        print(f"\n{'─'*70}")
        print(f"  WARNINGS ({len(all_warnings)}):")
        print(f"{'─'*70}")
        for w in all_warnings:
            print(f"  ⚠ {w}")

    print(f"\n{'─'*70}")
    if not all_errors and not all_warnings:
        print("  All checks passed.")
    elif not all_errors:
        print(f"  {len(all_warnings)} warning(s), 0 errors — good to continue annotating.")
    else:
        print(f"  {len(all_errors)} error(s) found — fix before proceeding.")
    print(f"{'─'*70}")

    if args.strict and all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
