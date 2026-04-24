"""
Phase 5 — Interactive CLI annotation helper.
Prompts field-by-field and appends to data/processed/eval_questions.jsonl.
Prevents schema drift across multiple annotation sessions.

Usage:
    python scripts/annotate_questions.py
    python scripts/annotate_questions.py --file data/processed/eval_questions.jsonl

Press Ctrl+C at any time to stop cleanly.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import EVAL_QUESTIONS  # noqa: E402

# ── Allowed values ────────────────────────────────────────────────────────────

TIERS = ["answerable", "partial", "ambiguous", "unanswerable"]

SUB_TIERS = {
    "answerable":   ["direct_lookup", "single_chunk_reasoning", "multi_chunk_synthesis"],
    "partial":      ["missing_specificity", "missing_subgroup", "missing_recent_update"],
    "ambiguous":    ["underspecified", "conflicting_sources"],
    "unanswerable": ["out_of_domain", "in_domain_absent"],
}

HALLUCINATION_TARGETS = {
    "answerable":   "factual_drift",
    "partial":      "gap_filling",
    "ambiguous":    "false_certainty",
    "unanswerable": "fabrication",
}

EXPECTED_BEHAVIORS = {
    "answerable":   "cite_and_answer",
    "partial":      "acknowledge_gap",
    "ambiguous":    "present_options",
    "unanswerable": "refuse",
}

DOMAINS = [
    "infectious_disease", "cardiology", "oncology", "hepatology",
    "pulmonology", "nephrology", "orthopedics", "cross_domain",
]

TARGET_COUNTS = {
    ("answerable",   "direct_lookup"):          10,
    ("answerable",   "single_chunk_reasoning"): 15,
    ("answerable",   "multi_chunk_synthesis"):   5,
    ("partial",      "missing_specificity"):    15,
    ("partial",      "missing_subgroup"):       10,
    ("partial",      "missing_recent_update"):   5,
    ("ambiguous",    "underspecified"):         15,
    ("ambiguous",    "conflicting_sources"):     5,
    ("unanswerable", "out_of_domain"):          10,
    ("unanswerable", "in_domain_absent"):       20,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def next_question_id(existing: list[dict]) -> str:
    if not existing:
        return "q_001"
    last = existing[-1]["question_id"]
    n = int(last.split("_")[1]) + 1
    return f"q_{n:03d}"


def progress_report(existing: list[dict]):
    from collections import Counter
    counts = Counter((r["tier"], r["sub_tier"]) for r in existing)
    print(f"\n{'─'*60}")
    print(f"  Progress ({len(existing)} / 110 questions annotated)")
    print(f"{'─'*60}")
    print(f"  {'Tier':<14} {'Sub-tier':<28} {'Done':>4} / {'Target':>6}")
    print(f"  {'─'*14}  {'─'*28}  {'─'*4}   {'─'*6}")
    for (tier, sub), target in TARGET_COUNTS.items():
        done = counts.get((tier, sub), 0)
        flag = " ✓" if done >= target else f" ({target - done} left)"
        print(f"  {tier:<14}  {sub:<28}  {done:>4} / {target:>4}{flag}")
    print(f"{'─'*60}\n")


def prompt(label: str, options: list[str] | None = None,
           default: str | None = None, multiline: bool = False) -> str:
    if options:
        abbrevs = {o[0]: o for o in options}
        opts_str = " | ".join(
            f"[{o[0]}]{o[1:]}" if len(o) > 1 else f"[{o}]" for o in options
        )
        while True:
            val = input(f"  {label} ({opts_str}): ").strip().lower()
            if val in options:
                return val
            if val in abbrevs:
                return abbrevs[val]
            if default and val == "":
                return default
            print(f"    ✗ Must be one of: {options}")
    if multiline:
        print(f"  {label} (type answer, then blank line to finish):")
        lines = []
        while True:
            line = input("    ")
            if line == "" and lines:
                break
            lines.append(line)
        return " ".join(lines).strip()
    while True:
        val = input(f"  {label}: ").strip()
        if val:
            return val
        if default is not None:
            return default
        print("    ✗ Required.")


def prompt_sources() -> list[str]:
    print("  gold_sources — paste chunk UUIDs (space or newline separated).")
    print("  Tip: run python scripts/validate_questions.py to check IDs later.")
    print("  Leave blank to skip (fill in later).")
    raw = input("  Chunk IDs: ").strip()
    if not raw:
        return []
    return raw.split()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(EVAL_QUESTIONS),
                        help="Output JSONL file (default: data/processed/eval_questions.jsonl)")
    args = parser.parse_args()
    out_path = Path(args.file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing(out_path)

    print("=" * 60)
    print("  Phase 5 — Question Annotation CLI")
    print("=" * 60)
    progress_report(existing)
    print("  Starting annotation. Press Ctrl+C to stop at any time.\n")

    try:
        while True:
            qid = next_question_id(existing)
            print(f"{'─'*60}")
            print(f"  Question {qid}")
            print(f"{'─'*60}")

            question = prompt("question")

            tier = prompt("tier", TIERS)
            sub_tier = prompt("sub_tier", SUB_TIERS[tier])
            domain = prompt("domain", DOMAINS)

            print(f"\n  gold_answer — what the CORRECT answer is (from the corpus).")
            gold_answer = prompt("gold_answer", multiline=True)

            gold_sources = prompt_sources()

            exp_behavior = EXPECTED_BEHAVIORS[tier]
            print(f"  expected_behavior → {exp_behavior}  (auto-set from tier)")

            difficulty = prompt("difficulty", ["1", "2", "3"], default="2")
            notes = input("  notes (optional, press Enter to skip): ").strip()

            record = {
                "question_id":        qid,
                "question":           question,
                "tier":               tier,
                "sub_tier":           sub_tier,
                "hallucination_target": HALLUCINATION_TARGETS[tier],
                "gold_answer":        gold_answer,
                "gold_sources":       gold_sources,
                "expected_behavior":  exp_behavior,
                "domain":             domain,
                "notes":              notes,
                "annotated_on":       date.today().isoformat(),
                "difficulty":         int(difficulty),
            }

            print(f"\n  Preview:")
            print(f"    {json.dumps(record, indent=4)}")
            confirm = input("\n  Save this question? [y/n] (default y): ").strip().lower()
            if confirm in ("", "y", "yes"):
                with open(out_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                existing.append(record)
                print(f"  ✓ Saved. Total: {len(existing)} / 110\n")
            else:
                print("  Skipped.\n")

    except KeyboardInterrupt:
        print(f"\n\n  Stopped. {len(existing)} questions saved to {out_path}")
        progress_report(existing)


if __name__ == "__main__":
    main()
