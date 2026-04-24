"""
Phase 5 — Corpus topic inventory.
Run this BEFORE writing any evaluation questions to understand which topics
are densely covered (→ answerable), thinly covered (→ partial), and absent
(→ unanswerable).

Usage:
    python scripts/explore_corpus.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import CHUNKS_CLEAN  # noqa: E402

# Clinical terms to track — covers the 7 corpus domains
CLINICAL_TERMS = [
    # infectious disease
    "tuberculosis", "influenza", "pneumonia", "sepsis", "hiv", "covid",
    "antibiotic", "vaccination", "vaccine", "antiviral", "infection",
    "bacterial", "viral", "fungal", "pathogen",
    # cardiology
    "hypertension", "myocardial", "infarction", "arrhythmia", "atrial",
    "fibrillation", "statin", "beta.blocker", "heart.failure", "stroke",
    "cholesterol", "cardiovascular", "coronary", "angina", "thrombosis",
    # oncology
    "chemotherapy", "carcinoma", "metastasis", "lymphoma", "leukemia",
    "radiotherapy", "immunotherapy", "biopsy", "tumor", "cancer",
    # hepatology
    "cirrhosis", "hepatitis", "fibrosis", "liver", "bilirubin",
    "portal.hypertension", "ascites", "steatosis",
    # pulmonology
    "asthma", "copd", "bronchitis", "emphysema", "spirometry",
    "inhaler", "corticosteroid", "pulmonary", "respiratory",
    # nephrology
    "nephropathy", "dialysis", "creatinine", "glomerular", "proteinuria",
    "kidney", "renal", "hemodialysis",
    # orthopedics
    "fracture", "osteoporosis", "arthritis", "cartilage", "bone.density",
    "orthopedic", "joint", "spine",
    # diabetes / metabolic (cross-domain)
    "metformin", "insulin", "diabetes", "glycemic", "hba1c",
    "hyperglycemia", "obesity", "metabolic",
]

THIN_THRESHOLD = 50  # domains with fewer chunks are "thin coverage"


def load_chunks():
    with open(CHUNKS_CLEAN) as f:
        return [json.loads(line) for line in f]


def term_freq(chunks):
    """Count how many chunks mention each clinical term."""
    counts = Counter()
    for chunk in chunks:
        text = chunk["text"].lower()
        for term in CLINICAL_TERMS:
            pattern = term.replace(".", r"[\s\-]?")
            if re.search(pattern, text):
                counts[term] += 1
    return counts


def top_entities_per_domain(chunks, top_n=12):
    """Top keyword frequency per domain (words ≥7 chars)."""
    domain_text = defaultdict(list)
    for c in chunks:
        domain_text[c["metadata"]["domain"]].append(c["text"].lower())

    generic = {
        "analysis", "studies", "reported", "therapy", "factors", "outcomes",
        "compared", "performed", "specific", "evidence", "patient", "control",
        "findings", "response", "included", "baseline", "research", "through",
        "primary", "further", "activity", "hospital", "between", "however",
        "patients", "results", "treatment", "disease", "clinical", "diabetes",
        "associated", "increased", "observed", "participants", "significant",
    }

    result = {}
    for domain, texts in domain_text.items():
        combined = " ".join(texts)
        words = re.findall(r"[a-z]{7,}", combined)
        freq = Counter(w for w in words if w not in generic)
        result[domain] = freq.most_common(top_n)
    return result


def main():
    print("=" * 70)
    print("  Phase 5 — Corpus Topic Inventory")
    print("=" * 70)

    chunks = load_chunks()
    total = len(chunks)

    # ── Domain distribution ──────────────────────────────────────────────
    domain_counts = Counter(c["metadata"]["domain"] for c in chunks)

    print(f"\n{'─'*70}")
    print("  DOMAIN DISTRIBUTION")
    print(f"{'─'*70}")
    print(f"  {'Domain':<25} {'Chunks':>6}  {'%':>5}  Coverage")
    print(f"  {'─'*25}  {'─'*6}  {'─'*5}  {'─'*8}")
    for domain, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        tag = "THIN" if n < THIN_THRESHOLD else "dense" if n > 200 else "moderate"
        print(f"  {domain:<25} {n:>6}  {n/total*100:>4.1f}%  {tag}")
    print(f"  {'─'*25}  {'─'*6}")
    print(f"  {'TOTAL':<25} {total:>6}")

    # ── Clinical term coverage ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CLINICAL TERM COVERAGE  (# chunks mentioning the term)")
    print(f"{'─'*70}")
    freq = term_freq(chunks)
    dense, thin, absent = [], [], []
    for term in CLINICAL_TERMS:
        n = freq.get(term, 0)
        if n >= 30:
            dense.append((term, n))
        elif n >= 5:
            thin.append((term, n))
        else:
            absent.append((term, n))

    print(f"\n  DENSE (≥30 chunks) — good for ANSWERABLE questions:")
    for t, n in sorted(dense, key=lambda x: -x[1]):
        print(f"    {t:<25} {n:>4} chunks")

    print(f"\n  THIN (5–29 chunks) — good for PARTIAL questions:")
    for t, n in sorted(thin, key=lambda x: -x[1]):
        print(f"    {t:<25} {n:>4} chunks")

    print(f"\n  ABSENT (<5 chunks) — good for UNANSWERABLE questions:")
    for t, n in sorted(absent, key=lambda x: -x[1]):
        label = "0 chunks" if n == 0 else f"{n} chunk{'s' if n != 1 else ''}"
        print(f"    {t:<25} {label}")

    # ── Top entities per domain ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  TOP KEYWORDS PER DOMAIN  (use these to write questions)")
    print(f"{'─'*70}")
    top = top_entities_per_domain(chunks)
    for domain in sorted(top.keys()):
        n_chunks = domain_counts[domain]
        tag = " [THIN]" if n_chunks < THIN_THRESHOLD else ""
        print(f"\n  {domain.upper()}{tag}  ({n_chunks} chunks)")
        kws = ", ".join(f"{w}({c})" for w, c in top[domain])
        print(f"    {kws}")

    # ── Question writing guide ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  QUESTION WRITING GUIDE")
    print(f"{'─'*70}")
    print("""
  ANSWERABLE (30):  Use DENSE terms above. Write questions whose answers
                    are directly stated in ≥1 chunk. Confirm with
                    validate_questions.py (gold source in retriever top-10).

  PARTIAL (30):     Use THIN terms, or ask about a SUBGROUP or DOSAGE that
                    the corpus mentions only in passing (e.g., drug used for
                    T2D but not for PCOS → partial for PCOS question).

  AMBIGUOUS (20):   Ask about a topic where the corpus has multiple chunks
                    with different (not conflicting) perspectives, or where
                    the question is underspecified (e.g., "What is the dose
                    of metformin?" without specifying indication).

  UNANSWERABLE (30):  Use ABSENT terms (out_of_domain), or ask a clinically
                    plausible question about a topic that simply isn't in the
                    corpus (in_domain_absent). The model should refuse, not guess.
""")

    print("=" * 70)
    print("  Done. Use this output to write docs/annotation_guidelines.md")
    print("  then start annotating with: python scripts/annotate_questions.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
