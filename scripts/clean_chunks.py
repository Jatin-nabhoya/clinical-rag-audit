# scripts/clean_chunks.py
import json
import re
from collections import Counter

from utils import CHUNKS_FILE as INPUT, CHUNKS_CLEAN as OUTPUT

# ── 1. LaTeX cleaner ──────────────────────────────────────────────────────────

LATEX_PATTERNS = [
    r"\\usepackage\{[^}]*\}",          # \usepackage{...}
    r"\\begin\{[^}]*\}",               # \begin{document}
    r"\\end\{[^}]*\}",                 # \end{document}
    r"\\[a-zA-Z]+\{[^}]*\}",           # any \command{...}
    r"\\[a-zA-Z]+",                    # bare \command
    r"\$\$.*?\$\$",                    # display math $$...$$
    r"\$[^$]+\$",                      # inline math $...$
    r"\\setlength\{[^}]*\}\{[^}]*\}",  # \setlength{}{} (two-arg commands)
]

LATEX_RE = re.compile("|".join(LATEX_PATTERNS), re.DOTALL)

def clean_latex(text: str) -> str:
    text = LATEX_RE.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)   # collapse whitespace
    return text.strip()

def has_heavy_latex(text: str) -> bool:
    """True if chunk is predominantly LaTeX (>15% of chars are markup)"""
    markup_chars = len("".join(LATEX_RE.findall(text)))
    return markup_chars / max(len(text), 1) > 0.15

# ── 2. Domain relabeler (title-weighted) ──────────────────────────────────────

DOMAIN_KEYWORDS = {
    "infectious_disease": [
        "tuberculosis", "HIV", "sepsis", "influenza", "COVID", "infection",
        "antiviral", "antibiotic", "pathogen", "bacteremia", "pneumonia",
        "antimicrobial", "endemic", "epidemic", "pandemic", "viral load"
    ],
    "oncology": [
        "cancer", "tumor", "carcinoma", "melanoma", "lymphoma", "leukemia",
        "chemotherapy", "immunotherapy", "metastasis", "malignant", "sarcoma",
        "regorafenib", "durvalumab", "carboplatin", "paclitaxel", "BRAF"
    ],
    "cardiology": [
        "cardiac", "cardiovascular", "heart failure", "stroke", "TAVR",
        "myocardial", "arrhythmia", "hypertension", "atherosclerosis",
        "LDL", "bempedoic", "vascular", "coronary", "aortic"
    ],
    "hepatology": [
        "liver", "hepatic", "portal", "fibrosis", "cirrhosis", "TIPS",
        "MELD", "steatosis", "hepatitis", "cholestasis", "transplant",
        "dendritic cell", "leukapheresis"
    ],
    "nephrology": [
        "kidney", "renal", "podocyte", "preeclampsia", "urinary", "TCF21",
        "glomerular", "creatinine", "dialysis", "proteinuria"
    ],
    "orthopedics": [
        "osteoarthritis", "knee", "joint", "cartilage", "bone", "orthopedic",
        "musculoskeletal", "fracture", "ligament"
    ],
    "pulmonology": [
        "COPD", "asthma", "respiratory", "pulmonary", "lung", "OSA",
        "sleep apnea", "spirometry", "bronchial"
    ],
}

TITLE_WEIGHT = 5   # title keyword hit = 5 body keyword hits

def infer_domain(text: str, title: str) -> str:
    title_lower = title.lower()
    body_lower  = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            kw_l = kw.lower()
            if kw_l in title_lower:
                scores[domain] += TITLE_WEIGHT
            if kw_l in body_lower:
                scores[domain] += 1

    best = max(scores, key=scores.get)
    # If top score is 0 — nothing matched — keep existing label
    if scores[best] == 0:
        return None
    return best

# ── 3. Main cleaning pass ──────────────────────────────────────────────────────

MIN_TOKENS = 100   # drop chunks below this

stats = {
    "total_in":        0,
    "dropped_latex":   0,
    "dropped_short":   0,
    "relabeled":       0,
    "total_out":       0,
}

cleaned = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        stats["total_in"] += 1

        text  = chunk["text"]
        title = chunk["metadata"].get("title", "")

        # ── Drop LaTeX-heavy chunks entirely ──
        if has_heavy_latex(text):
            stats["dropped_latex"] += 1
            continue

        # ── Clean residual LaTeX from surviving chunks ──
        text = clean_latex(text)
        chunk["text"] = text

        # ── Drop micro-chunks ──
        token_count = chunk["metadata"].get("token_count", len(text.split()))
        if token_count < MIN_TOKENS:
            stats["dropped_short"] += 1
            continue

        # ── Relabel domain using title-weighted scorer ──
        old_domain = chunk["metadata"].get("domain", "unknown")
        new_domain = infer_domain(text, title)
        if new_domain and new_domain != old_domain:
            chunk["metadata"]["domain"]     = new_domain
            chunk["metadata"]["domain_old"] = old_domain   # audit trail
            stats["relabeled"] += 1

        cleaned.append(chunk)

# ── 4. Save ───────────────────────────────────────────────────────────────────

with open(OUTPUT, "w", encoding="utf-8") as f:
    for chunk in cleaned:
        f.write(json.dumps(chunk) + "\n")

# ── 5. Report ─────────────────────────────────────────────────────────────────

stats["total_out"] = len(cleaned)
domain_dist = Counter(c["metadata"]["domain"] for c in cleaned)

print("\n" + "="*60)
print("  CLEANING REPORT")
print("="*60)
print(f"  Input chunks      : {stats['total_in']}")
print(f"  Dropped (LaTeX)   : {stats['dropped_latex']}")
print(f"  Dropped (short)   : {stats['dropped_short']}")
print(f"  Relabeled         : {stats['relabeled']}")
print(f"  Output chunks     : {stats['total_out']}")
print(f"\nDomain distribution:")
for domain, count in domain_dist.most_common():
    pct = count / stats["total_out"] * 100
    bar = "█" * int(pct / 2)
    print(f"  {domain:20s} {count:5d}  ({pct:.1f}%)  {bar}")
print("="*60)
print(f"\nSaved -> {OUTPUT}")