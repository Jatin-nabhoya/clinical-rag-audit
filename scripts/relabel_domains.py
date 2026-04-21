# Relabel domain tags on existing chunks using keyword scoring

import csv
import json
from collections import defaultdict

from utils import CHUNKS_FILE as CHUNKS_PATH, METADATA_CSV

DOMAIN_MAP: dict[str, list[str]] = {
    "cardiology":         ["cardiac", "cardiovascular", "heart", "stroke",
                           "tavr", "vascular", "coronary", "arrhythmia"],
    "infectious_disease": ["sepsis", "infection", "candida", "bacterial",
                           "viral", "hiv", "tuberculosis", "influenza",
                           "covid", "antimicrobial"],
    "nephrology":         ["kidney", "renal", "podocyte", "preeclampsia",
                           "urinary", "glomerular"],
    "hepatology":         ["liver", "hepatic", "portal", "fibrosis",
                           "tips", "meld", "cirrhosis", "transplant"],
    "oncology":           ["cancer", "tumor", "breast cancer", "carcinoma",
                           "malignant", "chemotherapy", "lymphoma"],
}

TITLE_WEIGHT = 3  # title keyword hits count this many times


def score_text(text: str) -> dict[str, int]:
    lower = text.lower()
    return {domain: sum(lower.count(kw) for kw in kws)
            for domain, kws in DOMAIN_MAP.items()}


def pick_domain(scores: dict[str, int]) -> str:
    max_score = max(scores.values(), default=0)
    if max_score == 0:
        return "general"
    winners = [d for d, s in scores.items() if s == max_score]
    return winners[0] if len(winners) == 1 else "general"


def build_doc_domain_map() -> dict[str, str]:
    doc_scores: dict[str, dict[str, int]] = defaultdict(
        lambda: {d: 0 for d in DOMAIN_MAP}
    )
    seen_titles: set[str] = set()

    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        for raw in fh:
            chunk = json.loads(raw)
            meta  = chunk["metadata"]
            doc_id = meta["doc_id"]

            # Weight title 3x, but only once per doc
            if doc_id not in seen_titles:
                seen_titles.add(doc_id)
                for dom, s in score_text(meta["title"]).items():
                    doc_scores[doc_id][dom] += s * TITLE_WEIGHT

            for dom, s in score_text(chunk["text"]).items():
                doc_scores[doc_id][dom] += s

    return {doc_id: pick_domain(scores) for doc_id, scores in doc_scores.items()}


def rewrite_chunks(doc_domain: dict[str, str]) -> tuple[dict, dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    before: dict[str, int] = {}
    out_lines = []

    for raw in raw_lines:
        chunk = json.loads(raw)
        old = chunk["metadata"]["domain"]
        before[old] = before.get(old, 0) + 1
        doc_id = chunk["metadata"]["doc_id"]
        if doc_id in doc_domain:
            chunk["metadata"]["domain"] = doc_domain[doc_id]
        out_lines.append(json.dumps(chunk, ensure_ascii=False))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")

    after: dict[str, int] = {}
    for line in out_lines:
        d = json.loads(line)["metadata"]["domain"]
        after[d] = after.get(d, 0) + 1

    return before, after


def update_metadata_csv(doc_domain: dict[str, str]) -> None:
    fieldnames = [
        "doc_id", "source", "title", "url", "download_date",
        "publication_date", "license", "file_path", "format", "domain_tag",
    ]
    with open(METADATA_CSV, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    updated = 0
    for row in rows:
        if row["doc_id"] in doc_domain:
            new_tag = doc_domain[row["doc_id"]]
            if row["domain_tag"] != new_tag:
                row["domain_tag"] = new_tag
                updated += 1

    with open(METADATA_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[metadata] updated domain_tag for {updated} row(s)")


def print_dist(label: str, counts: dict[str, int]) -> None:
    total = sum(counts.values())
    print(f"\n{label} ({total} chunks):")
    for domain, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * (n * 40 // max(total, 1))
        print(f"  {domain:<20s} {n:5d}  {bar}")


if __name__ == "__main__":
    print(f"[relabel] scoring {CHUNKS_PATH} ...")
    doc_domain = build_doc_domain_map()

    from collections import Counter
    doc_dist = Counter(doc_domain.values())
    print(f"[relabel] {len(doc_domain)} unique doc_ids → predicted domains:")
    for dom, n in doc_dist.most_common():
        print(f"  {dom}: {n} docs")

    before, after = rewrite_chunks(doc_domain)
    print_dist("BEFORE", before)
    print_dist("AFTER",  after)

    update_metadata_csv(doc_domain)
    print(f"\n[relabel] done. Saved → {CHUNKS_PATH}")
