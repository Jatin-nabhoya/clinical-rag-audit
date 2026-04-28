# Fetch additional PMC articles for underrepresented domains, then re-run document ingestion

import csv
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import METADATA_CSV                                                  # noqa: E402
from download_pmc import search_pmids, fetch_xml, parse_title, DELAY, OUT_DIR  # noqa: E402
from log_metadata import append_row                                             # noqa: E402

FETCH_PLAN: list[tuple[str, str, int]] = [
    ("tuberculosis treatment",         "infectious_disease", 10),
    ("HIV antiretroviral therapy",     "infectious_disease", 10),
    ("sepsis management",              "infectious_disease", 10),
    ("influenza antiviral treatment",  "infectious_disease", 10),
    ("COVID-19 clinical outcomes",     "infectious_disease", 10),
    ("pneumonia treatment clinical",   "infectious_disease", 10),
]


def load_existing_doc_ids() -> set[str]:
    if not METADATA_CSV.exists():
        return set()
    with open(METADATA_CSV, encoding="utf-8", newline="") as fh:
        return {row["doc_id"] for row in csv.DictReader(fh)}


def run_ingestion() -> None:
    script = ROOT / "scripts" / "ingest_documents.py"
    print(f"\n[augment_corpus] re-running document ingestion ...")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT / "scripts"),
        check=False,
    )
    if result.returncode != 0:
        print(f"[augment_corpus] ingestion exited with code {result.returncode}", file=sys.stderr)
    else:
        print("[augment_corpus] ingestion complete.")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_doc_ids()
    print(f"[augment_corpus] {len(existing)} doc_ids already in metadata.csv")

    total_new = 0

    for query, domain_tag, max_results in FETCH_PLAN:
        print(f"\n[augment_corpus] query='{query}'  domain={domain_tag}  max={max_results}")
        pmids = search_pmids(query, max_results)
        print(f"  → {len(pmids)} PMC IDs returned")

        for pmcid in pmids:
            doc_id = f"pmc_{pmcid}"

            if doc_id in existing:
                print(f"  [skip] {doc_id} already in metadata.csv")
                continue

            xml_path = fetch_xml(pmcid, OUT_DIR)
            time.sleep(DELAY)

            if xml_path is None:
                continue

            title = parse_title(xml_path)
            append_row({
                "doc_id":           doc_id,
                "source":           "pmc",
                "title":            title,
                "url":              f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                "download_date":    date.today().isoformat(),
                "publication_date": "",
                "license":          "open-access",
                "file_path":        str(xml_path.relative_to(ROOT)),
                "format":           "xml",
                "domain_tag":       domain_tag,
            })

            existing.add(doc_id)
            total_new += 1

    print(f"\n[augment_corpus] fetched {total_new} new article(s).")

    if total_new > 0:
        run_ingestion()
    else:
        print("[augment_corpus] no new articles — skipping ingestion re-run.")
