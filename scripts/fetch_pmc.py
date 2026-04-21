# Fetch additional PMC articles for underrepresented domains, then re-run pipeline

import csv
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from download_pmc import search_pmids, fetch_xml, parse_title, DELAY, OUT_DIR  # noqa: E402
from log_metadata import append_row                                             # noqa: E402

METADATA_CSV = ROOT / "data" / "metadata.csv"

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


def run_pipeline() -> None:
    script = ROOT / "scripts" / "pipeline.py"
    print(f"\n[fetch_pmc] re-running pipeline ...")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT / "scripts"),
        check=False,
    )
    if result.returncode != 0:
        print(f"[fetch_pmc] pipeline exited with code {result.returncode}", file=sys.stderr)
    else:
        print("[fetch_pmc] pipeline complete.")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_doc_ids()
    print(f"[fetch_pmc] {len(existing)} doc_ids already in metadata.csv")

    total_new = 0

    for query, domain_tag, max_results in FETCH_PLAN:
        print(f"\n[fetch_pmc] query='{query}'  domain={domain_tag}  max={max_results}")
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

    print(f"\n[fetch_pmc] fetched {total_new} new article(s).")

    if total_new > 0:
        run_pipeline()
    else:
        print("[fetch_pmc] no new articles — skipping pipeline re-run.")
