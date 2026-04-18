"""
log_metadata.py — append a row to data/metadata.csv.

Usage (from project root):
    python scripts/log_metadata.py \
        --doc_id pmc_001 \
        --source pmc \
        --title "Some Article" \
        --url https://... \
        --publication_date 2023-01-15 \
        --license CC-BY \
        --file_path data/raw/pmc/pmc_001.xml \
        --format xml \
        --domain_tag cardiology
"""

import argparse
import csv
import os
from datetime import date

METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.csv")
FIELDNAMES = [
    "doc_id", "source", "title", "url", "download_date",
    "publication_date", "license", "file_path", "format", "domain_tag",
]


def append_row(row: dict) -> None:
    path = os.path.abspath(METADATA_PATH)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[metadata] logged → {row['doc_id']}")


def main():
    parser = argparse.ArgumentParser(description="Append a document row to metadata.csv")
    for field in FIELDNAMES:
        parser.add_argument(f"--{field}", default="")
    args = parser.parse_args()
    row = vars(args)
    if not row["download_date"]:
        row["download_date"] = date.today().isoformat()
    append_row(row)


if __name__ == "__main__":
    main()
