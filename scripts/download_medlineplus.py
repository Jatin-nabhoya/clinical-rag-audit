"""
download_medlineplus.py — fetch health topic pages from MedlinePlus Connect API.

MedlinePlus Connect returns XML (HL7 InfoButton) for a given topic code or
free-text search. We use the web service to pull topic summaries.

Usage:
    python scripts/download_medlineplus.py --topics "diabetes" "hypertension" "asthma"
    python scripts/download_medlineplus.py --topics_file configs/medlineplus_topics.txt

API docs: https://medlineplus.gov/xml.html
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from log_metadata import append_row  # noqa: E402

OUT_DIR = ROOT / "data" / "raw" / "medlineplus"
BASE_URL = "https://connect.medlineplus.gov/service"
DELAY = 1.0


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")


def fetch_topic(topic: str, out_dir: Path) -> Path | None:
    params = {
        "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.177",
        "mainSearchCriteria.v.dn": topic,
        "knowledgeResponseType": "application/json",
        "informationRecipient": "PATNFO",
    }
    url = f"{BASE_URL}?{urlencode(params)}"
    out_path = out_dir / f"{slug(topic)}.json"
    if out_path.exists():
        return out_path
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return out_path
    except Exception as exc:
        print(f"  [warn] {topic}: {exc}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="*", default=[])
    parser.add_argument("--topics_file", default="")
    parser.add_argument("--domain_tag", default="general")
    args = parser.parse_args()

    topics = list(args.topics)
    if args.topics_file:
        p = Path(args.topics_file)
        if p.exists():
            topics += [l.strip() for l in p.read_text().splitlines() if l.strip()]

    if not topics:
        parser.error("Provide --topics or --topics_file")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[medlineplus] fetching {len(topics)} topics → {OUT_DIR}")

    for topic in tqdm(topics, unit="topic"):
        out_path = fetch_topic(topic, OUT_DIR)
        time.sleep(DELAY)
        if out_path:
            append_row({
                "doc_id": f"mlp_{slug(topic)}",
                "source": "medlineplus",
                "title": topic,
                "url": f"https://medlineplus.gov/connect/service?mainSearchCriteria.v.dn={topic}",
                "download_date": date.today().isoformat(),
                "publication_date": "",
                "license": "public-domain",
                "file_path": str(out_path.relative_to(ROOT)),
                "format": "json",
                "domain_tag": args.domain_tag,
            })

    print("[medlineplus] done.")


if __name__ == "__main__":
    main()
