"""
download_pmc.py — fetch open-access articles from PubMed Central via NCBI E-utilities.

Usage:
    python scripts/download_pmc.py --query "diabetes type 2" --max_results 50
    python scripts/download_pmc.py --pmids 12345678 87654321
    python scripts/download_pmc.py --query "hypertension" --max_results 20 --format xml

Requires: biopython, tqdm, requests
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

from tqdm import tqdm

# Add project root to path so log_metadata is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from log_metadata import append_row  # noqa: E402

try:
    from Bio import Entrez
except ImportError:
    sys.exit("biopython not installed — run: pip install biopython")

OUT_DIR = ROOT / "data" / "raw" / "pmc"
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "user@example.com")  # set in .env
Entrez.email = ENTREZ_EMAIL
DELAY = 0.4  # NCBI rate limit: max 3 req/s without API key


def search_pmids(query: str, max_results: int) -> list[str]:
    handle = Entrez.esearch(db="pmc", term=query + " AND open access[filter]",
                            retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_xml(pmcid: str, out_dir: Path) -> Path | None:
    out_path = out_dir / f"PMC{pmcid}.xml"
    if out_path.exists():
        return out_path
    try:
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
        content = handle.read()
        handle.close()
        out_path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return out_path
    except Exception as exc:
        print(f"  [warn] PMC{pmcid}: {exc}", file=sys.stderr)
        return None


def parse_title(xml_path: Path) -> str:
    try:
        from lxml import etree
        tree = etree.parse(str(xml_path))
        titles = tree.xpath("//article-title")
        if titles:
            return "".join(titles[0].itertext()).strip()
    except Exception:
        pass
    return xml_path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="")
    parser.add_argument("--pmids", nargs="*", default=[])
    parser.add_argument("--max_results", type=int, default=20)
    parser.add_argument("--domain_tag", default="general")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pmids = args.pmids or search_pmids(args.query, args.max_results)
    print(f"[pmc] downloading {len(pmids)} articles → {OUT_DIR}")

    for pmid in tqdm(pmids, unit="article"):
        xml_path = fetch_xml(pmid, OUT_DIR)
        time.sleep(DELAY)
        if xml_path:
            title = parse_title(xml_path)
            append_row({
                "doc_id": f"pmc_{pmid}",
                "source": "pmc",
                "title": title,
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}/",
                "download_date": date.today().isoformat(),
                "publication_date": "",
                "license": "open-access",
                "file_path": str(xml_path.relative_to(ROOT)),
                "format": "xml",
                "domain_tag": args.domain_tag,
            })

    print("[pmc] done.")


if __name__ == "__main__":
    main()
