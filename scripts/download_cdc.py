"""
download_cdc.py — scrape CDC fact-sheet pages and save as HTML.

Provide a list of CDC URLs (one per line in a file, or directly on the CLI).
Each page is saved as raw HTML; downstream processing (Phase 2) extracts text.

Usage:
    python scripts/download_cdc.py --urls_file configs/cdc_urls.txt
    python scripts/download_cdc.py --urls https://www.cdc.gov/diabetes/basics/diabetes.html

Note: WHO pages can also be passed — the script is source-agnostic.
"""

import argparse
import hashlib
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from log_metadata import append_row  # noqa: E402

OUT_DIR = ROOT / "data" / "raw" / "cdc_who"
HEADERS = {"User-Agent": "clinical-rag-audit/1.0 (research project)"}
DELAY = 1.5


def url_to_filename(url: str) -> str:
    short = url.rstrip("/").split("/")[-1] or hashlib.md5(url.encode()).hexdigest()[:8]
    if not short.endswith(".html"):
        short += ".html"
    return short


def fetch_page(url: str, out_dir: Path) -> tuple[Path | None, str]:
    filename = url_to_filename(url)
    out_path = out_dir / filename
    if out_path.exists():
        soup = BeautifulSoup(out_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        title = soup.title.string.strip() if soup.title else filename
        return out_path, title
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        soup = BeautifulSoup(resp.text, "lxml")
        title = soup.title.string.strip() if soup.title else filename
        return out_path, title
    except Exception as exc:
        print(f"  [warn] {url}: {exc}", file=sys.stderr)
        return None, ""


def infer_source(url: str) -> str:
    if "cdc.gov" in url:
        return "cdc"
    if "who.int" in url:
        return "who"
    return "cdc_who"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="*", default=[])
    parser.add_argument("--urls_file", default="")
    parser.add_argument("--domain_tag", default="general")
    args = parser.parse_args()

    urls = list(args.urls)
    if args.urls_file:
        p = Path(args.urls_file)
        if not p.exists():
            parser.error(f"--urls_file not found: {p}")
        urls += [l.strip() for l in p.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]

    if not urls:
        parser.error("Provide --urls or a non-empty --urls_file")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[cdc/who] fetching {len(urls)} pages → {OUT_DIR}")

    for url in tqdm(urls, unit="page"):
        out_path, title = fetch_page(url, OUT_DIR)
        time.sleep(DELAY)
        if out_path:
            source = infer_source(url)
            doc_id = f"{source}_{url_to_filename(url).replace('.html', '')}"
            append_row({
                "doc_id": doc_id,
                "source": source,
                "title": title,
                "url": url,
                "download_date": date.today().isoformat(),
                "publication_date": "",
                "license": "public-domain",
                "file_path": str(out_path.relative_to(ROOT)),
                "format": "html",
                "domain_tag": args.domain_tag,
            })

    print("[cdc/who] done.")


if __name__ == "__main__":
    main()
