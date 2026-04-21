# Step 3: Full Pipeline — Extract → Chunk → Attach Metadata → Save JSONL

import csv
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # clinical-rag-audit/

from extract_text import extract_html, extract_xml, extract_json_medlineplus
from chunk_documents import chunk_text, token_len

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def extract(file_path: str, fmt: str) -> str:
    if fmt == "html":
        return extract_html(file_path)
    elif fmt == "xml":
        return extract_xml(file_path)
    elif fmt == "json":
        return extract_json_medlineplus(file_path)
    raise ValueError(f"Unsupported format: {fmt}")

def save_jsonl(chunks: list[dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(metadata_csv: str, output_path: str) -> list[dict]:
    rows = load_metadata(metadata_csv)

    seen_files = set()
    all_chunks = []
    stats = {"processed": 0, "skipped_missing": 0, "skipped_empty": 0, "errors": 0}

    for row in rows:
        file_path = row["file_path"]

        # Deduplicate (cdc_index appears multiple times in metadata)
        if file_path in seen_files:
            continue
        seen_files.add(file_path)

        if not (ROOT / file_path).exists():
            stats["skipped_missing"] += 1
            continue

        try:
            text = extract(str(ROOT / file_path), row["format"])
        except Exception as e:
            print(f"  [ERROR] {file_path}: {e}")
            stats["errors"] += 1
            continue

        if not text.strip():
            stats["skipped_empty"] += 1
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id":     str(uuid.uuid4()),
                "text":         chunk,
                "metadata": {
                    "doc_id":        row["doc_id"],
                    "source":        row["source"],
                    "title":         row["title"],
                    "url":           row["url"],
                    "domain":        row["domain_tag"],
                    "file_path":     file_path,
                    "format":        row["format"],
                    "license":       row["license"],
                    "chunk_index":   i,
                    "total_chunks":  len(chunks),
                    "token_count":   token_len(chunk),
                }
            })

        stats["processed"] += 1

    save_jsonl(all_chunks, output_path)
    return all_chunks, stats


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT = str(ROOT / "data/processed/chunks.jsonl")
    chunks, stats = run_pipeline(str(ROOT / "data/metadata.csv"), OUTPUT)

    from collections import Counter
    by_source = Counter(c["metadata"]["source"] for c in chunks)

    print(f"\n{'='*45}")
    print(f"  Documents processed : {stats['processed']}")
    print(f"  Missing files       : {stats['skipped_missing']}")
    print(f"  Empty after extract : {stats['skipped_empty']}")
    print(f"  Errors              : {stats['errors']}")
    print(f"{'='*45}")
    print(f"  Total chunks        : {len(chunks)}")
    print(f"\nBy source:")
    for src, count in by_source.most_common():
        avg = sum(c["metadata"]["token_count"] for c in chunks
                  if c["metadata"]["source"] == src) / count
        print(f"  {src:15s}  {count:4d} chunks  (avg {avg:.0f} tokens)")
    print(f"\nSaved -> {OUTPUT}")


'''
output: 
=============================================
  Documents processed : 56
  Missing files       : 3
  Empty after extract : 0
  Errors              : 0
=============================================
  Total chunks        : 1745

By source:
  pmc              1715 chunks  (avg 389 tokens)
  who                25 chunks  (avg 441 tokens)
  cdc                 5 chunks  (avg 440 tokens)

Saved -> /Users/jatinnabhoya/Desktop/UNH/Semester 3/NLP/Project/clinical-rag-audit/data/processed/chunks.jsonl
'''

'''
Pipeline ran cleanly:

1745 total chunks across 56 documents
PMC — 1715 chunks (50 XML articles, avg 389 tokens)
WHO — 25 chunks (5 HTML pages, avg 441 tokens)
CDC — 5 chunks (1 HTML page, avg 440 tokens)
3 missing files — the MedlinePlus JSONs (empty entries, as expected)
Output saved to data/processed/chunks.jsonl. Each line is a chunk with a chunk_id, text, and full metadata block (doc_id, source, title, url, domain, token_count, etc.).
'''