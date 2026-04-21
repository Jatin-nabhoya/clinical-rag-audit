# utils.py — shared path constants for all scripts

from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_CSV  = ROOT / "data" / "metadata.csv"
CHUNKS_FILE   = PROCESSED_DIR / "chunks.jsonl"
CHUNKS_CLEAN  = PROCESSED_DIR / "chunks_clean.jsonl"
