"""
Phase 3 — Embeddings & FAISS Index Builder
src/retrieval/embed.py

Builds separate FAISS indexes for general vs. medical embedding models.
Each model gets its own output directory — indexes never overwrite each other.

Usage:
  python src/retrieval/embed.py --model general   # baseline only
  python src/retrieval/embed.py --model medical   # ablation only
  python src/retrieval/embed.py --model all       # build both (default)

Output:
  data/vector_store/
  ├── general/
  │   ├── faiss_index.bin
  │   ├── chunks_meta.jsonl
  │   └── embed_config.json
  └── medical/
      ├── faiss_index.bin
      ├── chunks_meta.jsonl
      └── embed_config.json
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT        = Path(__file__).resolve().parent.parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_clean.jsonl"
BATCH_SIZE  = 64   # safe for CPU; increase to 128 on GPU

MODELS = {
    "general": {
        "model_name":  "sentence-transformers/all-MiniLM-L6-v2",
        "description": "General-purpose, fast (384-dim). Ablation baseline.",
        "output_dir":  ROOT / "data" / "vector_store" / "general",
    },
    "medical": {
        "model_name":  "pritamdeka/S-PubMedBert-MS-MARCO",
        "description": "PubMed + MS-MARCO fine-tuned (768-dim). Medical ablation.",
        "output_dir":  ROOT / "data" / "vector_store" / "medical",
    },
}


# ── Core Functions ─────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks):,} chunks from {path.name}")
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        vecs  = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        all_embeddings.append(vecs)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def build_and_save_index(model_key: str, model_cfg: dict, chunks: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"  [{model_key.upper()}]  {model_cfg['model_name']}")
    print(f"  {model_cfg['description']}")
    print(f"{'='*60}")

    # 1. Load model
    t0    = time.time()
    model = SentenceTransformer(model_cfg["model_name"])
    dim   = model.get_sentence_embedding_dimension()
    print(f"\n  Model ready in {time.time()-t0:.1f}s  |  dim={dim}")

    # 2. Embed
    embeddings = embed_chunks(chunks, model)

    # 3. L2-normalize so inner product = cosine similarity
    faiss.normalize_L2(embeddings)

    # 4. Build FAISS flat index (exact search)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"\n  FAISS index: {index.ntotal:,} vectors  |  type=IndexFlatIP")

    # 5. Save to disk
    out = Path(model_cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "faiss_index.bin"))

    with open(out / "chunks_meta.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    config = {
        "model_key":         model_key,
        "embedding_model":   model_cfg["model_name"],
        "embedding_dim":     dim,
        "num_chunks":        len(chunks),
        "index_type":        "IndexFlatIP",
        "normalized":        True,
        "similarity_metric": "cosine",
    }
    (out / "embed_config.json").write_text(json.dumps(config, indent=2))
    print(f"  Saved → {out}/")

    # 6. Smoke test
    sample = "What is the treatment for hypertension?"
    q_vec  = model.encode([sample], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, k=3)

    print(f"\n  Smoke test: '{sample}'")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        src     = chunks[idx]["metadata"].get("source", "?")
        domain  = chunks[idx]["metadata"].get("domain", "?")
        preview = chunks[idx]["text"][:100].replace("\n", " ")
        print(f"    Rank {rank} | score={score:.4f} | {src} | {domain}")
        print(f"           {preview}...")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which embedding model to build (default: all)",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Phase 3 — FAISS Index Builder")
    print("="*60)

    targets = list(MODELS.items()) if args.model == "all" else [(args.model, MODELS[args.model])]
    print(f"\n  Building : {[k for k, _ in targets]}")
    print(f"  Source   : {CHUNKS_PATH.name}\n")

    chunks = load_chunks(CHUNKS_PATH)

    for model_key, model_cfg in targets:
        build_and_save_index(model_key, model_cfg, chunks)

    # Summary table
    print(f"\n{'='*60}")
    print("  Build Summary")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Dim':>5}  {'Chunks':>7}  {'Size':>8}  Output")
    print(f"  {'-'*55}")
    for model_key, model_cfg in targets:
        out  = Path(model_cfg["output_dir"])
        cfg  = json.loads((out / "embed_config.json").read_text())
        size = (out / "faiss_index.bin").stat().st_size / 1e6
        print(f"  {model_key:<12} {cfg['embedding_dim']:>5}  {cfg['num_chunks']:>7,}  "
              f"{size:>6.1f} MB  {out.relative_to(ROOT)}/")
    print()


if __name__ == "__main__":
    main()
