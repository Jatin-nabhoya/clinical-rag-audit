"""
Phase 3 — Retrieval Function
src/retrieval/retriever.py

Loads a saved FAISS index and returns top-k chunks for a query.
Accepts vector_store_dir so the Phase 4 ablation runner can swap
between general and medical indexes with one argument change.

Usage:
    from src.retrieval.retriever import Retriever

    r = Retriever()                                        # general (default)
    r = Retriever("data/vector_store/medical")             # medical ablation

    results = r.retrieve("What is the treatment for sepsis?", k=5)
    context = r.format_context(results)
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_STORE = str(ROOT / "data" / "vector_store" / "general")


class Retriever:
    """
    Loads FAISS index + chunk metadata from disk.
    Model and index are loaded once at __init__ — all .retrieve() calls are fast.
    """

    def __init__(self, vector_store_dir: str = _DEFAULT_STORE):
        self.store_dir = Path(vector_store_dir)

        config_path = self.store_dir / "embed_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No embed_config.json in {vector_store_dir}.\n"
                "Run:  python src/retrieval/embed.py --model general"
            )

        self.config = json.loads(config_path.read_text())
        model_name  = self.config["embedding_model"]
        model_key   = self.config.get("model_key", "unknown")

        print(f"[Retriever:{model_key}] loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

        index_path = self.store_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        print(f"[Retriever:{model_key}] index: {self.index.ntotal:,} vectors  "
              f"dim={self.config['embedding_dim']}")

        meta_path   = self.store_dir / "chunks_meta.jsonl"
        self.chunks = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))
        print(f"[Retriever:{model_key}] {len(self.chunks):,} chunks loaded. Ready.\n")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Embed query and return top-k most similar chunks.

        Each result dict contains:
          chunk_id, text, score (cosine 0–1), rank, metadata
        """
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, k=k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "text":     chunk["text"],
                "score":    float(score),
                "rank":     rank,
                "metadata": chunk.get("metadata", {}),
            })
        return results

    def format_context(self, results: list[dict], max_chars: int = 3000) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.
        Stops adding chunks once max_chars is reached.
        """
        parts       = []
        total_chars = 0
        for res in results:
            source = res["metadata"].get("source", "unknown")
            title  = res["metadata"].get("title", "")
            block  = f"[Source: {source} | {title}]\n{res['text'].strip()}\n"
            if total_chars + len(block) > max_chars:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n---\n".join(parts)
