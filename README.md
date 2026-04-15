# Clinical RAG Hallucination Audit

DSCI 6004 NLP course project comparing hallucination behavior across Llama-3-8B, Mistral-7B, and Phi-3-mini in a clinical RAG pipeline.

**Team:** Jatin Nabhoya, Mohit Raiyani

## Setup
1. `python3.11 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Download data per `data/README.md`
4. See `notebooks/` for usage examples.

## Structure
- `src/ingestion/` — document loading & chunking
- `src/retrieval/` — embeddings & FAISS
- `src/generation/` — model wrappers
- `src/evaluation/` — RAGAS & manual scoring
- `notebooks/` — experiments
- `results/` — output tables & charts
- `configs/` — YAML configs for runs