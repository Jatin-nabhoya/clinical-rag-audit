# Clinical RAG Hallucination Audit — User Manual

> A reproducible pipeline for comparing hallucination behaviour across open-source LLMs in a clinical Retrieval-Augmented Generation setting.

**Authors:** Jatin Nabhoya · Mohit Raiyani  
**Institution:** University of New Haven — Graduate NLP Seminar, 2026  
**Paper:** `reports/paper/clinical_rag_paper.md`  
**Repository:** `github.com/Jatin-nabhoya/clinical-rag-audit`

---

## What this project does

This pipeline audits how three open-source LLMs hallucinate when deployed in a clinical RAG system:

| Model | Parameters | Overall correct |
|-------|-----------|----------------|
| Mistral-7B-Instruct-v0.2 | 7B | **52.7%** |
| Llama-3-8B-Instruct | 8B | 39.1% |
| Phi-3-mini-4k-instruct | 3.8B | 36.4% |

**Key finding:** Over-refusal (35–55%) is the dominant failure mode across all models — substantially larger than fabrication (≤1.8%). Safety-tuned open-source LLMs sacrifice utility, not safety, in clinical RAG settings.

---

## Quick start — reproduce our analysis without a GPU

If you just want to run the scoring and visualisation on our pre-generated outputs:

```bash
git clone https://github.com/Jatin-nabhoya/clinical-rag-audit.git
cd clinical-rag-audit
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Score the pre-generated responses (CPU only)
python scripts/score_hallucinations.py

# Regenerate all 8 publication charts
python scripts/visualize_results.py

# Regenerate all 5 result tables
python reports/phase8/generate_analysis.py
```

Output goes to `results/reports/figures/` (charts) and `reports/phase8/tables/` (CSV tables).

---

## Full pipeline — reproduce from scratch (requires CUDA GPU)

### 1. Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | Other versions untested |
| CUDA GPU | ≥16 GB VRAM | Kaggle T4 free tier works |
| HuggingFace account | — | Required for Llama-3 (gated model) |
| NCBI account | — | Optional; provide email for higher rate limits |

### 2. Installation

```bash
git clone https://github.com/Jatin-nabhoya/clinical-rag-audit.git
cd clinical-rag-audit

# Create and activate virtual environment
python3.11 -m venv .clinical-rag-audit
source .clinical-rag-audit/bin/activate        # macOS / Linux
# .clinical-rag-audit\Scripts\activate.bat     # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 3. Configure secrets

Create a `.env` file in the project root (never commit this):

```bash
ENTREZ_EMAIL=your@email.com   # for NCBI E-utilities polite use
HF_TOKEN=hf_...               # HuggingFace token — required for Llama-3
```

### 4. Phase 1 — Data collection

```bash
# PubMed Central articles (run once per domain)
python scripts/download_pmc.py --query "hypertension management"   --max_results 50 --domain_tag cardiology
python scripts/download_pmc.py --query "tuberculosis treatment"    --max_results 20 --domain_tag infectious_disease
python scripts/download_pmc.py --query "hepatocellular carcinoma"  --max_results 20 --domain_tag oncology

# CDC and WHO fact sheets
python scripts/download_cdc.py --urls_file configs/cdc_urls.txt --domain_tag infectious_disease

# MedlinePlus topics
python scripts/download_medlineplus.py --topics "diabetes" "asthma" "stroke"

# Verify everything downloaded correctly
python scripts/verify_metadata.py
```

Expected output: `data/metadata.csv` with 110 rows.

### 5. Phase 2 — Preprocessing and chunking

```bash
python scripts/ingest_documents.py          # extract text → chunk → save chunks.jsonl
python scripts/augment_corpus.py         # fetch additional infectious disease articles
python scripts/relabel_domains.py   # fix domain labels with keyword scoring
python scripts/clean_chunks.py      # remove LaTeX noise + micro-chunks

# Verify output
python scripts/inspect_corpus.py
```

Expected output: `data/processed/chunks_clean.jsonl` with 2,753 chunks.

### 6. Phase 3 — Build vector indexes

```bash
# Build the PubMedBERT medical index (used in all evaluations) — ~3 min on CPU
python src/retrieval/embed.py --model medical

# Build the MiniLM general baseline index — ~1 min
python src/retrieval/embed.py --model general

# Sanity check both indexes
python src/retrieval/inspect_index.py --index all
```

Expected output: `data/vector_store/medical/` and `data/vector_store/general/`.

### 7. Phase 4 — Pipeline smoke test (requires GPU)

```bash
# Test one model before running the full evaluation
python scripts/smoke_test.py --model mistral
```

### 8. Phase 5 — Build and validate the evaluation set

```bash
# Inspect corpus topic coverage
python scripts/explore_corpus.py

# Generate the 110-question evaluation set
python scripts/generate_eval_questions.py

# Validate all questions against the FAISS index
python scripts/validate_questions.py         # full retrieval validation (activate venv first)
python scripts/validate_questions.py --no-retrieval  # fast schema-only check
```

Expected output: `data/processed/eval_questions.jsonl` (110 questions).

### 9. Phase 5 — Run evaluation (requires CUDA GPU, ~2–3 hours on T4)

```bash
python scripts/run_inference.py              # all 3 models, 110 questions each
python scripts/run_inference.py --model mistral   # single model
```

Expected output: `results/eval_hallucination_audit/*/generations.jsonl` (110 outputs per model).

### 10. Phase 5 — Analyse results (CPU, no GPU required)

```bash
python scripts/analyze_hallucinations.py    # ROUGE-L, refusal rates, keyword recall
python scripts/generate_report.py           # cross-model audit report
```

### 11. Phase 6 — Hallucination scoring and visualisation (CPU)

```bash
python scripts/score_hallucinations.py      # 7-category taxonomy + bootstrap CIs
python scripts/visualize_results.py         # 8 publication-quality charts (300 DPI)
```

Expected output: `results/eval_hallucination_audit/taxonomy.csv`, `results/reports/figures/*.png`.

---

## Project structure

```
clinical-rag-audit/
├── .env                          ← Secrets (never commit — git-ignored)
├── requirements.txt              ← Full pip dependency list
├── README_USER_MANUAL.md         ← This file
│
├── configs/
│   └── cdc_urls.txt              ← 20 CDC/WHO fact-sheet URLs
│
├── data/
│   ├── metadata.csv              ← Ground truth: 110 docs, one row each
│   ├── raw/                      ← Downloaded files (git-ignored)
│   │   ├── pmc/                  ← 94 PMC XML articles
│   │   ├── cdc_who/              ← 13 CDC & WHO HTML pages
│   │   ├── medlineplus/          ← 3 MedlinePlus JSON responses
│   │   └── medquad/, bioasq/     ← Supplementary Q&A sources
│   ├── processed/
│   │   ├── chunks_clean.jsonl    ← ★ Use this — 2,753 production chunks
│   │   └── eval_questions.jsonl  ← ★ 110-question gold evaluation set
│   └── vector_store/             ← FAISS indexes (git-ignored, rebuild with embed.py)
│       ├── general/              ← MiniLM-L6-v2 baseline (dim=384)
│       └── medical/              ← PubMedBERT primary (dim=768)
│
├── scripts/                      ← All runnable pipeline scripts
│   ├── download_pmc.py           ← Phase 1: fetch PMC articles
│   ├── download_cdc.py           ← Phase 1: scrape CDC/WHO pages
│   ├── ingest_documents.py               ← Phase 2: Extract → Chunk → Save
│   ├── clean_chunks.py           ← Phase 2: quality filter
│   ├── generate_eval_questions.py← Phase 5: build gold eval set
│   ├── run_inference.py  ← Phase 5: run all 3 LLMs (GPU)
│   ├── analyze_hallucinations.py ← Phase 5: ROUGE-L, refusal rates
│   ├── score_hallucinations.py   ← Phase 6: 7-category taxonomy + CIs
│   └── visualize_results.py      ← Phase 6: 8 publication charts
│
├── src/
│   ├── retrieval/
│   │   ├── embed.py              ← Build FAISS indexes
│   │   └── retriever.py          ← Retriever class (retrieve + format_context)
│   ├── generation/
│   │   ├── config.py             ← Model registry + quantization config
│   │   ├── prompts.py            ← RAG + no-RAG prompt templates
│   │   ├── llm_wrapper.py        ← Unified LLMWrapper for all 3 models
│   │   └── rag_ingest_documents.py       ← RAGPipeline.answer() — full retrieve→generate
│   └── evaluation/
│       └── ragas_scorer.py       ← RAGAS metrics (optional, GPU + API key)
│
├── docs/
│   ├── taxonomy_definitions.md   ← 7-category hallucination taxonomy (v1.1)
│   └── annotation_guidelines.md  ← Eval set tier philosophy and edge cases
│
├── results/
│   └── eval_hallucination_audit/ ← Primary evaluation outputs
│       ├── llama3_8b/generations.jsonl
│       ├── mistral_7b/generations.jsonl
│       ├── phi3_mini/generations.jsonl
│       ├── combined_results.csv
│       ├── taxonomy.csv          ← 330 rows: one label per (model, question)
│       └── scoring_summary.json  ← Per-model × per-tier stats + bootstrap CIs
│
└── reports/
    ├── paper/
    │   └── clinical_rag_paper.md ← Full ACL-style conference paper
    └── phase8/
        ├── final_report.md       ← Technical report
        ├── tables/               ← 5 CSV result tables
        └── figures/              ← 8 publication charts (300 DPI PNG)
```

---

## Evaluation set — format reference

Each question in `data/processed/eval_questions.jsonl` follows this schema:

```json
{
  "question_id": "q_001",
  "question": "What surface antigens do seasonal influenza vaccines primarily target?",
  "tier": "answerable",
  "sub_tier": "direct_lookup",
  "hallucination_target": "factual_drift",
  "gold_answer": "Seasonal influenza vaccines primarily target...",
  "expected_behavior": "cite_and_answer",
  "domain": "infectious_disease",
  "difficulty": 1,
  "annotated_on": "2026-04-24"
}
```

**Tiers:** `answerable` (30) · `partial` (31) · `ambiguous` (20) · `unanswerable` (29)

---

## Corpus sources and licences

| Source | Docs | Format | Licence |
|--------|------|--------|---------|
| PubMed Central (PMC) | 94 | XML | Open Access / CC-BY |
| CDC Fact Sheets | 8 | HTML | Public Domain (US Gov) |
| WHO Fact Sheets | 5 | HTML | CC-BY-NC-SA 3.0 IGO |
| MedlinePlus | 3 | JSON | Public Domain (NLM/NIH) |

Raw data files are git-ignored. Rebuild using the Phase 1 download scripts above.

---

## Using your own documents

To run the pipeline on your own clinical corpus:

1. Place your documents in `data/raw/<your_source>/`
2. Add a row per document to `data/metadata.csv` (follow the existing schema)
3. Add a parser to `scripts/extract_text.py` if your format is not XML/HTML/JSON
4. Run `python scripts/ingest_documents.py` then `python scripts/clean_chunks.py`
5. Rebuild the FAISS index: `python src/retrieval/embed.py --model medical`
6. Run generation and scoring as normal

---

## Using your own LLM

The `LLMWrapper` in `src/generation/llm_wrapper.py` supports any HuggingFace instruction-tuned model. Add your model to the registry in `src/generation/config.py`:

```python
MODEL_REGISTRY = {
    "my_model": {
        "hf_id": "org/my-model-name",
        "chat_template": "llama",   # or "mistral" or "phi"
    },
    ...
}
```

Then run: `python scripts/run_inference.py --model my_model`

---

## Results summary

| Metric | Llama-3-8B | Mistral-7B | Phi-3-mini |
|--------|-----------|-----------|-----------|
| Overall correct | 39.1% | **52.7%** | 36.4% |
| Over-refusal | **54.5%** | 35.5% | 42.7% |
| Fabrication | 0.0% | 1.8% | 0.9% |
| Gap-filling | 5.5% | 4.5% | **10.0%** |
| Context overlap (mean) | **0.567** | 0.483 | 0.199 |
| ROUGE-L (mean) | 0.132 | **0.160** | 0.099 |

Full tables and charts: `reports/phase8/tables/` and `results/reports/figures/`.

---

## Citation

If you use this pipeline or dataset in your research:

```bibtex
@misc{nabhoya2026clinicalrag,
  title   = {Auditing Hallucination in Clinical Retrieval-Augmented Generation:
             A Comparative Study of Three Open-Source LLMs},
  author  = {Nabhoya, Jatin and Raiyani, Mohit},
  year    = {2026},
  note    = {University of New Haven NLP Seminar Project},
  url     = {https://github.com/Jatin-nabhoya/clinical-rag-audit}
}
```

---

## Troubleshooting

**`FileNotFoundError: No embed_config.json`**
→ Run `python src/retrieval/embed.py --model medical` to build the FAISS index first.

**`CUDA out of memory`**
→ Only one model loads at a time. Ensure no other GPU processes are running. The pipeline automatically unloads each model before loading the next.

**`HF 401 Unauthorized`**
→ Add your HuggingFace token to `.env` as `HF_TOKEN=hf_...` and accept the Llama-3 licence at `huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`.

**`chunks_clean.jsonl not found`**
→ Run the full Phase 2 pipeline: `ingest_documents.py` → `augment_corpus.py` → `clean_chunks.py`.

---

## Licence

Code: MIT. See `LICENSE`.  
Clinical documents: licences per source — see Corpus table above.
