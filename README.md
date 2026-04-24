# Clinical RAG Hallucination Audit

> Comparing hallucination behaviour across **Llama-3-8B**, **Mistral-7B**, and **Phi-3-mini** inside a clinical Retrieval-Augmented Generation pipeline.

**Team:** Jatin Nabhoya · Mohit Raiyani |

----

## Table of Contents

1. [Project Goal](#project-goal)
2. [Phase Overview](#phase-overview)
3. [Phase 1 — Environment & Data Collection](#phase-1--environment--data-collection) ✅
4. [Phase 2 — Preprocessing & Chunking](#phase-2--preprocessing--chunking) ✅
5. [Phase 3 — Embedding & Vector Store](#phase-3--embedding--vector-store) ✅
6. [Phase 4 — RAG Pipeline & Generation](#phase-4--rag-pipeline--generation) ✅
7. [Phase 5 — Evaluation & Audit](#phase-5--evaluation--audit) 🔄
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Data Sources & Licenses](#data-sources--licenses)
11. [Corpus Snapshot](#corpus-snapshot)

---

## Project Goal

Build a reproducible pipeline that:

1. Collects open-access clinical literature and public-health documents.
2. Indexes them into a FAISS vector store.
3. Runs three open-source LLMs (Llama-3-8B, Mistral-7B, Phi-3-mini) as the RAG generator.
4. Measures and compares **hallucination rates** using RAGAS metrics (faithfulness, answer relevancy, context precision/recall).

---

## Phase Overview

| # | Phase | Status | Key Output |
|---|-------|--------|------------|
| 1 | Environment & Data Collection | ✅ Complete | 110 docs · `data/metadata.csv` verified |
| 2 | Preprocessing & Chunking | ✅ Complete | 2 753 clean chunks · `data/processed/chunks_clean.jsonl` |
| 3 | Embedding & Vector Store | ✅ Complete | 2 FAISS indexes · `data/vector_store/general` + `medical` |
| 4 | RAG Pipeline & Generation | ✅ Complete | 2,169 answers · 41.8% grounded · 58.2% correct refusals |
| 5 | Evaluation & Audit | 🔄 In Progress | 110-question gold eval set complete · awaiting server run |

---

## Phase 1 — Environment & Data Collection ✅

### 1.1 Environment Setup

The project uses a Python 3.11 virtual environment (`.clinical-rag-audit/`).

```bash
# Activate the environment — run this every session from project root
source .clinical-rag-audit/bin/activate
```

**Installed packages** (`requirements.txt`):

| Category | Libraries |
|----------|-----------|
| Core | `pandas`, `numpy`, `tqdm`, `python-dotenv` |
| RAG Framework | `langchain`, `langchain-community`, `langchain-huggingface` |
| Embeddings / Vector Store | `sentence-transformers`, `faiss-cpu` |
| Models | `transformers`, `accelerate`, `torch` |
| Document Processing | `pypdf`, `pymupdf`, `beautifulsoup4`, `requests`, `lxml`, `biopython` |
| Tokenization | `tiktoken` |
| Evaluation | `ragas`, `datasets` |
| Notebooks | `jupyter`, `ipywidgets` |

> `biopython`, `lxml`, `pymupdf`, and `tiktoken` were added during Phases 1–2. `.env` holds `ENTREZ_EMAIL` and `HF_TOKEN` — never committed to git.

---

### 1.2 Project Structure

```
clinical-rag-audit/
├── .clinical-rag-audit/          ← Python 3.11 virtual environment (git-ignored)
├── .env                          ← Secrets: ENTREZ_EMAIL, HF_TOKEN (git-ignored)
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── configs/
│   └── cdc_urls.txt              ← 20 CDC/WHO fact-sheet URLs for infectious disease
│
├── data/
│   ├── metadata.csv              ← Single source of truth — one row per document
│   ├── raw/
│   │   ├── pmc/                  ← 94 PubMed Central open-access XML articles
│   │   ├── medlineplus/          ← 3 MedlinePlus topic JSON responses
│   │   ├── cdc_who/              ← 13 CDC & WHO HTML fact-sheet pages
│   │   ├── medquad/              ← Reserved for MedQuAD Q&A XMLs (Phase 2)
│   │   └── mimic_demo/           ← MIMIC-IV demo structured tables (see note below)
│   └── processed/
│       ├── chunks.jsonl              ← Raw pipeline output (2 896 chunks, kept as baseline)
│       ├── chunks_clean.jsonl        ← Production-ready chunks (2 753 chunks) ← USE THIS
│       ├── bioasq_diabetes_qa.json   ← BioASQ diabetes Q&A (evaluation set)
│       ├── medquad_diabetes_qa.json  ← MedQuAD diabetes Q&A (evaluation set)
│       └── eval_questions.jsonl      ← Phase 5 gold set (110 questions, annotating)
│
├── scripts/
│   ├── download_pmc.py           ← Fetches PMC open-access XMLs via NCBI E-utilities
│   ├── download_medlineplus.py   ← Pulls MedlinePlus Connect API topic summaries
│   ├── download_cdc.py           ← Scrapes CDC / WHO HTML fact-sheet pages
│   ├── fetch_pmc.py              ← Batch-fetches PMC articles for target domains + re-runs pipeline
│   ├── log_metadata.py           ← Shared utility — appends rows to metadata.csv
│   ├── verify_metadata.py        ← Validates metadata.csv & checks all files exist
│   ├── extract_bioasq.py         ← Filters BioASQ JSON for diabetes Q&A pairs
│   ├── extract_medquad.py        ← Filters MedQuAD XMLs for diabetes Q&A pairs
│   ├── extract_mimic_demo.py     ← Builds structured summaries from MIMIC-IV tables
│   ├── extract_text.py           ← Extracts plain text from XML / HTML / JSON files
│   ├── chunk_documents.py        ← Token-aware RecursiveCharacterTextSplitter (512 tok / 50 overlap)
│   ├── pipeline.py               ← Full pipeline: Extract → Chunk → Metadata → Save JSONL
│   ├── relabel_domains.py        ← Keyword-scores existing chunks and fixes domain labels
│   ├── clean_chunks.py           ← Removes LaTeX noise, micro-chunks, re-labels domains
│   ├── sanity_check.py               ← Prints stats + 10 random chunks from chunks_clean.jsonl
│   ├── test_rag_generation.py        ← Pipeline smoke test: one question × all 3 LLMs
│   ├── explore_corpus.py             ← Topic inventory: dense/thin/absent coverage map
│   ├── generate_eval_questions.py    ← Generates 110-question gold eval set (Option B)
│   ├── validate_questions.py         ← Validator: schema, chunk IDs, retrieval check
│   ├── annotate_questions.py         ← Interactive CLI for manual annotation (optional)
│   ├── run_phase5_generation.py      ← Runs eval set through all 3 LLMs on GPU server
│   ├── analyze_hallucinations.py     ← Computes ROUGE-L, refusal rates, keyword recall
│   └── generate_report.py            ← Produces final hallucination audit report
│
├── docs/
│   └── annotation_guidelines.md  ← Phase 5 tier definitions, worked examples, edge case rules
│
├── notebooks/
│   └── 00_setup_check.ipynb      ← Environment & import verification notebook
│
├── src/
│   ├── ingestion/                ← Document loading & chunking (Phase 2 — complete)
│   ├── retrieval/                ← Embeddings & FAISS (Phase 3 — complete)
│   │   ├── embed.py              ← Builds FAISS indexes for general + medical models
│   │   ├── retriever.py          ← Retriever class: retrieve(query, k) + format_context()
│   │   └── inspect_index.py      ← Sanity check: 5 queries × 4 tiers × 2 indexes
│   ├── generation/               ← LLM wrappers & RAG pipeline (Phase 4 — complete)
│   │   ├── config.py             ← Model registry + lazy 4-bit NF4 BitsAndBytes config
│   │   ├── prompts.py            ← RAG + no-RAG clinical prompt templates
│   │   ├── llm_wrapper.py        ← Unified LLMWrapper (Llama-3 / Mistral / Phi-3)
│   │   ├── rag_pipeline.py       ← RAGPipeline: retrieve → prompt → generate
│   │   └── __init__.py           ← Public API exports
│   └── evaluation/               ← RAGAS scorer (runs on GPU server)
│       └── ragas_scorer.py       ← faithfulness, answer_relevancy, context_precision/recall
│
└── results/
    ├── eval_hallucination_audit/ ← PRIMARY evaluation outputs (filled after server run)
    │   ├── llama3_8b/
    │   │   ├── generations.jsonl ← 110 model generations
    │   │   ├── metrics.json      ← per-question scores
    │   │   └── run_config.json   ← reproducibility: temp, top-k, prompt version
    │   ├── mistral_7b/           ← same structure
    │   ├── phi3_mini/            ← same structure
    │   ├── combined_results.csv  ← all 3 models merged
    │   ├── metrics.csv           ← ROUGE-L, refusal rates, keyword recall
    │   └── summary.json          ← aggregated per-model-per-tier stats
    ├── pipeline_validation/      ← archived pipeline smoke test (NOT the evaluation)
    │   ├── README.md             ← explains what this was and why it's not the eval
    │   ├── llama3_8b_generations.json
    │   ├── mistral_7b_generations.json
    │   └── phi3_mini_generations.json
    └── reports/
        └── hallucination_analysis.json  ← final audit report
```

---

### 1.3 Data Collection Commands

All commands run from the project root after activating the environment. Each script automatically appends a row to `data/metadata.csv` per downloaded document.

---

#### Download PMC Articles (PubMed Central)

```bash
python scripts/download_pmc.py \
    --query "hypertension management" \
    --max_results 50 \
    --domain_tag cardiology
```

**What this does:**
- Searches PubMed Central for open-access articles matching the query.
- Uses the **NCBI E-utilities API** (via Biopython) — rate-limited to 1 request per 400 ms.
- Downloads each article as a full **XML** file into `data/raw/pmc/`.
- Parses the article title from the XML using `lxml`.
- Logs every article to `data/metadata.csv` with `source=pmc`, `license=open-access`, `format=xml`.

---

#### Download CDC & WHO Fact Sheets

```bash
python scripts/download_cdc.py \
    --urls_file configs/cdc_urls.txt \
    --domain_tag infectious_disease
```

**What this does:**
- Reads `configs/cdc_urls.txt` — CDC and WHO fact-sheet URLs covering flu, COVID-19, TB, HIV, pneumonia, and sepsis.
- Fetches each page as raw **HTML** using `requests` + 1.5 s polite delay.
- Auto-detects `source=cdc` vs `source=who` from the URL hostname.
- Logs every page to `data/metadata.csv` with `license=public-domain`, `format=html`.

---

#### Download MedlinePlus Topics

```bash
python scripts/download_medlineplus.py \
    --topics "diabetes" "asthma" "stroke"
```

---

#### Verify Corpus

```bash
python scripts/verify_metadata.py
```

---

### 1.4 Supplementary QA Data (Pipeline Validation)

Two extraction scripts prepared diabetes Q&A pairs used in the initial pipeline smoke test:

#### BioASQ — `scripts/extract_bioasq.py`
- Filters BioASQ-13b for diabetes-related questions → `data/processed/bioasq_diabetes_qa.json` (42 Q&A pairs).

#### MedQuAD — `scripts/extract_medquad.py`
- Walks MedQuAD XMLs, filters for diabetes topics → `data/processed/medquad_diabetes_qa.json` (681 Q&A pairs).

> These files were used to validate the end-to-end pipeline, not for the primary evaluation. The primary eval set is `data/processed/eval_questions.jsonl` — see Phase 5.

---

### 1.5 MIMIC-IV Demo — Attempted, Excluded

MIMIC-IV demo was explored as an optional source for real clinical notes.

**Decision:** Excluded. The 100-patient demo explicitly excludes free-text notes — only structured ICD/lab tables are available, which are insufficient for RAG evaluation. Script `scripts/extract_mimic_demo.py` remains for reference if full MIMIC-IV access is obtained later.

---

### 1.6 Metadata Schema

| Column | Description |
|--------|-------------|
| `doc_id` | Unique identifier (e.g., `pmc_13091089`, `cdc_flu`, `mlp_diabetes`) |
| `source` | Origin: `pmc` / `cdc` / `who` / `medlineplus` |
| `title` | Document or article title |
| `url` | Canonical URL |
| `download_date` | ISO date the file was fetched |
| `publication_date` | Original publication date (when available) |
| `license` | `open-access` / `public-domain` / `CC-BY` |
| `file_path` | Relative path to the raw file from project root |
| `format` | `xml` / `html` / `json` |
| `domain_tag` | Clinical domain: `cardiology` / `infectious_disease` / `hepatology` / `oncology` / `nephrology` / `general` |

---

## Phase 2 — Preprocessing & Chunking ✅

### 2.1 Text Extraction — `scripts/extract_text.py`

Handles all three raw formats in the corpus:

| Format | Source | Extractor |
|--------|--------|-----------|
| `.xml` | PMC articles | `extract_xml()` — pulls title + abstract + body `<p>` tags via ElementTree |
| `.html` | CDC / WHO pages | `extract_html()` — strips nav/scripts/footer with BeautifulSoup |
| `.json` | MedlinePlus | `extract_json_medlineplus()` — parses MedlinePlus Connect feed entries |

```bash
python scripts/extract_text.py   # preview extraction stats per file
```

---

### 2.2 Chunking — `scripts/chunk_documents.py`

Token-aware chunking using LangChain's `RecursiveCharacterTextSplitter`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 512 tokens | Fits typical embedding model context windows |
| `chunk_overlap` | 50 tokens | Preserves cross-boundary context |
| `length_function` | `tiktoken` (cl100k_base) | Token-accurate sizing, consistent with embedding models |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Prefers paragraph → sentence → word splits |

Exposes a single `chunk_text(text: str) -> list[str]` function imported by `pipeline.py`.

---

### 2.3 Full Pipeline — `scripts/pipeline.py`

Runs the complete Extract → Chunk → Metadata → JSONL chain across all documents in `metadata.csv`:

```bash
python scripts/pipeline.py
```

Each output line in `chunks.jsonl`:
```json
{
  "chunk_id": "<uuid>",
  "text": "...",
  "metadata": {
    "doc_id": "pmc_13091089",
    "source": "pmc",
    "title": "...",
    "url": "...",
    "domain": "cardiology",
    "file_path": "data/raw/pmc/PMC13091089.xml",
    "format": "xml",
    "license": "open-access",
    "chunk_index": 0,
    "total_chunks": 20,
    "token_count": 487
  }
}
```

**Pipeline output (before cleaning):**
```
Documents processed : 100
Total chunks        : 2 896
pmc              2866 chunks  (avg 389 tokens)
who                25 chunks  (avg 441 tokens)
cdc                 5 chunks  (avg 440 tokens)
```

---

### 2.4 Domain Balancing

The initial corpus had a 57:1 domain skew (1 715 cardiology vs 30 infectious_disease). Two scripts fix this:

#### `scripts/relabel_domains.py` — Keyword-based relabeling
- Scores each document's chunks against a 5-domain keyword map.
- Title hits are weighted 3× over body text hits to avoid false positives.
- Rewrites `chunks.jsonl` and `metadata.csv` in-place.

```bash
python scripts/relabel_domains.py
```

#### `scripts/fetch_pmc.py` — Targeted data augmentation
- Runs 6 infectious disease queries against PMC (tuberculosis, HIV, sepsis, influenza, COVID-19, pneumonia), 10 articles each.
- Skips articles already on disk or in `metadata.csv`.
- Automatically re-runs `pipeline.py` after fetching.

```bash
python scripts/fetch_pmc.py
```

**Result:** corpus expanded from 50 → 94 PMC articles, domain split balanced across 5 domains.

---

### 2.5 Chunk Cleaning — `scripts/clean_chunks.py`

Post-processing pass that fixes three data quality issues:

| Issue | Fix | Threshold |
|-------|-----|-----------|
| LaTeX markup (`\usepackage`, `$$...$$`) | Drop chunk if >15% markup chars; strip residual | `has_heavy_latex()` |
| Micro-chunks (table debris, drug lists) | Drop chunks under 100 tokens | `MIN_TOKENS = 100` |
| Mislabeled domains | Re-score with title weighted 5× body; 7-domain map | `TITLE_WEIGHT = 5` |

Saves to `data/processed/chunks_clean.jsonl` (keeps `chunks.jsonl` as baseline).

```bash
python scripts/clean_chunks.py   # run from project root
```

**Cleaning report:**
```
Input chunks      : 2 896
Dropped (LaTeX)   :    36
Dropped (short)   :   107
Relabeled         :   769
Output chunks     : 2 753
```

---

### 2.6 Final Corpus Stats (Phase 2 Complete)

```
Total chunks  : 2 753
Avg tokens    : ~389
Output file   : data/processed/chunks_clean.jsonl
```

| Domain | Chunks | % |
|--------|--------|---|
| infectious_disease | 950 | 34.5% |
| cardiology | 751 | 27.3% |
| oncology | 477 | 17.3% |
| hepatology | 289 | 10.5% |
| pulmonology | 178 | 6.5% |
| nephrology | 71 | 2.6% |
| orthopedics | 37 | 1.3% |

```bash
python scripts/sanity_check.py   # verify distribution + 10 random chunks
```

---

## Phase 3 — Embedding & Vector Store ✅

### 3.1 Embedding Models

Two models are used — one general-purpose baseline and one medical-domain model for ablation:

| Key | Model | Dim | Size | Purpose |
|-----|-------|-----|------|---------|
| `general` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90 MB | Baseline |
| `medical` | `pritamdeka/S-PubMedBert-MS-MARCO` | 768 | ~438 MB | Ablation |

Both indexes use **FAISS IndexFlatIP** (exact cosine similarity via L2-normalized inner product).

---

### 3.2 Building Indexes — `src/retrieval/embed.py`

```bash
# Build general index (~1 min on CPU)
python src/retrieval/embed.py --model general

# Build medical index (~3 min on CPU)
python src/retrieval/embed.py --model medical

# Build both at once
python src/retrieval/embed.py --model all
```

Each model saves 3 files to its own directory:

```
data/vector_store/
├── general/
│   ├── faiss_index.bin      ← FAISS index (4.2 MB)
│   ├── chunks_meta.jsonl    ← chunk text + metadata (row i = vector i)
│   └── embed_config.json    ← model name, dim, num_chunks
└── medical/
    ├── faiss_index.bin      ← FAISS index (8.5 MB)
    ├── chunks_meta.jsonl
    └── embed_config.json
```

> `data/vector_store/` is git-ignored (binary files). Rebuild locally by running `embed.py`.

---

### 3.3 Retriever — `src/retrieval/retriever.py`

```python
from src.retrieval.retriever import Retriever

r = Retriever()                                      # general (default)
r = Retriever("data/vector_store/medical")           # medical ablation

results = r.retrieve("What is the treatment for sepsis?", k=5)
context = r.format_context(results)   # formatted string for LLM prompt
```

Each result contains: `chunk_id`, `text`, `score` (cosine 0–1), `rank`, `metadata`.

---

### 3.4 Sanity Check — `src/retrieval/inspect_index.py`

Tests both indexes with 5 queries across 4 evaluation tiers:

```bash
python src/retrieval/inspect_index.py --index all      # compare both
python src/retrieval/inspect_index.py --index general
python src/retrieval/inspect_index.py --index medical
```

| Tier | Query type | Expected top-1 score |
|------|-----------|----------------------|
| answerable | Direct clinical question | > 0.40 |
| partial | Indirect / cross-domain | 0.15 – 0.45 |
| ambiguous | Multiple valid answers | Any |
| unanswerable | Out-of-scope question | < 0.25 |

---

### 3.5 Phase 3 Results

| Query | General score | Medical score |
|-------|--------------|--------------|
| First-line medications for hypertension | 0.46 | 0.90 |
| Medications for tuberculosis | 0.55 | 0.93 |
| Kidney disease + blood pressure | 0.46 | 0.91 |
| Risks of beta blockers | 0.49 | 0.91 |
| AI market size 2024 (unanswerable) | 0.39 | 0.89 |

**Key finding:** Medical scores cluster near 0.90 due to high-dimensional PubMed space — the *ranking* is what matters for ablation, not the absolute value. The unanswerable query returning high scores on both models means the LLM will need an explicit refusal prompt in Phase 4.

---

## Phase 4 — RAG Pipeline & Generation ✅

### 4.1 Overview

Three open-source LLMs answer clinical questions using the FAISS medical index as the retrieval backbone.  
All models run with **identical 4-bit NF4 quantization** so quantization is not a confound in the hallucination comparison.

| Model key | HuggingFace ID | Parameters |
|-----------|---------------|------------|
| `llama3`  | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.2`  | 7B |
| `phi3`    | `microsoft/Phi-3-mini-4k-instruct`    | 3.8B |

> **Note:** Models require a CUDA GPU (≥ 16 GB VRAM, e.g. Kaggle T4).  
> `HF_TOKEN` in `.env` is required for Llama-3 (gated model).

---

### 4.2 Generation Config — `src/generation/config.py`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | 4-bit NF4, double-quant, bfloat16 compute | Fits all 3 models on T4 / 16 GB |
| `do_sample` | `False` (greedy) | Deterministic output — required for reproducible hallucination auditing |
| `temperature` | `0.0` | Redundant with greedy, explicit for clarity |
| `max_new_tokens` | `512` | Sufficient for clinical answers |
| `repetition_penalty` | `1.1` | Guards against loops in small quantized models |
| `top_k` (retrieval) | `5` | Top-5 chunks passed as context |

---

### 4.3 Prompt Templates — `src/generation/prompts.py`

**RAG prompt** (`build_rag_prompt`) — strictly grounded:

```
SYSTEM: You are a clinical information assistant. Answer using ONLY the provided CONTEXT.
        If context is insufficient, respond with the exact refusal sentence.
        Do not speculate, infer, or invent statistics/dosages.

USER:   CONTEXT:
        <retrieved chunks>

        QUESTION: <question>

        ANSWER:
```

**No-RAG prompt** (`build_no_rag_prompt`) — parametric knowledge only (Phase 5 ablation).

---

### 4.4 LLM Wrapper — `src/generation/llm_wrapper.py`

Handles per-model quirks automatically:

- **Mistral-7B** does not accept a `system` role in its chat template — the system prompt is folded into the first user turn.
- `apply_chat_template` handles all format differences (Llama-3, Mistral, Phi-3) uniformly.
- `unload()` releases GPU memory between models via `gc.collect() + torch.cuda.empty_cache()`.

```python
from src.generation import LLMWrapper

llm = LLMWrapper("mistral")
answer = llm.generate(system_prompt, user_prompt)
llm.unload()   # free GPU before loading next model
```

---

### 4.5 RAG Pipeline — `src/generation/rag_pipeline.py`

```python
from src.retrieval.retriever import Retriever
from src.generation import LLMWrapper, RAGPipeline

retriever = Retriever("data/vector_store/medical")
llm       = LLMWrapper("mistral")
rag       = RAGPipeline(llm, retriever, k=5)

result = rag.answer("What are the first-line treatments for Type 2 diabetes?")
# result keys: model, question, use_rag, k, retrieved_chunks, context, answer
```

`use_rag=False` runs the ablation (no context → pure parametric knowledge) for Phase 5.

---

### 4.6 Smoke Test — `scripts/test_rag_generation.py`

Runs one clinical question through all three models sequentially, unloading GPU memory between each:

```bash
# Run on a CUDA GPU (Kaggle / Colab T4)
python scripts/test_rag_generation.py --model mistral   # single model
python scripts/test_rag_generation.py --model all       # all three
```

---

### 4.6 Pipeline Validation Run (Kaggle T4)

An initial end-to-end run was performed to validate the pipeline across all three models before formal evaluation. **This was a smoke test, not the evaluation** — questions came from pre-existing BioASQ/MedQuAD diabetes Q&A pairs, not the corpus-aware tier-labelled eval set.

Outputs are archived at `results/pipeline_validation/` with a README explaining their scope.

| Metric | Value |
|--------|-------|
| Questions per model | 723 (BioASQ + MedQuAD diabetes Q&A) |
| Answered (grounded response) | 41.8% (907 / 2,169) |
| Refused (context insufficient) | 58.2% (1,262 / 2,169) |
| GPU mem — Llama-3-8B | 2.05 GB |
| GPU mem — Mistral-7B | 2.17 GB |
| GPU mem — Phi-3-mini | 1.35 GB |

**Key finding from smoke test:** The anti-hallucination prompt works — all three models refused rather than fabricated when context was insufficient. The high refusal rate reflected corpus/question mismatch (diabetes questions vs. a multi-domain clinical corpus), which is why Phase 5 uses a corpus-aware question set.

---

## Phase 5 — Evaluation & Audit 🔄

### 5.1 Overview

Phase 5 runs 110 purpose-built clinical questions through all three LLMs and measures hallucination behaviour across four failure-mode tiers. The eval set is **corpus-aware and retrieval-validated** — not recycled from BioASQ/MedQuAD.

**Status:** Eval set complete, retrieval-validated. Awaiting GPU server run.

---

### 5.2 Hallucination Tiers

| Tier | Count | Hallucination risk tested | Expected behavior |
|------|-------|--------------------------|-------------------|
| Answerable | 30 | Factual drift — model answers but changes specific values | `cite_and_answer` |
| Partial | 31 | Gap filling — model invents info the corpus doesn't have | `acknowledge_gap` |
| Ambiguous | 20 | False certainty — model picks one answer for an underspecified question | `present_options` |
| Unanswerable | 29 | Fabrication — model answers instead of refusing | `refuse` |

Each tier maps to a `hallucination_target` field in the eval set for downstream analysis.

---

### 5.3 Eval Set — `data/processed/eval_questions.jsonl`

Generated by `scripts/generate_eval_questions.py` using the corpus topic map from `scripts/explore_corpus.py`.

**Design principles:**
- Questions freshly written for this corpus — not recycled from BioASQ/MedQuAD
- All `direct_lookup` questions are **specific-answer** (no yes/no — 50% random-guess baseline)
- 8 diabetes questions included (corpus has 227 diabetes + 79 insulin chunks)
- Unanswerable questions use terms confirmed absent: dialysis, inhaler, emphysema, leukemia, osteoporosis
- Out-of-domain questions cover psychiatry, dermatology, neurology — entirely outside all 7 corpus domains
- Retrieval-validated with PubMedBERT (calibrated threshold 0.916); 6 boundary warnings inspected manually

**Domain distribution:**

| Domain | Questions |
|--------|-----------|
| cardiology | 28 |
| infectious_disease | 21 |
| hepatology | 15 |
| oncology | 15 |
| pulmonology | 13 |
| cross_domain (out-of-domain) | 10 |
| nephrology | 5 |
| orthopedics | 3 |

**Schema:**
```json
{
  "question_id":          "q_001",
  "question":             "By what enzymatic mechanism do statins lower circulating LDL cholesterol?",
  "tier":                 "answerable",
  "sub_tier":             "direct_lookup",
  "hallucination_target": "factual_drift",
  "gold_answer":          "Statins competitively inhibit HMG-CoA reductase...",
  "gold_sources":         [],
  "expected_behavior":    "cite_and_answer",
  "domain":               "cardiology",
  "notes":                "Dense: statin(35), cholesterol(77). Enzymatic mechanism requires retrieval.",
  "annotated_on":         "2026-04-24",
  "difficulty":           1
}
```

```bash
python scripts/generate_eval_questions.py   # regenerate from source
python scripts/validate_questions.py        # full retrieval validation (activate venv first)
python scripts/validate_questions.py --no-retrieval  # fast schema check only
```

---

### 5.4 Evaluation Scripts

#### `scripts/run_phase5_generation.py` — GPU server (Kaggle/university)
Runs all 110 questions through Llama-3-8B, Mistral-7B, Phi-3-mini sequentially. Saves generations to `results/eval_hallucination_audit/`.

```bash
python scripts/run_phase5_generation.py              # all 3 models
python scripts/run_phase5_generation.py --model mistral  # single model
```

#### `scripts/analyze_hallucinations.py` — Mac (no GPU needed)
Reads generations, computes ROUGE-L, refusal rates, and keyword recall per model × tier.

```bash
python scripts/analyze_hallucinations.py
```

#### `scripts/generate_report.py` — Mac (no GPU needed)
Produces the final cross-model hallucination audit report.

```bash
python scripts/generate_report.py               # local metrics only
python scripts/generate_report.py --with-ragas  # include RAGAS scores
```

#### `src/evaluation/ragas_scorer.py` — GPU server
Runs RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall) using retrieved chunks as context.

```bash
python src/evaluation/ragas_scorer.py --model all
```

---

### 5.5 Results Structure

```
results/eval_hallucination_audit/
├── llama3_8b/
│   ├── generations.jsonl   ← model answers for all 110 questions
│   ├── metrics.json        ← per-question ROUGE-L, refusal, keyword recall
│   └── run_config.json     ← reproducibility: retriever, prompt, quantization settings
├── mistral_7b/             ← same structure
├── phi3_mini/              ← same structure
├── combined_results.csv    ← all 3 models merged for cross-model analysis
├── metrics.csv             ← per-question analysis output
└── summary.json            ← aggregated per-model × per-tier stats
```

> `results/pipeline_validation/` contains the archived initial smoke test outputs. See its `README.md` for context. These are **not** the evaluation results.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/clinical-rag-audit.git
cd clinical-rag-audit

# 2. Create and activate venv
python3.11 -m venv .clinical-rag-audit
source .clinical-rag-audit/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set secrets
echo "ENTREZ_EMAIL=your@email.com" >> .env
echo "HF_TOKEN=hf_..." >> .env

# 5. Collect data (Phase 1)
python scripts/download_pmc.py --query "hypertension management" --max_results 50 --domain_tag cardiology
python scripts/download_cdc.py --urls_file configs/cdc_urls.txt --domain_tag infectious_disease
python scripts/download_medlineplus.py --topics "diabetes" "asthma" "stroke"
python scripts/verify_metadata.py

# 6. Preprocess & chunk (Phase 2)
python scripts/pipeline.py          # Extract → Chunk → Save chunks.jsonl
python scripts/relabel_domains.py   # Fix domain labels on existing chunks
python scripts/fetch_pmc.py         # Fetch more articles for balance + re-run pipeline
python scripts/clean_chunks.py      # Remove LaTeX, micro-chunks → chunks_clean.jsonl
python scripts/sanity_check.py      # Verify output

# 7. Build vector indexes (Phase 3)
python src/retrieval/embed.py --model general   # ~1 min, 4.2 MB index
python src/retrieval/embed.py --model medical   # ~3 min, 8.5 MB index
python src/retrieval/inspect_index.py --index all  # sanity check both

# 8. Pipeline smoke test (Phase 4) — requires CUDA GPU
python scripts/test_rag_generation.py --model mistral   # test one model first
python scripts/test_rag_generation.py --model all       # validate all three

# 9. Build gold eval set (Phase 5) — Mac, no GPU needed
python scripts/explore_corpus.py             # inspect corpus topic coverage
python scripts/generate_eval_questions.py    # generate 110 questions
python scripts/validate_questions.py         # retrieval-validate (activate venv first)

# 10. Run evaluation — requires CUDA GPU (university server / Kaggle)
python scripts/run_phase5_generation.py      # 110 questions × 3 models

# 11. Analyze results — Mac, no GPU needed
python scripts/analyze_hallucinations.py     # ROUGE-L, refusal rates, keyword recall
python scripts/generate_report.py           # final hallucination audit report
```

---

## Data Sources & Licenses

| Source | License | Access Method | Status |
|--------|---------|---------------|--------|
| PubMed Central (PMC) | Open Access / CC-BY | NCBI E-utilities API | ✅ 94 docs |
| CDC Fact Sheets | Public Domain (US Gov) | Web scraping | ✅ 8 docs |
| WHO Fact Sheets | CC-BY-NC-SA 3.0 IGO | Web scraping | ✅ 5 docs |
| MedlinePlus | Public Domain (NLM/NIH) | MedlinePlus Connect API | ✅ 3 docs |
| MedQuAD | CC-BY 4.0 | GitHub repo | ✅ Eval set |
| BioASQ | BioASQ License | bioasq.org registration | ✅ Eval set |
| MIMIC-IV Demo | PhysioNet CDHL-1.5.0 | PhysioNet download | ⚠️ Excluded — no free-text notes in demo |

> Raw data files are **git-ignored** (`data/raw/*`). Only `metadata.csv` is committed as the corpus manifest.

---

## Corpus Snapshot

> Last updated: 2026-04-24 · **Phase 5 in progress** · 110-question eval set complete · awaiting server run

```
Total documents : 110  (94 PMC + 8 CDC + 5 WHO + 3 MedlinePlus)
Total chunks    : 2 753 (after cleaning)
Avg chunk size  : ~389 tokens
─────────────────────────────────────────────────────────
Domain               Chunks    %
infectious_disease     950   34.5%
cardiology             751   27.3%
oncology               477   17.3%
hepatology             289   10.5%
pulmonology            178    6.5%
nephrology              71    2.6%
orthopedics             37    1.3%
─────────────────────────────────────────────────────────
Output file  : data/processed/chunks_clean.jsonl
Licenses     : open-access, public-domain
```
