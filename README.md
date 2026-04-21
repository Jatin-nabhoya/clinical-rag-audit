# Clinical RAG Hallucination Audit

> Comparing hallucination behaviour across **Llama-3-8B**, **Mistral-7B**, and **Phi-3-mini** inside a clinical Retrieval-Augmented Generation pipeline.

**Team:** Jatin Nabhoya · Mohit Raiyani |

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Phase Overview](#phase-overview)
3. [Phase 1 — Environment & Data Collection](#phase-1--environment--data-collection) ✅
4. [Phase 2 — Preprocessing & Chunking](#phase-2--preprocessing--chunking) ✅
5. [Phase 3 — Embedding & Vector Store](#phase-3--embedding--vector-store) 🔜
6. [Phase 4 — RAG Pipeline & Generation](#phase-4--rag-pipeline--generation) 🔜
7. [Phase 5 — Evaluation & Audit](#phase-5--evaluation--audit) 🔜
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
| 3 | Embedding & Vector Store | 🔜 Next | FAISS index |
| 4 | RAG Pipeline & Generation | 🔜 Upcoming | Model answer JSONs |
| 5 | Evaluation & Audit | 🔜 Upcoming | RAGAS scores, hallucination report |

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
│       └── medquad_diabetes_qa.json  ← MedQuAD diabetes Q&A (evaluation set)
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
│   └── sanity_check.py           ← Prints stats + 10 random chunks from chunks_clean.jsonl
│
├── notebooks/
│   └── 00_setup_check.ipynb      ← Environment & import verification notebook
│
├── src/
│   ├── ingestion/                ← Document loading & chunking (Phase 2 — complete)
│   ├── retrieval/                ← Embeddings & FAISS (Phase 3)
│   ├── generation/               ← LLM wrappers (Phase 4)
│   └── evaluation/               ← RAGAS & manual scoring (Phase 5)
│
└── results/                      ← Output tables & charts (Phase 5)
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

### 1.4 Evaluation Data Preparation

Two extraction scripts prepare the **evaluation QA set**:

#### BioASQ — `scripts/extract_bioasq.py`
- Filters BioASQ-13b for diabetes-related questions → `data/processed/bioasq_diabetes_qa.json`.

#### MedQuAD — `scripts/extract_medquad.py`
- Walks MedQuAD XMLs, filters for diabetes topics → `data/processed/medquad_diabetes_qa.json`.

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

## Phase 3 — Embedding & Vector Store

> **Status: 🔜 Next**

Planned work:
- Embed chunks with `sentence-transformers` (candidate: `BioLORD-2023`).
- Build a FAISS flat-L2 index over `chunks_clean.jsonl`.
- Persist index to disk for reproducible retrieval.
- Update `src/retrieval/` modules.

---

## Phase 4 — RAG Pipeline & Generation

> **Status: 🔜 Upcoming**

Planned work:
- Load each LLM via HuggingFace `transformers` (Llama-3-8B, Mistral-7B, Phi-3-mini).
- Wire retriever → prompt template → generator using LangChain.
- Run on the fixed clinical QA test set from BioASQ / MedQuAD.
- Save model answers to `results/`.

---

## Phase 5 — Evaluation & Audit

> **Status: 🔜 Upcoming**

Planned work:
- Score each answer with **RAGAS** metrics: faithfulness, answer relevancy, context precision, context recall.
- Manual review of sampled hallucinated answers.
- Cross-model comparison tables and visualizations.
- Final report in `results/`.

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

> Last updated: 2026-04-21 · **Phase 2 complete**

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
