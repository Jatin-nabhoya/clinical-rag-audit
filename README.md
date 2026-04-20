# Clinical RAG Hallucination Audit

> **DSCI 6004 · NLP Course Project**
> Comparing hallucination behaviour across **Llama-3-8B**, **Mistral-7B**, and **Phi-3-mini** inside a clinical Retrieval-Augmented Generation pipeline.

**Team:** Jatin Nabhoya · Mohit Raiyani

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Phase Overview](#phase-overview)
3. [Phase 1 — Environment & Data Collection](#phase-1--environment--data-collection) ✅
4. [Phase 2 — Preprocessing & Chunking](#phase-2--preprocessing--chunking) 🔜
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
| 1 | Environment & Data Collection | ✅ Done | 66 docs in `data/metadata.csv` |
| 2 | Preprocessing & Chunking | 🔜 Next | `data/processed/` clean text chunks |
| 3 | Embedding & Vector Store | 🔜 Upcoming | FAISS index |
| 4 | RAG Pipeline & Generation | 🔜 Upcoming | Model answer JSONs |
| 5 | Evaluation & Audit | 🔜 Upcoming | RAGAS scores, hallucination report |

---

## Phase 1 — Environment & Data Collection

### 1.1 Environment Setup

The project uses a Python 3.11 virtual environment (`.clinical-rag-audit/`).

```bash
# Activate the environment (run from project root every session)
source .clinical-rag-audit/bin/activate
```

**Installed packages** (`requirements.txt`):

| Category | Libraries |
|----------|-----------|
| Core | `pandas`, `numpy`, `tqdm`, `python-dotenv` |
| RAG Framework | `langchain`, `langchain-community`, `langchain-huggingface` |
| Embeddings / Vector Store | `sentence-transformers`, `faiss-cpu` |
| Models | `transformers`, `accelerate`, `torch` |
| Document Processing | `pypdf`, `beautifulsoup4`, `requests`, `lxml`, `biopython` |
| Evaluation | `ragas`, `datasets` |
| Notebooks | `jupyter`, `ipywidgets` |

> **Note:** `biopython` and `lxml` were added during Phase 1 for PMC XML parsing. `.env` holds `ENTREZ_EMAIL` and `HF_TOKEN` — never committed.

---

### 1.2 Directory Structure Created

```
clinical-rag-audit/
├── .clinical-rag-audit/        ← Python 3.11 virtual environment (git-ignored)
├── .env                        ← Secrets: ENTREZ_EMAIL, HF_TOKEN (git-ignored)
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── configs/
│   └── cdc_urls.txt            ← List of CDC/WHO fact-sheet URLs (20 entries)
│
├── data/
│   ├── metadata.csv            ← Single source of truth — one row per document
│   ├── raw/
│   │   ├── pmc/                ← PubMed Central XMLs (50 articles)
│   │   ├── medlineplus/        ← MedlinePlus topic JSON responses (3 topics)
│   │   ├── cdc_who/            ← CDC & WHO HTML fact sheets (13 pages)
│   │   ├── medquad/            ← Reserved for Q&A XMLs (Phase 2)
│   │   └── mimic_demo/         ← Reserved for MIMIC demo notes (optional)
│   └── processed/              ← Clean text chunks land here (Phase 2)
│
├── scripts/
│   ├── download_pmc.py         ← Fetches PMC open-access XMLs via NCBI E-utilities
│   ├── download_medlineplus.py ← Pulls MedlinePlus Connect API topic summaries
│   ├── download_cdc.py         ← Scrapes CDC / WHO HTML fact-sheet pages
│   └── log_metadata.py         ← Shared utility — appends rows to metadata.csv
│
├── notebooks/
│   └── 00_setup_check.ipynb    ← Environment & import verification
│
├── src/
│   ├── ingestion/              ← Document loading & chunking (Phase 2)
│   ├── retrieval/              ← Embeddings & FAISS (Phase 3)
│   ├── generation/             ← LLM wrappers (Phase 4)
│   └── evaluation/             ← RAGAS & manual scoring (Phase 5)
│
└── results/                    ← Output tables & charts (Phase 5)
```

---

### 1.3 Data Collection Commands

All three commands below were run after activating the environment. Each script automatically appends a row to `data/metadata.csv` for every document it downloads.

---

#### Download PMC Articles (PubMed Central)

```bash
python scripts/download_pmc.py \
    --query "hypertension management" \
    --max_results 50 \
    --domain_tag cardiology
```

**What this does:**
- Searches PubMed Central for open-access articles matching `"hypertension management"`.
- Uses the **NCBI E-utilities API** (via Biopython) at a safe rate of 1 request per 400 ms.
- Downloads each article as a full **XML** file into `data/raw/pmc/`.
- Parses the article title from the XML using `lxml`.
- Logs every downloaded article to `data/metadata.csv` with `source=pmc`, `license=open-access`, `format=xml`, `domain_tag=cardiology`.
- Result: **50 PMC XML articles** collected.

---

#### Download CDC & WHO Fact Sheets

```bash
python scripts/download_cdc.py \
    --urls_file configs/cdc_urls.txt \
    --domain_tag infectious_disease
```

**What this does:**
- Reads 20 URLs from `configs/cdc_urls.txt` (CDC and WHO fact-sheet pages covering flu, COVID-19, TB, HIV, pneumonia, sepsis).
- Fetches each page as raw **HTML** using `requests` + a 1.5 s delay (polite scraping).
- Extracts the page `<title>` using BeautifulSoup and saves the HTML to `data/raw/cdc_who/`.
- Auto-detects `source=cdc` vs `source=who` from the URL hostname.
- Logs every page to `data/metadata.csv` with `license=public-domain`, `format=html`, `domain_tag=infectious_disease`.
- Result: **8 CDC pages + 5 WHO pages = 13 HTML files** collected.

---

#### Download MedlinePlus Topics

```bash
python scripts/download_medlineplus.py \
    --topics "diabetes" "asthma" "stroke"
```

**What this does:**
- Calls the **MedlinePlus Connect API** (HL7 InfoButton format) for each topic.
- Returns structured health-topic summaries as **JSON** and saves them to `data/raw/medlineplus/`.
- Each topic becomes one JSON file (e.g., `diabetes.json`).
- Logs every topic to `data/metadata.csv` with `source=medlineplus`, `license=public-domain`, `format=json`, `domain_tag=general`.
- Result: **3 MedlinePlus topic JSON files** collected.

---

### 1.4 Metadata Log

Every download is tracked in `data/metadata.csv` — the single source of truth for the corpus:

```
doc_id,source,title,url,download_date,publication_date,license,file_path,format,domain_tag
```

| Column | Description |
|--------|-------------|
| `doc_id` | Unique identifier (e.g., `pmc_13091089`, `cdc_flu`, `mlp_diabetes`) |
| `source` | Origin: `pmc` / `cdc` / `who` / `medlineplus` / `medquad` |
| `title` | Document or article title |
| `url` | Canonical URL |
| `download_date` | ISO date the file was fetched |
| `publication_date` | Original publication date (when available) |
| `license` | `open-access` / `public-domain` / `CC-BY` |
| `file_path` | Relative path to the raw file |
| `format` | `xml` / `html` / `json` / `pdf` |
| `domain_tag` | Clinical domain label for filtering |

**Current corpus (Phase 1 complete):**

| Source | Domain | Files | Format |
|--------|--------|------:|--------|
| PubMed Central | cardiology | 50 | XML |
| CDC | infectious_disease | 8 | HTML |
| WHO | infectious_disease | 5 | HTML |
| MedlinePlus | general | 3 | JSON |
| **Total** | | **66** | |

---

## Phase 2 — Preprocessing & Chunking

> **Status: 🔜 Next**

Planned work:
- Parse XML (PMC), HTML (CDC/WHO), and JSON (MedlinePlus) into plain text.
- Clean text (strip boilerplate, normalize whitespace, remove citations noise).
- Chunk into ~512-token windows with 50-token overlap.
- Save chunks to `data/processed/` as JSONL.
- Update `src/ingestion/` modules.

---

## Phase 3 — Embedding & Vector Store

> **Status: 🔜 Upcoming**

Planned work:
- Embed chunks with `sentence-transformers` (model TBD, e.g., `BioLORD-2023`).
- Build a FAISS flat-L2 index.
- Persist index to disk for reproducible retrieval.
- Update `src/retrieval/` modules.

---

## Phase 4 — RAG Pipeline & Generation

> **Status: 🔜 Upcoming**

Planned work:
- Load each LLM via HuggingFace `transformers` (Llama-3-8B, Mistral-7B, Phi-3-mini).
- Wire retriever → prompt template → generator using LangChain.
- Run on a fixed clinical QA test set.
- Save model answers to `results/`.
- Update `src/generation/` modules.

---

## Phase 5 — Evaluation & Audit

> **Status: 🔜 Upcoming**

Planned work:
- Score each answer with **RAGAS** metrics: faithfulness, answer relevancy, context precision, context recall.
- Manual review of sampled hallucinated answers.
- Cross-model comparison tables and visualizations.
- Final report in `results/`.
- Update `src/evaluation/` modules.

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
cp .env.example .env        # then fill in ENTREZ_EMAIL and HF_TOKEN

# 5. Collect data (Phase 1)
python scripts/download_pmc.py --query "hypertension management" --max_results 50 --domain_tag cardiology
python scripts/download_cdc.py --urls_file configs/cdc_urls.txt --domain_tag infectious_disease
python scripts/download_medlineplus.py --topics "diabetes" "asthma" "stroke"

# 6. Verify corpus
python -c "import pandas as pd; df=pd.read_csv('data/metadata.csv'); print(df.groupby(['source','domain_tag']).size())"
```

---

## Data Sources & Licenses

| Source | License | Access Method |
|--------|---------|---------------|
| PubMed Central (PMC) | Open Access / CC-BY | NCBI E-utilities API |
| CDC Fact Sheets | Public Domain (US Gov) | Web scraping |
| WHO Fact Sheets | CC-BY-NC-SA 3.0 IGO | Web scraping |
| MedlinePlus | Public Domain (NLM/NIH) | MedlinePlus Connect API |
| MedQuAD | CC-BY 4.0 | GitHub (Phase 2) |
| MIMIC-III Demo | PhysioNet Credentialed | Optional (Phase 2+) |

> Raw data files are **git-ignored** (`data/raw/*`). Only `metadata.csv` is committed as the corpus manifest.

---

## Corpus Snapshot

> Last updated: 2026-04-19 · Phase 1 complete

```
Total documents : 66
─────────────────────────────────
PMC        (cardiology)          50 XML
CDC        (infectious_disease)   8 HTML
WHO        (infectious_disease)   5 HTML
MedlinePlus (general)             3 JSON
─────────────────────────────────
```
