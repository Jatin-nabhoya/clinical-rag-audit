# Clinical RAG Hallucination Audit

> Comparing hallucination behaviour across **Llama-3-8B**, **Mistral-7B**, and **Phi-3-mini** inside a clinical Retrieval-Augmented Generation pipeline.

**Team:** Jatin Nabhoya · Mohit Raiyani | 

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
| 1 | Environment & Data Collection | ✅ Complete | 66 docs · `data/metadata.csv` verified |
| 2 | Preprocessing & Chunking | 🔜 Next | `data/processed/` clean text chunks |
| 3 | Embedding & Vector Store | 🔜 Upcoming | FAISS index |
| 4 | RAG Pipeline & Generation | 🔜 Upcoming | Model answer JSONs |
| 5 | Evaluation & Audit | 🔜 Upcoming | RAGAS scores, hallucination report |

---

## Phase 1 — Environment & Data Collection ✅

### 1.1 Environment Setup

The project uses a Python 3.11 virtual environment (`.clinical-rag-audit/`).


# Activate the environment — run this every session from project root
```bash
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

> `biopython` and `lxml` were added during Phase 1 for PMC XML parsing. `.env` holds `ENTREZ_EMAIL` and `HF_TOKEN` — never committed to git.

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
│   │   ├── pmc/                  ← 50 PubMed Central open-access XML articles
│   │   ├── medlineplus/          ← 3 MedlinePlus topic JSON responses
│   │   ├── cdc_who/              ← 13 CDC & WHO HTML fact-sheet pages
│   │   ├── medquad/              ← Reserved for MedQuAD Q&A XMLs (Phase 2)
│   │   └── mimic_demo/           ← MIMIC-IV demo structured tables (see note below)
│   └── processed/                ← Clean text chunks land here (Phase 2)
│       ├── bioasq_diabetes_qa.json   ← BioASQ diabetes Q&A (evaluation set, Phase 2)
│       └── medquad_diabetes_qa.json  ← MedQuAD diabetes Q&A (evaluation set, Phase 2)
│
├── scripts/
│   ├── download_pmc.py           ← Fetches PMC open-access XMLs via NCBI E-utilities
│   ├── download_medlineplus.py   ← Pulls MedlinePlus Connect API topic summaries
│   ├── download_cdc.py           ← Scrapes CDC / WHO HTML fact-sheet pages
│   ├── log_metadata.py           ← Shared utility — appends rows to metadata.csv
│   ├── verify_metadata.py        ← Validates metadata.csv & checks all files exist
│   ├── extract_bioasq.py         ← Filters BioASQ JSON for diabetes Q&A pairs
│   ├── extract_medquad.py        ← Filters MedQuAD XMLs for diabetes Q&A pairs
│   └── extract_mimic_demo.py     ← Builds structured summaries from MIMIC-IV tables
│
├── notebooks/
│   └── 00_setup_check.ipynb      ← Environment & import verification notebook
│
├── src/
│   ├── ingestion/                ← Document loading & chunking (Phase 2)
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
- Searches PubMed Central for open-access articles matching `"hypertension management"`.
- Uses the **NCBI E-utilities API** (via Biopython) — rate-limited to 1 request per 400 ms.
- Downloads each article as a full **XML** file into `data/raw/pmc/`.
- Parses the article title from the XML using `lxml`.
- Logs every article to `data/metadata.csv` with `source=pmc`, `license=open-access`, `format=xml`, `domain_tag=cardiology`.
- **Result: 50 PMC XML articles collected.**

---

#### Download CDC & WHO Fact Sheets

```bash
python scripts/download_cdc.py \
    --urls_file configs/cdc_urls.txt \
    --domain_tag infectious_disease
```

**What this does:**
- Reads `configs/cdc_urls.txt` — 20 CDC and WHO fact-sheet URLs covering flu, COVID-19, TB, HIV, pneumonia, and sepsis.
- Fetches each page as raw **HTML** using `requests` + 1.5 s polite delay.
- Extracts the page `<title>` with BeautifulSoup and saves HTML to `data/raw/cdc_who/`.
- Auto-detects `source=cdc` vs `source=who` from the URL hostname.
- Logs every page to `data/metadata.csv` with `license=public-domain`, `format=html`, `domain_tag=infectious_disease`.
- **Result: 8 CDC pages + 5 WHO pages = 13 HTML files collected.**

---

#### Download MedlinePlus Topics

```bash
python scripts/download_medlineplus.py \
    --topics "diabetes" "asthma" "stroke"
```

**What this does:**
- Calls the **MedlinePlus Connect API** (HL7 InfoButton format) for each topic.
- Returns structured health-topic summaries as **JSON** into `data/raw/medlineplus/`.
- Each topic becomes one JSON file (e.g., `diabetes.json`).
- Logs every topic to `data/metadata.csv` with `source=medlineplus`, `license=public-domain`, `format=json`, `domain_tag=general`.
- **Result: 3 MedlinePlus topic JSON files collected.**

---

#### Verify Corpus

```bash
python scripts/verify_metadata.py
```

**What this does:**
- Reads `data/metadata.csv` and prints a count breakdown by source.
- Checks every `file_path` entry actually exists on disk (`Missing files: 0`).
- Reports all unique license types in the corpus.

**Output (Phase 1 complete):**
```
source
cdc             8
medlineplus     3
pmc            50
who             5
dtype: int64
Total docs: 66
Licenses: ['open-access' 'public-domain']
Missing files: 0
```

---

### 1.4 Evaluation Data Preparation (Phase 2 Preview)

Two extraction scripts were written during Phase 1 to prepare the **evaluation QA set** for Phase 2+:

#### BioASQ — `scripts/extract_bioasq.py`
- Filters the BioASQ-13b training set for diabetes-related questions.
- Saves to `data/processed/bioasq_diabetes_qa.json`.
- Captures question, ideal answer, exact answer, snippets, and question type (factoid / yesno / list / summary).
- **Requires:** BioASQ dataset downloaded to `data/raw/bioasq/` (registration at bioasq.org).

#### MedQuAD — `scripts/extract_medquad.py`
- Walks all MedQuAD XML subfolders, filters for diabetes-related topics by keyword.
- Saves to `data/processed/medquad_diabetes_qa.json`.
- Captures question, answer, topic focus, and source collection.
- **Requires:** MedQuAD dataset cloned to `data/raw/medquad/` (GitHub, CC-BY 4.0).

---

### 1.5 MIMIC-IV Demo — Attempted, Excluded

MIMIC-IV demo was explored as an optional source for real clinical notes.

**What was found:**
- The MIMIC-IV Clinical Database Demo (100 patients) **explicitly excludes free-text clinical notes** (stated in its own README).
- The separate MIMIC-IV Note Demo module (`physionet.org/content/mimic-iv-note-demo/`) returned a **404 — page not found**.
- Without free-text notes, only structured tables (ICD codes, prescriptions, lab events) are available.

**Decision:** Excluded from corpus. The 100-patient demo with structured data only is insufficient for meaningful RAG evaluation. The script `scripts/extract_mimic_demo.py` remains for reference — it builds structured clinical summaries from ICD + prescription + lab tables and can be used if the full credentialed MIMIC-IV is obtained later.

> To use MIMIC-IV in future: complete CITI training → request access at physionet.org → download full dataset → run `extract_mimic_demo.py`.

---

### 1.6 Metadata Schema

Every downloaded document is logged in `data/metadata.csv`:

| Column | Description |
|--------|-------------|
| `doc_id` | Unique identifier (e.g., `pmc_13091089`, `cdc_flu`, `mlp_diabetes`) |
| `source` | Origin: `pmc` / `cdc` / `who` / `medlineplus` / `mimic_demo` |
| `title` | Document or article title |
| `url` | Canonical URL |
| `download_date` | ISO date the file was fetched |
| `publication_date` | Original publication date (when available) |
| `license` | `open-access` / `public-domain` / `CC-BY` / `PhysioNet-CDHL-1.5.0` |
| `file_path` | Relative path to the raw file from project root |
| `format` | `xml` / `html` / `json` / `txt` |
| `domain_tag` | Clinical domain label for filtering (cardiology / infectious_disease / general / diabetes) |

---

## Phase 2 — Preprocessing & Chunking

> **Status: 🔜 Next**

Planned work:
- Parse XML (PMC), HTML (CDC/WHO), and JSON (MedlinePlus) into plain text.
- Clean text: strip boilerplate, normalize whitespace, remove citation noise.
- Chunk into ~512-token windows with 50-token overlap.
- Save chunks to `data/processed/` as JSONL.
- Run `extract_bioasq.py` and `extract_medquad.py` to finalize the evaluation QA set.
- Update `src/ingestion/` modules.

---

## Phase 3 — Embedding & Vector Store

> **Status: 🔜 Upcoming**

Planned work:
- Embed chunks with `sentence-transformers` (candidate: `BioLORD-2023`).
- Build a FAISS flat-L2 index.
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

# 4. Set secrets (.env file)
echo "ENTREZ_EMAIL=your@email.com" >> .env
echo "HF_TOKEN=hf_..." >> .env

# 5. Collect data (Phase 1)
python scripts/download_pmc.py --query "hypertension management" --max_results 50 --domain_tag cardiology
python scripts/download_cdc.py --urls_file configs/cdc_urls.txt --domain_tag infectious_disease
python scripts/download_medlineplus.py --topics "diabetes" "asthma" "stroke"

# 6. Verify corpus
python scripts/verify_metadata.py
```

---

## Data Sources & Licenses

| Source | License | Access Method | Status |
|--------|---------|---------------|--------|
| PubMed Central (PMC) | Open Access / CC-BY | NCBI E-utilities API | ✅ 50 docs |
| CDC Fact Sheets | Public Domain (US Gov) | Web scraping | ✅ 8 docs |
| WHO Fact Sheets | CC-BY-NC-SA 3.0 IGO | Web scraping | ✅ 5 docs |
| MedlinePlus | Public Domain (NLM/NIH) | MedlinePlus Connect API | ✅ 3 docs |
| MedQuAD | CC-BY 4.0 | GitHub repo | 🔜 Phase 2 eval set |
| BioASQ | BioASQ License | bioasq.org registration | 🔜 Phase 2 eval set |
| MIMIC-IV Demo | PhysioNet CDHL-1.5.0 | PhysioNet download | ⚠️ Excluded — no free-text notes in demo |

> Raw data files are **git-ignored** (`data/raw/*`). Only `metadata.csv` is committed as the corpus manifest.

---

## Corpus Snapshot

> Last updated: 2026-04-20 · **Phase 1 complete**

```
Total documents : 66
Verified missing: 0
─────────────────────────────────────────────
PMC          (cardiology)           50  XML
CDC          (infectious_disease)    8  HTML
WHO          (infectious_disease)    5  HTML
MedlinePlus  (general)               3  JSON
─────────────────────────────────────────────
Licenses : open-access, public-domain
```