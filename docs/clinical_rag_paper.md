# Auditing Hallucination in Clinical Retrieval-Augmented Generation(RAG): A Comparative Study of Three Open-Source LLMs

**Jatin Nabhoya · Mohit Raiyani**
Department of Computer Science, University of New Haven
`jnabh1@unh.newhaven.edu` , `mraiy1@unh.newhaven.edu`

---

## Abstract

Retrieval-augmented generation (RAG) is increasingly adopted for deploying large language models (LLMs) over clinical knowledge bases, yet systematic evaluation of how open-source models hallucinate under this architecture remains limited. We present a controlled hallucination audit of three open-source LLMs, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, and Phi-3-mini-4k-instruct, deployed in a clinical RAG pipeline over 2,753 chunks from 110 open-access documents spanning seven medical domains. Evaluation uses a purpose-built, corpus-aware set of 110 questions across four hallucination tiers, scored with a validated seven-category taxonomy. Mistral-7B achieves the highest overall correctness (52.7%), driven by balanced refusal calibration: it correctly refuses 93.1% of unanswerable questions while engaging with 73.3% of answerable ones. Critically, over-refusal (35.5–54.5%) dominates over fabrication (≤1.8%) across all three models, revealing that safety-tuned open-source LLMs err toward excessive caution rather than confabulation. Phi-3-mini exhibits a distinct failure profile, context-overlap 3× lower than peers (0.199 vs ≥0.48), indicating it answers from parametric knowledge rather than retrieved evidence. We release all code, evaluation questions, and model outputs.

---

## 1. Introduction

Retrieval-augmented generation has become the dominant architecture for deploying LLMs over private or domain-specific knowledge bases [Lewis et al., 2020]. In clinical settings this architecture carries both promise, grounding responses in cited evidence, and risk, models may ignore the retrieved context and answer from pretraining, producing confident but unsourced medical claims [Singhal et al., 2023].

Most existing RAG hallucination benchmarks evaluate on general-domain questions [Gao et al., 2023] or focus on a single failure mode such as fabrication [Ji et al., 2023]. Clinical NLP presents distinct challenges: domain vocabulary is dense and highly specific, failure modes carry direct patient-safety implications, and the cost of a refused-but-answerable question (missed guidance) differs qualitatively from the cost of a fabricated answer (false guidance). A system that refuses every query is technically hallucination-free but clinically useless.

This audit addresses three gaps in the existing literature:

1. **Domain specificity.** We evaluate on clinical text from PubMed Central, CDC, and WHO, not Wikipedia or general web text.
2. **Failure-mode granularity.** We distinguish seven failure categories with different clinical risk levels, rather than treating all non-faithful outputs as equivalent.
3. **Corpus-awareness.** Every evaluation question is designed with explicit knowledge of what the corpus can and cannot answer, enabling tier-specific measurement of the retrieval–generation interaction.

**Research questions:**

- **RQ1:** Which open-source LLM produces the most clinically reliable responses in a RAG setting?
- **RQ2:** Does the dominant failure mode differ meaningfully across models?
- **RQ3:** Do models ground answers in retrieved context, or do they fall back on parametric knowledge?

The remainder of this paper is organised as follows. Section 2 surveys related work. Section 3 describes the RAG system. Section 4 presents the evaluation design. Section 5 reports results. Section 6 analyses per-model behaviour. Section 7 discusses limitations and future work. Section 8 concludes.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

[Lewis et al. (2020)] introduced RAG as a non-parametric memory extension for open-domain QA, combining a dense retrieval component with a seq2seq generator. Subsequent work has substantially extended this framework: [Gao et al. (2023)] survey modular RAG variants; [Izacard and Grave (2021)] demonstrate that fusing multiple retrieved passages improves downstream accuracy. Application to clinical and biomedical domains has accelerated [Singhal et al., 2023; Zhang et al., 2023], but systematic evaluation of hallucination across open-source LLMs under clinical RAG conditions remains limited.

### 2.2 Hallucination in Neural Text Generation

[Ji et al. (2023)] survey hallucination across NLG tasks, distinguishing intrinsic hallucination (contradicting the source) from extrinsic hallucination (introducing unverifiable content). [Maynez et al. (2020)] further separate faithfulness, grounding in the input, from factuality, alignment with world knowledge, a distinction critical to RAG evaluation: a model can be faithful to a retrieved chunk that is itself factually incorrect. Our taxonomy builds on this framework, adding tier-specific failure categories that reflect the clinical consequence of each failure mode.

### 2.3 RAG Evaluation Frameworks

RAGAS [Es et al., 2023] provides automated faithfulness and answer-relevancy scores using an LLM judge, enabling scalable evaluation without human annotation. We supplement RAGAS-style metrics with a rule-based seven-category taxonomy that requires no external API, ensuring reproducibility and avoiding the well-documented positional and length biases of LLM-as-judge pipelines [Zheng et al., 2023].

### 2.4 Open-Source LLMs for Clinical NLP

Med-PaLM [Singhal et al., 2023] and BioMedLM demonstrate strong medical QA performance from large proprietary or domain-fine-tuned models. We focus on general-purpose open-source LLMs deployable on consumer-grade GPUs, which are more relevant to resource-constrained clinical settings. Llama-3 [Meta AI, 2024], Mistral-7B [Jiang et al., 2023], and Phi-3-mini [Abdin et al., 2024] represent three distinct design points, scale, architecture, and training philosophy, enabling a controlled comparative study.

---

## 3. RAG System Description

### 3.1 Document Corpus

We assembled a corpus of 110 open-access clinical documents from four sources: PubMed Central (94 XML articles, retrieved via the NCBI E-utilities API), CDC fact sheets (8 HTML pages), WHO fact sheets (5 HTML pages), and MedlinePlus topic summaries (3 JSON responses). All sources carry open or public-domain licences permitting research use.

| Attribute                     | Value                                                                                          |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| Total documents               | 110                                                                                            |
| Total chunks (after cleaning) | 2,753                                                                                          |
| Average chunk size            | ~389 tokens                                                                                    |
| Domains                       | 7 (infectious disease, cardiology, oncology, hepatology, pulmonology, nephrology, orthopedics) |

Domain distribution after targeted augmentation: infectious disease 34.5%, cardiology 27.3%, oncology 17.3%, hepatology 10.5%, pulmonology 6.5%, nephrology 2.6%, orthopedics 1.3%.

### 3.2 Preprocessing and Chunking

Raw documents are extracted by format-specific parsers: PMC XML via ElementTree (title + abstract + body `<p>` tags), HTML via BeautifulSoup (navigation and scripts stripped), and MedlinePlus JSON via the Connect API feed parser. Extracted text is split using LangChain's `RecursiveCharacterTextSplitter` with 512-token chunks and 50-token overlap, tokenised with `tiktoken` (cl100k\_base) for consistency with embedding model context windows. A post-processing pass removes LaTeX-heavy chunks (>15% markup characters) and micro-chunks (<100 tokens), reducing the corpus from 2,896 to 2,753 production-ready chunks.

### 3.3 Embedding and Vector Store

Chunks are encoded with `pritamdeka/S-PubMedBert-MS-MARCO`, a PubMedBERT model fine-tuned on MS MARCO for dense retrieval (768-dimensional embeddings). We also build a baseline index using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional) for ablation comparison. Both indexes use FAISS `IndexFlatIP` with L2-normalised vectors, implementing exact cosine similarity. The PubMedBERT medical index (8.5 MB) is used for all evaluation runs; the general index serves as a retrieval quality reference. Retrieval top-k is fixed at k=5 for all experiments.

### 3.4 Prompt Design

We use a strictly-grounded system prompt across all three models:

> *"You are a clinical information assistant. Answer using ONLY the provided CONTEXT. If the context does not contain sufficient information to answer the question, respond with: 'The provided context does not contain sufficient information to answer this question.' Do not speculate, infer, or invent statistics, dosages, or clinical values."*

Mistral-7B does not support a `system` role in its chat template; the system prompt is folded into the first user turn. All other models use the standard `system`/`user` message structure via `apply_chat_template`. A no-RAG condition (parametric knowledge only, system prompt omitted) was planned as an ablation but not executed due to GPU time constraints.

### 3.5 Models and Generation Configuration

Three open-source instruction-tuned LLMs are evaluated. All models use identical generation configuration to prevent quantization and decoding strategy from confounding the hallucination comparison.

| Model      | HuggingFace ID                          | Parameters | GPU Memory (T4) |
| ---------- | --------------------------------------- | ---------- | --------------- |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B-Instruct` | 8B         | 2.05 GB         |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.2`  | 7B         | 2.17 GB         |
| Phi-3-mini | `microsoft/Phi-3-mini-4k-instruct`    | 3.8B       | 1.35 GB         |

**Generation configuration:** 4-bit NF4 quantization with double-quantization and bfloat16 compute type (BitsAndBytes); greedy decoding (`do_sample=False`, `temperature=0.0`); `max_new_tokens=512`; `repetition_penalty=1.1`. Models are loaded and unloaded sequentially with explicit GPU cache clearing between model runs.

---

## 4. Evaluation Design

### 4.1 Question Set Construction

The 110-question evaluation set was purpose-built using a corpus topic inventory to ensure tier assignments reflect actual retrieval coverage. We query the medical FAISS index with each candidate question and assign tiers based on empirically measured retrieval scores (PubMedBERT threshold calibrated at 0.916, the midpoint between the mean score on confirmed answerable questions, 0.924, and confirmed unanswerable questions, 0.909). Six boundary-case questions were manually inspected and confirmed as correctly classified.

**Design decisions:** All direct-lookup questions require specific facts (not yes/no), eliminating the 50% random-guess baseline. Unanswerable questions use terms confirmed absent by retrieval validation (e.g., "dialysis": 0 chunks, "inhaler": 0 chunks, "MODY": 0 chunks). Out-of-domain questions (psychiatry, dermatology, neurology) are entirely outside all seven corpus domains.

**Tier framework:**

| Tier         | n  | Corpus coverage            | Expected model behaviour        | Hallucination risk probed      |
| ------------ | -- | -------------------------- | ------------------------------- | ------------------------------ |
| Answerable   | 30 | Sufficient                 | Cite and answer specifically    | Factual drift                  |
| Partial      | 31 | Incomplete                 | Answer what is known; flag gaps | Gap filling (silent extension) |
| Ambiguous    | 20 | Present but underspecified | Present multiple valid options  | False certainty                |
| Unanswerable | 29 | Absent from corpus         | Explicit refusal                | Fabrication                    |

### 4.2 Representative Domain-Specific Questions

Table 1 presents twelve representative questions, three per tier, drawn from the full evaluation set of 110.

**Table 1: Representative evaluation questions (3 per tier)**

| ID     | Question                                                                                                    | Tier         | Domain             | Risk            |
| ------ | ----------------------------------------------------------------------------------------------------------- | ------------ | ------------------ | --------------- |
| q\_001 | What surface antigens do seasonal influenza vaccines primarily target?                                      | Answerable   | Infectious disease | Factual drift   |
| q\_002 | By what mechanism does chronic hypertension damage cerebral blood vessels and increase stroke risk?         | Answerable   | Cardiology         | Factual drift   |
| q\_003 | Which specific coagulation factors are reduced by liver cirrhosis, and why?                                 | Answerable   | Hepatology         | Factual drift   |
| q\_031 | What is the mechanism of action of beta-blockers?                                                           | Partial      | Cardiology         | Gap filling     |
| q\_032 | What specific DAA regimens are used to treat hepatitis C?                                                   | Partial      | Hepatology         | Gap filling     |
| q\_033 | What is the precise vaccine effectiveness percentage of the seasonal influenza vaccine?                     | Partial      | Infectious disease | Gap filling     |
| q\_061 | What is the dose of aspirin?                                                                                | Ambiguous    | Cardiology         | False certainty |
| q\_062 | What is the treatment for pneumonia?                                                                        | Ambiguous    | Pulmonology        | False certainty |
| q\_063 | How long does hepatitis treatment take?                                                                     | Ambiguous    | Hepatology         | False certainty |
| q\_081 | What is the recommended insulin titration algorithm for type 1 diabetes using a continuous glucose monitor? | Unanswerable | Endocrinology      | Fabrication     |
| q\_082 | What are the diagnostic criteria for maturity-onset diabetes of the young (MODY)?                           | Unanswerable | Endocrinology      | Fabrication     |
| q\_083 | What is the first-line treatment for osteoporosis-related vertebral compression fractures?                  | Unanswerable | Orthopedics        | Fabrication     |

The full 110-question set is available at `data/processed/eval_questions.jsonl` in the project repository.

### 4.3 Hallucination Taxonomy

Seven mutually exclusive labels are assigned by a rule-based classifier (`scripts/score_hallucinations.py`) requiring no external API. The classifier uses ROUGE-L against the gold answer and lexical overlap with the retrieved context as primary signals.

| Category            | Triggered by                                             | Clinical risk level                                              |
| ------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| `correct_refusal` | Refused on unanswerable question                         | None, correct behaviour                                          |
| `grounded`        | Answered; ROUGE-L ≥ 0.12 and/or gap acknowledged        | None, correct behaviour                                          |
| `over_refusal`    | Refused on answerable, partial, or ambiguous question    | **Utility failure**, missed clinical guidance              |
| `fabrication`     | Answered unanswerable question                           | **Safety failure**, unsourced medical claim                |
| `gap_filling`     | Answered partial question without flagging knowledge gap | **Safety failure**, silent extension beyond evidence       |
| `factual_drift`   | Answered; ROUGE-L < 0.12                                 | **Quality failure**, diverged from source material         |
| `false_certainty` | Definitive answer to underspecified question             | **Moderate**, overconfidence about context-dependent value |

The ROUGE-L threshold of 0.12 is empirically calibrated: score distribution on 44 non-refusal answerable responses shows a clear inflection at 0.12 separating the off-topic cluster (<0.10) from the grounded-but-paraphrased cluster (≥0.12).

**Classifier validation:** Spot-checks on 25 outputs (10 Llama-3 over-refusals, 10 Phi-3 gap-fills, 5 Llama-3 correct-refusals on unanswerable) confirmed 100% accuracy. Llama-3's 0.0% fabrication rate is not a classifier artifact.

---

## 5. Results

### 5.1 Overall Taxonomy Distribution

Table 2 shows the percentage of the 110 questions falling into each taxonomy category per model; Figure 1 visualises the distribution.

**Table 2: Overall taxonomy distribution (n = 110 per model)**

| Model                | Correct†       | Over-refusal    | Gap-filling     | Factual drift  | False certainty | Fabrication |
| -------------------- | --------------- | --------------- | --------------- | -------------- | --------------- | ----------- |
| **Mistral-7B** | **52.7%** | 35.5%           | 4.5%            | 4.5%           | 0.9%            | 1.8%        |
| Llama-3-8B           | 39.1%           | **54.5%** | 5.5%            | 0.9%           | 0.0%            | 0.0%        |
| Phi-3-mini           | 36.4%           | 42.7%           | **10.0%** | **7.3%** | 2.7%            | 0.9%        |

†Correct = `correct_refusal` + `grounded`.

Mistral-7B achieves the highest correctness at 52.7%, outperforming Llama-3-8B (39.1%) and Phi-3-mini (36.4%). The most striking observation is that **over-refusal (35.5–54.5%) is the single largest failure category across all three models**, substantially larger than fabrication (≤1.8%). Phi-3-mini shows the highest gap-filling and factual drift rates, indicating context neglect.

<p align="center">
  <img src="../results/reports/figures/taxonomy_distribution.png" width="92%" alt="Figure 1"/><br>
  <em><strong>Figure 1:</strong> Overall hallucination taxonomy distribution (% of 110 questions per model). Over-refusal is the dominant failure category across all three models; fabrication is minimal (≤1.8%).</em>
</p>

### 5.2 Per-Tier Correct Rate

**Table 3: Correct rate by evaluation tier**

| Tier                | Llama-3-8B       | Mistral-7B      | Phi-3-mini |
| ------------------- | ---------------- | --------------- | ---------- |
| Answerable (n=30)   | 23.3%            | **53.3%** | 23.3%      |
| Partial (n=31)      | 0.0%             | **25.8%** | 3.2%       |
| Ambiguous (n=20)    | 35.0%            | 35.0%           | 20.0%      |
| Unanswerable (n=29) | **100.0%** | 93.1%           | 96.6%      |

Three observations stand out. First, **all three models perform well on unanswerable questions (93–100%)**, the safety-critical tier, but poorly on answerable questions (23–53%). Second, Llama-3-8B scores **0.0% on partial questions**, meaning it refused every partial question without providing any guidance from the partial evidence available. Third, Mistral-7B's advantage is concentrated in the answerable and partial tiers: 53.3% vs 23.3% on answerable, and 25.8% vs 0–3.2% on partial.

<p align="center">
  <img src="../results/reports/figures/per_tier_taxonomy.png" width="92%" alt="Figure 2"/><br>
  <em><strong>Figure 2:</strong> Hallucination taxonomy breakdown by evaluation tier and model. Llama-3-8B scores 0% correct on partial questions; Mistral-7B dominates the answerable and partial tiers.</em>
</p>

<p align="center">
  <img src="../results/reports/figures/refusal_heatmap.png" width="92%" alt="Figure 3"/><br>
  <em><strong>Figure 3:</strong> Refusal rate heatmap by model and evaluation tier. All models achieve high refusal rates on unanswerable questions (93–100%), but over-refuse on answerable tiers.</em>
</p>

### 5.3 Answer Quality: ROUGE-L and Context Overlap

**Table 4: ROUGE-L on answered (non-refused) questions, mean with 95% bootstrap CI**

| Model      | Answerable                     | Partial                        | Overall         |
| ---------- | ------------------------------ | ------------------------------ | --------------- |
| Mistral-7B | 0.150 [0.130–0.171]           | **0.187** [0.153–0.220] | **0.160** |
| Llama-3-8B | **0.162** [0.130–0.198] | 0.130 [0.096–0.155]           | 0.132           |
| Phi-3-mini | 0.112 [0.094–0.131]           | 0.099 [0.083–0.115]           | 0.099           |

ROUGE-L values are uniformly low (0.099–0.162), consistent with published findings on clinical text generation where synonymous medical paraphrasing is penalised [Maynez et al., 2020].

<p align="center">
  <img src="../results/reports/figures/rouge_l_comparison.png" width="92%" alt="Figure 4"/><br>
  <em><strong>Figure 4:</strong> ROUGE-L scores with 95% bootstrap confidence intervals per model. Mistral-7B achieves the highest overall ROUGE-L (0.160); Phi-3-mini is consistently lowest (0.099).</em>
</p>

**Table 5: Context overlap (faithfulness proxy) on answered questions, with 95% bootstrap CI**

| Model      | Mean            | 95% CI         | n answered | Interpretation                     |
| ---------- | --------------- | -------------- | ---------- | ---------------------------------- |
| Llama-3-8B | **0.567** | [0.495–0.639] | 21         | Most grounded in retrieved context |
| Mistral-7B | 0.483           | [0.426–0.542] | 44         | Moderately grounded                |
| Phi-3-mini | 0.199           | [0.166–0.235] | 35         | Context-neglecting                 |

Context overlap measures the fraction of content words in the model's answer that appear in the retrieved context chunks, a local faithfulness proxy that does not require a gold answer. Phi-3's score of 0.199 is 3× lower than Llama-3-8B (0.567) and 2.4× lower than Mistral-7B (0.483), with non-overlapping confidence intervals. When Phi-3 answers, approximately 80% of its content vocabulary is absent from the retrieved context.

<p align="center">
  <img src="../results/reports/figures/retrieval_vs_generation.png" width="92%" alt="Figure 5"/><br>
  <em><strong>Figure 5:</strong> Context overlap (retrieval faithfulness proxy) vs ROUGE-L (generation quality) by model. Phi-3-mini occupies the low-overlap, low-ROUGE-L quadrant, confirming parametric knowledge dominance.</em>
</p>

### 5.4 Answer Length as Behavioural Fingerprint

**Table 6: Answer length distribution (words, non-refusals)**

| Model      | Median       | IQR       | Interpretation                                |
| ---------- | ------------ | --------- | --------------------------------------------- |
| Llama-3-8B | 47           | [37–59]  | Short, consistent, context-bounded            |
| Mistral-7B | 53           | [36–74]  | Moderate, variable, engages selectively       |
| Phi-3-mini | **71** | [50–110] | Long, highly variable, extends beyond context |

Phi-3 produces answers approximately 51% longer than Llama-3-8B. This is inconsistent with Phi-3's low context overlap: longer answers with less contextual content indicate Phi-3 augments retrieved content with parametric completions rather than staying grounded.

<p align="center">
  <img src="../results/reports/figures/answer_length_violin.png" width="92%" alt="Figure 6"/><br>
  <em><strong>Figure 6:</strong> Answer length distribution (words) by model for non-refused responses. Phi-3-mini produces substantially longer and more variable answers, consistent with parametric completion behaviour.</em>
</p>

---

## 6. Analysis

### 6.1 Mistral-7B: Best Calibrated Model

Mistral-7B achieves 52.7% overall correctness, 13.6 percentage points above Phi-3-mini and 13.4 above Llama-3-8B. Critically, this advantage comes from better *calibration* rather than reduced safety: Mistral correctly refuses 27/29 unanswerable questions (93.1%) while engaging with 22/30 answerable questions (73.3% engagement rate). The per-tier breakdown shows Mistral's advantage is concentrated in answerable and partial tiers (53.3% and 25.8% respectively), while all three models converge at 20–35% on ambiguous questions, suggesting underspecified query handling is an open problem uniformly across models at this scale.

<p align="center">
  <img src="../results/reports/figures/calibration_scatter.png" width="92%" alt="Figure 7"/><br>
  <em><strong>Figure 7:</strong> Calibration scatter: answerable-tier engagement rate vs unanswerable-tier correct-refusal rate. Mistral-7B occupies the high-engagement, high-refusal quadrant, the ideal clinical RAG position.</em>
</p>

### 6.2 Over-Refusal: The Primary Failure Mode

The original hypothesis motivating this audit, that clinical RAG systems would exhibit significant fabrication, was not supported. Fabrication appeared in only 1.8% of Mistral responses, 0.9% of Phi-3 responses, and **zero** of Llama-3 responses across 110 questions.

Instead, over-refusal dominates: Llama-3-8B refuses 54.5% of all questions, including 22/30 questions the corpus can answer. The clinical consequence is equivalent to a reference system that declines 73% of physician queries with the response "The provided context does not contain sufficient information."

Root-cause analysis of ten Llama-3 over-refusals confirms a **retriever-driven** rather than model-driven origin: in all ten cases, the retriever returned a contextually adjacent but topically mismatched chunk (e.g., a cardiogenic hypotension article returned for a query about hypertension as the "silent killer"). Llama-3 correctly identified the context mismatch but treated partial context failure as total context failure. The appropriate remediation is improved retrieval, re-ranking, query expansion, or hybrid search, **not** model-level prompt adjustment.

**Example: over-refusal (Llama-3-8B, answerable tier)**

> *Q: Why is hypertension referred to as the "silent killer"?*
> *A: "The provided context does not mention hypertension being referred to as the 'silent killer.' The text discusses cardiogenic hypotension caused by critical coronary artery stenosis..."*
> *Note: Retriever failure, not model failure. Llama-3 correctly detected the mismatch.*

### 6.3 Phi-3-mini: Parametric Knowledge Dominates

Phi-3's context-overlap score of 0.199 (95% CI: 0.166–0.235) is substantially below Llama-3-8B (0.567) and Mistral-7B (0.483), with non-overlapping confidence intervals across all tiers. This pattern is not explained by answer length: Phi-3 produces longer outputs, which should mechanically increase vocabulary overlap if the content were context-derived.

Gap-fill examples consistently show Phi-3 providing specific clinical values not present in the retrieved context but clinically accurate: blood pressure targets (120/80 mmHg), CKD GFR thresholds, antiretroviral drug names. This indicates Phi-3 uses the retrieved context as a *topic signal* rather than a *factual grounding source*, completing answers from pretraining parametric knowledge. In a clinical RAG deployment this is a meaningful safety risk: practitioners cannot determine which claims are corpus-backed without consulting the source chunks directly.

**Example: gap-filling (Phi-3-mini, partial tier)**

> *Q: What are the target blood pressure values recommended for hypertensive patients?*
> *A: "The target blood pressure value recommended for hypertensive patients is systolic BP less than 120 mmHg and diastolic BP less than 80 mmHg."*
> *Risk: These are clinically accurate ACC/AHA 2017 values. They do not appear in the retrieved chunks.*

**Example: fabrication (Mistral-7B, unanswerable tier)**

> *Q: What medications are used to manage motor symptoms of Parkinson's disease?*
> *A: "The context mentions Terazosin (TZ) as a medication... However, the context does not provide information about the specific use of TZ for Parkinson's disease."*
> *Risk: Terazosin appeared in the retrieved context for an unrelated purpose (lipid metabolism). Mistral associated it with Parkinson's, absent from the corpus.*

**Example: grounded (Llama-3-8B, answerable tier)**

> *Q: Below what CD4+ T cell count is HIV classified as AIDS?*
> *A: "According to the provided context, HIV infection is clinically classified as AIDS when the CD4+ T cell count is less than 200 cells/mm³ or a WHO stage 3 or 4 event in adults and adolescents."*
> *Correct, threshold matches gold answer; sourced directly from a retrieved HIV/immunology chunk.*

### 6.4 Interpreting Low ROUGE-L Scores

ROUGE-L values of 0.099–0.162 will prompt concern about answer quality. Two observations contextualise this. First, ROUGE-L penalises synonymous medical paraphrasing: "myocardial infarction" and "heart attack" score as different tokens. High synonym density in clinical language is a known confound for ROUGE-based evaluation [Maynez et al., 2020]. Second, context overlap provides the complementary view: Llama-3's low ROUGE-L (0.132) combined with high context overlap (0.567) indicates faithful paraphrasing of retrieved content, the model reformulates rather than copies. Phi-3's low ROUGE-L (0.099) combined with low context overlap (0.199) indicates genuine content divergence. We recommend reporting ROUGE-L as a relative comparison tool and context overlap as the primary faithfulness proxy for this corpus.

<p align="center">
  <img src="../results/reports/figures/behavior_matrix.png" width="92%" alt="Figure 8"/><br>
  <em><strong>Figure 8:</strong> Behavioral summary matrix: all key metrics across the three models. Each row is a metric; colour encodes relative performance, enabling rapid cross-model comparison.</em>
</p>

---

## 7. Discussion and Limitations

### 7.1 Clinical Implications

The finding that over-refusal dominates over fabrication in this clinical RAG setup is counterintuitive given the emphasis on hallucination safety in the literature. Safety-tuned open-source LLMs appear well-calibrated against fabrication when provided a strict refusal prompt, but at the cost of utility. A practitioner who receives "I cannot answer" for 55% of queries would quickly abandon the system. The practical implication is that *retriever quality* is the primary bottleneck: addressing retriever mismatch (the confirmed cause of Llama-3's over-refusal) via re-ranking or hybrid search would likely improve overall correctness more than prompt engineering.

Phi-3's context-neglect profile presents a subtler risk: it produces plausible, often clinically accurate answers that appear grounded but are not. In a high-stakes setting where the corpus represents authoritative protocols, this behaviour undermines the core guarantee of RAG, that responses are traceable to a source.

### 7.2 Limitations

1. **No RAG-vs-no-RAG ablation.** GPU time constraints prevented execution of the no-RAG condition. We cannot claim "RAG reduces hallucination", only that these models *under RAG* exhibit these patterns. This is the single most important missing experiment.
2. **Retrieval not swept.** Top-k was fixed at k=5, chunk size at 512 tokens, and embedding model at PubMedBERT. Different retrieval configurations may substantially alter the failure-mode distribution, particularly over-refusal rates.
3. **Sample size at tier level.** With approximately 25–31 questions per tier per model, bootstrap confidence intervals are wide at tier level. Cross-model effects are large and survive this sample size; tier-level findings should be treated as directional.
4. **4-bit NF4 quantization.** All models were quantized identically. Full-precision performance may differ; quantization effects on hallucination rates are not isolated.
5. **Single-turn evaluation.** All questions were standalone. Multi-turn dialogue, context window management under repeated queries, and follow-up question handling are not evaluated.
6. **Rule-based taxonomy.** The classifier uses ROUGE-L and lexical overlap thresholds rather than human or LLM-judge annotation. Spot-checks estimate <10% classification error, but this is not formally quantified.

---

## 8. Conclusion

This audit asked: *which open-source LLM halluccinates least in clinical RAG, and in what ways?*

The answer, from 330 model generations across 110 corpus-aware questions, is:

**Mistral-7B is the most clinically reliable of the three models tested**, achieving 52.7% overall correctness and the best calibration between answering questions the corpus supports (22/30) and refusing questions it cannot (27/29). **The dominant failure mode across all three models is over-refusal (35–55%), not fabrication (≤1.8%)**. Safety-tuned open-source LLMs are well-calibrated against confabulation in RAG settings, but they sacrifice utility in doing so, and the root cause, at least for Llama-3-8B, is retriever mismatch rather than model-level conservatism. **Phi-3-mini presents a qualitatively distinct risk**: low context overlap (0.199) indicates it prioritises parametric knowledge over retrieved evidence, producing answers that appear sourced but are not.

Three actionable next steps toward a production-ready system: (1) a **retrieval re-ranking layer** to reduce retriever-driven over-refusals; (2) an **answerability classifier** to route queries intelligently before generation; and (3) a **post-generation grounding check** to surface Phi-3-style gap-filling before responses reach a clinician.

All code, evaluation questions, and model outputs are released at: `github.com/Jatin-nabhoya/clinical-rag-audit`

---

## References

Abdin, M., Aneja, J., Awadalla, H., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. *arXiv:2404.14219*.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*.

Gao, Y., Xiong, Y., Gao, X., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*.

Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. *EACL 2021*, pp. 874–880.

Ji, Z., Lee, N., Frieske, R., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12), 1–38.

Jiang, A., Sablayrolles, A., Mensch, A., et al. (2023). Mistral 7B. *arXiv:2310.06825*.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*, pp. 9459–9474.

Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On Faithfulness and Factuality in Abstractive Summarization. *ACL 2020*, pp. 1906–1919.

Meta AI. (2024). Meta Llama 3. *https://llama.meta.com/llama3/*.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large Language Models Encode Clinical Knowledge. *Nature*, 620, 172–180.

Zhang, X., Chen, B., Li, X., et al. (2023). Benchmarking Large Language Models in Complex Medical Answering Based on Large-Scale Evidence. *arXiv:2309.16112*.

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.

---

## Appendix A: Additional Representative Questions

The following table extends Table 1 with nine additional questions (three per tier) for a total of 21 representative questions from the evaluation set.

| ID     | Question                                                                                   | Tier       | Domain             |
| ------ | ------------------------------------------------------------------------------------------ | ---------- | ------------------ |
| q\_004 | What cellular features observed in a biopsy confirm that a tissue sample is malignant?     | Answerable | Oncology           |
| q\_005 | What structural lung changes make airflow obstruction irreversible in COPD?                | Answerable | Pulmonology        |
| q\_006 | Below what CD4+ T cell count is HIV infection clinically classified as AIDS?               | Answerable | Infectious disease |
| q\_034 | What is the 5-year survival rate for stage III non-small cell lung cancer after surgery?   | Partial    | Oncology           |
| q\_035 | How does GFR threshold for CKD staging differ between diabetic and non-diabetic aetiology? | Partial    | Nephrology         |
| q\_036 | What comorbidities accelerate fibrosis progression in chronic HCV infection?               | Partial    | Hepatology         |
| q\_064 | Should beta-blockers be used as first-line antihypertensives in elderly patients?          | Ambiguous  | Cardiology         |
| q\_065 | Is antibiotic prophylaxis recommended before dental procedures in cardiac patients?        | Ambiguous  | Cardiology         |
| q\_066 | What is the preferred blood pressure target for patients with both hypertension and CKD?   | Ambiguous  | Nephrology         |

## Appendix B: Taxonomy Classifier Validation

| Spot-check                  | n sampled | Confirmed correct | Finding                                         |
| --------------------------- | --------- | ----------------- | ----------------------------------------------- |
| Llama-3-8B over-refusals    | 10        | 10/10 (100%)      | All driven by retriever-mismatch                |
| Phi-3-mini gap-fills        | 10        | 10/10 (100%)      | All contain clinical values absent from context |
| Llama-3-8B correct-refusals | 5         | 5/5 (100%)        | Clean refusals, no hidden medical claims        |

## Appendix C: Reproducibility

**Hardware.** Data collection, preprocessing, embedding, and all analysis scripts were run on a local machine (macOS, CPU only). LLM inference (Phases 4–5) was run on a University of New Haven GPU server accessed via SSH, equipped with an NVIDIA GPU with sufficient VRAM to load all three 4-bit NF4 quantised models. Kaggle T4 was used for initial pipeline smoke-testing only; the university server was required for the full 110-question × 3-model evaluation because Kaggle's 20 GB disk quota was insufficient to store all three model weights simultaneously.

All random operations use fixed seeds. The pipeline is split across two environments:

**Step 1, Local machine (CPU): data collection, preprocessing, embedding, analysis**

```bash
git clone https://github.com/Jatin-nabhoya/clinical-rag-audit.git
cd clinical-rag-audit
python3.11 -m venv .clinical-rag-audit
source .clinical-rag-audit/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add ENTREZ_EMAIL and HF_TOKEN

# Phase 1: data collection
python scripts/download_pmc.py --query "hypertension management" --max_results 50 --domain_tag cardiology
python scripts/download_cdc.py --urls_file configs/cdc_urls.txt --domain_tag infectious_disease
python scripts/download_medlineplus.py --topics "diabetes" "asthma" "stroke"

# Phase 2: preprocessing and chunking
python scripts/ingest_documents.py
python scripts/augment_corpus.py
python scripts/clean_chunks.py

# Phase 3: build PubMedBERT vector index (CPU, ~3 min)
python src/retrieval/embed.py --model medical

# Phase 5: build and validate evaluation set
python scripts/generate_eval_questions.py
python scripts/validate_questions.py --no-retrieval
```

**Step 2,personal GPU server (SSH): LLM generation only**

```bash
# SSH into the university GPU server
ssh <netid>@<personal-gpu-server-hostname>

# Clone the repo on the server (or rsync your local copy)
git clone https://github.com/Jatin-nabhoya/clinical-rag-audit.git
cd clinical-rag-audit
pip install -r requirements.txt
echo "HF_TOKEN=hf_..." > .env    # Llama-3 requires HuggingFace token

# Transfer the vector index and eval set from local machine
# (run from your local terminal)
# scp -r data/vector_store/medical/ <netid>@<server>:clinical-rag-audit/data/vector_store/
# scp data/processed/eval_questions.jsonl <netid>@<server>:clinical-rag-audit/data/processed/

# On the server: run all three models sequentially (~2–3 hours)
python scripts/run_inference.py

# Transfer results back to local machine
# scp -r <netid>@<server>:clinical-rag-audit/results/eval_hallucination_audit/ results/
```

**Step 3, Local machine (CPU): scoring and figures**

```bash
# Phase 6: hallucination taxonomy + publication charts
python scripts/score_hallucinations.py
python scripts/visualize_results.py
python reports/phase8/generate_analysis.py
```

Key versioned artifacts: eval set `data/processed/eval_questions.jsonl` (v2, 2026-04-24); taxonomy `docs/taxonomy_definitions.md` (v1.1); generations `results/eval_hallucination_audit/*/generations.jsonl`.
