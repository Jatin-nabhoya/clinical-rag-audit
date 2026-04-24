# Pipeline Validation — Archived Smoke Test

This directory contains outputs from an initial end-to-end pipeline validation
run, not from the primary evaluation.

## What this was

An early Kaggle T4 run to confirm that the retrieval + generation pipeline
worked correctly across all three models before the formal evaluation was
designed. Questions came from pre-existing BioASQ/MedQuAD diabetes Q&A pairs.

## Why it is NOT the evaluation

- Questions are diabetes-only (BioASQ + MedQuAD), not corpus-aware
- No tier labels (answerable / partial / ambiguous / unanswerable)
- No gold sources or gold answers
- Not designed to probe specific hallucination failure modes

## Primary evaluation

See `results/eval_hallucination_audit/` for the actual evaluation outputs.
The eval set is at `data/processed/eval_questions.jsonl`.

## Files

| File | Description |
|------|-------------|
| `llama3_8b_generations.json` | 723 generations — Llama-3-8B |
| `mistral_7b_generations.json` | 723 generations — Mistral-7B |
| `phi3_mini_generations.json` | 723 generations — Phi-3-mini |
| `combined_results.csv` | All 3 models merged (2,169 rows) |
| `run.log` | Kaggle execution log |
| `log_*.txt` | Per-model generation logs |
