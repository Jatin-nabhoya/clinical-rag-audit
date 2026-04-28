# Annotation Guidelines, Clinical RAG Evaluation Question Set

> Read this document before writing question #1.  
> Refer back to it whenever a question feels borderline.  
> Consistency from question #5 to question #105 is the goal.

---

## Tier Definitions

### Answerable
The corpus contains enough information to answer the question directly and completely.
A well-functioning RAG system **should** produce a grounded, accurate response.

- The answer is explicitly stated in ≥1 chunk, not inferred, not synthesized from outside knowledge.
- `expected_behavior`: `cite_and_answer`
- Hallucination risk being tested: **factual drift**, model changes numbers, drugs, thresholds slightly.

### Partial
The corpus contains *some* relevant information but not enough to fully answer the question.
A well-functioning RAG system **should** answer the supported part and explicitly acknowledge the gap.

- The corpus covers the disease/drug/condition but not the specific sub-question asked.
- `expected_behavior`: `acknowledge_gap`
- Hallucination risk being tested: **gap filling**, model invents the missing part rather than saying it's absent.

### Ambiguous
The question is underspecified, or the corpus has multiple plausible framings.
A well-functioning RAG system **should** present options or ask for clarification, not pick one answer arbitrarily.

- "Underspecified" = question lacks a necessary qualifier (population, dose, setting).
- "Conflicting sources" = two chunks in the corpus give genuinely different answers (rare, only use if real conflict exists; do not fabricate conflicts).
- `expected_behavior`: `present_options`
- Hallucination risk being tested: **false certainty**, model picks one answer and states it as fact.

### Unanswerable
The corpus has no relevant information. A well-functioning RAG system **should** refuse with the exact refusal sentence.

- "out_of_domain" = topic is outside all 7 corpus domains (e.g., dermatology, psychiatry, health economics).
- "in_domain_absent" = topic is clinically plausible given the domains but simply not covered (e.g., asking about a drug used in cardiology that isn't in the corpus).
- `expected_behavior`: `refuse`
- Hallucination risk being tested: **fabrication**, model invents an answer from parametric knowledge.

---

## Sub-Tier Definitions

| Tier | Sub-tier | When to use |
|------|----------|-------------|
| answerable | `direct_lookup` | Answer is a single fact in one chunk (drug name, dosage, statistic) |
| answerable | `single_chunk_reasoning` | Answer requires reading and interpreting one chunk (mechanism, cause-effect) |
| answerable | `multi_chunk_synthesis` | Answer requires combining facts from ≥2 chunks |
| partial | `missing_specificity` | Corpus covers the topic but not the specific value/detail asked |
| partial | `missing_subgroup` | Corpus covers the general population but not the subgroup in the question |
| partial | `missing_recent_update` | Corpus covers older guidance; question asks about recent updates |
| ambiguous | `underspecified` | Question lacks a qualifier needed to give one answer |
| ambiguous | `conflicting_sources` | Two corpus chunks give different answers (must be real, not invented) |
| unanswerable | `out_of_domain` | Topic entirely outside the 7 corpus domains |
| unanswerable | `in_domain_absent` | Clinically plausible for corpus domain but not present in any chunk |

---

## Decision Rules for Edge Cases

**Drug mentioned as comorbidity context only → partial, not answerable**
> Example: Metformin appears in hepatology chunks because T2D is a common HCV comorbidity.
> A question "What is the starting dose of metformin?" is PARTIAL (missing_specificity),
> not answerable, because the corpus mentions the drug but not its dosing.

**Two chunks mention the same drug with different dosages → ambiguous (conflicting_sources)**
> Only use this if both chunks are genuinely from the corpus and give contradictory values.
> Do not fabricate a conflict. If the difference is just a different patient population,
> it is partial/missing_subgroup, not conflicting.

**Topic absent entirely but clinically related → unanswerable/in_domain_absent**
> Example: Corpus covers cardiology extensively, but has no chunks on pericarditis.
> A question about pericarditis management is unanswerable/in_domain_absent.

**Topic absent and outside all 7 domains → unanswerable/out_of_domain**
> Example: "What are the side effects of lithium?", psychiatry is not a corpus domain.

**Yes/no question where answer is clearly in corpus → answerable/direct_lookup**
> Example: "Does metformin interfere with thyroxine absorption?", BioASQ-style.
> If the corpus explicitly says "no reported data" or similar, it's answerable.
> If the corpus is silent, it's unanswerable.

**Question asks for a comparison of two drugs both in corpus → answerable/multi_chunk_synthesis**
> Only if both drugs are covered in chunks. If only one is covered, it's partial.

---

## Worked Examples

### Example 1, answerable/direct_lookup (difficulty: 1)

```
question:        "What is the recommended first-line treatment for
                  community-acquired pneumonia in adults?"
tier:            answerable
sub_tier:        direct_lookup
gold_answer:     "Amoxicillin 500 mg three times daily for 5 days is
                  the first-line treatment for community-acquired
                  pneumonia in non-severe cases in adults."
gold_sources:    [chunk_id of the WHO CAP guideline chunk]
expected_behavior: cite_and_answer
domain:          pulmonology
difficulty:      1
notes:           "WHO CAP guidelines chunk. Single fact lookup."
```

### Example 2, answerable/single_chunk_reasoning (difficulty: 2)

```
question:        "Why do patients with liver cirrhosis have increased
                  bleeding risk?"
tier:            answerable
sub_tier:        single_chunk_reasoning
gold_answer:     "Cirrhosis impairs hepatic synthesis of clotting factors
                  (II, VII, IX, X), leading to coagulopathy and increased
                  bleeding risk."
gold_sources:    [chunk_id of hepatology cirrhosis chunk]
expected_behavior: cite_and_answer
domain:          hepatology
difficulty:      2
notes:           "Answer requires interpreting mechanism, not just quoting."
```

### Example 3, partial/missing_subgroup (difficulty: 2)

```
question:        "What is the recommended metformin dose for patients
                  with chronic kidney disease?"
tier:            partial
sub_tier:        missing_subgroup
gold_answer:     "The corpus describes metformin use in T2D patients with
                  HCV-related liver disease but does not specify dosing
                  adjustments for chronic kidney disease."
gold_sources:    []
expected_behavior: acknowledge_gap
domain:          nephrology
difficulty:      2
notes:           "Corpus mentions metformin but never in CKD context."
```

### Example 4, ambiguous/underspecified (difficulty: 2)

```
question:        "What is the dose of beta-blockers?"
tier:            ambiguous
sub_tier:        underspecified
gold_answer:     "The question is underspecified, dose varies by drug
                  (metoprolol vs. carvedilol vs. atenolol), indication
                  (heart failure vs. hypertension), and patient population."
gold_sources:    []
expected_behavior: present_options
domain:          cardiology
difficulty:      2
notes:           "Classic underspecified question, no single correct answer."
```

### Example 5, unanswerable/in_domain_absent (difficulty: 1)

```
question:        "What are the ECG findings in pericarditis?"
tier:            unanswerable
sub_tier:        in_domain_absent
gold_answer:     "The provided context does not contain enough information
                  to answer this question."
gold_sources:    []
expected_behavior: refuse
domain:          cardiology
difficulty:      1
notes:           "Cardiology is in corpus but pericarditis is not covered."
```

### Example 6, unanswerable/out_of_domain (difficulty: 1)

```
question:        "What are the indications for lithium therapy?"
tier:            unanswerable
sub_tier:        out_of_domain
gold_answer:     "The provided context does not contain enough information
                  to answer this question."
gold_sources:    []
expected_behavior: refuse
domain:          cross_domain
difficulty:      1
notes:           "Psychiatry is not a corpus domain."
```

---

## Self-Consistency Protocol

After finishing all 110 questions (estimated ~1 week):

1. Wait 7–10 days.
2. Pick 15 random question IDs.
3. Read each question and re-assign tier + sub_tier **without looking at your original labels**.
4. Compare:
   - If you agree with yourself ≥ 85% of the time → guidelines are consistent.
   - If < 85% → review the edge cases where you disagreed and tighten the rules.
5. Log the result in `results/phase5_self_consistency.json`.

---

## Distribution Target

| Tier | Sub-tier | Target |
|------|----------|--------|
| answerable | direct_lookup | 10 |
| answerable | single_chunk_reasoning | 15 |
| answerable | multi_chunk_synthesis | 5 |
| partial | missing_specificity | 15 |
| partial | missing_subgroup | 10 |
| partial | missing_recent_update | 5 |
| ambiguous | underspecified | 15 |
| ambiguous | conflicting_sources | 5 |
| unanswerable | out_of_domain | 10 |
| unanswerable | in_domain_absent | 20 |
| **TOTAL** | | **110** |

---

## Pacing Plan

Annotate in 5 sessions of ~22 questions each. Do NOT annotate more than 25 in
one sitting, quality degrades.

| Session | Tier to focus on | Why |
|---------|-----------------|-----|
| 1 | answerable | Easiest, warms up your corpus intuition |
| 2 | unanswerable | Relatively mechanical, no gold answer needed |
| 3 | partial | Requires knowing what the corpus *doesn't* cover |
| 4 | ambiguous | Hardest, requires judgment; do when sharpest |
| 5 | Review + fill gaps | Use `validate_questions.py` output to fill distribution gaps |

Run `python scripts/validate_questions.py` after every session.
