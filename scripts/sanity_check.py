# Step 4: Sanity Check — Print 10 Random Chunks

import json
import random
from collections import Counter

from utils import CHUNKS_CLEAN as CHUNKS_PATH

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

total = len(chunks)
avg_tokens = sum(c["metadata"]["token_count"] for c in chunks) / total
by_source = Counter(c["metadata"]["source"] for c in chunks)
by_domain = Counter(c["metadata"]["domain"] for c in chunks)

print(f"Total chunks  : {total}")
print(f"Avg tokens    : {avg_tokens:.1f}")
print(f"By source     : {dict(by_source)}")
print(f"By domain     : {dict(by_domain)}")
print()

sample = random.sample(chunks, min(10, total))
for i, chunk in enumerate(sample, 1):
    m = chunk["metadata"]
    print(f"{'='*60}")
    print(f"[{i}] {m['source'].upper()} | {m['domain']} | chunk {m['chunk_index']+1}/{m['total_chunks']} | {m['token_count']} tokens")
    print(f"    Title : {m['title'][:80]}")
    print(f"    Text  : {chunk['text'][:300]}")
    print()



r'''output  :
Total chunks  : 1745
Avg tokens    : 389.4
By source     : {'pmc': 1715, 'cdc': 5, 'who': 25}
By domain     : {'cardiology': 1715, 'infectious_disease': 30}

============================================================
[1] PMC | cardiology | chunk 2/45 | 499 tokens
    Title : Donor-derived regulatory dendritic cell infusion and early immunosuppressive dru
    Text  : Regulatory dendritic cell therapy in organ transplantation
Tolerogenic dendritic cell therapy in organ transplantation
Endogenous dendritic cells mediate the effects of intravenously injected therapeutic immunosuppressive dendritic cells in transplantation
Orchestration of transplantation tolerance 

============================================================
[2] PMC | cardiology | chunk 11/53 | 502 tokens
    Title : Hepatic Arterial Flow-Induced Portal Tract Fibrosis in Portal Hypertension: The 
    Text  : As a downstream molecule in a flow-induced mechano-signaling pathway, VCAM-1 is highly expressed in vascular endothelial cells (ECs) and plays a key role in immune cell recruitment.27 Our IHC analysis of livers following PPVL revealed a significant increase in VCAM-1 expression in HA ECs, peaking at

============================================================
[3] PMC | cardiology | chunk 39/79 | 375 tokens
    Title : A hybrid dual-stream CNN framework with dynamic data augmentation and improved M
    Text  : \usepackage{amsbsy}
				\usepackage{mathrsfs}
				\usepackage{upgreek}
				\setlength{\oddsidemargin}{-69pt}
				\begin{document}$$\mathrm{k}$$\end{document} is a scaling factor (e.g., 0.001 × 255) used to control the base noise level. This formulation ensures that brighter and low-contrast images r

============================================================
[4] PMC | cardiology | chunk 18/33 | 28 tokens
    Title : Stroke Risk After TAVR
    Text  : Procedural Details
Values are n (%) or mean ± SD.
NA = not applicable; other abbreviations as in Table 1.

============================================================
[5] PMC | cardiology | chunk 18/20 | 434 tokens
    Title : A Cohort Study on Cardiovascular Disease Mortality in Breast Cancer Patients Wit
    Text  : In conclusion, our study reveals that triple‐negative BC patients demonstrate significantly elevated CVD mortality compared to other subtypes. These findings highlight the importance of incorporating cardiovascular risk monitoring into the clinical management of these patients. The analysis further 

============================================================
[6] PMC | cardiology | chunk 158/245 | 81 tokens
    Title : Japanese clinical practice guidelines for vascular tumors, vascular malformation
    Text  : ■ Everolimus
Chiguer et al. (2020) reported a case that did not respond to prednisolone, acetylsalicylic acid, and ticlopidine but responded to everolimus at 0.1 mg/kg [507]. In these cases, everolimus was used, because sirolimus was not available
■ Vincristine

============================================================
[7] PMC | cardiology | chunk 11/20 | 362 tokens
    Title : A Cohort Study on Cardiovascular Disease Mortality in Breast Cancer Patients Wit
    Text  : Temporal trends in the proportions of cardiovascular disease mortality and breast cancer mortality in different subtypes of breast cancer patients. Note: CVD, cardiovascular disease. The figure illustrates the temporal trends in the proportions of cardiovascular disease mortality and breast cancer m

============================================================
[8] PMC | cardiology | chunk 26/38 | 350 tokens
    Title : Oral microbiome profiling of primary oral candidiasis during infection and post-
    Text  : Prevalence and differential abundance analysis of oral bacteria at phylum level
Cyanobacteria was excluded from the analysis as most of the taxa identified are plant-based
OT, primary oral candidiasis; AT, after treatment; C, control
*: Fold change < -1.5 or > 1.5 and p < 0.05 (Wald test) is conside

============================================================
[9] PMC | cardiology | chunk 18/46 | 299 tokens
    Title : Urinary transcription factor 21 (TCF21) as a non-invasive biomarker of podocyte 
    Text  : Within preeclampsia cases, urinary TCF21 showed inverse associations with diastolic blood pressure (r = − 0.244, p = 0.002), convulsions (r = − 0.193, p = 0.014), and preeclampsia severity (r = − 0.258, p < 0.001), and a weak positive association with platelet count (r = 0.194, p = 0.014) (Table 2).

============================================================
[10] PMC | cardiology | chunk 8/43 | 489 tokens
    Title : Comparative performance of ReMELD-Na, MELD 3.0 and established scores after TIPS
    Text  : The primary endpoint was the combined event of death or LTx within 90 days after TIPS, defined as the occurrence of either death or LTx. The secondary endpoint was the combined event of death or LTx within 1-year after TIPS. Both events were considered as outcomes defining failure. Area under the re

'''