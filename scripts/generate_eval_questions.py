"""
Phase 5 — Generate the 110-question gold evaluation set (Option B).

Questions are freshly written to cover all 7 corpus domains using the
topic map from explore_corpus.py. Three reviewer-driven fixes applied:
  1. All direct_lookup questions are specific-answer (not yes/no)
  2. 8 diabetes questions added — corpus has 227 diabetes + 79 insulin chunks
  3. Mislabels fixed: q_075 (kidney reversible → unanswerable),
     inhaler question re-framed to test fabrication risk

These questions are NOT from BioASQ/MedQuAD. They must be run through
the models on Kaggle (scripts/run_inference.py).

Usage:
    python scripts/generate_eval_questions.py
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import EVAL_QUESTIONS  # noqa: E402

TODAY = date.today().isoformat()

# ── 110 questions ─────────────────────────────────────────────────────────────
# (question, tier, sub_tier, domain, gold_answer,
#  expected_behavior, hallucination_target, difficulty, notes)

QUESTIONS = [

    # ══ ANSWERABLE — direct_lookup (10) ═══════════════════════════════════════
    # All specific-answer — not yes/no. Models must retrieve the specific fact.

    ("What surface antigens do seasonal influenza vaccines primarily target?",
     "answerable", "direct_lookup", "infectious_disease",
     "Seasonal influenza vaccines primarily target two surface antigens: haemagglutinin (H), which mediates viral entry into host cells, and neuraminidase (N), which enables viral release. Vaccine strains are updated annually based on circulating H and N subtypes.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: viral(275), vaccination(161), influenza(119). Specific antigen names require retrieval."),

    ("By what mechanism does chronic hypertension damage cerebral blood vessels and increase stroke risk?",
     "answerable", "direct_lookup", "cardiology",
     "Chronic hypertension causes sustained mechanical stress on arterial walls, leading to endothelial dysfunction, arteriosclerosis, hypertrophy of smooth muscle, and microaneurysm formation. These changes predispose to both ischaemic stroke (thromboembolism from atherosclerotic plaques) and haemorrhagic stroke (microaneurysm rupture).",
     "cite_and_answer", "factual_drift", 1,
     "Dense: hypertension(219), stroke(144), cardiovascular(219). Mechanism requires reasoning."),

    ("Which specific coagulation factors are reduced by liver cirrhosis, and why?",
     "answerable", "direct_lookup", "hepatology",
     "Liver cirrhosis reduces hepatic synthesis of vitamin K–dependent coagulation factors (II, VII, IX, X) and factor V. Because the liver is the primary site of clotting factor production, cirrhotic damage directly impairs haemostasis, causing coagulopathy and increased bleeding risk.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: cirrhosis(59), fibrosis(104), liver(382). Specific factor numbers require retrieval."),

    ("What cellular features observed in a biopsy specimen confirm that a tissue sample is malignant?",
     "answerable", "direct_lookup", "oncology",
     "Malignancy is confirmed by histological features including nuclear pleomorphism and atypia, abnormal or excessive mitotic figures, loss of normal cellular differentiation (anaplasia), and evidence of invasion through the basement membrane into surrounding tissue.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: biopsy(51), carcinoma(78), tumor(223). Histological criteria require retrieval."),

    ("What structural lung changes make airflow obstruction irreversible in COPD?",
     "answerable", "direct_lookup", "pulmonology",
     "Irreversibility in COPD results from two processes: emphysema, in which alveolar walls are permanently destroyed reducing elastic recoil; and airway remodelling from chronic bronchitis, where wall thickening, fibrosis, and mucus gland hyperplasia permanently narrow airways.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: copd(59), respiratory(230), pulmonary(139). Structural mechanism requires retrieval."),

    ("Below what CD4+ T cell count is HIV infection clinically classified as AIDS?",
     "answerable", "direct_lookup", "infectious_disease",
     "AIDS is defined as HIV infection with a CD4+ T cell count below 200 cells per microlitre, or the occurrence of any AIDS-defining opportunistic illness regardless of CD4 count. This threshold reflects severe immune compromise.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: hiv(193). Specific numerical threshold requires retrieval — won't be guessed correctly by all models."),

    ("By what enzymatic mechanism do statins lower circulating LDL cholesterol?",
     "answerable", "direct_lookup", "cardiology",
     "Statins competitively inhibit HMG-CoA reductase, the rate-limiting enzyme in the mevalonate pathway for hepatic cholesterol synthesis. Reduced intracellular cholesterol upregulates LDL receptors on hepatocytes, increasing clearance of LDL particles from the bloodstream.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: statin(35), cholesterol(77), cardiovascular(219). Enzymatic mechanism requires retrieval."),

    ("What proportion of people with chronic HCV infection develop liver cirrhosis over 20 years?",
     "answerable", "direct_lookup", "hepatology",
     "Approximately 15–30% of people with chronic HCV infection develop liver cirrhosis over 20–30 years. Progression is accelerated by alcohol use, HIV co-infection, obesity, and metabolic factors such as steatosis.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: hepatitis(88), cirrhosis(59), fibrosis(104). Specific percentage requires retrieval."),

    ("By what mechanism does ionising radiation in radiotherapy kill cancer cells?",
     "answerable", "direct_lookup", "oncology",
     "Ionising radiation causes double-strand DNA breaks in cancer cells, preventing replication and triggering apoptosis. Free radicals generated from water radiolysis amplify this damage. Hypoxic tumour regions are relatively radioresistant because oxygen is required to fix radiation-induced damage.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: radiotherapy(44), tumor(223), cancer(274). Mechanism (DSBs, free radicals, hypoxia) requires retrieval."),

    ("What post-bronchodilator spirometric criterion confirms reversible airflow obstruction in asthma?",
     "answerable", "direct_lookup", "pulmonology",
     "Reversible airflow obstruction in asthma is confirmed when spirometry shows an increase in FEV1 of ≥12% and ≥200 mL above baseline after bronchodilator administration. This distinguishes asthma from COPD, in which post-bronchodilator reversibility is absent or minimal.",
     "cite_and_answer", "factual_drift", 1,
     "Dense: asthma(64), copd(59), respiratory(230). Specific FEV1 criteria require retrieval."),

    # ══ ANSWERABLE — single_chunk_reasoning (15) ══════════════════════════════

    ("How does seasonal influenza vaccination reduce community disease burden?",
     "answerable", "single_chunk_reasoning", "infectious_disease",
     "Influenza vaccination stimulates antibody production against circulating strains before exposure, providing direct protection and contributing to herd immunity by reducing the number of susceptible individuals. This interrupts transmission chains, lowering incidence and severity of disease across the population.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: vaccination(161), influenza(119), infection(431)."),

    ("Why is hypertension referred to as the silent killer?",
     "answerable", "single_chunk_reasoning", "cardiology",
     "Hypertension is called the silent killer because it typically produces no symptoms while progressively damaging the vasculature, heart, kidneys, and brain over years. Patients are often diagnosed only after a cardiovascular event such as myocardial infarction, stroke, or heart failure.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: hypertension(219), cardiovascular(219), stroke(144)."),

    ("What is HbA1c and why is it the standard measure for long-term glycaemic control in diabetes?",
     "answerable", "single_chunk_reasoning", "cardiology",
     "HbA1c (glycated haemoglobin) is formed when glucose irreversibly binds to haemoglobin. Because red blood cells survive approximately 90–120 days, HbA1c reflects average blood glucose over the preceding 2–3 months. This makes it more informative for monitoring diabetes control than single fasting glucose measurements, which reflect only the moment of testing.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: diabetes(227), hba1c(31), metabolic(293). Diabetes gap filled."),

    ("What is the mechanism by which liver fibrosis progresses to cirrhosis?",
     "answerable", "single_chunk_reasoning", "hepatology",
     "Repeated hepatocyte injury triggers chronic inflammation and activation of hepatic stellate cells, which transdifferentiate into myofibroblasts and deposit excess collagen. Sustained fibrogenesis distorts the normal lobular architecture, replacing functional liver tissue with fibrous bands and regenerative nodules — the defining structure of cirrhosis.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: fibrosis(104), cirrhosis(59), liver(382)."),

    ("How does chemotherapy work to destroy cancer cells?",
     "answerable", "single_chunk_reasoning", "oncology",
     "Chemotherapy agents target rapidly dividing cells through several mechanisms: alkylating agents cross-link DNA strands; antimetabolites inhibit nucleotide synthesis; topoisomerase inhibitors trap DNA cleavage complexes; taxanes stabilise microtubules preventing mitotic spindle function. All mechanisms ultimately block cell division and induce apoptosis.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: chemotherapy(70), tumor(223), carcinoma(78)."),

    ("What is the pathophysiology of sepsis?",
     "answerable", "single_chunk_reasoning", "infectious_disease",
     "Sepsis results from a dysregulated host response to infection: pathogens trigger massive cytokine release (cytokine storm), causing systemic endothelial injury, microvascular dysfunction, impaired oxygen delivery, and eventually multi-organ failure. The dysregulation — not the infection itself — drives morbidity and mortality.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: sepsis(105), infection(431), pathogen(171), bacterial(94)."),

    ("Why does obesity increase cardiovascular disease risk?",
     "answerable", "single_chunk_reasoning", "cardiology",
     "Obesity promotes cardiovascular risk through multiple pathways: insulin resistance and type 2 diabetes, dyslipidaemia (elevated LDL and triglycerides, low HDL), hypertension, and systemic low-grade inflammation from adipokine dysregulation. These factors collectively accelerate atherosclerosis and increase rates of myocardial infarction, stroke, and heart failure.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: obesity(125), cardiovascular(219), metabolic(293)."),

    ("What complications can develop from liver cirrhosis?",
     "answerable", "single_chunk_reasoning", "hepatology",
     "Cirrhosis complications include portal hypertension (leading to oesophageal varices and ascites), spontaneous bacterial peritonitis, hepatic encephalopathy from impaired ammonia metabolism, hepatorenal syndrome, coagulopathy, and markedly elevated risk of hepatocellular carcinoma.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: cirrhosis(59), ascites(31), liver(382)."),

    ("How does COVID-19 cause respiratory failure in severe cases?",
     "answerable", "single_chunk_reasoning", "pulmonology",
     "In severe COVID-19, SARS-CoV-2 triggers an exaggerated immune response — a cytokine storm — causing diffuse alveolar damage, flooding alveoli with inflammatory exudate, and producing acute respiratory distress syndrome (ARDS). Gas exchange fails as alveolar surface area is lost, requiring mechanical ventilation.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: covid(136), respiratory(230), pulmonary(139), viral(275)."),

    ("How does insulin resistance in type 2 diabetes develop at the cellular level?",
     "answerable", "single_chunk_reasoning", "cardiology",
     "In type 2 diabetes, excess circulating free fatty acids and adipose-derived inflammatory cytokines (TNF-α, IL-6) impair insulin receptor substrate-1 (IRS-1) phosphorylation. This post-receptor signalling defect prevents GLUT-4 translocation to the cell membrane in muscle and adipose cells, blocking glucose uptake and causing compensatory hyperinsulinaemia.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: diabetes(227), insulin(79), metabolic(293). Cellular mechanism of T2D — diabetes gap filled."),

    ("What makes Mycobacterium tuberculosis difficult to eradicate with treatment?",
     "answerable", "single_chunk_reasoning", "infectious_disease",
     "M. tuberculosis is difficult to eradicate due to its slow replication rate, ability to enter a metabolically dormant state inside macrophages, a thick lipid-rich cell wall that limits antibiotic penetration, the need for prolonged multi-drug combination therapy to prevent resistance, and increasing rates of multi-drug resistant (MDR-TB) and extensively drug-resistant (XDR-TB) strains.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: tuberculosis(32), bacterial(94), infection(431)."),

    ("Why is early cancer detection associated with better survival outcomes?",
     "answerable", "single_chunk_reasoning", "oncology",
     "Early-stage cancers are typically localised, smaller, and have not spread to lymph nodes or distant organs, making them more amenable to curative treatment (surgery, radiotherapy, or chemotherapy). Five-year survival rates are substantially higher for localised compared to metastatic disease across most cancer types.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: cancer(274), carcinoma(78), metastasis(19), tumor(223)."),

    ("How does atrial fibrillation increase the risk of ischaemic stroke?",
     "answerable", "single_chunk_reasoning", "cardiology",
     "Atrial fibrillation causes disorganised atrial contractions that allow blood to pool and form thrombi, particularly in the left atrial appendage. These thrombi can detach and embolise to cerebral arteries, causing cardioembolic ischaemic stroke. AF increases stroke risk approximately fivefold compared to sinus rhythm.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: atrial(32), stroke(144), cardiovascular(219)."),

    ("What is hepatic steatosis and what metabolic conditions cause it?",
     "answerable", "single_chunk_reasoning", "hepatology",
     "Hepatic steatosis is the pathological accumulation of triglycerides within hepatocytes, exceeding 5% of liver weight. It is caused by insulin resistance, obesity, type 2 diabetes, dyslipidaemia, and alcohol use — conditions that impair fatty acid oxidation and promote hepatic lipid deposition.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: steatosis(48), metabolic(293), liver(382), obesity(125)."),

    ("How does glomerular injury lead to proteinuria?",
     "answerable", "single_chunk_reasoning", "nephrology",
     "The glomerular filtration barrier — comprising podocyte foot processes, the glomerular basement membrane, and fenestrated endothelium — normally restricts protein passage based on size and charge. Glomerular injury damages podocyte architecture and disrupts this barrier, allowing albumin and other proteins to leak into the filtrate, producing proteinuria.",
     "cite_and_answer", "factual_drift", 2,
     "Dense: glomerular(39), proteinuria(13), kidney(89), renal(78)."),

    # ══ ANSWERABLE — multi_chunk_synthesis (5) ════════════════════════════════

    ("What overlapping risk factors link hypertension and coronary artery disease?",
     "answerable", "multi_chunk_synthesis", "cardiology",
     "Hypertension and coronary artery disease share risk factors including smoking, obesity, dyslipidaemia, type 2 diabetes, physical inactivity, older age, and family history. Hypertension itself accelerates coronary atherosclerosis by increasing haemodynamic shear stress, promoting endothelial dysfunction and plaque formation.",
     "cite_and_answer", "factual_drift", 3,
     "Dense: hypertension(219), cardiovascular(219), coronary(82), myocardial(100)."),

    ("How do vaccination and antiviral therapy complement each other in influenza management?",
     "answerable", "multi_chunk_synthesis", "infectious_disease",
     "Vaccination prevents infection by priming adaptive immunity before exposure, contributing to herd immunity at population level. Antiviral drugs (e.g., neuraminidase inhibitors) reduce illness duration and severity after infection occurs by limiting viral replication. Together they address prevention and treatment complementarily.",
     "cite_and_answer", "factual_drift", 3,
     "Dense: vaccination(161), antiviral(95), influenza(119), infection(431)."),

    ("What is the progression from viral hepatitis to cirrhosis to hepatocellular carcinoma?",
     "answerable", "multi_chunk_synthesis", "hepatology",
     "Chronic viral hepatitis (HBV or HCV) sustains hepatocellular inflammation and necrosis, activating stellate cells to deposit collagen. Progressive fibrosis advances to cirrhosis, creating a microenvironment of genomic instability, regenerative nodule formation, and chronic oxidative stress that substantially elevates hepatocellular carcinoma risk.",
     "cite_and_answer", "factual_drift", 3,
     "Dense: hepatitis(88), fibrosis(104), cirrhosis(59), tumor(223), liver(382)."),

    ("How do chemotherapy and immunotherapy differ mechanistically in cancer treatment?",
     "answerable", "multi_chunk_synthesis", "oncology",
     "Chemotherapy uses cytotoxic drugs to kill rapidly dividing cells non-selectively, affecting both tumour and healthy tissues. Immunotherapy removes immunosuppressive signals (e.g., checkpoint inhibitors targeting PD-1/PD-L1 or CTLA-4) that tumour cells exploit, enabling T cells to selectively destroy cancer cells. This selectivity produces a distinct — and often more durable — toxicity and response profile.",
     "cite_and_answer", "factual_drift", 3,
     "Dense: chemotherapy(70), immunotherapy(41), tumor(223), cancer(274)."),

    ("What is the relationship between obesity, metabolic syndrome, and cardiovascular mortality?",
     "answerable", "multi_chunk_synthesis", "cardiology",
     "Obesity drives metabolic syndrome through central adiposity, hypertension, dyslipidaemia, insulin resistance, and chronic inflammation. Each component independently elevates cardiovascular risk; together they synergistically accelerate atherosclerosis, increase rates of myocardial infarction, heart failure, and stroke, and raise all-cause cardiovascular mortality.",
     "cite_and_answer", "factual_drift", 3,
     "Dense: obesity(125), metabolic(293), cardiovascular(219), myocardial(100)."),

    # ══ PARTIAL — missing_specificity (15) ════════════════════════════════════

    ("What is the mechanism of action of beta-blockers?",
     "partial", "missing_specificity", "cardiology",
     "The corpus describes beta-blockers in cardiovascular management but does not explain the specific receptor pharmacology: competitive antagonism at beta-1 adrenergic receptors (reducing heart rate and contractility) and beta-2 receptors (bronchospasm risk), or the difference between selective (metoprolol) and non-selective (carvedilol) agents.",
     "acknowledge_gap", "gap_filling", 2,
     "Thin: beta.blocker(10 chunks). Drug mentioned but mechanism absent."),

    ("What specific DAA regimens are used to treat hepatitis C?",
     "partial", "missing_specificity", "hepatology",
     "The corpus confirms that direct-acting antivirals (DAAs) cure HCV and improve liver outcomes, but does not specify regimen components, drug combinations (e.g., sofosbuvir/ledipasvir, glecaprevir/pibrentasvir), treatment durations (8–12 weeks), or genotype-specific protocol choices.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hepatitis(88). Regimen-level detail absent."),

    ("What is the precise vaccine effectiveness percentage of the seasonal influenza vaccine?",
     "partial", "missing_specificity", "infectious_disease",
     "The corpus confirms influenza vaccination reduces morbidity and mortality in high-risk groups, but does not provide season-specific vaccine effectiveness percentages, confidence intervals, or comparative efficacy data between vaccine formulations.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: vaccination(161), influenza(119). Efficacy percentages absent."),

    ("What are the specific toxicity grades and management of chemotherapy adverse effects?",
     "partial", "missing_specificity", "oncology",
     "The corpus describes chemotherapy in cancer treatment and notes that adverse effects occur, but does not provide a systematic toxicity grading framework (CTCAE grades 1–4), nor management protocols for common side effects such as febrile neutropenia, nausea, peripheral neuropathy, or cardiotoxicity.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: chemotherapy(70). Toxicity grading system absent."),

    ("What are the stepwise pharmacological treatments for COPD by disease severity?",
     "partial", "missing_specificity", "pulmonology",
     "The corpus mentions COPD and references bronchodilators and corticosteroids, but does not provide a GOLD-stage-based stepwise algorithm specifying when to initiate SABAs, LABAs, LAMAs, ICS combinations, or escalation criteria based on spirometry and symptom burden.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: copd(59), corticosteroid(46). Stepwise algorithm absent."),

    ("What are the specific statin drugs and their comparative LDL-lowering potencies?",
     "partial", "missing_specificity", "cardiology",
     "The corpus confirms statins lower LDL cholesterol and reduce cardiovascular risk, but does not name individual agents, compare high-intensity (rosuvastatin, atorvastatin) vs moderate-intensity statins, provide NNT data, or differentiate primary from secondary prevention recommendations.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: statin(35), cholesterol(77), cardiovascular(219). Drug-level comparison absent."),

    ("Which specific antibiotic classes are first-line for community-acquired pneumonia?",
     "partial", "missing_specificity", "pulmonology",
     "The corpus references antibiotic use in respiratory infections broadly but does not specify first-line choices for community-acquired pneumonia (e.g., amoxicillin for outpatient mild disease; macrolide or fluoroquinolone for atypical pathogens; beta-lactam + macrolide for hospitalised patients), nor differentiate by severity.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: antibiotic(45), pneumonia(49). Specific agents and severity tiers absent."),

    ("What are the HbA1c targets for elderly patients with type 2 diabetes?",
     "partial", "missing_specificity", "cardiology",
     "The corpus discusses HbA1c as a monitoring tool and diabetes management broadly, but does not provide age-stratified HbA1c targets for elderly patients. Guidelines recommend less stringent targets (e.g., <8.0–8.5%) in frail elderly with hypoglycaemia risk, compared to <7.0% for younger patients — this nuance is absent from the corpus.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hba1c(31), diabetes(227). Elderly-specific glycaemic targets absent. Diabetes gap filled."),

    ("What specific checkpoint inhibitor drugs are currently approved for immunotherapy?",
     "partial", "missing_specificity", "oncology",
     "The corpus describes checkpoint inhibitor immunotherapy as an effective cancer treatment, but does not list approved agents (e.g., pembrolizumab, nivolumab, ipilimumab, atezolizumab), their molecular targets (PD-1, PD-L1, CTLA-4), or tumour-specific approval indications.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: immunotherapy(41). Drug names and targets absent."),

    ("What are the standard antiretroviral regimen combinations for HIV treatment?",
     "partial", "missing_specificity", "infectious_disease",
     "The corpus discusses HIV pathogenesis and treatment broadly, but does not specify current first-line antiretroviral regimens, drug classes (NRTI backbone + integrase strand transfer inhibitor), preferred combinations (e.g., bictegravir/emtricitabine/TAF), or when to switch regimens.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hiv(193). ART regimen specifics absent."),

    ("What are the CKD stages and their corresponding GFR thresholds?",
     "partial", "missing_specificity", "nephrology",
     "The corpus discusses kidney disease, creatinine, glomerular dysfunction, and proteinuria, but does not define the KDIGO CKD staging system with its five eGFR-based stages (G1 ≥90, G2 60–89, G3a 45–59, G3b 30–44, G4 15–29, G5 <15 mL/min/1.73m²) or their associated management thresholds.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: creatinine(52), glomerular(39), renal(78). CKD staging system absent."),

    ("What are the target blood pressure values recommended for hypertensive patients?",
     "partial", "missing_specificity", "cardiology",
     "The corpus identifies hypertension as a major cardiovascular risk factor and discusses stroke and coronary disease risk, but does not specify guideline-recommended blood pressure targets (e.g., <130/80 mmHg per ACC/AHA 2017, or <140/90 mmHg per ESC/ESH), treatment initiation thresholds, or how targets vary by comorbidity.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hypertension(219). Numeric targets and guideline thresholds absent."),

    ("How is asthma severity formally classified into clinical categories?",
     "partial", "missing_specificity", "pulmonology",
     "The corpus describes asthma pathophysiology and management principles, but does not present the formal GINA severity classification (intermittent, mild persistent, moderate persistent, severe persistent), its spirometric and symptomatic criteria, or the step-up/step-down treatment framework derived from severity assessment.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: asthma(64). Severity classification framework absent."),

    ("What is the serological window period for HIV testing after exposure?",
     "partial", "missing_specificity", "infectious_disease",
     "The corpus covers HIV pathogenesis and antiretroviral treatment, but does not address the window period — the interval between infection and detectable antibodies (14–21 days for 4th-generation assays, up to 45 days for antibody-only tests) — which is critical for interpreting a negative post-exposure test.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hiv(193). Window period/testing timeline absent."),

    ("What are the cardiovascular benefits of SGLT2 inhibitors beyond glucose lowering?",
     "partial", "missing_specificity", "cardiology",
     "The corpus discusses diabetes, cardiovascular risk, and metabolic treatment broadly, but does not specifically detail the cardiovascular outcome trial data for SGLT2 inhibitors (EMPA-REG OUTCOME, CANVAS, DAPA-HF), which demonstrated reduced heart failure hospitalisation, cardiovascular death, and kidney disease progression independent of glucose lowering.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: diabetes(227), cardiovascular(219). SGLT2i CV outcome data absent. Diabetes gap filled."),

    # ══ PARTIAL — missing_subgroup (10) ═══════════════════════════════════════

    ("How does influenza vaccination efficacy differ in adults over 65?",
     "partial", "missing_subgroup", "infectious_disease",
     "The corpus confirms influenza vaccination reduces disease burden and identifies elderly patients as a high-risk population requiring vaccination, but does not provide age-stratified efficacy data or compare standard-dose vs adjuvanted/high-dose formulations specifically in adults over 65.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: vaccination(161), influenza(119). Elderly-specific efficacy absent."),

    ("What cardiovascular risks does hypertension pose specifically during pregnancy?",
     "partial", "missing_subgroup", "cardiology",
     "The corpus covers hypertension and cardiovascular risk broadly but does not specifically address gestational hypertension, preeclampsia, HELLP syndrome, or the distinct management approach required during pregnancy (avoiding ACE inhibitors, ARBS, and certain diuretics).",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hypertension(219). Pregnancy-specific content absent."),

    ("How is chemotherapy dosing adjusted in patients with renal impairment?",
     "partial", "missing_subgroup", "oncology",
     "The corpus describes chemotherapy and kidney disease as separate topics, but does not address pharmacokinetic dose adjustments for renally-cleared cytotoxic agents (e.g., carboplatin AUC-based dosing using Calvert formula, methotrexate dose reduction by eGFR) in patients with impaired renal function.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: chemotherapy(70), renal(78). Renal-adjusted dosing absent."),

    ("What are the hepatitis B management considerations specific to immunocompromised patients?",
     "partial", "missing_subgroup", "hepatology",
     "The corpus discusses hepatitis B and antiviral treatment generally, but does not specifically address reactivation risk in immunocompromised patients (chemotherapy, biologics, transplant immunosuppression) or prophylactic antiviral strategies to prevent HBV reactivation in this group.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hepatitis(88), antiviral(95). Immunocompromised management absent."),

    ("What are the COVID-19 vaccination schedule recommendations for immunocompromised individuals?",
     "partial", "missing_subgroup", "infectious_disease",
     "The corpus describes COVID-19 vaccination generally, but does not detail additional primary series doses, booster schedules, or precautions specifically recommended for immunocompromised patients (organ transplant recipients, haematological malignancies, patients on biological agents).",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: covid(136), vaccination(161). Immunocompromised-specific schedule absent."),

    ("How does heart failure management differ between HFrEF and HFpEF?",
     "partial", "missing_subgroup", "cardiology",
     "The corpus covers heart failure and cardiovascular management principles, but does not differentiate the evidence-based treatment strategies for heart failure with reduced ejection fraction (HFrEF: ACE inhibitors, beta-blockers, MRAs, SGLT2i, ICD/CRT) versus preserved ejection fraction (HFpEF: largely symptomatic management, diuretics, treating comorbidities).",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: heart.failure(71). HFrEF vs HFpEF distinction absent."),

    ("What asthma treatment modifications are needed specifically for children under 5 years?",
     "partial", "missing_subgroup", "pulmonology",
     "The corpus covers asthma pathophysiology and management in general adult terms but does not address paediatric-specific protocols for children under 5, including age-appropriate inhaler devices (spacer with face mask), weight-based dosing, or challenges in differentiating preschool wheeze from asthma.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: asthma(64). Paediatric-specific content absent."),

    ("What blood pressure targets are recommended for hypertensive patients with both CKD and diabetes?",
     "partial", "missing_subgroup", "nephrology",
     "The corpus discusses hypertension, CKD, and diabetes as overlapping topics, but does not address the combined CKD-diabetes subgroup, the preferred antihypertensive class (RAS blockade to reduce proteinuria), or specific BP targets (typically <130/80 mmHg) for this high-cardiovascular-risk combination.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hypertension(219), renal(78), kidney(89). CKD+DM subgroup absent."),

    ("What are the immune-related adverse event profiles of checkpoint inhibitors in elderly cancer patients?",
     "partial", "missing_subgroup", "oncology",
     "The corpus discusses immunotherapy in cancer and mentions elderly patients as a cancer-affected group, but does not address age-specific patterns of immune-related adverse events (irAEs), evidence on checkpoint inhibitor efficacy or toxicity in patients over 75, or dosing modifications for frail elderly patients.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: immunotherapy(41), cancer(274). Elderly irAE data absent."),

    ("How does sepsis management differ in paediatric versus adult patients?",
     "partial", "missing_subgroup", "infectious_disease",
     "The corpus covers sepsis pathophysiology and management generally, but does not address paediatric sepsis: age-specific diagnostic criteria (paediatric SIRS vs Sepsis-3 applicability), weight-based fluid resuscitation volumes, or paediatric-specific antibiotic dosing and target organ dysfunction thresholds.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: sepsis(105), infection(431). Paediatric-specific content absent."),

    # ══ PARTIAL — missing_recent_update (5) ═══════════════════════════════════

    ("What are the current WHO booster vaccination recommendations for COVID-19?",
     "partial", "missing_recent_update", "infectious_disease",
     "The corpus contains COVID-19 vaccination data from earlier pandemic phases but does not reflect current WHO recommendations for booster doses, bivalent vaccine formulations targeting Omicron subvariants, or updated priority-group guidance for boosting.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: covid(136), vaccination(161). Booster/bivalent guidance absent."),

    ("What do the latest ACC/AHA guidelines recommend for hypertension management?",
     "partial", "missing_recent_update", "cardiology",
     "The corpus discusses hypertension and cardiovascular risk extensively, but does not contain specific current ACC/AHA guideline thresholds (the 2017 reclassification to ≥130/80 mmHg for Stage 1 hypertension), updated treatment algorithms, or preferred first-line antihypertensive drug class recommendations.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hypertension(219). Current guideline specifics absent."),

    ("What immunotherapy combinations have recently been approved for non-small cell lung cancer?",
     "partial", "missing_recent_update", "oncology",
     "The corpus discusses immunotherapy in cancer broadly, but does not contain recently approved NSCLC regimens (e.g., pembrolizumab monotherapy for PD-L1 ≥50%, pembrolizumab + platinum-based chemotherapy, or atezolizumab + bevacizumab + chemotherapy) or their specific approval criteria.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: immunotherapy(41), cancer(274). NSCLC-specific approvals absent."),

    ("What are the updated DAA regimens for hepatitis C genotype 3 infection?",
     "partial", "missing_recent_update", "hepatology",
     "The corpus confirms DAAs are effective for HCV, but does not contain current genotype-3-specific recommendations (e.g., sofosbuvir/velpatasvir 12 weeks; or sofosbuvir/velpatasvir/voxilaprevir for NS5A-resistant strains) from updated treatment guidelines.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: hepatitis(88). Genotype-specific updated regimens absent."),

    ("What are the current CDC recommendations for latent tuberculosis preventive therapy?",
     "partial", "missing_recent_update", "infectious_disease",
     "The corpus discusses tuberculosis treatment broadly but does not contain the most recent CDC short-course preventive therapy regimens for latent TB infection (e.g., 1HP: weekly isoniazid + rifapentine for 12 weeks; 3HP regimen), which replaced the older 9-month isoniazid monotherapy as preferred options.",
     "acknowledge_gap", "gap_filling", 2,
     "Dense: tuberculosis(32). Short-course LTBI regimens absent."),

    # ══ AMBIGUOUS — underspecified (20) ═══════════════════════════════════════

    ("What is the dose of aspirin?",
     "ambiguous", "underspecified", "cardiology",
     "This question is underspecified. Aspirin dosing varies by indication: 75–100 mg daily for cardiovascular prevention, 300–325 mg for acute myocardial infarction or stroke, 500–1000 mg for analgesia or antipyresis. Without specifying the indication and patient profile, no single dose applies.",
     "present_options", "false_certainty", 2,
     "Classic underspecified dosing question — model should ask which indication."),

    ("What is the treatment for pneumonia?",
     "ambiguous", "underspecified", "pulmonology",
     "Treatment depends on multiple unspecified factors: severity (mild outpatient vs severe ICU), setting (community-acquired vs hospital-acquired), likely pathogen (bacterial vs viral vs atypical), and patient comorbidities. No single treatment regimen applies without these qualifiers.",
     "present_options", "false_certainty", 2,
     "Dense: pneumonia(49), antibiotic(45). Classic underspecified clinical question."),

    ("How long does hepatitis treatment take?",
     "ambiguous", "underspecified", "hepatology",
     "Duration varies entirely by hepatitis type: Hepatitis A is self-limiting and requires no antiviral treatment; chronic Hepatitis B may require lifelong suppressive therapy; Hepatitis C DAA regimens run 8–12 weeks. Without specifying the hepatitis type, a duration cannot be stated.",
     "present_options", "false_certainty", 2,
     "Dense: hepatitis(88). Type not specified — genuinely ambiguous."),

    ("What is the survival rate for cancer?",
     "ambiguous", "underspecified", "oncology",
     "Survival rates vary enormously by cancer type and stage: 5-year survival exceeds 90% for localised thyroid or prostate cancer but falls below 10% for metastatic pancreatic cancer. A single survival figure across all cancers is clinically meaningless without specifying type and stage.",
     "present_options", "false_certainty", 2,
     "Dense: cancer(274), tumor(223). Type and stage not specified."),

    ("What inhaler should a patient with breathing problems use?",
     "ambiguous", "underspecified", "pulmonology",
     "Inhaler selection requires specifying the diagnosis (asthma vs COPD), severity, drug class needed (SABA for rescue, ICS for controller, LABA/LAMA for maintenance), patient age, and inhaler technique ability (MDI vs DPI vs spacer). The question cannot be answered without a diagnosis and clinical context.",
     "present_options", "false_certainty", 2,
     "Inhaler(0 chunks) but question is genuinely underspecified — tests if model fabricates a specific inhaler or acknowledges ambiguity."),

    ("Is antibiotic treatment effective?",
     "ambiguous", "underspecified", "infectious_disease",
     "Antibiotic effectiveness depends on the pathogen (ineffective against viruses), specific bacterial species, susceptibility profile, antibiotic class, infection site, and patient immune status. A general yes or no is misleading without specifying the organism and clinical context.",
     "present_options", "false_certainty", 2,
     "Dense: antibiotic(45), bacterial(94). Pathogen/organism not specified."),

    ("When should antihypertensive medication be started?",
     "ambiguous", "underspecified", "cardiology",
     "Initiation thresholds depend on absolute blood pressure levels, total cardiovascular risk score, presence of target organ damage, and comorbidities such as diabetes or CKD. Stage 1 hypertension may be managed with lifestyle changes alone in low-risk patients; Stage 2 requires prompt pharmacological treatment regardless of risk.",
     "present_options", "false_certainty", 2,
     "Dense: hypertension(219). Risk stratification context not specified."),

    ("What diet should a patient with kidney disease follow?",
     "ambiguous", "underspecified", "nephrology",
     "Dietary recommendations depend on CKD stage, dialysis status, and individual metabolic abnormalities: protein restriction (pre-dialysis advanced CKD), potassium restriction (hyperkalaemia), phosphate restriction (hyperphosphataemia), sodium restriction, and fluid restriction (oliguric stages). Stage and dialysis status are unspecified.",
     "present_options", "false_certainty", 2,
     "Dense: kidney(89), renal(78). Stage and dialysis status unspecified."),

    ("Is chemotherapy safe?",
     "ambiguous", "underspecified", "oncology",
     "Safety depends on the specific agent, cumulative dose, patient organ function (renal, hepatic, cardiac reserve), age, and performance status. All chemotherapy carries risk — the question requires specifying the regimen and patient to have clinical meaning.",
     "present_options", "false_certainty", 2,
     "Dense: chemotherapy(70). Agent, patient context unspecified."),

    ("What foods should patients with liver disease avoid?",
     "ambiguous", "underspecified", "hepatology",
     "Dietary restrictions depend on the specific liver condition: cirrhosis requires sodium and often protein moderation; alcoholic liver disease requires complete alcohol cessation; fatty liver disease requires caloric and saturated fat reduction. The condition and severity are not specified.",
     "present_options", "false_certainty", 2,
     "Dense: liver(382), cirrhosis(59), steatosis(48). Condition unspecified."),

    ("What is the best treatment for heart disease?",
     "ambiguous", "underspecified", "cardiology",
     "Treatment varies entirely by heart disease type: coronary artery disease (statins, antiplatelet agents, PCI or CABG); heart failure (ACE inhibitors, beta-blockers, diuretics, device therapy); atrial fibrillation (rate or rhythm control, anticoagulation); valvular disease (repair or replacement). Type is not specified.",
     "present_options", "false_certainty", 2,
     "Dense: cardiovascular(219), heart.failure(71), coronary(82)."),

    ("How contagious is tuberculosis?",
     "ambiguous", "underspecified", "infectious_disease",
     "TB transmissibility depends on pulmonary vs extrapulmonary disease, sputum smear positivity, presence of cavitary lesions, ventilation of the exposure environment, and duration of contact. Without these variables, a contagiousness rating is not meaningful.",
     "present_options", "false_certainty", 2,
     "Dense: tuberculosis(32). Transmission context unspecified."),

    ("What is the role of steroids in infection management?",
     "ambiguous", "underspecified", "infectious_disease",
     "Corticosteroid use in infection is highly context-dependent: beneficial in severe COVID-19 (dexamethasone), TB meningitis, and Pneumocystis pneumonia in HIV-positive patients; potentially harmful in other contexts (worsens outcome in some bacterial infections). The infection type is not specified.",
     "present_options", "false_certainty", 2,
     "Dense: corticosteroid(46), infection(431), sepsis(105). Infection type unspecified."),

    ("What is the optimal blood glucose target for a diabetic patient?",
     "ambiguous", "underspecified", "cardiology",
     "Glucose targets depend on diabetes type, patient age, comorbidities (CKD, cardiovascular disease), hypoglycaemia risk, whether fasting or postprandial targets are discussed, and whether HbA1c or point-of-care glucose is being referenced. No single universal target exists.",
     "present_options", "false_certainty", 2,
     "Dense: diabetes(227), insulin(79), hba1c(31). Diabetes gap filled. Classic underspecified question."),

    ("Should a diabetic patient take insulin?",
     "ambiguous", "underspecified", "cardiology",
     "Insulin use depends on diabetes type (type 1 always requires insulin; type 2 may after oral agent failure), duration of diabetes, degree of beta-cell function remaining, and individual clinical factors. The corpus cannot give a yes/no answer without knowing the diabetes type, current HbA1c, and treatment history.",
     "present_options", "false_certainty", 2,
     "Dense: diabetes(227), insulin(79). Diabetes gap filled. Type and treatment stage unspecified."),

    ("What is the prognosis of liver cancer?",
     "ambiguous", "underspecified", "hepatology",
     "Prognosis depends on the type (hepatocellular carcinoma vs cholangiocarcinoma vs metastatic disease), stage at diagnosis (resectable vs locally advanced vs metastatic), underlying liver function (cirrhotic vs non-cirrhotic), and available treatments. Without these qualifiers, prognosis cannot be stated.",
     "present_options", "false_certainty", 2,
     "Dense: tumor(223), cancer(274), liver(382), cirrhosis(59)."),

    ("What is the treatment for liver failure?",
     "ambiguous", "underspecified", "hepatology",
     "Treatment depends on whether liver failure is acute (paracetamol toxicity: N-acetylcysteine; autoimmune: steroids; viral: antivirals; all: transplant evaluation) or chronic decompensation (diuretics for ascites, lactulose for encephalopathy, antibiotics for SBP, transplant listing). Aetiology and acuity are not specified.",
     "present_options", "false_certainty", 2,
     "Dense: liver(382), cirrhosis(59), hepatitis(88)."),

    ("How much exercise is beneficial for patients with cardiovascular disease?",
     "ambiguous", "underspecified", "cardiology",
     "Exercise recommendations for cardiovascular patients depend on the specific condition (stable angina vs recent MI vs heart failure vs arrhythmia), functional capacity (assessed by NYHA class or exercise testing), treatment status, and cardiac rehabilitation eligibility. No single prescription applies.",
     "present_options", "false_certainty", 2,
     "Dense: cardiovascular(219). Diagnosis and functional class unspecified."),

    ("Should cancer patients receive vaccinations?",
     "ambiguous", "underspecified", "oncology",
     "Vaccination decisions for cancer patients depend on cancer type, treatment phase (chemotherapy vs immunotherapy vs surveillance), immune status, and vaccine type — live vaccines are contraindicated in immunocompromised patients; inactivated vaccines are generally safe but may have reduced immunogenicity during chemotherapy.",
     "present_options", "false_certainty", 2,
     "Dense: vaccination(161), cancer(274), immunotherapy(41). Context not specified."),

    ("Can sepsis be prevented?",
     "ambiguous", "underspecified", "infectious_disease",
     "Sepsis prevention varies by setting and population: vaccination reduces infectious triggers; hand hygiene and care bundles reduce hospital-acquired sepsis; early antibiotic stewardship and source control reduce progression. Without specifying setting and patient population, multiple valid prevention strategies apply.",
     "present_options", "false_certainty", 2,
     "Dense: sepsis(105), infection(431). Setting and population unspecified."),

    # ══ UNANSWERABLE — in_domain_absent (20) ══════════════════════════════════

    ("What is the recommended insulin titration algorithm for type 1 diabetes using a continuous glucose monitor?",
     "unanswerable", "in_domain_absent", "cardiology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Dense: diabetes(227), insulin(79). CGM-based titration algorithms absent. Diabetes gap filled."),

    ("What are the diagnostic criteria for maturity-onset diabetes of the young (MODY)?",
     "unanswerable", "in_domain_absent", "cardiology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Dense: diabetes(227). MODY genetic diagnostic criteria absent. Diabetes gap filled."),

    ("What is the first-line treatment for osteoporosis-related vertebral compression fractures?",
     "unanswerable", "in_domain_absent", "orthopedics",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Absent: osteoporosis(2 chunks), bone.density(1). Orthopedics domain is THIN (37 total chunks)."),

    ("What are the indications for total knee arthroplasty in osteoarthritis?",
     "unanswerable", "in_domain_absent", "orthopedics",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Thin: orthopedic(7), joint(35), arthritis(36). Surgical criteria absent."),

    ("What are the stepwise management protocols for emphysema?",
     "unanswerable", "in_domain_absent", "pulmonology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Absent: emphysema(3 chunks). Pulmonology domain exists but emphysema management absent."),

    ("How should correct inhaler technique be taught to patients with obstructive lung disease?",
     "unanswerable", "in_domain_absent", "pulmonology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Absent: inhaler(0 chunks). Explicitly absent from corpus."),

    ("What are the induction chemotherapy regimens for acute myeloid leukaemia?",
     "unanswerable", "in_domain_absent", "oncology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Absent: leukemia(3 chunks). Oncology domain exists but AML induction absent."),

    ("What are the diagnostic criteria for chronic lymphocytic leukaemia?",
     "unanswerable", "in_domain_absent", "oncology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Absent: leukemia(3 chunks). CLL diagnostic criteria absent."),

    ("What ECG changes are characteristic of acute pericarditis?",
     "unanswerable", "in_domain_absent", "cardiology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Cardiology domain is dense but pericarditis ECG findings not in corpus."),

    ("What is the management strategy for hypertrophic obstructive cardiomyopathy?",
     "unanswerable", "in_domain_absent", "cardiology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Cardiology domain is dense but hypertrophic cardiomyopathy management absent."),

    ("What prophylaxis regimens are recommended for malaria prevention in travellers?",
     "unanswerable", "in_domain_absent", "infectious_disease",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Infectious disease domain is dense but malaria prophylaxis absent."),

    ("What is the treatment protocol for Clostridioides difficile colitis?",
     "unanswerable", "in_domain_absent", "infectious_disease",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Infectious disease domain exists but C. difficile treatment absent."),

    ("What are the diagnostic criteria for autoimmune hepatitis?",
     "unanswerable", "in_domain_absent", "hepatology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Hepatology domain exists but autoimmune hepatitis criteria absent."),

    ("What is the role of the TIPS procedure in managing variceal bleeding from portal hypertension?",
     "partial", "missing_specificity", "hepatology",
     "The corpus mentions the TIPS (transjugular intrahepatic portosystemic shunt) implantation technique — noting a transjugular approach with puncture needle — in the context of portal hypertension, but does not provide evidence-based guidance on when TIPS is indicated for variceal bleeding, its comparative efficacy versus endoscopic band ligation, patient selection criteria, or complication rates.",
     "acknowledge_gap", "gap_filling", 2,
     "Retrieval-validated: corpus has TIPS content (score=0.937) but only technical procedure description, not clinical indication guidelines."),

    ("What immunosuppression protocols are used after kidney transplantation?",
     "unanswerable", "in_domain_absent", "nephrology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Transplant(117) chunks exist but are liver transplant. Kidney transplant immunosuppression absent."),

    ("What are the diagnostic criteria for idiopathic pulmonary fibrosis?",
     "unanswerable", "in_domain_absent", "pulmonology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Dense: pulmonary(139), fibrosis(104). IPF diagnostic criteria absent."),

    ("What is the standard rehabilitation protocol after anterior cruciate ligament reconstruction?",
     "unanswerable", "in_domain_absent", "orthopedics",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Thin domain: orthopedics(37 total). ACL rehabilitation absent."),

    ("What is the recommended post-exposure prophylaxis protocol for rabies?",
     "unanswerable", "in_domain_absent", "infectious_disease",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Infectious disease domain exists but rabies PEP protocol absent."),

    ("What are the bronchitis antibiotic prescribing guidelines?",
     "unanswerable", "in_domain_absent", "pulmonology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Absent: bronchitis(1 chunk). Pulmonology domain exists but bronchitis guidelines absent."),

    ("What are the current device therapy guidelines for ventricular tachycardia?",
     "unanswerable", "in_domain_absent", "cardiology",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Cardiology domain is dense but ICD/device therapy for VT guidelines absent."),

    # ══ UNANSWERABLE — out_of_domain (10) ═════════════════════════════════════

    ("What are the first-line pharmacological treatments for major depressive disorder?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Psychiatry is entirely outside the 7 corpus domains."),

    ("What are the DSM-5 diagnostic criteria for schizophrenia?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Psychiatry/DSM-5 is entirely outside the 7 corpus domains."),

    ("What are the topical and systemic treatment options for moderate psoriasis?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Dermatology is entirely outside the 7 corpus domains."),

    ("What medications are used to treat attention deficit hyperactivity disorder (ADHD) in adults?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Psychiatry/neurodevelopmental disorders are outside the 7 corpus domains. Replaces glaucoma question (glaucoma AI papers found in corpus via retrieval validation)."),

    ("What are the indications for root canal treatment?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Dentistry is entirely outside the 7 corpus domains."),

    ("What anticonvulsant medications are first-line for newly diagnosed epilepsy?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Neurology/epilepsy is outside the 7 corpus domains."),

    ("What is the clinical staging system for Alzheimer's disease?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Neurology/dementia is outside the 7 corpus domains."),

    ("What are the standard pharmacological treatments for bipolar disorder?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Psychiatry is entirely outside the 7 corpus domains."),

    ("What are the treatment options for atopic dermatitis (eczema) in adults?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Dermatology is entirely outside the 7 corpus domains."),

    ("What medications are used to manage motor symptoms of Parkinson's disease?",
     "unanswerable", "out_of_domain", "cross_domain",
     "The provided context does not contain enough information to answer this question.",
     "refuse", "fabrication", 1,
     "Neurology/movement disorders is outside the 7 corpus domains."),
]


def main():
    print("=" * 70)
    print("  Phase 5 — Generate Option B Evaluation Question Set (v2)")
    print("=" * 70)
    print("""
  Fixes applied vs v1:
    - All 10 direct_lookup questions are now specific-answer (not yes/no)
    - 8 diabetes questions added across all 4 tiers
    - q_075 (kidney reversible) corrected to unanswerable
    - q_065 (inhaler) kept as ambiguous — genuinely underspecified,
      tests fabrication vs ambiguity acknowledgement
""")

    assert len(QUESTIONS) == 110, f"Expected 110 questions, got {len(QUESTIONS)}"

    EVAL_QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for i, (question, tier, sub_tier, domain, gold_answer,
            expected_behavior, hallucination_target, difficulty, notes) in enumerate(QUESTIONS, 1):
        records.append({
            "question_id":          f"q_{i:03d}",
            "question":             question,
            "tier":                 tier,
            "sub_tier":             sub_tier,
            "hallucination_target": hallucination_target,
            "gold_answer":          gold_answer,
            "gold_sources":         [],
            "expected_behavior":    expected_behavior,
            "domain":               domain,
            "notes":                notes,
            "annotated_on":         TODAY,
            "difficulty":           difficulty,
        })

    with open(EVAL_QUESTIONS, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    tier_counts   = Counter(r["tier"] for r in records)
    sub_counts    = Counter(r["sub_tier"] for r in records)
    domain_counts = Counter(r["domain"] for r in records)

    print(f"  Written {len(records)} questions → {EVAL_QUESTIONS}")

    print(f"\n{'─'*70}")
    print("  TIER DISTRIBUTION")
    print(f"{'─'*70}")
    for tier in ["answerable", "partial", "ambiguous", "unanswerable"]:
        print(f"  {tier:<14} {tier_counts[tier]:>3}")

    print(f"\n{'─'*70}")
    print("  SUB-TIER DISTRIBUTION")
    print(f"{'─'*70}")
    for sub, n in sorted(sub_counts.items(), key=lambda x: -x[1]):
        print(f"  {sub:<32} {n:>3}")

    print(f"\n{'─'*70}")
    print("  DOMAIN DISTRIBUTION")
    print(f"{'─'*70}")
    for dom, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {dom:<25} {n:>3}")

    diabetes_qs = [r for r in records if "diabetes" in r["notes"].lower() or
                   "hba1c" in r["notes"].lower() or "insulin" in r["notes"].lower()]
    print(f"\n  Diabetes questions: {len(diabetes_qs)} across all tiers")

    print(f"\n{'═'*70}")
    print("  NEXT STEPS")
    print(f"{'═'*70}")
    print("""
  1. Validate (no GPU needed):
       python scripts/validate_questions.py --no-retrieval

  2. Push to GitHub, then run on Kaggle:
       python scripts/run_inference.py

  3. After Kaggle run, pull results:
       python scripts/phase5_analysis.py
       python scripts/hallucination_report.py
""")


if __name__ == "__main__":
    main()
