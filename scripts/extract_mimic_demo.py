"""
extract_mimic_demo.py — build readable clinical summaries from MIMIC-IV Demo
structured tables (no free-text notes in the demo package).

Joins: diagnoses_icd → admissions → prescriptions → labevents
Filters to diabetes patients, writes one text summary per admission.

Usage:
    python scripts/extract_mimic_demo.py

Output:
    data/processed/mimic_diabetes_notes.csv   — one row per admission
    data/raw/mimic_demo/summaries/            — one .txt per admission
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from log_metadata import append_row  # noqa: E402

MIMIC_DIR = ROOT / "data" / "raw" / "mimic_demo" / "mimic-iv-clinical-database-demo-2.2"
HOSP = MIMIC_DIR / "hosp"
OUT_CSV = ROOT / "data" / "processed" / "mimic_diabetes_notes.csv"
OUT_TXT = ROOT / "data" / "raw" / "mimic_demo" / "summaries"

# ICD-10 diabetes prefixes (also catches ICD-9 250.x)
DIABETES_PREFIXES = ("E10", "E11", "E12", "E13", "E14", "250")

DIABETES_MEDS = {
    "metformin", "insulin", "glipizide", "glimepiride", "glyburide",
    "sitagliptin", "linagliptin", "empagliflozin", "dapagliflozin",
    "liraglutide", "semaglutide", "pioglitazone", "lantus", "humalog",
    "novolog", "levemir", "toujeo", "tresiba", "basaglar",
}

DIABETES_LABS = {
    "glucose", "hba1c", "hemoglobin a1c", "glycosylated",
    "c-peptide", "insulin level",
}


def load(table: str) -> pd.DataFrame:
    gz = HOSP / f"{table}.csv.gz"
    csv = HOSP / f"{table}.csv"
    path = gz if gz.exists() else csv
    print(f"  loading {path.name} ...", end=" ", flush=True)
    df = pd.read_csv(path, compression="gzip" if str(path).endswith(".gz") else None,
                     low_memory=False)
    print(f"{len(df):,} rows")
    return df


def is_diabetes_icd(code: str) -> bool:
    if pd.isna(code):
        return False
    return any(str(code).upper().startswith(p) for p in DIABETES_PREFIXES)


def build_summary(hadm_id, adm_row, diag_rows, rx_rows, lab_rows) -> str:
    lines = []
    lines.append(f"CLINICAL SUMMARY — Admission ID: {hadm_id}")
    lines.append(f"Admission type : {adm_row.get('admission_type', 'N/A')}")
    lines.append(f"Admission location: {adm_row.get('admission_location', 'N/A')}")
    lines.append(f"Discharge location: {adm_row.get('discharge_location', 'N/A')}")
    lines.append(f"Insurance : {adm_row.get('insurance', 'N/A')}")
    lines.append(f"Language  : {adm_row.get('language', 'N/A')}")
    lines.append(f"Marital status : {adm_row.get('marital_status', 'N/A')}")
    lines.append("")

    if len(diag_rows):
        lines.append("DIAGNOSES (ICD):")
        for _, r in diag_rows.iterrows():
            flag = " ← DIABETES" if is_diabetes_icd(str(r.get("icd_code", ""))) else ""
            lines.append(f"  [{r.get('icd_version','?')}] {r.get('icd_code','')} — "
                         f"{r.get('long_title', r.get('icd_code',''))}{flag}")
    lines.append("")

    if len(rx_rows):
        lines.append("PRESCRIPTIONS:")
        seen = set()
        for _, r in rx_rows.iterrows():
            drug = str(r.get("drug", "")).strip()
            if drug and drug not in seen:
                seen.add(drug)
                dose = r.get("dose_val_rx", "")
                unit = r.get("dose_unit_rx", "")
                route = r.get("route", "")
                lines.append(f"  {drug} {dose}{unit} {route}".strip())
    lines.append("")

    if len(lab_rows):
        lines.append("LAB RESULTS (selected):")
        for _, r in lab_rows.iterrows():
            lines.append(f"  {r.get('label','')}: {r.get('value','')} "
                         f"{r.get('valueuom','')} (ref: {r.get('ref_range_lower','?')}–"
                         f"{r.get('ref_range_upper','?')})")
    return "\n".join(lines)


def main():
    if not HOSP.exists():
        sys.exit(f"[error] MIMIC hosp directory not found: {HOSP}\n"
                 "Make sure you unzipped the MIMIC-IV demo into data/raw/mimic_demo/")

    print("[mimic] loading tables ...")
    admissions = load("admissions")
    diagnoses  = load("diagnoses_icd")
    icd_desc   = load("d_icd_diagnoses")
    prescripts = load("prescriptions")
    labevents  = load("labevents")
    lab_items  = load("d_labitems")

    # Merge ICD descriptions
    diagnoses = diagnoses.merge(
        icd_desc[["icd_code", "icd_version", "long_title"]],
        on=["icd_code", "icd_version"], how="left"
    )

    # Find admissions with a diabetes diagnosis
    diabetes_hadm = set(
        diagnoses[diagnoses["icd_code"].apply(is_diabetes_icd)]["hadm_id"]
    )
    print(f"\n[mimic] admissions with diabetes ICD: {len(diabetes_hadm)}")

    # Merge lab item names
    labevents = labevents.merge(
        lab_items[["itemid", "label", "ref_range_lower", "ref_range_upper"]],
        on="itemid", how="left"
    )
    # Keep only diabetes-relevant labs
    labevents = labevents[
        labevents["label"].str.lower().str.contains(
            "|".join(DIABETES_LABS), na=False
        )
    ]

    OUT_TXT.mkdir(parents=True, exist_ok=True)
    records = []

    for hadm_id in sorted(diabetes_hadm):
        adm = admissions[admissions["hadm_id"] == hadm_id]
        if adm.empty:
            continue
        adm_row = adm.iloc[0].to_dict()

        diag_rows = diagnoses[diagnoses["hadm_id"] == hadm_id]
        rx_rows   = prescripts[prescripts["hadm_id"] == hadm_id]
        lab_rows  = labevents[labevents["hadm_id"] == hadm_id]

        summary = build_summary(hadm_id, adm_row, diag_rows, rx_rows, lab_rows)

        txt_path = OUT_TXT / f"hadm_{hadm_id}.txt"
        txt_path.write_text(summary, encoding="utf-8")

        records.append({
            "hadm_id": hadm_id,
            "subject_id": adm_row.get("subject_id", ""),
            "admission_type": adm_row.get("admission_type", ""),
            "n_diagnoses": len(diag_rows),
            "n_prescriptions": len(rx_rows),
            "n_labs": len(lab_rows),
            "text": summary,
        })

        append_row({
            "doc_id": f"mimic_{hadm_id}",
            "source": "mimic_demo",
            "title": f"Clinical Summary — Admission {hadm_id}",
            "url": "",
            "download_date": date.today().isoformat(),
            "publication_date": "",
            "license": "PhysioNet-CDHL-1.5.0",
            "file_path": str(txt_path.relative_to(ROOT)),
            "format": "txt",
            "domain_tag": "diabetes",
        })

    result = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    print(f"\n[mimic] done.")
    print(f"  diabetes admissions : {len(result)}")
    print(f"  summaries written   : {OUT_TXT}")
    print(f"  processed CSV       : {OUT_CSV}")

    if len(result):
        print("\n--- Sample summary (first admission) ---")
        print(result.iloc[0]["text"][:800])
        print("...")


if __name__ == "__main__":
    main()