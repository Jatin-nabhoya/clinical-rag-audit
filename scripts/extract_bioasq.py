import json, os

# Load the full BioASQ dataset
with open("data/raw/bioasq/BioASQ-training13b.json", "r") as f:
    data = json.load(f)

questions = data["questions"]
print(f"Total BioASQ questions: {len(questions)}")

# Filter for diabetes-related
keywords = ["diabetes", "diabetic", "glucose", "insulin", "metformin",
            "a1c", "hba1c", "hyperglycemia", "hypoglycemia",
            "blood sugar", "type 2 diabetes", "t2dm"]

diabetes_qs = []
for q in questions:
    body = q.get("body", "").lower()
    if any(k in body for k in keywords):
        diabetes_qs.append({
            "question": q.get("body"),
            "type": q.get("type"),  # yesno, factoid, list, summary
            "ideal_answer": q.get("ideal_answer", ""),
            "exact_answer": q.get("exact_answer", []),
            "snippets": [s.get("text", "") for s in q.get("snippets", [])],
            "bioasq_id": q.get("id")
        })

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/bioasq_diabetes_qa.json", "w") as f:
    json.dump(diabetes_qs, f, indent=2)

# Summary
print(f"Diabetes-related questions: {len(diabetes_qs)}")
from collections import Counter
types = Counter(q["type"] for q in diabetes_qs)
print(f"By type: {dict(types)}")