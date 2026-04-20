import os, json, glob
from lxml import etree

qa_pairs = []

# Walk through all XML files in all subfolders
for xml_file in glob.glob("data/raw/medquad/**/*.xml", recursive=True):
    try:
        tree = etree.parse(xml_file)
        root = tree.getroot()

        # Get the topic focus
        focus = root.findtext(".//Focus", default="").lower()

        # Filter for diabetes-related topics
        keywords = ["diabetes", "diabetic", "glucose", "insulin", "a1c",
                     "hemoglobin a1c", "hyperglycemia", "hypoglycemia",
                     "metformin", "blood sugar"]

        if any(k in focus for k in keywords):
            source_folder = os.path.basename(os.path.dirname(xml_file))
            for qa in root.findall(".//QAPair"):
                question = qa.findtext("Question", default="").strip()
                answer = qa.findtext("Answer", default="").strip()

                if question:  # some pairs have questions but no answers — keep them
                    qa_pairs.append({
                        "question": question,
                        "answer": answer if answer else "[ANSWER REMOVED - COPYRIGHT]",
                        "focus": focus,
                        "source_collection": source_folder,
                        "file": os.path.basename(xml_file),
                        "has_answer": bool(answer)
                    })
    except Exception as e:
        print(f"Skipped {xml_file}: {e}")

# Save
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/medquad_diabetes_qa.json", "w") as f:
    json.dump(qa_pairs, f, indent=2)

# Summary
total = len(qa_pairs)
with_answers = sum(1 for q in qa_pairs if q["has_answer"])
print(f"Total diabetes Q&A pairs: {total}")
print(f"With answers: {with_answers}")
print(f"Without answers (questions only): {total - with_answers}")
print(f"Source folders: {set(q['source_collection'] for q in qa_pairs)}")