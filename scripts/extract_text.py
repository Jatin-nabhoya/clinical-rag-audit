# Step 1: Extract Text from HTML / XML / JSON

from bs4 import BeautifulSoup
from pathlib import Path
import json
import xml.etree.ElementTree as ET


def extract_html(path: str) -> str:
    """Extract clean text from CDC/WHO HTML files."""
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_xml(path: str) -> str:
    """Extract body text from PMC XML files."""
    tree = ET.parse(path)
    root = tree.getroot()
    texts = []
    # Pull title
    for t in root.iter("article-title"):
        if t.text:
            texts.append(t.text.strip())
    # Pull abstract paragraphs
    for t in root.iter("abstract"):
        for p in t.iter("p"):
            if p.text:
                texts.append(p.text.strip())
    # Pull body paragraphs
    for t in root.iter("body"):
        for p in t.iter("p"):
            text = "".join(p.itertext()).strip()
            if text:
                texts.append(text)
    return "\n".join(texts)


def extract_json_medlineplus(path: str) -> str:
    """Extract text from MedlinePlus JSON files."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    # MedlinePlus Connect response structure
    feed = data.get("feed", {})
    for entry in feed.get("entry", []):
        title = entry.get("title", {})
        if isinstance(title, dict):
            texts.append(title.get("_value", ""))
        summary = entry.get("summary", {})
        if isinstance(summary, dict):
            raw_html = summary.get("_value", "")
            soup = BeautifulSoup(raw_html, "html.parser")
            texts.append(soup.get_text(separator="\n", strip=True))
    return "\n".join(texts)


if __name__ == "__main__":
    raw = Path("data/raw")

    # --- HTML: CDC / WHO ---
    html_files = list((raw / "cdc_who").glob("*.html"))
    print(f"\n=== HTML files ({len(html_files)}) ===")
    for p in html_files:
        text = extract_html(str(p))
        print(f"  {p.name}: {len(text)} chars, ~{len(text.split())} words")

    # --- XML: PMC articles ---
    xml_files = list((raw / "pmc").glob("*.xml"))
    print(f"\n=== XML files ({len(xml_files)}) ===")
    for p in xml_files[:3]:  # preview first 3
        text = extract_xml(str(p))
        print(f"  {p.name}: {len(text)} chars, ~{len(text.split())} words")
    if len(xml_files) > 3:
        print(f"  ... and {len(xml_files) - 3} more")

    # --- JSON: MedlinePlus ---
    json_files = list((raw / "medlineplus").glob("*.json"))
    print(f"\n=== JSON files ({len(json_files)}) ===")
    for p in json_files:
        text = extract_json_medlineplus(str(p))
        print(f"  {p.name}: {len(text)} chars, ~{len(text.split())} words")
