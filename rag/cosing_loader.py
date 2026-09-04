"""
Loader for the EU CosIng cosmetic ingredient/fragrance inventory.

CosIng (Cosmetic Ingredient Database) is the European Commission's open
database of cosmetic substances used in the EU, covering INCI names,
CAS/EINECS identifiers, declared function(s), regulatory restrictions and a
short description for each ingredient. This mirrors the CSV published by the
Open Beauty Facts project (which republishes the official EC export), turning
~24k ingredient rows into short RAG-ready documents that sit alongside the
PDF-derived chunks from `pdf_loader.load_pdfs`.

Source: https://github.com/openfoodfacts/openbeautyfacts/tree/develop/cosing
"""

import csv
from pathlib import Path


def _clean(value):
    if value is None:
        return ""
    value = " ".join(value.split())
    return value.strip().strip("-").strip()


def load_cosing(csv_path, max_ingredients=None):
    """Load CosIng ingredient rows into documents shaped like `load_pdfs` output.

    Each ingredient becomes one document: {"file": ..., "text": ...}. Rows with
    no function and no description are skipped since they carry no answerable
    content, only an INCI name.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"CosIng CSV not found at {csv_path}. Skipping ingredient database.")
        return []

    documents = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header_seen = False
            for row in reader:
                if not row:
                    continue

                if not header_seen:
                    if row[0].strip() == "COSING Ref No":
                        header_seen = True
                    continue

                if len(row) < 9:
                    continue

                inci = _clean(row[1])
                cas = _clean(row[4])
                einecs = _clean(row[5])
                description = _clean(row[6])
                restriction = _clean(row[7])
                function = _clean(row[8])

                if not inci or (not description and not function):
                    continue

                parts = [f"INCI name: {inci}."]
                if function:
                    parts.append(f"Function(s): {function}.")
                if cas:
                    parts.append(f"CAS No: {cas}.")
                if einecs:
                    parts.append(f"EINECS/ELINCS No: {einecs}.")
                if restriction:
                    parts.append(f"Restriction: {restriction}.")
                if description:
                    parts.append(f"Description: {description}")

                text = " ".join(parts)
                if len(text.split()) < 8:
                    continue

                documents.append({
                    "file": f"CosIng:{inci}",
                    "text": text,
                    "word_count": len(text.split()),
                })

                if max_ingredients and len(documents) >= max_ingredients:
                    break
    except Exception as e:
        print(f"Error reading CosIng CSV {csv_path}: {e}. Skipping ingredient database.")
        return []

    print(f"Loaded {len(documents)} CosIng ingredient records from {path.name}.")
    return documents
