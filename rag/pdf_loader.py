import PyPDF2
from pathlib import Path

def load_pdfs(folder_path):
    folder = Path(folder_path)
    documents = []

    for pdf_path in folder.glob("*.pdf"):
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        documents.append({"file": pdf_path.name, "text": text})

    return documents
