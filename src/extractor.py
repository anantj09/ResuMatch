"""
src/extractor.py

Extract raw text from PDF, DOCX, or plain-text inputs.
"""

import io
import fitz
import docx


def extract_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file given its raw bytes."""
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()


def extract_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file given its raw bytes."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text(uploaded_file) -> str:
    """
    Dispatch extraction based on Streamlit UploadedFile type.
    Supports .pdf, .docx, and .txt.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".pdf"):
        return extract_from_pdf(raw)
    elif name.endswith(".docx"):
        return extract_from_docx(raw)
    elif name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore").strip()
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")
