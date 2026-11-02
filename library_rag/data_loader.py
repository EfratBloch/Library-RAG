"""Read the library PDFs and the book catalog into LlamaIndex Documents."""

import re

import pandas as pd
from llama_index.core import Document
from pypdf import PdfReader

from library_rag.config import CATALOG_CSV, DATA_DIR, INFO_PDF, RULES_PDF
from library_rag.pdf_fixes import BREADCRUMB_MARK, LINE_FIXES, NOISE_LINES

PDF_DOC_TYPES = {INFO_PDF: "library_info", RULES_PDF: "library_rules"}
CATALOG_COLUMNS = [
    "מזהה_ספר",
    "שם_הספר",
    "שם_המחבר",
    "קטגוריה",
    "גיל_מומלץ",
    "תקציר",
    "שנת_פרסום",
    "סטטוס_נתונים",
    "שייכות_לספרייה",
]

PRIVATE_USE_CHARS = re.compile("[\ue000-\uf8ff]")
MIRRORED_PARENS = re.compile(r"\)([^()]*)\(")
MID_SENTENCE_ENDINGS = ("–", ",")


def clean_page_lines(raw_text, fixes, used_fixes):
    """Remove website noise from one page and repair its scrambled lines."""
    lines = []
    for raw_line in raw_text.split("\n"):
        line = PRIVATE_USE_CHARS.sub("", raw_line).strip()
        if not line or line in NOISE_LINES or BREADCRUMB_MARK in line:
            continue
        if line in fixes:
            used_fixes.add(line)
            line = fixes[line]
        lines.append(MIRRORED_PARENS.sub(r"(\1)", line))
    return lines


def move_split_sentences(pages):
    """Keep a sentence that crosses a page break together on its first page."""
    for current, following in zip(pages, pages[1:]):
        while current and following and current[-1].endswith(MID_SENTENCE_ENDINGS):
            current.append(following.pop(0))
            while following and not current[-1].endswith((".", ":", "!")):
                current.append(following.pop(0))


def load_pdf_documents(data_dir=DATA_DIR):
    """Return one Document per PDF page that has real content."""
    documents = []
    for file_name, doc_type in PDF_DOC_TYPES.items():
        fixes = LINE_FIXES[file_name]
        used_fixes = set()
        reader = PdfReader(data_dir / file_name)
        pages = [
            clean_page_lines(page.extract_text() or "", fixes, used_fixes)
            for page in reader.pages
        ]
        unused_fixes = set(fixes) - used_fixes
        if unused_fixes:
            raise ValueError(
                f"{file_name}: {len(unused_fixes)} text corrections no longer match the "
                "PDF. The file probably changed, so review LINE_FIXES."
            )
        move_split_sentences(pages)

        for page_number, lines in enumerate(pages, start=1):
            if lines:
                documents.append(
                    Document(
                        text="\n".join(lines),
                        metadata={
                            "source_file": file_name,
                            "doc_type": doc_type,
                            "page": page_number,
                        },
                    )
                )
    return documents


def parse_age_range(age_text):
    """Turn a catalog age such as '8–10' into its numeric limits."""
    match = re.fullmatch(r"\s*(\d+)\s*[–-]\s*(\d+)\s*", str(age_text))
    if not match:
        raise ValueError(f"Unrecognized recommended age: {age_text!r}")
    return int(match.group(1)), int(match.group(2))


def load_book_documents(data_dir=DATA_DIR):
    """Return one Document per catalog record, with structured metadata for filtering."""
    catalog = pd.read_csv(data_dir / CATALOG_CSV, encoding="utf-8-sig")
    missing_columns = set(CATALOG_COLUMNS) - set(catalog.columns)
    if missing_columns:
        raise ValueError(f"{CATALOG_CSV}: missing columns {sorted(missing_columns)}")

    documents = []
    for _, row in catalog.iterrows():
        age_min, age_max = parse_age_range(row["גיל_מומלץ"])
        text = "\n".join(
            [
                f"שם הספר: {row['שם_הספר']}",
                f"מחבר: {row['שם_המחבר']}",
                f"קטגוריה: {row['קטגוריה']}",
                f"גיל מומלץ: {row['גיל_מומלץ']}",
                f"שנת פרסום: {row['שנת_פרסום']}",
                f"תקציר: {row['תקציר']}",
            ]
        )
        documents.append(
            Document(
                text=text,
                metadata={
                    "source_file": CATALOG_CSV,
                    "doc_type": "book",
                    "book_id": row["מזהה_ספר"],
                    "title": row["שם_הספר"],
                    "author": row["שם_המחבר"],
                    "category": row["קטגוריה"],
                    "age_range": row["גיל_מומלץ"],
                    "age_min": age_min,
                    "age_max": age_max,
                    "year": int(row["שנת_פרסום"]),
                    "data_status": row["סטטוס_נתונים"],
                    "library_affiliation": row["שייכות_לספרייה"],
                },
            )
        )
    return documents
