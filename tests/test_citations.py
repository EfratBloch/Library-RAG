"""Checks for building citations from retrieved nodes. They make no API calls.

Run from the project root: python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core.schema import NodeWithScore, TextNode

from library_rag.citations import build_citations


def retrieved(text, metadata, node_id, score=0.8):
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=metadata), score=score)


class CitationTests(unittest.TestCase):
    def test_pdf_citation_keeps_file_page_and_excerpt(self):
        source = retrieved(
            "משך שאילת כל ספר הוא 30 ימים.",
            {"source_file": "libarary_rules.pdf", "doc_type": "library_rules", "page": 2},
            "libarary_rules.pdf-p2-c2",
        )
        citation = build_citations([source])[0]
        self.assertEqual(citation.kind, "pdf")
        self.assertEqual(citation.source_file, "libarary_rules.pdf")
        self.assertEqual(citation.page, 2)
        self.assertEqual(citation.excerpt, "משך שאילת כל ספר הוא 30 ימים.")

    def test_book_citation_keeps_title_id_and_catalog_details(self):
        source = retrieved(
            "שם הספר: ספר בדיקה\nתקציר: תקציר הבדיקה.",
            {
                "source_file": "books_catalog_demo_100.csv",
                "doc_type": "book",
                "book_id": "DEMO-999",
                "title": "ספר בדיקה",
                "author": "מחבר בדיקה",
                "category": "מדע",
                "age_range": "8–10",
                "year": 2020,
            },
            "DEMO-999",
        )
        citation = build_citations([source])[0]
        self.assertEqual(citation.kind, "book")
        self.assertEqual(citation.title, "ספר בדיקה")
        self.assertEqual(citation.book_id, "DEMO-999")
        self.assertEqual(citation.excerpt, "תקציר הבדיקה.")
        self.assertEqual(citation.details["קטגוריה"], "מדע")
        self.assertEqual(citation.details["גיל מומלץ"], "8–10")

    def test_retrieval_order_is_kept(self):
        metadata = {"source_file": "libarary_info.pdf", "doc_type": "library_info", "page": 1}
        sources = [retrieved("א", metadata, "first", 0.9), retrieved("ב", metadata, "second", 0.7)]
        self.assertEqual([c.node_id for c in build_citations(sources)], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
