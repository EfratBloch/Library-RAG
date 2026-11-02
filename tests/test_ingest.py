"""Checks for the data loading and chunking. They make no API calls.

Run from the project root: python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from library_rag.config import CATALOG_CSV, INFO_PDF, RULES_PDF
from library_rag.data_loader import load_book_documents, load_pdf_documents, parse_age_range
from library_rag.ingest import build_nodes


class IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_documents = load_pdf_documents()
        cls.book_documents = load_book_documents()
        cls.nodes = build_nodes(cls.pdf_documents, cls.book_documents)
        cls.pdf_text = "\n".join(document.text for document in cls.pdf_documents)

    def page_text(self, file_name, page):
        return next(
            document.text
            for document in self.pdf_documents
            if document.metadata["source_file"] == file_name and document.metadata["page"] == page
        )

    def test_scrambled_values_are_repaired(self):
        info = self.page_text(INFO_PDF, 2)
        self.assertIn("ימים א’, ב’, ה’ בשעות 09:00–12:00", info)
        self.assertIn("ימים א’, ג’, ד’, ה’ בשעות 16:00–18:45", info)
        self.assertIn("ברחוב לוטם 10 מעל", info)
        self.assertIn("library@harish.muni.il", info)
        rules = self.page_text(RULES_PDF, 2)
        self.assertIn("לשאול/להחזיק 4 ספרים בו בזמן", rules)
        self.assertIn("משך שאילת כל ספר הוא 30 ימים", rules)
        self.assertIn("פחות מ-18 תתאפשר", rules)

    def test_website_noise_is_removed(self):
        self.assertNotIn("«", self.pdf_text)
        self.assertNotIn("הצהרת נגישות", self.pdf_text)
        self.assertNotIn("הצהרת מדיניות", self.pdf_text)
        self.assertFalse(any("" <= char <= "" for char in self.pdf_text))

    def test_pages_without_content_are_skipped(self):
        pages = {(d.metadata["source_file"], d.metadata["page"]) for d in self.pdf_documents}
        self.assertNotIn((INFO_PDF, 3), pages)
        self.assertNotIn((RULES_PDF, 4), pages)

    def test_sentence_across_pages_stays_together(self):
        first_page = self.page_text(RULES_PDF, 1)
        self.assertIn("תביא להרחקה מהספרייה", first_page)
        self.assertFalse(self.page_text(RULES_PDF, 2).startswith("תביא להרחקה"))

    def test_catalog_has_one_node_per_book(self):
        book_nodes = [node for node in self.nodes if node.metadata["doc_type"] == "book"]
        self.assertEqual(len(book_nodes), 100)
        self.assertEqual(len({node.metadata["book_id"] for node in book_nodes}), 100)
        for node in book_nodes:
            self.assertEqual(node.metadata["source_file"], CATALOG_CSV)
            self.assertIn(node.metadata["title"], node.text)
            self.assertEqual(node.text.count("שם הספר:"), 1)

    def test_pdf_nodes_keep_source_metadata(self):
        pdf_nodes = [node for node in self.nodes if node.metadata["doc_type"] != "book"]
        self.assertTrue(pdf_nodes)
        for node in pdf_nodes:
            self.assertIn(node.metadata["source_file"], (INFO_PDF, RULES_PDF))
            self.assertIsInstance(node.metadata["page"], int)

    def chunks_containing(self, text):
        return [node for node in self.nodes if text in node.text]

    def test_rules_stay_with_their_heading(self):
        for text, heading in (
            ("זכאים להירשם לספרייה ללא עלות", "ב. הרשמה לספרייה"),
            ("המינוי ייפתח לאחר 20 ימים", "ז. איחור בהחזרת ספר מושאל - איחור שני"),
            ("המינוי ייפתח לאחר 5 ימים", "ז. איחור בהחזרת ספר מושאל - איחור ראשון"),
            ("משך שאילת כל ספר הוא 30 ימים", "ג. שאילת ספרים"),
            ("בשעות 16:00–18:45", "שעות קבלת קהל:"),
        ):
            chunks = self.chunks_containing(text)
            self.assertTrue(chunks, text)
            for node in chunks:
                self.assertTrue(node.text.startswith(heading), node.node_id)

    def test_section_continues_across_a_page_break(self):
        chunks = self.chunks_containing("חל איסור לעשן בכל שטח הספרייה")
        self.assertEqual(chunks[0].metadata["page"], 2)
        self.assertEqual(chunks[0].metadata["section"], "א. כללי התנהגות")

    def test_sections_do_not_mix_in_a_chunk(self):
        for node in self.nodes:
            if node.metadata["doc_type"] == "library_rules":
                headings = [line for line in node.text.split("\n")[1:] if line[:2] in ("א.", "ב.", "ג.", "ד.", "ה.", "ו.", "ז.", "ח.", "ט.")]
                self.assertEqual(headings, [], node.node_id)

    def test_node_ids_are_unique(self):
        self.assertEqual(len({node.node_id for node in self.nodes}), len(self.nodes))

    def test_parse_age_range(self):
        self.assertEqual(parse_age_range("8–10"), (8, 10))
        self.assertEqual(parse_age_range("14-16"), (14, 16))
        with self.assertRaises(ValueError):
            parse_age_range("מבוגרים")


if __name__ == "__main__":
    unittest.main()
