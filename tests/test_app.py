"""Checks for the Streamlit screen. Gemini is replaced by fakes, so they make no API calls.

Run from the project root: python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st
from google.genai.errors import APIError
from streamlit.testing.v1 import AppTest

from library_rag.citations import Citation
from library_rag.rag import Answer, IndexNotReadyError

APP = str(PROJECT_DIR / "app.py")
PDF_CITATION = Citation(
    kind="pdf",
    node_id="node",
    score=0.8,
    title="libarary_rules.pdf",
    excerpt="משך שאילת כל ספר הוא 30 ימים.",
    source_file="libarary_rules.pdf",
    page=2,
)


def ask(question, answer=None, error=None, load_error=None):
    """Run the screen with a typed question and fake backend results."""
    st.cache_resource.clear()
    with (
        patch("library_rag.rag.load_index", return_value="index", side_effect=load_error),
        patch("library_rag.rag.get_llm", return_value="llm"),
        patch("library_rag.rag.answer_question", return_value=answer, side_effect=error),
    ):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.text_input[0].set_value(question)
        app.button[0].click()
        return app.run()


class AppTests(unittest.TestCase):
    def test_title_and_controls_are_shown(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "עוזר הספרייה החכם")
        self.assertEqual(len(app.text_input), 1)

    def test_answer_and_sources_are_shown(self):
        app = ask("כמה זמן?", answer=Answer("30 ימים.", True, [PDF_CITATION]))
        self.assertFalse(app.exception)
        text = " ".join(item.value for item in app.markdown)
        self.assertIn("30 ימים.", text)
        self.assertIn("libarary_rules.pdf", text)
        self.assertIn("עמוד 2", text)

    def test_unsupported_answer_is_a_warning_without_sources(self):
        app = ask("מזג אוויר?", answer=Answer("לא נמצא מידע.", False))
        self.assertEqual(app.warning[0].value, "לא נמצא מידע.")
        self.assertEqual(len(app.expander), 0)

    def test_empty_question_asks_for_text(self):
        app = ask("   ", answer=Answer("לא אמור להופיע", True))
        self.assertIn("יש לכתוב שאלה", app.warning[0].value)

    def test_quota_error_is_explained(self):
        error = APIError(429, {"error": {"message": "secret detail", "status": "RESOURCE_EXHAUSTED"}})
        app = ask("שאלה", error=error)
        self.assertIn("מכסת", app.error[0].value)
        self.assertNotIn("secret detail", app.error[0].value)

    def test_missing_index_is_explained(self):
        app = ask("שאלה", load_error=IndexNotReadyError("missing"))
        self.assertIn("python -m library_rag.ingest", app.error[0].value)

    def test_connection_error_is_explained(self):
        app = ask("שאלה", error=ConnectionError("no network"))
        self.assertIn("החיבור ל-Gemini נכשל", app.error[0].value)

    def test_internal_error_is_not_reported_as_a_network_problem(self):
        app = ask("שאלה", error=ValueError("a bug"))
        self.assertIn("תקלה פנימית", app.error[0].value)
        self.assertNotIn("נכשל", app.error[0].value)

    def test_index_is_opened_again_after_a_rebuild(self):
        st.cache_resource.clear()
        with (
            patch("library_rag.rag.load_index", return_value="index") as load_index,
            patch("library_rag.rag.get_llm", return_value="llm"),
            patch("library_rag.rag.answer_question", return_value=Answer("תשובה", True)),
        ):
            for version in (1, 1, 2):
                with patch("library_rag.rag.index_version", return_value=version):
                    app = AppTest.from_file(APP, default_timeout=30).run()
                    app.text_input[0].set_value("שאלה")
                    app.button[0].click()
                    app.run()
        self.assertEqual(load_index.call_count, 2)


if __name__ == "__main__":
    unittest.main()
