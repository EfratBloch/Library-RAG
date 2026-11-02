"""Checks for reading the catalog conditions from a question. They make no API calls.

Run from the project root: python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from library_rag.config import CATALOG_AGE_RANGES_TEXT
from library_rag.data_loader import load_book_documents
from library_rag.query_parser import build_filters, parse_book_search
from library_rag.rag import NETWORK_ATTEMPTS, NO_MATCHING_BOOKS_MESSAGE, answer_question


class ParseBookSearchTests(unittest.TestCase):
    def test_age_and_category_are_detected(self):
        search = parse_book_search("תמליץ לי על ספר הרפתקאות לילד בן 9")
        self.assertTrue(search.is_book_search)
        self.assertEqual(search.age, 9)
        self.assertEqual(search.category, "הרפתקאות")

    def test_age_with_prefix_is_detected(self):
        self.assertEqual(parse_book_search("אני מחפשת ספר לגיל 15").age, 15)

    def test_questions_about_rules_are_not_book_searches(self):
        for question in (
            "כמה ספרים מותר לשאול בו זמנית?",
            "מה קורה אם מאחרים בהחזרת ספר?",
            "איך מזמינים ספר?",
            "ילד בן 5 יכול להגיע לספרייה לבד?",
            "מה עושים אם ספר אבד?",
            "האפשר להיכנס עם משפחה?",
        ):
            self.assertFalse(parse_book_search(question).is_book_search, question)

    def test_adults_are_a_book_search_above_the_catalog_ages(self):
        for question in ("איזה ספרים מתאימים למבוגרים?", "איזה ספרים יש עבור מבוגרים?"):
            search = parse_book_search(question)
            self.assertTrue(search.is_book_search, question)
            self.assertEqual(search.age, 18, question)

    def test_adults_in_a_rules_question_are_not_a_book_search(self):
        self.assertFalse(parse_book_search("ילד יכול להגיע עם מבוגר?").is_book_search)
        self.assertFalse(parse_book_search("מבוגרים יכולים לשאול ספרים?").is_book_search)

    def test_age_ranges_in_the_message_match_the_catalog(self):
        ranges = {document.metadata["age_range"] for document in load_book_documents()}
        for age_range in ranges:
            self.assertIn(age_range, CATALOG_AGE_RANGES_TEXT)
        self.assertEqual(len(ranges), 3)

    def test_filters_require_book_age_and_category(self):
        filters = build_filters(parse_book_search("ספר בלשים לגיל 15"))
        conditions = {(item.key, item.operator.value, item.value) for item in filters.filters}
        self.assertEqual(
            conditions,
            {
                ("doc_type", "==", "book"),
                ("age_min", "<=", 15),
                ("age_max", ">=", 15),
                ("category", "==", "בלשים"),
            },
        )

    def test_library_question_has_no_filters(self):
        self.assertIsNone(build_filters(parse_book_search("מה שעות הפתיחה?")))


class NetworkRetryTests(unittest.TestCase):
    def test_brief_network_failure_is_retried(self):
        failure = httpx.ConnectError("connection closed")
        with patch("library_rag.rag.retrieve", side_effect=[failure, []]), patch("library_rag.rag.time.sleep"):
            answer = answer_question("index", "llm", "ספר לילד בן 5")
        self.assertEqual(answer.text, NO_MATCHING_BOOKS_MESSAGE)

    def test_lasting_network_failure_is_raised(self):
        failure = httpx.ConnectError("connection closed")
        with patch("library_rag.rag.retrieve", side_effect=failure) as retrieve, patch("library_rag.rag.time.sleep"):
            with self.assertRaises(httpx.ConnectError):
                answer_question("index", "llm", "שאלה")
        self.assertEqual(retrieve.call_count, NETWORK_ATTEMPTS)

    def test_other_errors_are_not_retried(self):
        with patch("library_rag.rag.retrieve", side_effect=ValueError("a bug")) as retrieve:
            with self.assertRaises(ValueError):
                answer_question("index", "llm", "שאלה")
        self.assertEqual(retrieve.call_count, 1)


if __name__ == "__main__":
    unittest.main()
