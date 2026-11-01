"""Read from a question whether it searches the catalog and what it requires."""

import re
from dataclasses import dataclass

from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

CATEGORIES = (
    "הרפתקאות",
    "מדע",
    "טבע",
    "היסטוריה",
    "בלשים",
    "משפחה",
    "חברות",
    "טכנולוגיה",
    "ספורט",
    "אמנות",
)

# Hebrew letters that can attach to the start of a word, such as "ל" in "לגיל".
HEBREW_PREFIX = r"(?<![א-ת])[ובהלמש]?"
AGE_PATTERN = re.compile(rf"{HEBREW_PREFIX}(?:בן|בת|גיל)\s*(\d{{1,2}})")
ADULT_PATTERN = re.compile(rf"{HEBREW_PREFIX}מבוגר(?:ים|ות|ת)?(?![א-ת])")
ADULT_AGE = 18
BOOK_WORD_PATTERN =re.compile(rf"{HEBREW_PREFIX}ספר(?:ים|י)?(?![א-ת])")
RECOMMEND_PATTERN = re.compile(r"המלצ|תמליץ|תמליצ|ממליצ|מחפש|מתאים|מתאימ|בשם|(?:ספר|ספרים|ספרי)\s+(?:על|בנושא|ש)")
# Questions about handling books are library rules, not catalog searches.
RULES_PATTERN = re.compile(r"לשאול|לשאיל|שאילת|להחזיר|החזרת|להאריך|הארכת|להזמין|הזמנת|איחור|מינוי|אבד|פגום")


@dataclass
class BookSearch:
    """What a question asks of the catalog, read from its wording."""

    is_book_search: bool = False
    age: int | None = None
    category: str | None = None


def parse_book_search(question):
    """Detect a catalog search and any age or category it requires."""
    age_match = AGE_PATTERN.search(question)
    age = int(age_match.group(1)) if age_match else None
    if age is None and ADULT_PATTERN.search(question):
        age = ADULT_AGE
    category = next(
        (name for name in CATEGORIES if re.search(rf"{HEBREW_PREFIX}{name}(?![א-ת])", question)),
        None,
    )
    mentions_book = bool(BOOK_WORD_PATTERN.search(question))
    is_about_rules = bool(RULES_PATTERN.search(question))
    is_book_search = (
        mentions_book
        and not is_about_rules
        and bool(RECOMMEND_PATTERN.search(question) or age is not None or category)
    )
    if not is_book_search:
        return BookSearch()
    return BookSearch(is_book_search=True, age=age, category=category)


def build_filters(search):
    """Turn a catalog search into exact conditions on the stored book fields."""
    if not search.is_book_search:
        return None
    conditions = [MetadataFilter(key="doc_type", value="book", operator=FilterOperator.EQ)]
    if search.age is not None:
        conditions.append(MetadataFilter(key="age_min", value=search.age, operator=FilterOperator.LTE))
        conditions.append(MetadataFilter(key="age_max", value=search.age, operator=FilterOperator.GTE))
    if search.category:
        conditions.append(MetadataFilter(key="category", value=search.category, operator=FilterOperator.EQ))
    return MetadataFilters(filters=conditions)
