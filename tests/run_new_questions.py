"""Measure the system on new questions that were not used to adjust it. This calls Gemini.

The expected facts were written from the source files before the first run.
Run from the project root: python tests/run_new_questions.py results_file_name
"""

import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR.parent))

from data_loader import load_book_documents
from rag import NOT_FOUND_MESSAGE, answer_question, get_llm, load_index

SECONDS_BETWEEN_QUESTIONS = 5

# (question, any of these must appear in the answer)
FACT_QUESTIONS = [
    ("אפשר להיכנס לספרייה עם כלב?", ["נחייה"]),
    ("מותר לאכול בספרייה?", ["לא", "איסור", "אסור"]),
    ("מותר לדבר בטלפון בספרייה?", ["לא", "איסור", "אסור"]),
    ("מה צריך להביא כדי להירשם לספרייה?", ["תעודת זהות"]),
    ("איבדתי ספר, מה עושים?", ["עותק חדש"]),
    ("לכמה זמן שומרים לי ספר שהזמנתי?", ["5"]),
    ("איפה משאירים אופניים?", ["מחוץ"]),
    ("למי אפשר לערער על חסימת מינוי?", ["מנהלת"]),
    ("אפשר להעביר את המינוי שלי לאחותי?", ["לא", "אישי"]),
    ("מתי נועלים את הדלתות של הספרייה?", ["רבע שעה"]),
    ("מה עושים כששומעים אזעקה?", ["מרחב המוגן", "למרחב"]),
    ("איפה הספרייה נמצאת?", ["לוטם 10"]),
    ("מה כתובת המייל של הספרייה?", ["library@harish.muni.il"]),
    ("כמה ספרים יש בספרייה?", ["20,000"]),
    ("אפשר לקרוא בספרייה ספרים דיגיטליים?", ["דיגיטל"]),
    ("מה מספר הטלפון לשאלות כלליות?", ["6719"]),
    ("מי כתב את הספר צבעים בחלון על שפת הים?", ["מחבר/ת לדוגמה 100"]),
    ("יש ספר בשם תעלומת המפתח ליד הנחל?", ["תעלומת המפתח ליד הנחל"]),
]

# (question, category, age): every retrieved book must meet these conditions
BOOK_QUESTIONS = [
    ("תמליץ על ספר מדע לילד בן 12", "מדע", 12),
    ("אני מחפשת ספר ספורט לילדה בת 14", "ספורט", 14),
    ("ספר טבע לילד בן 8", "טבע", 8),
]

ABSENT_QUESTIONS = [
    "מה השם של מנהלת הספרייה?",
    "יש בספרייה חוג שחמט?",
    "כמה עולה לשכור את האולם של הספרייה?",
    "מי ראש העיר של חריש?",
]


def check_titles(answer, catalog_titles):
    """Every catalog title in the answer must be among the retrieved books."""
    mentioned = {title for title in catalog_titles if title in answer.text}
    return mentioned <= {citation.title for citation in answer.citations}


def check_fact(answer, expected):
    return answer.is_supported and any(fact in answer.text for fact in expected)


def check_books(answer, category, age):
    if not answer.citations:
        return False
    for citation in answer.citations:
        low, high = citation.details["גיל מומלץ"].split("–")
        if citation.kind != "book" or citation.details["קטגוריה"] != category:
            return False
        if not int(low) <= age <= int(high):
            return False
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    output_path = TESTS_DIR / sys.argv[1]
    index = load_index()
    llm = get_llm()
    catalog_titles = {document.metadata["title"] for document in load_book_documents()}

    cases = [("עובדה", q, lambda a, e=e: check_fact(a, e)) for q, e in FACT_QUESTIONS]
    cases += [("המלצה", q, lambda a, c=c, g=g: check_books(a, c, g)) for q, c, g in BOOK_QUESTIONS]
    cases += [("לא קיים", q, lambda a: a.text == NOT_FOUND_MESSAGE) for q in ABSENT_QUESTIONS]

    lines = ["# שאלות חדשות שלא שימשו לכיוון המערכת\n"]
    passed = 0
    for number, (kind, question, check) in enumerate(cases, start=1):
        if number > 1:
            time.sleep(SECONDS_BETWEEN_QUESTIONS)
        try:
            answer = answer_question(index, llm, question)
            is_ok = check(answer) and check_titles(answer, catalog_titles)
            text = answer.text
        except Exception as error:
            is_ok = False
            text = f"שגיאה בהרצה: {type(error).__name__}"
        passed += is_ok
        status = "עבר" if is_ok else "נכשל"
        print(f"{number:2d} [{status}] ({kind}) {question}\n    {text}")
        lines.append(f"## {number}. {question}\n\n- סוג: {kind}\n- תוצאה אוטומטית: {status}\n\n{text}\n")

    summary = f"עברו {passed} מתוך {len(cases)}"
    print(f"\n{summary}")
    lines.insert(1, f"{summary}. הבדיקה האוטומטית בודקת רק שהעובדה הצפויה מופיעה, ולכן צריך גם לקרוא את התשובות.\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
