"""Run the evaluation questions on the real system and save the raw results. This calls Gemini.

Run from the project root: python tests/run_evaluation.py results_file_name
The results are saved in tests/ as they were received, without any editing.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR.parent))

from library_rag.config import EMBED_MODEL, LLM_MODEL
from library_rag.rag import answer_question, get_llm, load_index

SECONDS_BETWEEN_QUESTIONS = 5

# Each question is listed with the type the assignment requires.
QUESTIONS = [
    ("כללי השאלה", "כמה ספרים מותר לשאול בו זמנית ולכמה זמן?"),
    ("שעות פתיחה", "מה שעות הפתיחה של הספרייה ביום שלישי?"),
    ("שעות פתיחה", "הספריה פתוחה בשעות הצהרים?"),
    ("שירותי הספרייה", "איך אפשר להאריך תקופת השאלה של ספר?"),
    ("המלצה על ספר", "תמליץ לי על ספר הרפתקאות לילד בן 9"),
    ("פרשנות של תקציר", "אני מחפשת ספר על ילדים שעובדים יחד כדי לפתור בעיה"),
    ("פרשנות של נוהל", "אפשר לבוא עם תינוקים לשם?"),
    ("פרשנות של נוהל", "מה קורה אם מאחרים בהחזרת ספר בפעם השנייה?"),
    ("פרשנות של נוהל", "כמה עולה מנוי?"),
    ("מידע שלא קיים במסמכים", "כמה עולה קנס על איחור בהחזרת ספר?"),
    ("מידע שלא קיים במסמכים", "מה מזג האוויר מחר בחריש?"),
    ("מידע שלא קיים במסמכים", "יש לכם ספר על דינוזאורים?"),
]


def format_citation(citation):
    if citation.kind == "book":
        heading = f"{citation.title} | מזהה {citation.book_id} | " + " | ".join(
            f"{name}: {value}" for name, value in citation.details.items()
        )
    else:
        heading = f"{citation.source_file} | עמוד {citation.page}"
    excerpt = citation.excerpt.replace("\n", "\n> ")
    return f"**{heading}** (ציון דמיון {citation.score:.3f}, `{citation.node_id}`)\n\n> {excerpt}\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    output_path = TESTS_DIR / sys.argv[1]
    index = load_index()
    llm = get_llm()

    sections = [
        "# תוצאות הרצה גולמיות\n",
        f"- זמן ההרצה: {datetime.now():%Y-%m-%d %H:%M}",
        f"- מודל תשובות: `{LLM_MODEL}`",
        f"- מודל Embeddings: `{EMBED_MODEL}`\n",
    ]
    for number, (question_type, question) in enumerate(QUESTIONS, start=1):
        if number > 1:
            time.sleep(SECONDS_BETWEEN_QUESTIONS)
        sections.append(f"## {number}. {question_type}\n\n**שאלה:** {question}\n")
        try:
            answer = answer_question(index, llm, question)
        except Exception as error:
            sections.append(f"**שגיאה בהרצה:** `{type(error).__name__}`\n")
            print(f"{number}: ERROR {type(error).__name__}")
            continue
        sections.append(f"**תשובה:**\n\n{answer.text}\n")
        sections.append(f"**is_supported:** {answer.is_supported}\n")
        sections.append(f"**מקורות שנשלפו ({len(answer.citations)}):**\n")
        sections.extend(format_citation(citation) for citation in answer.citations)
        print(f"{number}: done, {len(answer.citations)} citations")

    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
