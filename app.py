"""Streamlit interface of the library assistant. Start it with: streamlit run app.py"""

import html
import logging

import streamlit as st
from google.genai.errors import APIError

from library_rag.rag import NETWORK_ERRORS, IndexNotReadyError, answer_question, get_llm, index_version, load_index

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = {
    "מידע על הספרייה": [
        "מה שעות הפתיחה של הספרייה?",
        "כמה ספרים מותר לשאול בו זמנית?",
        "מה קורה אם מאחרים בהחזרת ספר?",
    ],
    "חיפוש והמלצות על ספרים": [
        "ספר הרפתקאות לילד בן 9",
        "ספר על בלשים לגיל 15",
        "יש לכם ספר על ילדה שאוהבת יצירה?",
    ],
}

RTL_STYLE = """
<style>
.stApp, .stMarkdown, .stTextInput input, .stAlert, [data-testid="stExpander"] {
    direction: rtl;
    text-align: right;
}
[data-testid="stExpander"] summary { flex-direction: row; }
.source-excerpt {
    white-space: pre-wrap;
    border-right: 3px solid #8a8f98;
    padding: 0.25rem 0.75rem;
    margin: 0.25rem 0 0.75rem;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_resources(index_version):
    """Open the index and the model once per index version, and reuse them for every question."""
    return load_index(), get_llm()


def set_example_question(question):
    st.session_state.question = question
    st.session_state.run_question = True


def request_answer():
    st.session_state.run_question = True


def error_message(error):
    """Explain a failure in Hebrew, without exposing technical details."""
    if isinstance(error, IndexNotReadyError):
        return "מאגר המידע עדיין לא נבנה. יש להריץ פעם אחת את הפקודה: python -m library_rag.ingest"
    if isinstance(error, RuntimeError):
        return "לא נמצא מפתח API. יש להוסיף את GOOGLE_API_KEY לקובץ .env."
    if isinstance(error, APIError) and error.code == 429:
        return "עברנו את מכסת השימוש של Gemini. מכסה לדקה מתחדשת תוך דקה, ומכסה יומית מתחדשת רק כעבור כמה שעות."
    if isinstance(error, APIError):
        return f"השירות של Gemini החזיר שגיאה (קוד {error.code}). אפשר לנסות שוב בעוד רגע."
    if isinstance(error, NETWORK_ERRORS):
        return "החיבור ל-Gemini נכשל גם אחרי ניסיון נוסף. כדאי לבדוק את החיבור לאינטרנט ולנסות שוב."
    return "אירעה תקלה פנימית במערכת, והיא לא קשורה לחיבור לאינטרנט. הפרטים נרשמו ביומן השרת."


def show_citation(citation):
    """Show one retrieved source. All its values come from the stored metadata."""
    if citation.kind == "book":
        st.markdown(f"**{html.escape(citation.title)}** · מזהה: {citation.book_id}")
        st.caption(" · ".join(f"{name}: {value}" for name, value in citation.details.items()))
    else:
        st.markdown(f"**{citation.source_file}** · עמוד {citation.page}")
        if citation.section:
            st.caption(f"סעיף: {citation.section}")
    st.markdown(
        f'<div class="source-excerpt">{html.escape(citation.excerpt)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"ציון דמיון: {citation.score:.2f}")


def show_answer(answer):
    st.subheader("תשובה", anchor=False)
    if not answer.is_supported:
        st.warning(answer.text)
        return
    st.markdown(answer.text)
    with st.expander(f"מקורות שנשלפו ({len(answer.citations)})"):
        st.caption(
            "אלה הקטעים שנמצאו בחיפוש ונשלחו למודל. ייתכן שלא כולם שימשו לתשובה, "
            "וציון הדמיון אינו מבטיח שהמידע נכון."
        )
        for citation in answer.citations:
            show_citation(citation)


def main():
    st.set_page_config(page_title="עוזר הספרייה החכם", page_icon="📚")
    st.markdown(RTL_STYLE, unsafe_allow_html=True)
    st.title("עוזר הספרייה החכם", anchor=False)
    st.write("אפשר לשאול על שעות הפתיחה, על תקנון הספרייה ועל ספרים מהקטלוג.")

    st.text_input("השאלה שלך", key="question", on_change=request_answer)
    submitted = st.button("שליחה", type="primary", on_click=request_answer)

    for title, questions in EXAMPLE_QUESTIONS.items():
        st.caption(title)
        for column, question in zip(st.columns(len(questions)), questions):
            column.button(
                question,
                key=f"example-{question}",
                use_container_width=True,
                on_click=set_example_question,
                args=(question,),
            )

    should_run = st.session_state.pop("run_question", False) or submitted
    question = st.session_state.get("question", "").strip()
    if should_run:
        st.session_state.pop("answer", None)
        if not question:
            st.warning("יש לכתוב שאלה לפני השליחה.")
        else:
            try:
                with st.spinner("מחפש במקורות..."):
                    index, llm = get_resources(index_version())
                    st.session_state.answer = answer_question(index, llm, question)
            except Exception as error:
                logger.exception("Answering a question failed")
                st.error(error_message(error))

    if "answer" in st.session_state:
        show_answer(st.session_state.answer)

    st.divider()
    st.caption("קטלוג הספרים מתאר ספרים בלבד, ואינו מאמת שהם זמינים בספרייה העירונית.")


if __name__ == "__main__":
    main()
