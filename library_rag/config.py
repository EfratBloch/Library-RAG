"""Settings shared by the whole application."""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
FINGERPRINT_FILE = CHROMA_DIR / "index_fingerprint.txt"
COLLECTION_NAME = "library"
EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 20
EMBED_RETRIES = 6
# The free tier allows 100 embedded texts per minute, and the index has more than that.
EMBED_TEXTS_PER_MINUTE = 80

INFO_PDF = "libarary_info.pdf"
RULES_PDF = "libarary_rules.pdf"
CATALOG_CSV = "books_catalog_demo_100.csv"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 60

CATALOG_AGE_RANGES_TEXT = "8–10, 11–13 ו-14–16"

DEFAULT_TOP_K = 5

LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0.1
