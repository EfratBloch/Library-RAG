"""Build the persistent vector index from the library PDFs and the book catalog.

Usage:
    python -m library_rag.ingest            build the index, or skip if it is already up to date
    python -m library_rag.ingest --rebuild  build the index again even if it is up to date
    python -m library_rag.ingest --preview  print the chunks without calling any API
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import chromadb
from dotenv import load_dotenv
from google.genai.errors import APIError
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from library_rag.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    EMBED_RETRIES,
    EMBED_TEXTS_PER_MINUTE,
    FINGERPRINT_FILE,
    PROJECT_DIR,
)
from library_rag.data_loader import load_book_documents, load_pdf_documents

# Section headings of the PDFs. A heading such as "ג. שאילת ספרים" opens a section in the rules.
RULES_HEADING = re.compile(r"^[א-ט]\. \S")
SUB_HEADINGS = {"איחור ראשון", "איחור שני", "סייג"}
INFO_HEADINGS = {
    "שעות קבלת קהל:",
    "אוספי הספרייה",
    "הספרייה העירונית בקדמת הטכנולוגיה",
    "הספר הדיגיטלי",
    "קטלוג הספרייה",
    "הספרייה כמרכז תרבות ופנאי",
}


def make_node(text, metadata, node_id):
    """Create a node whose embedding is based on its text, without the metadata."""
    return TextNode(
        id_=node_id,
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
        excluded_llm_metadata_keys=list(metadata),
    )


def group_lines_by_section(pdf_documents):
    """Yield each page's text section by section, carrying a section across page breaks."""
    source_file = None
    main_heading = section = ""
    for document in pdf_documents:
        if document.metadata["source_file"] != source_file:
            source_file = document.metadata["source_file"]
            main_heading = section = ""
        lines = []
        for line in document.text.split("\n"):
            is_main_heading = RULES_HEADING.match(line) or line in INFO_HEADINGS
            if not is_main_heading and line not in SUB_HEADINGS:
                lines.append(line)
                continue
            if lines:
                yield document, section, "\n".join(lines)
                lines = []
            if is_main_heading:
                main_heading = section = line
            else:
                section = f"{main_heading} - {line}"
        if lines:
            yield document, section, "\n".join(lines)


def build_nodes(pdf_documents, book_documents):
    """Split PDF sections into overlapping chunks and keep every book as a single node."""
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = []
    sections = group_lines_by_section(pdf_documents)
    for section_number, (document, section, text) in enumerate(sections):
        metadata = {**document.metadata, "section": section}
        for index, chunk in enumerate(splitter.split_text(text)):
            node_id = f"{metadata['source_file']}-p{metadata['page']}-s{section_number}-c{index}"
            # The heading is part of the chunk, so its rules are found and read in context.
            chunk_text = f"{section}\n{chunk}" if section else chunk
            nodes.append(make_node(chunk_text, dict(metadata), node_id))
    for document in book_documents:
        nodes.append(make_node(document.text, dict(document.metadata), document.metadata["book_id"]))
    return nodes


def get_embed_model():
    """Create the Gemini embedding model using the key from .env."""
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to the .env file.")
    return GoogleGenAIEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=EMBED_BATCH_SIZE,
        retries=EMBED_RETRIES,
    )


def get_collection(client):
    """Open the Chroma collection, comparing vectors by cosine distance."""
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def compute_fingerprint(nodes):
    """Hash everything that affects the stored vectors, to detect a stale index."""
    content = [EMBED_MODEL] + [[node.node_id, node.text, node.metadata] for node in nodes]
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def embed_nodes_paced(nodes, embed_model):
    """Embed the nodes in timed batches. Each text counts as one request in the free tier."""
    seconds_per_batch = EMBED_BATCH_SIZE * 60 / EMBED_TEXTS_PER_MINUTE
    for start in range(0, len(nodes), EMBED_BATCH_SIZE):
        started_at = time.monotonic()
        batch = nodes[start : start + EMBED_BATCH_SIZE]
        texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in batch]
        for node, embedding in zip(batch, embed_model.get_text_embedding_batch(texts)):
            node.embedding = embedding
        done = start + len(batch)
        print(f"Embedded {done}/{len(nodes)}", flush=True)
        if done < len(nodes):
            time.sleep(max(0, seconds_per_batch - (time.monotonic() - started_at)))


def build_index(nodes, rebuild=False):
    """Embed the nodes into ChromaDB. Return False if the index was already up to date."""
    fingerprint = compute_fingerprint(nodes)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    is_current = (
        FINGERPRINT_FILE.exists()
        and FINGERPRINT_FILE.read_text(encoding="utf-8") == fingerprint
        and get_collection(client).count() == len(nodes)
    )
    if is_current and not rebuild:
        return False

    # Embed first, so a quota or network failure leaves any existing index untouched.
    embed_model = get_embed_model()
    embed_nodes_paced(nodes, embed_model)

    # The fingerprint is written last, so an interrupted build is redone on the next run.
    FINGERPRINT_FILE.unlink(missing_ok=True)
    if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=get_collection(client))
    VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=embed_model,
    )
    FINGERPRINT_FILE.write_text(fingerprint, encoding="utf-8")
    return True


def print_preview(nodes):
    for node in nodes:
        if node.metadata["doc_type"] != "book":
            print(f"\n--- {node.node_id} ({len(node.text)} chars) ---")
            print(node.text)
    print("\n--- sample book node ---")
    print(nodes[-1].node_id, nodes[-1].metadata)
    print(nodes[-1].text)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build the library vector index.")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if up to date")
    parser.add_argument("--preview", action="store_true", help="print chunks, no API calls")
    args = parser.parse_args()

    pdf_documents = load_pdf_documents()
    book_documents = load_book_documents()
    nodes = build_nodes(pdf_documents, book_documents)
    print(f"PDF pages with content: {len(pdf_documents)}")
    print(f"Catalog records: {len(book_documents)}")
    print(f"Nodes: {len(nodes)}")

    if args.preview:
        print_preview(nodes)
        return

    try:
        if build_index(nodes, rebuild=args.rebuild):
            print(f"Index built and saved to {CHROMA_DIR}")
        else:
            print("Index is already up to date. Nothing was sent to Gemini.")
    except RuntimeError as error:
        sys.exit(f"Error: {error}")
    except APIError as error:
        sys.exit(f"Gemini API error {error.code}: {error.message}. Run the command again later.")


if __name__ == "__main__":
    main()
