"""Turn retrieved chunks into source citations. Gemini takes no part in this."""

from dataclasses import dataclass, field

BOOK_SUMMARY_LABEL = "תקציר: "


@dataclass
class Citation:
    """One retrieved source, built only from the stored text and metadata."""

    kind: str
    node_id: str
    score: float
    title: str
    excerpt: str
    source_file: str
    page: int | None = None
    section: str = ""
    book_id: str | None = None
    details: dict = field(default_factory=dict)


def book_summary(text):
    """Return the summary part of a book chunk."""
    _, _, summary = text.partition(BOOK_SUMMARY_LABEL)
    return summary.strip() or text


def build_citation(source):
    """Create a citation from one retrieved node."""
    node = source.node
    metadata = node.metadata
    if metadata["doc_type"] == "book":
        return Citation(
            kind="book",
            node_id=node.node_id,
            score=source.score,
            title=metadata["title"],
            excerpt=book_summary(node.text),
            source_file=metadata["source_file"],
            book_id=metadata["book_id"],
            details={
                "מחבר": metadata["author"],
                "קטגוריה": metadata["category"],
                "גיל מומלץ": metadata["age_range"],
                "שנת פרסום": metadata["year"],
            },
        )
    return Citation(
        kind="pdf",
        node_id=node.node_id,
        score=source.score,
        title=metadata["source_file"],
        excerpt=node.text,
        source_file=metadata["source_file"],
        page=metadata["page"],
        section=metadata.get("section", ""),
    )


def build_citations(sources):
    """Create citations for the retrieved nodes, keeping their retrieval order."""
    return [build_citation(source) for source in sources]
