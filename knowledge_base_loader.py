"""
Knowledge Base Loader — loads, chunks, and categorizes documents from the
knowledge_base/ directory for the RAG engine.

Produces four lists of Document objects:
  1. planning_docs   — for BM25 Planner RAG
  2. technical_docs   — for Vector Executor RAG
  3. evaluation_docs  — for Hybrid Judge RAG
  4. policy_docs      — shared across Executor and Judge RAG
"""

import os
from pathlib import Path
from rag_engine import Document
from rich.console import Console

console = Console()

# Root directory for knowledge base files
KB_ROOT = Path(__file__).parent / "knowledge_base"

# Chunk configuration
CHUNK_SIZE = 500       # approximate tokens per chunk (words as proxy)
CHUNK_OVERLAP = 50     # overlap tokens between chunks


def _read_file(filepath: Path) -> str:
    """Read a text file with encoding fallback."""
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return filepath.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    Each chunk is approximately chunk_size words, with overlap words
    carried over from the previous chunk for context continuity.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Advance by (chunk_size - overlap) to create overlap
        start += chunk_size - overlap

    return chunks


def _load_category(category_dir: Path, category_name: str) -> list[Document]:
    """
    Load all .txt files from a category directory and chunk them.
    Returns a list of Document objects with metadata.
    """
    docs = []

    if not category_dir.exists():
        console.print(f"[dim]  ⚠ Knowledge base directory not found: {category_dir}[/dim]")
        return docs

    txt_files = list(category_dir.glob("*.txt"))
    if not txt_files:
        console.print(f"[dim]  ⚠ No .txt files in {category_dir}[/dim]")
        return docs

    for filepath in sorted(txt_files):
        text = _read_file(filepath)
        if not text.strip():
            continue

        chunks = _chunk_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                metadata={
                    "source": filepath.name,
                    "category": category_name,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "filepath": str(filepath),
                },
            )
            docs.append(doc)

    console.print(
        f"[dim]  📄 Loaded {category_name}: {len(txt_files)} files → {len(docs)} chunks[/dim]"
    )
    return docs


def load_knowledge_base() -> tuple[list[Document], list[Document], list[Document], list[Document]]:
    """
    Load the entire knowledge base from disk, chunked and categorized.

    Returns:
        (planning_docs, technical_docs, evaluation_docs, policy_docs)
    """
    console.print("[cyan]Loading knowledge base...[/cyan]")

    planning_docs = _load_category(KB_ROOT / "planning", "planning")
    technical_docs = _load_category(KB_ROOT / "technical", "technical")
    evaluation_docs = _load_category(KB_ROOT / "evaluation", "evaluation")
    policy_docs = _load_category(KB_ROOT / "policy", "policy")

    total = len(planning_docs) + len(technical_docs) + len(evaluation_docs) + len(policy_docs)
    console.print(f"[cyan]Knowledge base loaded: {total} total chunks[/cyan]")

    return planning_docs, technical_docs, evaluation_docs, policy_docs
