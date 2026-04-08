"""
RAG Engine — Three retrieval strategies for the multi-agent pipeline.

1. BM25Retriever   — keyword/TF-IDF matching  (Planner stage)
2. VectorRetriever — semantic similarity via ChromaDB + sentence-transformers (Executor stage)
3. HybridRetriever — BM25 + Vector with reciprocal rank fusion (Judge stage)

None of these touch resource_pool.py — RAG operates purely at the prompt level.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

console = Console()


# ─── Document Model ──────────────────────────────────────────────────────────

@dataclass
class Document:
    """A chunk of text from the knowledge base."""
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


# ─── BM25 Retriever (Planner Stage) ──────────────────────────────────────────

class BM25Retriever:
    """
    BM25 (Best Matching 25) — a probabilistic keyword retrieval algorithm.
    Pure Python, zero external dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}
        self._indexed = False

    def add_documents(self, docs: list[Document]):
        self.documents.extend(docs)
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _build_index(self):
        self._doc_term_freqs = []
        self._doc_lengths = []
        doc_count = len(self.documents)

        df: dict[str, int] = {}

        for doc in self.documents:
            tokens = self._tokenize(doc.content)
            self._doc_lengths.append(len(tokens))

            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._doc_term_freqs.append(tf)

            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        self._avg_dl = sum(self._doc_lengths) / max(doc_count, 1)

        self._idf = {}
        for term, count in df.items():
            self._idf[term] = math.log(
                (doc_count - count + 0.5) / (count + 0.5) + 1.0
            )

        self._indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        if not self._indexed or not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.documents):
            score = 0.0
            dl = self._doc_lengths[idx]
            tf_map = self._doc_term_freqs[idx]

            for token in query_tokens:
                if token not in self._idf:
                    continue

                tf = tf_map.get(token, 0)
                idf = self._idf[token]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score += idf * (numerator / denominator)

            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                doc = self.documents[idx]
                result = Document(
                    content=doc.content,
                    metadata=dict(doc.metadata),
                    score=score,
                )
                results.append(result)

        return results


# ─── Vector Retriever (Executor Stage) ────────────────────────────────────────

class VectorRetriever:
    """
    Semantic similarity retriever using ChromaDB + sentence-transformers.
    """

    def __init__(self, collection_name: str = "executor_knowledge"):
        self._collection_name = collection_name
        self._collection = None
        self._documents: list[Document] = []
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        try:
            from core.storage import get_chroma_client
            from chromadb.utils import embedding_functions

            self._client = get_chroma_client()

            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

            self._initialized = True
            console.print(f"[dim]✔ VectorRetriever initialized (PersistentClient + all-MiniLM-L6-v2)[/dim]")

        except ImportError as e:
            console.print(f"[bold yellow]⚠ VectorRetriever unavailable: {e}[/bold yellow]")
            console.print("[dim]Install with: pip install chromadb sentence-transformers[/dim]")
            raise

    def add_documents(self, docs: list[Document]):
        self._ensure_initialized()
        self._documents.extend(docs)

        if not docs:
            return

        ids = []
        documents = []
        metadatas = []
        base_count = self._collection.count()

        for i, doc in enumerate(docs):
            doc_id = f"doc_{base_count + i}"
            ids.append(doc_id)
            documents.append(doc.content)
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )

        docs = []
        if results and results["documents"]:
            for i, content in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1.0 - distance
                docs.append(Document(
                    content=content,
                    metadata=meta,
                    score=similarity,
                ))

        return docs


# ─── Hybrid Retriever (Judge Stage) ──────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 + Vector retrieval using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25: Optional[BM25Retriever] = None,
        vector: Optional[VectorRetriever] = None,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25 or BM25Retriever()
        self.vector = vector or VectorRetriever(collection_name="judge_knowledge")
        self.rrf_k = rrf_k

    def add_documents(self, docs: list[Document]):
        self.bm25.add_documents(docs)
        self.vector.add_documents(docs)

    def retrieve(self, query: str, top_k: int = 3) -> list[Document]:
        fetch_k = top_k * 3
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)
        vector_results = self.vector.retrieve(query, top_k=fetch_k)

        content_to_doc: dict[str, Document] = {}
        rrf_scores: dict[str, float] = {}

        for rank, doc in enumerate(bm25_results):
            key = doc.content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in content_to_doc:
                content_to_doc[key] = doc

        for rank, doc in enumerate(vector_results):
            key = doc.content[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in content_to_doc:
                content_to_doc[key] = doc

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for key in sorted_keys[:top_k]:
            doc = content_to_doc[key]
            results.append(Document(
                content=doc.content,
                metadata=dict(doc.metadata),
                score=rrf_scores[key],
            ))

        return results


# ─── Singleton Instances ──────────────────────────────────────────────────────

_planner_rag: Optional[BM25Retriever] = None
_executor_rag: Optional[VectorRetriever] = None
_judge_rag: Optional[HybridRetriever] = None
_initialized = False


def initialize_rag():
    global _planner_rag, _executor_rag, _judge_rag, _initialized

    if _initialized:
        return

    from core.knowledge_base_loader import load_knowledge_base

    console.print("\n[bold cyan]Initializing RAG Engine...[/bold cyan]")

    planning_docs, technical_docs, evaluation_docs, policy_docs = load_knowledge_base()

    _planner_rag = BM25Retriever()
    _planner_rag.add_documents(planning_docs)
    console.print(f"[dim]  ✔ Planner BM25 RAG: {len(planning_docs)} planning chunks indexed[/dim]")

    _executor_rag = VectorRetriever(collection_name="executor_knowledge")
    executor_docs = technical_docs + policy_docs
    _executor_rag.add_documents(executor_docs)
    console.print(f"[dim]  ✔ Executor Vector RAG: {len(executor_docs)} technical+policy chunks indexed[/dim]")

    _judge_rag = HybridRetriever(
        bm25=BM25Retriever(),
        vector=VectorRetriever(collection_name="judge_knowledge"),
    )
    judge_docs = evaluation_docs + technical_docs + policy_docs
    _judge_rag.add_documents(judge_docs)
    console.print(f"[dim]  ✔ Judge Hybrid RAG: {len(judge_docs)} eval+technical+policy chunks indexed[/dim]")

    _initialized = True
    console.print("[bold green]✔ RAG Engine ready (BM25 + Vector + Hybrid)[/bold green]\n")


def get_planner_rag() -> BM25Retriever:
    if not _initialized:
        initialize_rag()
    return _planner_rag


def get_executor_rag() -> VectorRetriever:
    if not _initialized:
        initialize_rag()
    return _executor_rag


def get_judge_rag() -> HybridRetriever:
    if not _initialized:
        initialize_rag()
    return _judge_rag


def format_rag_context(docs: list[Document], max_chars: int = 3000) -> str:
    if not docs:
        return "(No relevant knowledge base documents found.)"

    parts = []
    total_chars = 0

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        category = doc.metadata.get("category", "general")
        header = f"[Source: {source} | Category: {category} | Relevance: {doc.score:.3f}]"
        chunk = f"{header}\n{doc.content}"

        if total_chars + len(chunk) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 200:
                chunk = chunk[:remaining] + "\n... (truncated)"
                parts.append(chunk)
            break

        parts.append(chunk)
        total_chars += len(chunk)

    return "\n\n---\n\n".join(parts)


def add_user_interaction(query: str, answer: str):
    if not _initialized:
        return

    try:
        from core.knowledge_base_loader import _chunk_text

        combined = f"User Query: {query}\n\nAnswer: {answer}"
        chunks = _chunk_text(combined)

        if not chunks:
            return

        docs = [
            Document(
                content=chunk,
                metadata={
                    "source": "user_interaction",
                    "category": "user_interactions",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "query": query[:200],
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        if _planner_rag:
            _planner_rag.add_documents(docs)
        if _executor_rag:
            _executor_rag.add_documents(docs)
        if _judge_rag:
            _judge_rag.add_documents(docs)

        console.print(
            f"[dim]📚 RAG knowledge base updated: +{len(docs)} chunks from user interaction[/dim]"
        )

    except Exception as e:
        console.print(f"[dim]⚠ Failed to add user interaction to RAG: {e}[/dim]")
