"""
Query Cache — Two-level caching to avoid redundant pipeline executions.

Level 1 (L1): Exact hash match (SHA-256 in SQLite)
Level 2 (L2): Semantic similarity match (ChromaDB + sentence-transformers)
"""

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# ─── Cache Configuration ─────────────────────────────────────────────────────

DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_SEMANTIC_THRESHOLD = 0.90
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "multiagent.db"


@dataclass
class CacheEntry:
    """A cached query-answer pair."""
    query_hash: str
    query_text: str
    cached_answer: str
    hit_count: int
    created_at: str
    last_hit_at: str
    cache_level: str
    similarity: float


class QueryCache:
    """Two-level query cache: L1 (exact hash) + L2 (semantic similarity)."""

    def __init__(
        self,
        db_path: Path = DB_PATH,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ):
        self._db_path = db_path
        self._ttl = ttl_seconds
        self._semantic_threshold = semantic_threshold
        self._db_lock = threading.Lock()

        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()
        self._init_chroma()

    def _init_sqlite(self):
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        query_hash    TEXT    PRIMARY KEY,
                        query_text    TEXT    NOT NULL,
                        cached_answer TEXT    NOT NULL,
                        hit_count     INTEGER DEFAULT 0,
                        created_at    TEXT    DEFAULT (datetime('now', 'localtime')),
                        last_hit_at   TEXT    DEFAULT (datetime('now', 'localtime'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_cache_created
                        ON query_cache(created_at);
                """)
                conn.commit()
            finally:
                conn.close()

    def _init_chroma(self):
        try:
            from core.storage import get_chroma_client
            from chromadb.utils import embedding_functions

            client = get_chroma_client()
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._cache_collection = client.get_or_create_collection(
                name="query_cache_embeddings",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._l2_available = True
            console.print(
                f"[dim]✔ Semantic cache collection: "
                f"{self._cache_collection.count()} cached embeddings[/dim]"
            )
        except Exception as e:
            console.print(f"[bold yellow]⚠ Semantic cache (L2) unavailable: {e}[/bold yellow]")
            self._cache_collection = None
            self._l2_available = False

    @staticmethod
    def _normalize(query: str) -> str:
        import re
        text = query.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def lookup(self, query: str) -> Optional[CacheEntry]:
        normalized = self._normalize(query)

        entry = self._l1_lookup(normalized)
        if entry is not None:
            self._l1_hits += 1
            return entry

        entry = self._l2_lookup(query)
        if entry is not None:
            self._l2_hits += 1
            return entry

        self._misses += 1
        return None

    def _l1_lookup(self, normalized_query: str) -> Optional[CacheEntry]:
        query_hash = self._hash(normalized_query)

        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.execute(
                    """SELECT query_hash, query_text, cached_answer,
                              hit_count, created_at, last_hit_at
                       FROM query_cache
                       WHERE query_hash = ?""",
                    (query_hash,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                created_str = row[4]
                try:
                    import datetime
                    created = datetime.datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
                    age_seconds = (datetime.datetime.now() - created).total_seconds()
                    if age_seconds > self._ttl:
                        conn.execute(
                            "DELETE FROM query_cache WHERE query_hash = ?",
                            (query_hash,),
                        )
                        conn.commit()
                        console.print(f"[dim]⏰ Cache L1: expired entry removed (age: {age_seconds:.0f}s)[/dim]")
                        return None
                except (ValueError, TypeError):
                    pass

                conn.execute(
                    """UPDATE query_cache
                       SET hit_count = hit_count + 1,
                           last_hit_at = datetime('now', 'localtime')
                       WHERE query_hash = ?""",
                    (query_hash,),
                )
                conn.commit()

                return CacheEntry(
                    query_hash=row[0],
                    query_text=row[1],
                    cached_answer=row[2],
                    hit_count=row[3] + 1,
                    created_at=row[4],
                    last_hit_at=row[5],
                    cache_level="L1_exact",
                    similarity=1.0,
                )

            finally:
                conn.close()

    def _l2_lookup(self, query: str) -> Optional[CacheEntry]:
        if not self._l2_available or self._cache_collection is None:
            return None

        if self._cache_collection.count() == 0:
            return None

        try:
            results = self._cache_collection.query(
                query_texts=[query],
                n_results=1,
            )

            if not results or not results["documents"] or not results["documents"][0]:
                return None

            distance = results["distances"][0][0]
            similarity = 1.0 - distance

            if similarity < self._semantic_threshold:
                console.print(
                    f"[dim]🔍 Cache L2: closest match has similarity {similarity:.3f} "
                    f"(below threshold {self._semantic_threshold})[/dim]"
                )
                return None

            meta = results["metadatas"][0][0]
            cached_answer = meta.get("cached_answer", "")
            original_query = meta.get("original_query", "")
            query_hash = meta.get("query_hash", "")

            if not cached_answer:
                return None

            self._update_hit_count(query_hash)

            console.print(
                f"[dim]🔍 Cache L2: semantic match found "
                f"(similarity: {similarity:.3f}, threshold: {self._semantic_threshold})[/dim]"
            )

            return CacheEntry(
                query_hash=query_hash,
                query_text=original_query,
                cached_answer=cached_answer,
                hit_count=1,
                created_at="",
                last_hit_at="",
                cache_level="L2_semantic",
                similarity=similarity,
            )

        except Exception as e:
            console.print(f"[dim]⚠ Cache L2 lookup error: {e}[/dim]")
            return None

    def _update_hit_count(self, query_hash: str):
        if not query_hash:
            return
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute(
                    """UPDATE query_cache
                       SET hit_count = hit_count + 1,
                           last_hit_at = datetime('now', 'localtime')
                       WHERE query_hash = ?""",
                    (query_hash,),
                )
                conn.commit()
            finally:
                conn.close()

    def store(self, query: str, answer: str):
        normalized = self._normalize(query)
        query_hash = self._hash(normalized)

        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO query_cache
                       (query_hash, query_text, cached_answer, hit_count, created_at, last_hit_at)
                       VALUES (?, ?, ?, 0, datetime('now', 'localtime'), datetime('now', 'localtime'))""",
                    (query_hash, query, answer),
                )
                conn.commit()
            finally:
                conn.close()

        if self._l2_available and self._cache_collection is not None:
            try:
                truncated_answer = answer[:8000] if len(answer) > 8000 else answer

                self._cache_collection.upsert(
                    ids=[f"cache_{query_hash}"],
                    documents=[query],
                    metadatas=[{
                        "query_hash": query_hash,
                        "original_query": query[:500],
                        "cached_answer": truncated_answer,
                    }],
                )
            except Exception as e:
                console.print(f"[dim]⚠ Cache L2 store error: {e}[/dim]")

        console.print(f"[dim]💾 Cached query → answer (hash: {query_hash[:12]}...)[/dim]")

    def invalidate(self, query: str):
        normalized = self._normalize(query)
        query_hash = self._hash(normalized)

        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("DELETE FROM query_cache WHERE query_hash = ?", (query_hash,))
                conn.commit()
            finally:
                conn.close()

        if self._l2_available and self._cache_collection is not None:
            try:
                self._cache_collection.delete(ids=[f"cache_{query_hash}"])
            except Exception:
                pass

    def clear(self):
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("DELETE FROM query_cache")
                conn.commit()
            finally:
                conn.close()

        if self._l2_available and self._cache_collection is not None:
            try:
                from core.storage import get_chroma_client
                client = get_chroma_client()
                client.delete_collection("query_cache_embeddings")
                self._init_chroma()
            except Exception as e:
                console.print(f"[dim]⚠ Failed to clear L2 cache: {e}[/dim]")

        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
        console.print("[bold green]✔ Cache cleared (L1 + L2)[/bold green]")

    def stats(self) -> dict:
        l1_count = 0
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM query_cache")
                l1_count = cursor.fetchone()[0]
            finally:
                conn.close()

        l2_count = 0
        if self._l2_available and self._cache_collection is not None:
            l2_count = self._cache_collection.count()

        total_lookups = self._l1_hits + self._l2_hits + self._misses
        hit_rate = (
            (self._l1_hits + self._l2_hits) / total_lookups * 100
            if total_lookups > 0
            else 0.0
        )

        return {
            "l1_entries": l1_count,
            "l2_entries": l2_count,
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "total_lookups": total_lookups,
            "hit_rate_pct": hit_rate,
        }

    def print_stats(self):
        s = self.stats()
        console.print(f"\n[bold cyan]📊 Cache Statistics[/bold cyan]")
        console.print(f"  L1 (exact hash):      {s['l1_entries']} entries, {s['l1_hits']} hits")
        console.print(f"  L2 (semantic):         {s['l2_entries']} entries, {s['l2_hits']} hits")
        console.print(f"  Misses:                {s['misses']}")
        console.print(f"  Total lookups:         {s['total_lookups']}")
        console.print(f"  Hit rate:              {s['hit_rate_pct']:.1f}%")
