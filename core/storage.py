"""
Storage Manager — SQLite + ChromaDB persistent storage for the multi-agent system.

Responsibilities:
  1. SQLite: Structured chat history (conversations, messages, timing)
  2. ChromaDB: Persistent vector embeddings (knowledge base + user interactions)
  3. Knowledge growth: Final answers are chunked and fed back into the RAG engine
"""

import json
import sqlite3
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()

# ─── Paths (data/ directory at project root) ─────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "multiagent.db"
CHROMA_PATH = DATA_DIR / "chroma_db"


# ─── Shared ChromaDB Persistent Client (Singleton) ───────────────────────────

_chroma_client = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    """
    Get or create the shared ChromaDB PersistentClient.
    Thread-safe singleton — all modules share the same client instance.
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    with _chroma_lock:
        if _chroma_client is not None:
            return _chroma_client

        import chromadb
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        console.print(f"[dim]✔ ChromaDB PersistentClient initialized at {CHROMA_PATH}[/dim]")
        return _chroma_client


# ─── Conversation Data ────────────────────────────────────────────────────────

@dataclass
class ConversationRecord:
    """A saved conversation from the database."""
    id: int
    query: str
    plan_json: str
    final_answer: str
    model_used: str
    duration_seconds: float
    created_at: str


# ─── Storage Manager ─────────────────────────────────────────────────────────

class StorageManager:
    """
    Manages persistent storage for the multi-agent system.

    SQLite:  Structured chat history (conversations + messages)
    ChromaDB: User interaction embeddings (grows knowledge base over time)
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db_lock = threading.Lock()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._init_user_kb_collection()

    # ─── SQLite Schema ────────────────────────────────────────────────────

    def _init_db(self):
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        query           TEXT    NOT NULL,
                        plan_json       TEXT    DEFAULT '',
                        final_answer    TEXT    DEFAULT '',
                        model_used      TEXT    DEFAULT '',
                        duration_seconds REAL   DEFAULT 0.0,
                        created_at      TEXT    DEFAULT (datetime('now', 'localtime'))
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER NOT NULL,
                        role            TEXT    NOT NULL,
                        agent_name      TEXT    DEFAULT '',
                        content         TEXT    NOT NULL,
                        model           TEXT    DEFAULT '',
                        tier            INTEGER DEFAULT -1,
                        created_at      TEXT    DEFAULT (datetime('now', 'localtime')),
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_conv_created
                        ON conversations(created_at DESC);

                    CREATE INDEX IF NOT EXISTS idx_msg_conv_id
                        ON messages(conversation_id);
                """)
                conn.commit()
            finally:
                conn.close()

        console.print(f"[dim]✔ SQLite database initialized at {self._db_path}[/dim]")

    # ─── User Knowledge Base Collection (ChromaDB) ────────────────────────

    def _init_user_kb_collection(self):
        try:
            from chromadb.utils import embedding_functions

            client = get_chroma_client()
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._user_kb = client.get_or_create_collection(
                name="user_interactions",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            console.print(
                f"[dim]✔ User interactions collection: "
                f"{self._user_kb.count()} existing documents[/dim]"
            )
        except Exception as e:
            console.print(f"[bold yellow]⚠ User KB collection unavailable: {e}[/bold yellow]")
            self._user_kb = None

    # ─── Save Conversation ────────────────────────────────────────────────

    def save_conversation(
        self,
        query: str,
        plan_json: str,
        agent_outputs: list[dict],
        final_answer: str,
        model_used: str = "",
        duration_seconds: float = 0.0,
    ) -> int:
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO conversations
                       (query, plan_json, final_answer, model_used, duration_seconds)
                       VALUES (?, ?, ?, ?, ?)""",
                    (query, plan_json, final_answer, model_used, duration_seconds),
                )
                conv_id = cursor.lastrowid

                cursor.execute(
                    """INSERT INTO messages
                       (conversation_id, role, agent_name, content)
                       VALUES (?, 'user', '', ?)""",
                    (conv_id, query),
                )

                for output in agent_outputs:
                    cursor.execute(
                        """INSERT INTO messages
                           (conversation_id, role, agent_name, content, model, tier)
                           VALUES (?, 'agent', ?, ?, ?, ?)""",
                        (
                            conv_id,
                            output.get("agent_name", ""),
                            output.get("content", ""),
                            output.get("model", ""),
                            output.get("tier", -1),
                        ),
                    )

                cursor.execute(
                    """INSERT INTO messages
                       (conversation_id, role, agent_name, content, model)
                       VALUES (?, 'judge', 'Judge_Synthesizer', ?, ?)""",
                    (conv_id, final_answer, model_used),
                )

                conn.commit()
                console.print(
                    f"[dim]💾 Conversation #{conv_id} saved to SQLite "
                    f"({len(agent_outputs)} agents, {duration_seconds:.1f}s)[/dim]"
                )

            finally:
                conn.close()

        self._embed_user_interaction(query, final_answer, conv_id)
        return conv_id

    # ─── Embed User Interaction into ChromaDB ─────────────────────────────

    def _embed_user_interaction(self, query: str, answer: str, conv_id: int):
        if self._user_kb is None:
            return

        try:
            from core.knowledge_base_loader import _chunk_text
            chunks = _chunk_text(f"User Query: {query}\n\nAnswer: {answer}")

            if not chunks:
                return

            ids = [f"user_{conv_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": "user_interaction",
                    "category": "user_interactions",
                    "conversation_id": conv_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "query": query[:200],
                }
                for i in range(len(chunks))
            ]

            self._user_kb.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )

            console.print(
                f"[dim]📚 Embedded {len(chunks)} chunks from conversation #{conv_id} "
                f"into knowledge base (total: {self._user_kb.count()} docs)[/dim]"
            )

        except Exception as e:
            console.print(f"[bold yellow]⚠ Failed to embed user interaction: {e}[/bold yellow]")

    # ─── Get Chat History ─────────────────────────────────────────────────

    def get_history(self, limit: int = 20) -> list[ConversationRecord]:
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.execute(
                    """SELECT id, query, plan_json, final_answer,
                              model_used, duration_seconds, created_at
                       FROM conversations
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (limit,),
                )
                return [
                    ConversationRecord(
                        id=row[0], query=row[1], plan_json=row[2],
                        final_answer=row[3], model_used=row[4],
                        duration_seconds=row[5], created_at=row[6],
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    def get_conversation(self, conv_id: int) -> Optional[dict]:
        with self._db_lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?", (conv_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None

                conv = {
                    "id": row[0], "query": row[1], "plan_json": row[2],
                    "final_answer": row[3], "model_used": row[4],
                    "duration_seconds": row[5], "created_at": row[6],
                }

                msg_cursor = conn.execute(
                    """SELECT role, agent_name, content, model, tier, created_at
                       FROM messages
                       WHERE conversation_id = ?
                       ORDER BY created_at""",
                    (conv_id,),
                )
                conv["messages"] = [
                    {
                        "role": m[0], "agent_name": m[1], "content": m[2],
                        "model": m[3], "tier": m[4], "created_at": m[5],
                    }
                    for m in msg_cursor.fetchall()
                ]

                return conv
            finally:
                conn.close()

    # ─── Search History (Semantic) ────────────────────────────────────────

    def search_history(self, query: str, top_k: int = 3) -> list[ConversationRecord]:
        if self._user_kb is None or self._user_kb.count() == 0:
            return []

        try:
            results = self._user_kb.query(
                query_texts=[query],
                n_results=min(top_k * 2, self._user_kb.count()),
            )

            if not results or not results["metadatas"]:
                return []

            conv_ids = set()
            for meta in results["metadatas"][0]:
                cid = meta.get("conversation_id")
                if cid is not None:
                    conv_ids.add(cid)

            records = []
            for cid in list(conv_ids)[:top_k]:
                with self._db_lock:
                    conn = sqlite3.connect(str(self._db_path))
                    try:
                        cursor = conn.execute(
                            """SELECT id, query, plan_json, final_answer,
                                      model_used, duration_seconds, created_at
                               FROM conversations WHERE id = ?""",
                            (cid,),
                        )
                        row = cursor.fetchone()
                        if row:
                            records.append(ConversationRecord(
                                id=row[0], query=row[1], plan_json=row[2],
                                final_answer=row[3], model_used=row[4],
                                duration_seconds=row[5], created_at=row[6],
                            ))
                    finally:
                        conn.close()

            return records

        except Exception as e:
            console.print(f"[bold yellow]⚠ History search failed: {e}[/bold yellow]")
            return []

    # ─── Print History Table ──────────────────────────────────────────────

    def print_history(self, limit: int = 10):
        records = self.get_history(limit)

        if not records:
            console.print("[dim]No conversation history yet.[/dim]")
            return

        table = Table(
            title="📜 Conversation History",
            show_lines=True,
            title_style="bold cyan",
        )
        table.add_column("#", style="cyan", justify="right", width=4)
        table.add_column("Query", style="white", max_width=50)
        table.add_column("Model", style="yellow", max_width=25)
        table.add_column("Duration", style="green", justify="right", width=8)
        table.add_column("Time", style="dim", width=19)

        for rec in records:
            query_preview = rec.query[:47] + "..." if len(rec.query) > 50 else rec.query
            table.add_row(
                str(rec.id),
                query_preview,
                rec.model_used or "—",
                f"{rec.duration_seconds:.1f}s",
                rec.created_at,
            )

        console.print(table)

    # ─── Get User KB Document Count ───────────────────────────────────────

    def get_user_kb_count(self) -> int:
        if self._user_kb is None:
            return 0
        return self._user_kb.count()
