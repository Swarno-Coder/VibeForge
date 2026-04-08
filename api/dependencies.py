"""
API Dependencies — Shared singleton state for the FastAPI backend.

Holds references to ResourcePool, StorageManager, QueryCache, and clients.
Initialized once during server startup via init_dependencies().
"""

import os
from dotenv import load_dotenv
from core.llm import get_clients
from core.resource_pool import ResourcePool
from core.storage import StorageManager
from core.cache import QueryCache
from core.rag_engine import initialize_rag

# ─── Singleton State ──────────────────────────────────────────────────────────

_clients = None
_pool = None
_storage = None
_cache = None
_initialized = False


def init_dependencies():
    """Initialize all shared dependencies. Called once at server startup."""
    global _clients, _pool, _storage, _cache, _initialized

    if _initialized:
        return

    load_dotenv()

    _clients = get_clients()
    _pool = ResourcePool(_clients)
    initialize_rag()
    _storage = StorageManager()
    _cache = QueryCache()
    _initialized = True


def get_default_clients():
    """Get the default API key clients."""
    if _clients is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _clients


def get_pool() -> ResourcePool:
    if _pool is None:
        raise RuntimeError("Dependencies not initialized.")
    return _pool


def get_storage() -> StorageManager:
    if _storage is None:
        raise RuntimeError("Dependencies not initialized.")
    return _storage


def get_cache() -> QueryCache:
    if _cache is None:
        raise RuntimeError("Dependencies not initialized.")
    return _cache


def is_initialized() -> bool:
    return _initialized
