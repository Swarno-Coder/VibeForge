"""
FastAPI Server — Multi-Agent System API Backend.

Provides REST endpoints for:
  - POST /api/query     — Run the full Planner → Executor → Judge pipeline
  - GET  /api/health    — System health (pool, cache, storage, RAG)
  - GET  /api/history   — List conversation history
  - GET  /api/history/N — Detailed conversation
  - GET  /api/cache/stats — Cache statistics
  - DELETE /api/cache   — Clear cache

Start with: uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.dependencies import init_dependencies
from api.routes import health, query, history, cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all dependencies on startup."""
    print("🚀 Starting Multi-Agent API Server...")
    init_dependencies()
    print("✅ All dependencies initialized. Server ready.")
    yield
    print("🛑 Shutting down Multi-Agent API Server...")


app = FastAPI(
    title="Multi-Agent System API",
    description=(
        "REST API for the Gemini Multi-Agent Pipeline. "
        "Supports BYOK (Bring Your Own Key) and provides access to "
        "Planner → Executor → Judge pipeline, conversation history, "
        "cache management, and system health monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(health.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(cache.router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "Multi-Agent System API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /api/query",
            "health": "GET /api/health",
            "history": "GET /api/history",
            "cache_stats": "GET /api/cache/stats",
            "clear_cache": "DELETE /api/cache",
        }
    }
