"""
Health Route — GET /api/health
"""

from fastapi import APIRouter
from api.models import HealthResponse, PoolStatusResponse, CacheStatsResponse
from api.dependencies import get_pool, get_storage, get_cache, is_initialized
import core.rag_engine as rag_module

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check — pool, cache, storage, RAG status."""
    if not is_initialized():
        return HealthResponse(status="initializing")

    pool = get_pool()
    storage = get_storage()
    cache = get_cache()

    pool_status = pool.get_pool_status()
    cache_stats = cache.stats()

    return HealthResponse(
        status="ok",
        pool=PoolStatusResponse(
            total_slots=pool_status["total_slots"],
            busy_slots=pool_status["busy_slots"],
            free_slots=pool_status["free_slots"],
            blacklisted_slots=pool_status["blacklisted_slots"],
            active_allocations=pool_status["active_allocations"],
        ),
        cache=CacheStatsResponse(**cache_stats),
        storage_conversations=len(storage.get_history(limit=9999)),
        storage_user_kb_docs=storage.get_user_kb_count(),
        rag_initialized=rag_module._initialized,
    )
