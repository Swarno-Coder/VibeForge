"""
Cache Routes — GET /api/cache/stats, DELETE /api/cache
"""

from fastapi import APIRouter
from api.models import CacheStatsResponse
from api.dependencies import get_cache

router = APIRouter(tags=["cache"])


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Get cache statistics (L1 + L2)."""
    cache = get_cache()
    stats = cache.stats()
    return CacheStatsResponse(**stats)


@router.delete("/cache")
async def clear_cache():
    """Clear the entire query cache (L1 + L2)."""
    cache = get_cache()
    cache.clear()
    return {"status": "ok", "message": "Cache cleared successfully"}
