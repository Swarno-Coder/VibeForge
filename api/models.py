"""
API Pydantic Models — Request/Response schemas for the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /api/query"""
    query: str = Field(..., description="The user's query to process through the multi-agent pipeline")
    api_keys: Optional[dict[int, str]] = Field(
        None,
        description="Optional BYOK keys: {1: 'key1', 2: 'key2', ...}. If omitted, uses default system keys."
    )


# ─── Response Models ─────────────────────────────────────────────────────────

class AgentOutput(BaseModel):
    """Individual agent execution result."""
    agent_name: str
    content: str
    model: str
    tier: int
    tier_label: str = ""
    key_index: int = 0
    criticality: int = 5
    attempts: int = 1


class PlanAgent(BaseModel):
    """An agent in the execution plan."""
    name: str
    role: str
    task: str
    tools_required: list[str] = []
    criticality: int = 5


class QueryResponse(BaseModel):
    """Response body for POST /api/query"""
    answer: str
    plan: list[PlanAgent]
    agent_outputs: list[AgentOutput]
    duration_seconds: float
    model_used: str
    cached: bool = False
    cache_level: Optional[str] = None
    cache_similarity: Optional[float] = None


class HistoryItem(BaseModel):
    """A conversation history entry."""
    id: int
    query: str
    final_answer: str = ""
    model_used: str = ""
    duration_seconds: float = 0.0
    created_at: str = ""


class HistoryDetailResponse(BaseModel):
    """Detailed conversation with messages."""
    id: int
    query: str
    plan_json: str = ""
    final_answer: str = ""
    model_used: str = ""
    duration_seconds: float = 0.0
    created_at: str = ""
    messages: list[dict] = []


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    l1_entries: int = 0
    l2_entries: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    misses: int = 0
    total_lookups: int = 0
    hit_rate_pct: float = 0.0


class PoolStatusResponse(BaseModel):
    """Resource pool status."""
    total_slots: int = 0
    busy_slots: int = 0
    free_slots: int = 0
    blacklisted_slots: int = 0
    active_allocations: int = 0


class HealthResponse(BaseModel):
    """System health/status response."""
    status: str = "ok"
    pool: PoolStatusResponse = PoolStatusResponse()
    cache: CacheStatsResponse = CacheStatsResponse()
    storage_conversations: int = 0
    storage_user_kb_docs: int = 0
    rag_initialized: bool = False
