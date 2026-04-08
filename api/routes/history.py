"""
History Routes — GET /api/history, GET /api/history/{id}
"""

from fastapi import APIRouter, HTTPException
from api.models import HistoryItem, HistoryDetailResponse
from api.dependencies import get_storage

router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[HistoryItem])
async def list_history(limit: int = 20):
    """List recent conversations."""
    storage = get_storage()
    records = storage.get_history(limit=limit)
    return [
        HistoryItem(
            id=r.id,
            query=r.query,
            final_answer=r.final_answer[:500] if r.final_answer else "",
            model_used=r.model_used,
            duration_seconds=r.duration_seconds,
            created_at=r.created_at,
        )
        for r in records
    ]


@router.get("/history/{conv_id}", response_model=HistoryDetailResponse)
async def get_conversation_detail(conv_id: int):
    """Get detailed conversation including all messages."""
    storage = get_storage()
    conv = storage.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found")

    return HistoryDetailResponse(
        id=conv["id"],
        query=conv["query"],
        plan_json=conv.get("plan_json", ""),
        final_answer=conv.get("final_answer", ""),
        model_used=conv.get("model_used", ""),
        duration_seconds=conv.get("duration_seconds", 0.0),
        created_at=conv.get("created_at", ""),
        messages=conv.get("messages", []),
    )
