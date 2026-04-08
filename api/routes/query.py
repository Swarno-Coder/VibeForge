"""
Query Route — POST /api/query

Runs the full Planner → Executor → Judge pipeline.
Supports BYOK (Bring Your Own Key) via request body.
"""

import json
import time
from fastapi import APIRouter, HTTPException
from api.models import QueryRequest, QueryResponse, PlanAgent, AgentOutput
from api.dependencies import get_default_clients, get_pool, get_storage, get_cache
from core.llm import get_clients
from core.resource_pool import ResourcePool
from core.planner import build_plan
from core.executor import execute_plan
from core.judge import evaluate_and_synthesize
from core.rag_engine import add_user_interaction

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """
    Execute a query through the multi-agent pipeline.

    If api_keys are provided (BYOK), creates temporary clients and pool.
    Otherwise uses the default system keys.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Resolve clients: BYOK or default
    if request.api_keys:
        try:
            clients = get_clients(api_keys=request.api_keys)
            pool = ResourcePool(clients)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid API keys: {e}")
    else:
        clients = get_default_clients()
        pool = get_pool()

    storage = get_storage()
    cache = get_cache()

    # ── Cache Lookup ──────────────────────────────────────────────
    cached = cache.lookup(query)
    if cached is not None:
        return QueryResponse(
            answer=cached.cached_answer,
            plan=[],
            agent_outputs=[],
            duration_seconds=0.0,
            model_used="cache",
            cached=True,
            cache_level=cached.cache_level,
            cache_similarity=cached.similarity,
        )

    # ── Full Pipeline ─────────────────────────────────────────────
    t_start = time.time()

    try:
        # Phase 1: Planning
        plan = build_plan(clients, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planning failed: {e}")

    # Phase 2: Execution
    agent_outputs_list = []
    final_context = execute_plan(clients, plan, pool, agent_outputs=agent_outputs_list)

    if not final_context.strip():
        raise HTTPException(status_code=500, detail="Agents failed to produce any context")

    # Phase 3: Synthesis
    final_answer, model_used = evaluate_and_synthesize(
        clients, pool, query, final_context
    )

    duration = time.time() - t_start

    # ── Persist ───────────────────────────────────────────────────
    if final_answer.strip():
        plan_json = json.dumps(
            {"agents": [a.model_dump() for a in plan.agents]},
            indent=2,
        )
        storage.save_conversation(
            query=query,
            plan_json=plan_json,
            agent_outputs=agent_outputs_list,
            final_answer=final_answer,
            model_used=model_used,
            duration_seconds=duration,
        )
        cache.store(query, final_answer)
        add_user_interaction(query, final_answer)

    # ── Build Response ────────────────────────────────────────────
    plan_agents = [
        PlanAgent(
            name=a.name,
            role=a.role,
            task=a.task[:500],  # truncate for response
            tools_required=a.tools_required,
            criticality=a.criticality,
        )
        for a in plan.agents
    ]

    agent_out_models = [
        AgentOutput(
            agent_name=ao.get("agent_name", ""),
            content=ao.get("content", "")[:2000],  # truncate
            model=ao.get("model", ""),
            tier=ao.get("tier", -1),
            tier_label=ao.get("tier_label", ""),
            key_index=ao.get("key_index", 0),
            criticality=ao.get("criticality", 5),
            attempts=ao.get("attempts", 1),
        )
        for ao in agent_outputs_list
    ]

    return QueryResponse(
        answer=final_answer,
        plan=plan_agents,
        agent_outputs=agent_out_models,
        duration_seconds=round(duration, 2),
        model_used=model_used,
        cached=False,
    )
