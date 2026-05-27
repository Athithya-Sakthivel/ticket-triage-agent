"""
FastAPI routes for the Agent Service.
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select, func

from db import AsyncSessionLocal, HumanOverride, Ticket
from state import AgentState, Context

log = logging.getLogger("agent-service")
router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    user_id: str | None = None


class OverrideRequest(BaseModel):
    ticket_id: str
    original_classification: dict
    corrected_classification: dict
    reason: str | None = None
    overridden_by: str | None = None


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()

    app = websocket.app
    graph = app.state.graph
    triage_program = app.state.triage_program
    resolver_lm = app.state.resolver_lm
    mcp_client = app.state.mcp_client
    tracer = app.state.tracer

    ctx = Context(
        triage_program=triage_program,
        mcp_client=mcp_client,
        resolver_lm=resolver_lm,
        tracer=tracer,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            query = data.get("query", "").strip()

            if not query:
                await websocket.send_json({"error": "query is required"})
                continue

            state: AgentState = {
                "messages": [{"role": "user", "content": query}],
                "query_text": query,
                "user_id": data.get("user_id"),
                "thread_id": session_id,
                "guardrail_rejected": False,
                "classification": None,
                "customer_context": None,
                "tool_results": [],
                "resolution_type": None,
                "ticket_id": None,
                "final_response": None,
                "error": None,
            }

            config = {"configurable": {"thread_id": session_id}}
            result = await graph.ainvoke(state, config, context=ctx)

            await websocket.send_json({
                "response": result.get("final_response", ""),
                "resolution_type": result.get("resolution_type"),
                "ticket_id": result.get("ticket_id"),
            })

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s", session_id)
    except Exception:
        log.exception("WebSocket error")
        await websocket.close()


@router.get("/admin/queue")
async def get_ticket_queue(limit: int = 50, offset: int = 0):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Ticket)
            .where(Ticket.status.in_(["open", "pending_human"]))
            .order_by(Ticket.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        tickets = result.scalars().all()
        return {
            "tickets": [
                {
                    "id": str(t.id),
                    "user_id": str(t.user_id),
                    "query_text": t.query_text,
                    "classification": t.classification,
                    "resolution_type": t.resolution_type,
                    "status": t.status,
                    "priority": t.priority,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in tickets
            ],
            "count": len(tickets),
        }


@router.post("/admin/override")
async def submit_override(body: OverrideRequest):
    async with AsyncSessionLocal() as session:
        override = HumanOverride(
            ticket_id=uuid.UUID(body.ticket_id),
            original_classification=body.original_classification,
            corrected_classification=body.corrected_classification,
            reason=body.reason,
            overridden_by=body.overridden_by,
        )
        session.add(override)
        await session.commit()
        return {"status": "stored", "id": str(override.id)}


@router.get("/admin/analytics")
async def get_analytics():
    async with AsyncSessionLocal() as session:
        total_result = await session.execute(select(func.count()).select_from(Ticket))
        total = total_result.scalar() or 0
        resolved_result = await session.execute(
            select(func.count())
            .select_from(Ticket)
            .where(Ticket.resolution_type == "auto_resolved")
        )
        auto_resolved = resolved_result.scalar() or 0
        override_result = await session.execute(select(func.count()).select_from(HumanOverride))
        overrides = override_result.scalar() or 0

    return {
        "total_tickets": total,
        "auto_resolved": auto_resolved,
        "auto_resolution_rate": round(auto_resolved / total * 100, 1) if total > 0 else 0,
        "human_overrides": overrides,
    }