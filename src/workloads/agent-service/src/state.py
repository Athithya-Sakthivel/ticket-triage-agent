"""
AgentState TypedDict, Context dataclass, and routing conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import MessagesState

from config import settings


class AgentState(MessagesState):
    """Full agent state flowing through every node."""

    user_id: str | None
    query_text: str
    thread_id: str

    guardrail_rejected: bool
    classification: dict[str, Any] | None

    customer_context: dict[str, Any] | None

    tool_results: list[dict[str, Any]]
    resolution_type: str | None
    ticket_id: str | None

    final_response: str | None
    error: str | None


@dataclass
class Context:
    """Runtime dependencies injected via graph.ainvoke(..., context=Context(...))."""
    triage_program: Any
    mcp_client: Any
    resolver_lm: Any
    tracer: Any


def route_after_guardrail(state: AgentState) -> str:
    if state.get("guardrail_rejected", False):
        return "human_escalate"

    classification = state.get("classification", {})
    if classification.get("urgency", 0) >= settings.urgency_escalate_threshold:
        return "human_escalate"

    if not classification.get("auto_resolvable", True):
        return "human_escalate"

    return "context_gatherer"