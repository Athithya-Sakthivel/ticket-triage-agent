"""
AgentState TypedDict for LangGraph.

Uses MessagesState as the base so message history is automatically
appended via the add_messages reducer.  Additional fields carry
business context through the graph.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class AgentState(MessagesState):
    """Full agent state flowing through every node.

    Inherits from MessagesState, which provides:
      - messages: list[AnyMessage] with add_messages reducer
    """

    # ── Input ─────────────────────────────────────────────────────
    user_id: str | None
    query_text: str
    thread_id: str  # idempotency key for LangGraph checkpointing

    # ── Guardrail + Classification ─────────────────────────────────
    guardrail_rejected: bool
    classification: dict[str, Any] | None  # {intent, urgency, sentiment, auto_resolvable}

    # ── Context ────────────────────────────────────────────────────
    customer_context: dict[str, Any] | None  # customer profile + orders

    # ── Resolution ─────────────────────────────────────────────────
    policy_chunks: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]  # raw MCP tool call results
    resolution_type: str | None  # "auto_resolved" | "escalated" | "deflected"
    ticket_id: str | None

    # ── Output ─────────────────────────────────────────────────────
    final_response: str | None
    error: str | None