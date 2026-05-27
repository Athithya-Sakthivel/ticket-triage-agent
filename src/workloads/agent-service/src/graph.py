"""
LangGraph graph definition for the ticket triage agent.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from config import settings
from nodes import (
    agentic_resolver,
    context_gatherer,
    guardrail_classifier,
    human_escalate,
)
from state import AgentState, Context, route_after_guardrail

log = logging.getLogger("agent-service")


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState, context_schema=Context)

    builder.add_node("guardrail_classifier", guardrail_classifier)
    builder.add_node("context_gatherer", context_gatherer)
    builder.add_node("agentic_resolver", agentic_resolver)
    builder.add_node("human_escalate", human_escalate)

    builder.add_edge(START, "guardrail_classifier")

    builder.add_conditional_edges(
        "guardrail_classifier",
        route_after_guardrail,
        {
            "human_escalate": "human_escalate",
            "context_gatherer": "context_gatherer",
        },
    )

    builder.add_edge("context_gatherer", "agentic_resolver")
    builder.add_edge("human_escalate", END)
    builder.add_edge("agentic_resolver", END)

    return builder


async def compile_graph(checkpointer: AsyncPostgresSaver | None = None):
    builder = build_graph()

    if checkpointer is None:
        async with AsyncPostgresSaver.from_conn_string(
            settings.database_url
        ) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(checkpointer=checkpointer)
            log.info("Graph compiled with AsyncPostgresSaver")
            return graph

    graph = builder.compile(checkpointer=checkpointer)
    log.info("Graph compiled with provided checkpointer")
    return graph