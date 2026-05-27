"""
LangGraph node implementations – production ready.

All nodes receive runtime dependencies via Runtime[Context] (LangGraph v1.2+).
Metrics are recorded for every node via the _record_node helper.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import nullcontext
from typing import Any

from langgraph.runtime import Runtime

from config import settings
from state import AgentState, Context
import observability

log = logging.getLogger("agent-service")


async def _record_node(node_name: str, coro):
    labels = {"node": node_name}
    start = time.perf_counter()
    observability.metrics_.request_counter.add(1, {**labels, "status": "started"})
    try:
        result = await coro
        elapsed = time.perf_counter() - start
        observability.metrics_.request_counter.add(1, {**labels, "status": "success"})
        observability.metrics_.request_duration.record(elapsed, labels)
        log.info("Node completed: %s (%.3fs)", node_name, elapsed)
        return result
    except Exception:
        elapsed = time.perf_counter() - start
        observability.metrics_.error_counter.add(1, labels)
        observability.metrics_.request_counter.add(1, {**labels, "status": "error"})
        observability.metrics_.request_duration.record(elapsed, labels)
        log.exception("Node failed: %s", node_name)
        raise


# ═══════════════════════════════════════════════════════════════════
# 1. GUARDRAIL + CLASSIFIER
# ═══════════════════════════════════════════════════════════════════
async def guardrail_classifier(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Safety check + ticket classification via DSPy TriageProgram."""

    triage_program = runtime.context.triage_program
    tracer = runtime.context.tracer

    async def _run():
        query = state["query_text"].strip()
        if not query or len(query) < 3:
            return {
                "guardrail_rejected": True,
                "final_response": "I couldn't understand your message. Could you please rephrase?",
                "resolution_type": "escalated",
            }

        with (tracer.start_as_current_span("guardrail_classifier") if tracer else nullcontext()) as span:
            if span:
                span.set_attribute("openinference.span.kind", "GUARDRAIL")

            result = await asyncio.get_event_loop().run_in_executor(
                None, triage_program, query
            )

            if span:
                span.set_attribute("llm.model_name", settings.llm_safeguard_model)
                span.set_attribute("guardrail.safety", result.get("safety", "UNKNOWN"))

        return result

    result = await _record_node("guardrail_classifier", _run())

    if result.get("safety") == "UNSAFE":
        return {
            "guardrail_rejected": True,
            "classification": result,
            "final_response": "Your message has been flagged for review. A human agent will respond shortly.",
            "resolution_type": "escalated",
        }

    return {
        "guardrail_rejected": False,
        "classification": result,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. CONTEXT GATHERER
# ═══════════════════════════════════════════════════════════════════
async def context_gatherer(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Fetch customer profile and recent orders via MCP tools in parallel."""

    mcp_client = runtime.context.mcp_client
    tracer = runtime.context.tracer

    async def _run():
        query = state["query_text"]
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query)
        email = email_match.group(0) if email_match else None
        user_id = state.get("user_id")

        if not email and not user_id:
            log.warning("No customer identifier – skipping context")
            return {"customer_context": None}

        with (tracer.start_as_current_span("context_gatherer") if tracer else nullcontext()) as span:
            if span:
                span.set_attribute("openinference.span.kind", "CHAIN")

            customer = None
            if email:
                customer = await mcp_client.call_tool("lookup_customer", {"email": email})
            if customer and customer.get("id"):
                user_id = customer["id"]

            orders = []
            if user_id:
                orders = await mcp_client.call_tool("get_recent_orders", {"user_id": user_id})

            customer_context = {"customer": customer, "orders": orders}
            if span:
                span.set_attribute("customer.found", customer is not None)
                span.set_attribute("orders.count", len(orders))

            return {"user_id": user_id, "customer_context": customer_context}

    return await _record_node("context_gatherer", _run())


# ═══════════════════════════════════════════════════════════════════
# 3. AGENTIC RESOLVER
# ═══════════════════════════════════════════════════════════════════
RESOLVER_SYSTEM_PROMPT = """You are a helpful, empathetic customer service agent for Kestral, an Indian e-commerce company.

You have access to these tools:
- search_policies(query) — search company policies (returns, refunds, delivery, warranty)
- check_refund_eligibility(order_id) — check if an order can be refunded
- issue_wallet_credit(user_id, amount, reason) — issue store credit (max Rs.500)
- schedule_return_pickup(order_id, pickup_date) — schedule a return pickup
- create_ticket(user_id, query_text, classification, priority, assigned_team) — create a support ticket
- escalate_to_human(ticket_id) — escalate a ticket to a human agent
- route_to_team(ticket_id, team) — assign a ticket to a specific team

Rules:
1. Gather information step by step. Never jump to conclusions.
2. Always ground your responses in retrieved policy documents.
3. Only use issue_wallet_credit for policy-driven compensation (delays, goodwill) and never more than Rs.500.
4. Only use schedule_return_pickup if the order is eligible for return.
5. For high-value claims (>Rs.10,000) or security issues, do NOT resolve automatically – create a ticket and escalate.
6. Use the customer's name when available. Include specific timelines and amounts from policies.
7. Output your reasoning, then the tool call or final answer.

Respond in JSON:
- If you need to call a tool: {"action": "tool_call", "tool": "<name>", "args": {<params>}, "thought": "<why>"}
- If you have enough information to respond: {"action": "final_answer", "response": "<message to customer>"}
"""

MAX_RESOLVER_STEPS = 5


async def agentic_resolver(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Agentic loop that resolves tickets by calling tools dynamically."""

    resolver_lm = runtime.context.resolver_lm
    mcp_client = runtime.context.mcp_client
    tracer = runtime.context.tracer

    async def _run():
        query = state["query_text"]
        classification = state.get("classification", {})
        customer_ctx = state.get("customer_context") or {}
        customer = customer_ctx.get("customer") or {}
        orders = customer_ctx.get("orders") or []

        messages = [
            {"role": "system", "content": RESOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": f"""Customer: {customer.get('full_name', 'Unknown')} ({customer.get('segment', 'unknown')})
Query: {query}
Classification: {json.dumps(classification)}
Recent orders: {json.dumps(orders[:3]) if orders else 'None'}"""},
        ]

        tool_results: list[dict] = []

        with (tracer.start_as_current_span("agentic_resolver") if tracer else nullcontext()) as root_span:
            if root_span:
                root_span.set_attribute("openinference.span.kind", "AGENT")

            for step in range(MAX_RESOLVER_STEPS):
                with (tracer.start_as_current_span(f"resolver_step_{step}") if tracer else nullcontext()) as step_span:
                    try:
                        raw = await resolver_lm.acall(messages=messages)
                        raw_text = raw[0] if isinstance(raw, list) else raw
                        result = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
                    except (json.JSONDecodeError, KeyError):
                        log.warning("Resolver step %d: bad JSON, forcing final answer", step)
                        messages.append({"role": "user", "content": "Please give a final answer now. Do not call more tools."})
                        raw = await resolver_lm.acall(messages=messages)
                        raw_text = raw[0] if isinstance(raw, list) else raw
                        return {
                            "final_response": raw_text if isinstance(raw_text, str) else str(raw_text),
                            "resolution_type": "auto_resolved",
                            "tool_results": tool_results,
                        }

                    if step_span:
                        step_span.set_attribute("resolver.action", result.get("action", "unknown"))

                    if result.get("action") == "final_answer":
                        if step_span:
                            step_span.set_attribute("resolver.steps", step + 1)
                        return {
                            "final_response": result["response"],
                            "resolution_type": "auto_resolved",
                            "tool_results": tool_results,
                        }

                    tool_name = result.get("tool")
                    tool_args = result.get("args", {})

                    if tool_name == "issue_wallet_credit" and tool_args.get("amount", 0) > settings.max_wallet_credit_amount:
                        tool_output = {"status": "rejected", "reason": f"Amount exceeds maximum of Rs.{settings.max_wallet_credit_amount}"}
                    else:
                        with (tracer.start_as_current_span(f"tool:{tool_name}") if tracer else nullcontext()) as tool_span:
                            if tool_span:
                                tool_span.set_attribute("openinference.span.kind", "TOOL")
                                tool_span.set_attribute("tool.name", tool_name)
                            try:
                                tool_output = await mcp_client.call_tool(tool_name, tool_args)
                                if tool_span:
                                    tool_span.set_attribute("tool.status", "success")
                            except Exception as exc:
                                tool_output = {"error": str(exc)}
                                if tool_span:
                                    tool_span.set_attribute("tool.status", "error")

                    tool_results.append({"tool": tool_name, "args": tool_args, "result": tool_output})
                    messages.append({"role": "assistant", "content": json.dumps(result)})
                    messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_output)}"})

            messages.append({"role": "user", "content": "Please give a final answer to the customer now."})
            raw = await resolver_lm.acall(messages=messages)
            raw_text = raw[0] if isinstance(raw, list) else raw
            return {
                "final_response": raw_text if isinstance(raw_text, str) else str(raw_text),
                "resolution_type": "auto_resolved",
                "tool_results": tool_results,
            }

    return await _record_node("agentic_resolver", _run())


# ═══════════════════════════════════════════════════════════════════
# 4. HUMAN ESCALATE
# ═══════════════════════════════════════════════════════════════════
TEAM_ROUTING = {
    "wrong_item_delivered": "order_fulfillment",
    "damaged_product": "service_center",
    "late_delivery": "logistics",
    "refund_status": "payments",
    "cancellation_request": "order_fulfillment",
    "return_request": "order_fulfillment",
    "warranty_claim": "service_center",
    "payment_issue": "payments",
    "account_issue": "senior_support",
    "general_inquiry": "general_support",
    "complaint": "senior_support",
}


async def human_escalate(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Create a ticket, escalate, and route to the correct team."""

    mcp_client = runtime.context.mcp_client
    tracer = runtime.context.tracer

    async def _run():
        classification = state.get("classification", {})
        customer_ctx = state.get("customer_context") or {}
        customer = customer_ctx.get("customer") or {}
        urgency = classification.get("urgency", 5)
        intent = classification.get("intent", "general_inquiry")

        priority = "critical" if urgency >= 9 else "high" if urgency >= 7 else "medium"
        sla = "2 hours" if urgency >= 9 else "4 hours" if urgency >= 7 else "24 hours"
        team = TEAM_ROUTING.get(intent, "general_support")

        with (tracer.start_as_current_span("human_escalate") if tracer else nullcontext()) as span:
            if span:
                span.set_attribute("openinference.span.kind", "CHAIN")

            try:
                ticket_id = await mcp_client.call_tool("create_ticket", {
                    "user_id": state.get("user_id", "unknown"),
                    "query_text": state["query_text"],
                    "classification": classification,
                    "priority": priority,
                    "assigned_team": team,
                })
                await mcp_client.call_tool("escalate_to_human", {"ticket_id": ticket_id})
                await mcp_client.call_tool("route_to_team", {"ticket_id": ticket_id, "team": team})
                if span:
                    span.set_attribute("ticket.id", ticket_id)
                    span.set_attribute("ticket.priority", priority)
                    span.set_attribute("ticket.team", team)
            except Exception:
                log.exception("Failed to create/escalate ticket")
                ticket_id = "unknown"

        response = (
            f"{customer.get('full_name', 'Hello')}, your issue has been flagged as "
            f"{priority} priority. A {team.replace('_', ' ')} specialist will review your case within {sla}. "
            f"Your reference number is {ticket_id}."
        )
        return {
            "ticket_id": ticket_id,
            "final_response": response,
            "resolution_type": "escalated",
        }

    return await _record_node("human_escalate", _run())