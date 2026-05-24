"""
MCP tool implementations — 4 context tools + 4 ops tools.

INVARIANT 3: Business calls propagate trace context via inject(headers).
Health checks do NOT propagate.

INVARIANT 4: Every external call gets a CLIENT child span with:
  http.method, http.url, http.status_code, results.count.

INVARIANT 11: Metric labels match span attribute names (tool, status).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastmcp import Context
from opentelemetry.propagate import inject

import db
import telemetry
from config import settings
from vector import hybrid_search

_tracer = telemetry.tracer


# ═══════════════════════════════════════════════════════════════════
# CONTEXT TOOLS (Read-Only)
# ═══════════════════════════════════════════════════════════════════

async def lookup_customer(
    email: str | None = None,
    phone: str | None = None,
    *,
    ctx: Context,
) -> dict[str, Any] | None:
    """Look up a customer by email or phone number."""
    pool = ctx.lifespan_context["pool"]

    with (_tracer.start_as_current_span("postgres lookup_customer") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({"db.operation": "SELECT", "db.table": "users"})

        if email:
            result = await db.get_user_by_email(pool, email)
        elif phone:
            result = await db.get_user_by_phone(pool, phone)
        else:
            result = None

        if span is not None:
            span.set_attribute("db.result", "found" if result else "not_found")

    return result


async def get_recent_orders(
    user_id: str,
    *,
    ctx: Context,
) -> list[dict[str, Any]]:
    """Return the 5 most recent orders for a customer."""
    pool = ctx.lifespan_context["pool"]

    with (_tracer.start_as_current_span("postgres get_recent_orders") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({"db.operation": "SELECT", "db.table": "orders"})

        result = await db.get_recent_orders(pool, user_id)

        if span is not None:
            span.set_attributes({"results.count": len(result)})

    return result


async def get_order_details(
    order_id: str,
    *,
    ctx: Context,
) -> dict[str, Any] | None:
    """Return full order details including product information."""
    pool = ctx.lifespan_context["pool"]

    with (_tracer.start_as_current_span("postgres get_order_details") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({"db.operation": "SELECT", "db.table": "orders JOIN products"})

        result = await db.get_order_with_product(pool, order_id)

        if span is not None:
            span.set_attribute("db.result", "found" if result else "not_found")

    return result


async def check_refund_eligibility(
    order_id: str,
    *,
    ctx: Context,
) -> dict[str, Any]:
    """Check whether an order is eligible for refund."""
    pool = ctx.lifespan_context["pool"]

    with (_tracer.start_as_current_span("postgres check_refund_eligibility") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({"db.operation": "SELECT", "db.table": "orders JOIN products + billing"})

        order = await db.get_order_with_product(pool, order_id)
        if not order:
            return {"eligible": False, "reason": "order_not_found"}

        billing_rows = await db.get_billing_by_order(pool, order_id)

        for br in billing_rows:
            if br["transaction_type"] == "refund" and br["status"] == "completed":
                return {
                    "eligible": False,
                    "reason": "already_refunded",
                    "refund_id": br["gateway_transaction_id"],
                }

        delivery_date = order.get("delivery_date")
        return_window = order.get("return_window_days", 10)

        if delivery_date and return_window:
            days_since = (datetime.now(timezone.utc) - delivery_date).days
            if days_since > return_window:
                return {
                    "eligible": False,
                    "reason": f"return_window_expired ({return_window} days)",
                }

        payment = next(
            (br for br in billing_rows if br["transaction_type"] == "payment"), None
        )
        if not payment:
            return {"eligible": False, "reason": "no_payment_found"}

        result = {
            "eligible": True,
            "reason": "within_return_window",
            "amount": payment["amount"],
            "method": order.get("payment_method", "upi"),
        }

        if span is not None:
            span.set_attribute("refund.eligible", True)

    return result


# ═══════════════════════════════════════════════════════════════════
# OPS TOOLS (Read/Write + External Calls)
# ═══════════════════════════════════════════════════════════════════

async def search_policies(
    query: str,
    top_k: int = 5,
    *,
    ctx: Context,
) -> list[dict[str, Any]]:
    """Search company policies via Qdrant vector search.

    INVARIANT 3: Qdrant is not an HTTP trace-context-aware service.
    No inject(headers) needed. We create a CLIENT span to capture latency.
    """
    with (_tracer.start_as_current_span("qdrant search_policies") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({
                "db.operation": "vector_search",
                "db.collection": settings.qdrant_collection,
                "query.length": len(query),
                "top_k": top_k,
            })

        results = await hybrid_search(query, top_k)

        if span is not None:
            span.set_attributes({"results.count": len(results)})

    return results


async def create_ticket(
    user_id: str,
    query_text: str,
    classification: dict,
    priority: str,
    *,
    ctx: Context,
) -> str:
    """Create a new support ticket in PostgreSQL."""
    pool = ctx.lifespan_context["pool"]

    with (_tracer.start_as_current_span("postgres create_ticket") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({
                "db.operation": "INSERT",
                "db.table": "tickets",
                "ticket.priority": priority,
            })

        ticket_id = await db.insert_ticket(pool, user_id, query_text, classification, priority)

        if span is not None:
            span.set_attribute("ticket.id", ticket_id)

    return ticket_id


async def escalate_to_human(
    ticket_id: str,
    *,
    ctx: Context,
) -> dict[str, Any]:
    """Escalate a ticket to a human agent."""
    pool = ctx.lifespan_context["pool"]

    with (_tracer.start_as_current_span("postgres escalate_to_human") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({
                "db.operation": "UPDATE",
                "db.table": "tickets",
                "ticket.id": ticket_id,
            })

        await db.update_ticket_status(pool, ticket_id, "pending_human")
        await db.set_ticket_priority(pool, ticket_id, "critical")

    return {"status": "escalated", "ticket_id": ticket_id}


async def process_auto_refund(
    order_id: str,
    *,
    ctx: Context,
) -> dict[str, Any]:
    """Process an automatic refund for an order."""
    pool = ctx.lifespan_context["pool"]

    # First check eligibility
    eligibility = await check_refund_eligibility(order_id=order_id, ctx=ctx)

    if not eligibility.get("eligible"):
        return {"status": "failed", "reason": eligibility.get("reason", "not_eligible")}

    amount = float(eligibility["amount"])

    with (_tracer.start_as_current_span("postgres process_refund") if _tracer else _noop()) as span:
        if span is not None:
            span.set_attributes({
                "db.operation": "INSERT",
                "db.table": "billing",
                "refund.amount": amount,
            })

        refund_id = await db.process_refund(pool, order_id, amount)

        if span is not None:
            span.set_attribute("refund.id", refund_id)

    return {
        "status": "completed",
        "refund_id": refund_id,
        "amount": amount,
        "method": eligibility.get("method", "upi"),
    }


# ── Helper for null context ──────────────────────────────────────
from contextlib import nullcontext as _noop