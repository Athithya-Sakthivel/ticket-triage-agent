"""
MCP Context Server — entrypoint.

Provides customer, order, and billing data tools to the agent-service
via the MCP protocol over SSE transport.

Architecture:
    1. Configure OpenTelemetry BEFORE importing FastMCP.
    2. Use @lifespan decorator to manage the asyncpg connection pool.
    3. Tools access the pool through ctx.lifespan_context["pool"].
    4. record_metrics decorator wraps each tool for OTel metrics + logs.
    5. Custom routes (/healthz, /readyz) for Kubernetes probes.

Testing:
    fastmcp call http://localhost:8001/sse lookup_customer email=priya.sharma@email.com
    fastmcp call http://localhost:8001/sse get_recent_orders user_id=a1b2c3d4-...
    curl http://localhost:8001/healthz
    curl http://localhost:8001/readyz
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time

# ═══════════════════════════════════════════════════════════════════
# 0.  Configure basic Python logging (stderr only; OTel bridge was
#     already installed at import‑time by telemetry).
# ═══════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("mcp-server")

# Silence noisy third‑party loggers — only suppress specific loggers,
# NOT the root logger.  Root stays at INFO so our application logs
# are visible and get exported to OTel with Trace IDs.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.sse").setLevel(logging.WARNING)
logging.getLogger("sse_starlette.sse").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════
# 1.  Initialise OTel traces & metrics BEFORE importing FastMCP.
#     (Logs were already set up at telemetry import time.)
# ═══════════════════════════════════════════════════════════════════
import telemetry

telemetry.init_otel()

# ═══════════════════════════════════════════════════════════════════
# 2.  Now it is safe to import FastMCP – its internal get_tracer()
#     will see the registered TracerProvider.
# ═══════════════════════════════════════════════════════════════════
from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan
from starlette.requests import Request
from starlette.responses import PlainTextResponse

import db as _db
import tools as _tools
from config import settings

# ═══════════════════════════════════════════════════════════════════
# Module-level pool reference — set during lifespan startup.
# Custom routes cannot access ctx.lifespan_context, so we store the
# pool here so /readyz can verify database connectivity.
# ═══════════════════════════════════════════════════════════════════
_pool: _db.asyncpg.Pool | None = None


# ═══════════════════════════════════════════════════════════════════
# 3.  Tool decorator — records OTel metrics + emits logs inside
#     FastMCP's auto‑created span.
#
#     FastMCP 3.3.1 auto‑creates a span for every MCP tool call.
#     This decorator wraps the actual tool function, so metrics
#     and logs run INSIDE that span.  LoggingInstrumentor (set up
#     in telemetry.py) automatically injects the span's trace_id
#     and span_id into every log record emitted here.
#
#     Each decorated tool records:
#       mcp_context.requests{status} += 1
#       mcp_context.duration (histogram, seconds)
#       mcp_context.errors += 1 (on exception)
#     And emits:
#       log.info("Tool call started: <name>")
#       log.info("Tool call completed: <name>")
# ═══════════════════════════════════════════════════════════════════

def record_metrics(tool_name: str):
    """Decorator that records OTel metrics and emits logs for a tool call."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            labels = {"tool": tool_name}
            start = time.perf_counter()

            log.info("Tool call started: %s", tool_name)
            telemetry.metrics_.request_counter.add(1, {**labels, "status": "started"})

            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                telemetry.metrics_.request_counter.add(1, {**labels, "status": "success"})
                telemetry.metrics_.request_duration.record(elapsed, labels)
                log.info("Tool call completed: %s (%.3fs)", tool_name, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                telemetry.metrics_.error_counter.add(1, labels)
                telemetry.metrics_.request_counter.add(1, {**labels, "status": "error"})
                telemetry.metrics_.request_duration.record(elapsed, labels)
                log.exception("Tool call failed: %s", tool_name)
                raise
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════
# 4.  Lifespan — manages the asyncpg pool
# ═══════════════════════════════════════════════════════════════════

@lifespan
async def app_lifespan(server: FastMCP):
    """Create the database pool on startup and close it on shutdown."""
    global _pool

    log.info(
        "Creating database pool (min=%d max=%d) …",
        settings.pool_min_size,
        settings.pool_max_size,
    )
    _pool = await _db.create_pool()
    log.info("Database pool ready")
    try:
        yield {"pool": _pool}
    finally:
        log.info("Closing database pool …")
        await _pool.close()
        _pool = None
        log.info("Database pool closed — flushing telemetry")
        telemetry.shutdown()
        log.info("Telemetry flushed — shutdown complete")


# ═══════════════════════════════════════════════════════════════════
# 5.  Create the FastMCP server and register all 8 tools
# ═══════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "mcp-server",
    lifespan=app_lifespan,
)

# ── Context Tools ────────────────────────────────────────────────

@mcp.tool(name="lookup_customer", description="Look up a customer by email or phone number")
@record_metrics("lookup_customer")
async def lookup_customer_tool(
    email: str | None = None,
    phone: str | None = None,
    ctx: Context = None,
) -> dict | None:
    return await _tools.lookup_customer(email=email, phone=phone, ctx=ctx)


@mcp.tool(name="get_recent_orders", description="Return the 5 most recent orders for a customer")
@record_metrics("get_recent_orders")
async def get_recent_orders_tool(user_id: str, ctx: Context = None) -> list[dict]:
    return await _tools.get_recent_orders(user_id=user_id, ctx=ctx)


@mcp.tool(name="get_order_details", description="Return full order details including product information")
@record_metrics("get_order_details")
async def get_order_details_tool(order_id: str, ctx: Context = None) -> dict | None:
    return await _tools.get_order_details(order_id=order_id, ctx=ctx)


@mcp.tool(name="check_refund_eligibility", description="Check whether an order is eligible for refund")
@record_metrics("check_refund_eligibility")
async def check_refund_eligibility_tool(order_id: str, ctx: Context = None) -> dict:
    return await _tools.check_refund_eligibility(order_id=order_id, ctx=ctx)

# ── Ops Tools ────────────────────────────────────────────────────

@mcp.tool(name="search_policies", description="Search company policies using semantic search")
@record_metrics("search_policies")
async def search_policies_tool(query: str, top_k: int = 5, ctx: Context = None) -> list[dict]:
    return await _tools.search_policies(query=query, top_k=top_k, ctx=ctx)


@mcp.tool(name="create_ticket", description="Create a new support ticket")
@record_metrics("create_ticket")
async def create_ticket_tool(
    user_id: str, query_text: str, classification: dict, priority: str, ctx: Context = None
) -> str:
    return await _tools.create_ticket(
        user_id=user_id, query_text=query_text, classification=classification, priority=priority, ctx=ctx
    )


@mcp.tool(name="escalate_to_human", description="Escalate a ticket to a human agent")
@record_metrics("escalate_to_human")
async def escalate_to_human_tool(ticket_id: str, ctx: Context = None) -> dict:
    return await _tools.escalate_to_human(ticket_id=ticket_id, ctx=ctx)


@mcp.tool(name="process_auto_refund", description="Process an automatic refund for an order")
@record_metrics("process_auto_refund")
async def process_auto_refund_tool(order_id: str, ctx: Context = None) -> dict:
    return await _tools.process_auto_refund(order_id=order_id, ctx=ctx)

# ═══════════════════════════════════════════════════════════════════
# 6.  Health‑check endpoints via @mcp.custom_route
#     Custom routes receive a Starlette Request — they do NOT have
#     access to Context or ctx.lifespan_context.  Use module-level
#     _pool instead.
# ═══════════════════════════════════════════════════════════════════

@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> PlainTextResponse:
    """Liveness probe – returns 200 if the process is alive."""
    return PlainTextResponse("ok")


@mcp.custom_route("/readyz", methods=["GET"])
async def readyz(request: Request) -> PlainTextResponse:
    """Readiness probe – verifies database connectivity."""
    if _pool is None:
        return PlainTextResponse("not ready: pool not initialised", status_code=503)
    try:
        async with _pool.acquire() as conn:
            await conn.fetchrow("SELECT 1")
        return PlainTextResponse("ready")
    except Exception as exc:
        return PlainTextResponse(f"not ready: {exc}", status_code=503)


# ═══════════════════════════════════════════════════════════════════
# 7.  Entrypoint
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info(
        "Starting mcp-server on %s:%s (transport=sse)",
        settings.host,
        settings.port,
    )
    
    mcp.run(
        transport="sse",
        host=settings.host,
        port=settings.port,
        log_level=LOG_LEVEL.lower(),
    )