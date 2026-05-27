"""
Agent Service – entrypoint.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("agent-service")

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openinference").setLevel(logging.WARNING)

import observability
observability.init_otel()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from compile_dspy import load_or_compile_triage
from config import settings, create_safeguard_lm, create_resolver_lm
from graph import compile_graph
from mcp_client import MCPClientManager
from routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Agent Service...")

    safeguard_lm = create_safeguard_lm()
    resolver_lm = create_resolver_lm()
    app.state.resolver_lm = resolver_lm

    triage_program = load_or_compile_triage(lm=safeguard_lm)
    app.state.triage_program = triage_program

    app.state.mcp_client = MCPClientManager()
    await app.state.mcp_client.connect()

    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url
    ) as checkpointer:
        await checkpointer.setup()
        app.state.graph = await compile_graph(checkpointer=checkpointer)
        app.state.tracer = observability.tracer

        try:
            yield
        finally:
            log.info("Shutting down...")
            await app.state.mcp_client.close()
            observability.shutdown()
            log.info("Shutdown complete")


app = FastAPI(
    title="agent-service",
    version=settings.service_version,
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if hasattr(app.state, "mcp_client") and app.state.mcp_client._client:
        return {"status": "ready"}
    return {"status": "not_ready"}, 503


if __name__ == "__main__":
    import uvicorn

    log.info("Starting on %s:%s", settings.host, settings.port)
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=LOG_LEVEL.lower(),
        log_config=None,
    )