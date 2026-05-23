"""
Minimal retriever service – dense embedding + Qdrant vector search only.

Returns {results: [{text, score}]} with no answer generation, citations,
or metadata.

OpenTelemetry traces, metrics, and logs exported via OTLP/gRPC.

═══════════════════════════════════════════════════════════════════════
CRITICAL: LOGS ARE SET UP AT MODULE‑IMPORT TIME.
This MUST happen before uvicorn starts, otherwise uvicorn's internal
logging.config.dictConfig() will remove the OTel log handler.
Traces & metrics are set up later in the startup event (see below).
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import telemetry
from store import QdrantStore

# ─── Basic Python logging config (stderr only, OTel bridge added below) ──
LOG_LEVEL: str = os.getenv("OTEL_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("retriever-minimal")

# ─── Service configuration ────────────────────────────────────────
DENSE_URL: str = os.getenv("DENSE_URL", "http://dense-svc:8200")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
DENSE_DIM: int = int(os.getenv("DENSE_DIM", "384"))
TOP_K: int = int(os.getenv("TOP_K", "5"))
HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "10.0"))

store = QdrantStore(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
    dense_dim=DENSE_DIM,
)

# ═══════════════════════════════════════════════════════════════════
# 1. OTel LOGS – MUST run at module‑import time
#    (before uvicorn touches logging, otherwise the handler is lost)
# ═══════════════════════════════════════════════════════════════════
from opentelemetry import _logs
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.logging.handler import LoggingHandler
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

_otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
_otel_insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
_service_name = os.getenv("OTEL_SERVICE_NAME", "retriever-minimal")

_resource = Resource.create(
    {
        "service.name": _service_name,
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
    }
)

_logger_provider = LoggerProvider(resource=_resource)
_logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=_otel_endpoint, insecure=_otel_insecure)
    )
)
_logs.set_logger_provider(_logger_provider)

# Attach the OTel handler to the root logger so ALL logs (including
# uvicorn.access, httpx, etc.) are exported as OTLP log records.
_handler = LoggingHandler(level=logging.NOTSET, logger_provider=_logger_provider)
logging.getLogger().setLevel(logging.NOTSET)
logging.getLogger().addHandler(_handler)

# Protect the handler from uvicorn's internal logging reconfiguration.
LoggingInstrumentor().instrument(set_logging_format=False)

log.info("Logs initialised — endpoint=%s", _otel_endpoint)

# ─── FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="retriever-minimal",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class RetrieveResult(BaseModel):
    text: str
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]


# ═══════════════════════════════════════════════════════════════════
# 2. STARTUP – initialise traces & metrics, then instrument FastAPI
# ═══════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup() -> None:
    # Traces & metrics (logs already running since import time)
    telemetry.init_otel()

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    # Pass the real TracerProvider so middleware sees it immediately.
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=telemetry._tracer_provider
    )
    # Instrument httpx so outbound calls to dense/qdrant get trace spans.
    HTTPXClientInstrumentor().instrument()

    log.info(
        "Retriever started dense=%s qdrant=%s collection=%s",
        DENSE_URL,
        QDRANT_URL,
        COLLECTION_NAME,
    )


# ═══════════════════════════════════════════════════════════════════
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")

    labels = {"model": "dense"}
    # Safe – metrics_ namespace always returns a usable object.
    telemetry.metrics_.requests_in_progress.add(1, labels)
    start = time.perf_counter()
    status = "success"

    try:
        # Use a real span if the tracer is available; else no‑op.
        if telemetry.tracer is not None:
            span = telemetry.tracer.start_as_current_span("POST /retrieve")
        else:
            from contextlib import nullcontext

            span = nullcontext()

        with span as s:
            if s is not None:
                s.set_attributes(
                    {"query.length": len(req.query), "top_k": req.top_k}
                )

            # 1. Get embedding from dense service
            query_vector = await _embed_query(req.query)

            # 2. Search Qdrant
            results = await store.search(query_vector, limit=req.top_k)

            if s is not None:
                s.set_attributes(
                    {
                        "embedding.dim": len(query_vector),
                        "results.count": len(results),
                    }
                )

            log.info(
                "Retrieve completed query_len=%d results=%d",
                len(req.query),
                len(results),
            )

        return {
            "results": [
                {"text": r["text"], "score": r["score"]} for r in results
            ]
        }

    except HTTPException:
        status = "client_error"
        raise
    except Exception:
        status = "server_error"
        telemetry.metrics_.error_counter.add(
            1, {**labels, "error_type": "exception"}
        )
        log.exception("Retrieve failed")
        raise HTTPException(500, "Internal server error")
    finally:
        elapsed = time.perf_counter() - start
        telemetry.metrics_.request_counter.add(1, {**labels, "status": status})
        telemetry.metrics_.request_duration.record(elapsed, labels)
        telemetry.metrics_.requests_in_progress.add(-1, labels)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, Any]:
    dense_ok = await _check_dense_health()
    qdrant_ok = await store.ping()
    return {
        "status": "ready" if (dense_ok and qdrant_ok) else "not_ready",
        "dense": dense_ok,
        "qdrant": qdrant_ok,
    }


# ─── Internal helpers ──────────────────────────────────────────────
async def _embed_query(text: str) -> list[float]:
    """Call dense embedding service, return L2‑normalised vector."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(f"{DENSE_URL}/embed", json={"texts": [text]})
        resp.raise_for_status()
        vectors = resp.json().get("vectors", [])
        if not vectors:
            raise RuntimeError("dense service returned empty vectors")

    vec = [float(x) for x in vectors[0]]
    if len(vec) != DENSE_DIM:
        raise RuntimeError(
            f"Dimension mismatch: expected {DENSE_DIM}, got {len(vec)}"
        )

    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.astype(float).tolist()


async def _check_dense_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DENSE_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# 3. SHUTDOWN – flush all telemetry
# ═══════════════════════════════════════════════════════════════════
@app.on_event("shutdown")
async def shutdown() -> None:
    log.info("Shutting down retriever")
    await store.close()
    telemetry.shutdown()


# ═══════════════════════════════════════════════════════════════════
# 4. ENTRYPOINT – log_config=None keeps our OTel handler alive
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8001")),
        log_level=LOG_LEVEL.lower(),
        log_config=None,  # ← CRITICAL: prevent uvicorn dictConfig wipe
        loop="uvloop",
        http="httptools",
    )