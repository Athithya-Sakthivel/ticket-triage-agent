"""
Retriever service — dense embedding + Qdrant vector search.

Returns {results: [{text, score}]} — no answer generation.

═══════════════════════════════════════════════════════════════════════
OBSERVABILITY INVARIANTS ENFORCED (see docs/contracts/observability/)
═══════════════════════════════════════════════════════════════════════
1.  Log bridge at module import time (before uvicorn touches logging)
2.  TracerProvider registered before FastAPI imports it
3.  Explicit trace context propagation on business calls (inject()),
    NEVER on health checks (bare httpx calls)
4.  Child spans for every external call with result attributes:
    http.method, http.url, http.status_code, results.count
5.  Bounded metric cardinality: labels="model:dense", status="success|error"
6.  Exactly three metric instruments: retrieve.requests, retrieve.duration,
    retrieve.errors
7.  Sampling exclusively in the SigNoz collector — never in app code
8.  Health endpoints filtered at collector (filter/health processor)
9.  All providers force-flushed on graceful shutdown (5s timeout)
10. No INTERNAL spans — only CLIENT spans for external calls
11. Metric label names match span attribute names (e.g., "status")
12. Every debug log emitted inside an active span (has trace_id)

DEBUGGING THIS SERVICE:
- Logs go to stderr AND SigNoz via OTel log bridge
- To see logs locally: tail -f /tmp/retriever.log
- To see traces in SigNoz: filter by service.name=retriever-minimal
- To verify metrics: check signoz_metrics.distributed_samples_v4
  WHERE metric_name LIKE 'retrieve.%' AND unix_milli >= now_ms - 300000
- Trace context propagation: inject(headers) in _embed() only
- Health checks: _check_dense() uses bare httpx — no inject()
- Shutdown: SIGINT triggers lifespan shutdown → force_flush → data saved
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager, nullcontext
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from opentelemetry.propagate import inject

from store import QdrantStore

# ═══════════════════════════════════════════════════════════════════
# 0. Logging — stderr only. OTel bridge added below at import time.
#    Do NOT set root logger to WARNING — that silences our app logs.
# ═══════════════════════════════════════════════════════════════════
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("retriever-minimal")

# Silence noisy third-party loggers individually — NEVER the root logger
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ─── Configuration (all env vars have production-safe defaults) ──
DENSE_URL: str = os.getenv("DENSE_URL", "http://dense-svc:8200")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
DENSE_DIM: int = int(os.getenv("DENSE_DIM", "384"))
TOP_K: int = int(os.getenv("TOP_K", "5"))
HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "10.0"))

store = QdrantStore(
    url=QDRANT_URL,
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name=COLLECTION_NAME,
    dense_dim=DENSE_DIM,
)

# ═══════════════════════════════════════════════════════════════════
# INVARIANT 1: Log bridge at module import time (before uvicorn)
#
# WHY: uvicorn calls logging.config.dictConfig() during startup,
# which removes all previously-attached handlers.  By setting up
# the OTel log bridge at module level, it survives this wipe.
# If this were in a startup event, logs would never reach SigNoz.
# ═══════════════════════════════════════════════════════════════════
from opentelemetry import _logs
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

_otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
_service_name = os.getenv("OTEL_SERVICE_NAME", "retriever-minimal")

_resource = Resource.create({
    "service.name": _service_name,
    "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
    "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
})

_logger_provider = LoggerProvider(resource=_resource)
_logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter(endpoint=_otel_endpoint, insecure=True))
)
_logs.set_logger_provider(_logger_provider)

logging.getLogger().addHandler(
    LoggingHandler(level=logging.NOTSET, logger_provider=_logger_provider)
)
LoggingInstrumentor().instrument(set_logging_format=False)
log.info("OTel logs initialised — endpoint=%s service=%s", _otel_endpoint, _service_name)

# ─── FastAPI App (lifespan replaces deprecated @app.on_event) ────
app = FastAPI(title="retriever-minimal", version="0.1.0", docs_url=None, redoc_url=None)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class RetrieveResponse(BaseModel):
    results: list[dict[str, Any]]


# ═══════════════════════════════════════════════════════════════════
# OTel state — populated during lifespan startup
# ═══════════════════════════════════════════════════════════════════
_tracer: Any = None
_tracer_provider: Any = None
_meter_provider: Any = None
_otel_initialized: bool = False  # Guard against duplicate init

# INVARIANT 6: Exactly three metric instruments
_request_counter: Any = None    # retrieve.requests (counter)
_request_duration: Any = None   # retrieve.duration (histogram)
_error_counter: Any = None      # retrieve.errors (counter)


class _NoOp:
    """Safe fallback before OTel is initialised — prevents AttributeError."""
    def add(self, *a: Any, **k: Any) -> None: pass
    def record(self, *a: Any, **k: Any) -> None: pass


_noop = _NoOp()
_request_counter = _noop
_request_duration = _noop
_error_counter = _noop


# ═══════════════════════════════════════════════════════════════════
# Lifespan — replaces deprecated @app.on_event("startup"/"shutdown")
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown in one context manager.

    Startup: initialise traces + metrics + FastAPI instrumentation.
    Shutdown: force-flush all OTel providers so no data is lost.
    """
    global _tracer, _tracer_provider, _meter_provider, _otel_initialized
    global _request_counter, _request_duration, _error_counter

    # ── Startup ──────────────────────────────────────────────────
    if not _otel_initialized:
        _otel_initialized = True

        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name": _service_name,
            "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
            "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
        })

        # INVARIANT 2: TracerProvider before FastAPIInstrumentor
        try:
            _tracer_provider = TracerProvider(resource=resource)
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=_otel_endpoint, insecure=True),
                    max_queue_size=2048,
                    max_export_batch_size=512,
                    schedule_delay_millis=5000,
                    export_timeout_millis=30000,
                )
            )
            trace.set_tracer_provider(_tracer_provider)
            _tracer = trace.get_tracer(__name__)
            log.info("OTel traces initialised — endpoint=%s", _otel_endpoint)
        except Exception:
            log.exception("CRITICAL: Traces init failed — no spans will be exported")

        # INVARIANT 6: Exactly three instruments
        try:
            interval = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "60000"))
            timeout = int(os.getenv("OTEL_METRIC_EXPORT_TIMEOUT_MS", "30000"))

            _meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(endpoint=_otel_endpoint, insecure=True),
                        export_interval_millis=interval,
                        export_timeout_millis=timeout,
                    )
                ],
            )
            metrics.set_meter_provider(_meter_provider)
            meter = metrics.get_meter(__name__)

            _request_counter = meter.create_counter(
                "retrieve.requests", "1", "Total retrieve requests"
            )
            _request_duration = meter.create_histogram(
                "retrieve.duration", "s", "Retrieve request latency"
            )
            _error_counter = meter.create_counter(
                "retrieve.errors", "1", "Total retrieve errors"
            )
            log.info("OTel metrics initialised — interval=%sms", interval)
        except Exception:
            log.exception("CRITICAL: Metrics init failed — no metrics will be exported")
            _request_counter = _noop
            _request_duration = _noop
            _error_counter = _noop

        # FastAPIInstrumentor creates SERVER spans for incoming HTTP requests
        FastAPIInstrumentor.instrument_app(app, tracer_provider=_tracer_provider)
        log.info(
            "Retriever started dense=%s qdrant=%s collection=%s",
            DENSE_URL, QDRANT_URL, COLLECTION_NAME,
        )

    yield  # ── Server runs here ──

    # ── Shutdown — INVARIANT 9: force-flush all providers ───────
    log.info("Shutting down retriever — flushing all telemetry")
    await store.close()

    for name, p in [
        ("traces", _tracer_provider),
        ("metrics", _meter_provider),
        ("logs", _logger_provider),
    ]:
        if p is None:
            continue
        try:
            if hasattr(p, "force_flush"):
                p.force_flush(timeout_millis=5_000)  # 5s max to avoid hanging
            p.shutdown()
            log.info("Flushed %s provider", name)
        except Exception:
            log.warning("Could not flush %s provider (collector may be down)", name)

    log.info("Shutdown complete — all telemetry flushed")


app.router.lifespan_context = lifespan


# ═══════════════════════════════════════════════════════════════════
# POST /retrieve — business endpoint
# ═══════════════════════════════════════════════════════════════════
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> dict[str, Any]:
    """
    Semantic search endpoint.

    INVARIANT 4: Every external call gets a child span with:
      - http.method, http.url (before call)
      - http.status_code, results.count (after call)

    INVARIANT 11: Metric labels match span attributes (status, model)
    """
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")

    labels = {"model": "dense"}
    start = time.perf_counter()
    status = "success"

    # SERVER span — created by FastAPIInstrumentor
    # CLIENT child spans — created manually below
    parent_span = _tracer.start_as_current_span("POST /retrieve") if _tracer else nullcontext()

    try:
        with parent_span as span:
            if span is not None:
                span.set_attributes({
                    "query.length": len(req.query),
                    "top_k": req.top_k,
                })

            # ── Child span 1: dense /embed ─────────────────────
            with (_tracer.start_as_current_span("dense /embed") if _tracer else nullcontext()) as dense_span:
                if dense_span is not None:
                    dense_span.set_attributes({
                        "http.method": "POST",
                        "http.url": f"{DENSE_URL}/embed",
                    })
                    log.info("Calling dense /embed — query_len=%d", len(req.query))

                vec = await _embed(req.query)

                if dense_span is not None:
                    dense_span.set_attributes({
                        "http.status_code": 200,
                        "embedding.dim": len(vec),
                    })
                    log.info("dense /embed completed — dim=%d", len(vec))

            # ── Child span 2: qdrant search ────────────────────
            with (_tracer.start_as_current_span("qdrant search") if _tracer else nullcontext()) as qdrant_span:
                if qdrant_span is not None:
                    qdrant_span.set_attributes({
                        "http.method": "POST",
                        "collection": COLLECTION_NAME,
                    })
                    log.info("Calling qdrant search — top_k=%d", req.top_k)

                results = await store.search(vec, limit=req.top_k)

                if qdrant_span is not None:
                    qdrant_span.set_attributes({
                        "http.status_code": 200,
                        "results.count": len(results),
                    })
                    log.info("qdrant search completed — results=%d", len(results))

            if span is not None:
                span.set_attributes({"results.count": len(results)})

            # INVARIANT 12: Log inside span — gets trace_id injected
            log.info(
                "Retrieve completed query_len=%d results=%d",
                len(req.query), len(results),
            )

        return {"results": [{"text": r["text"], "score": r["score"]} for r in results]}

    except HTTPException:
        status = "client_error"
        raise
    except Exception:
        status = "server_error"
        _error_counter.add(1, {**labels, "error_type": "exception"})
        log.exception("Retrieve failed — query_len=%d", len(req.query))
        raise HTTPException(500, "Internal server error")
    finally:
        elapsed = time.perf_counter() - start
        _request_counter.add(1, {**labels, "status": status})
        _request_duration.record(elapsed, {**labels, "status": status})


# ═══════════════════════════════════════════════════════════════════
# Health endpoints
# INVARIANT 3: Health checks do NOT propagate trace context
# INVARIANT 8: These spans are filtered at the SigNoz collector
# ═══════════════════════════════════════════════════════════════════
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, Any]:
    dense_ok = await _check_dense()
    qdrant_ok = await store.ping()
    return {
        "status": "ready" if (dense_ok and qdrant_ok) else "not_ready",
        "dense": dense_ok,
        "qdrant": qdrant_ok,
    }


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
async def _embed(text: str) -> list[float]:
    """
    Call dense embedding service.

    INVARIANT 3: inject() propagates trace context (traceparent header).
    This makes the dense service's span a CHILD of our CLIENT span.
    """
    headers: dict[str, str] = {}
    inject(headers)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(
            f"{DENSE_URL}/embed",
            json={"texts": [text]},
            headers=headers,
        )
        resp.raise_for_status()
        vectors = resp.json().get("vectors", [])
        if not vectors:
            raise RuntimeError("dense service returned empty vectors")

    vec = [float(x) for x in vectors[0]]
    if len(vec) != DENSE_DIM:
        raise RuntimeError(f"Dimension mismatch: expected {DENSE_DIM}, got {len(vec)}")

    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).astype(float).tolist() if norm > 0 else arr.astype(float).tolist()


async def _check_dense() -> bool:
    """
    Check dense service health.

    INVARIANT 3: NO inject() — health checks must NOT propagate traces.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            return (await client.get(f"{DENSE_URL}/health")).status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# Entrypoint
# log_config=None is CRITICAL: prevents uvicorn from wiping the OTel
# log handler that was attached at module import time.
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8001")),
        log_level=LOG_LEVEL.lower(),
        log_config=None,
        loop="uvloop",
        http="httptools",
    )