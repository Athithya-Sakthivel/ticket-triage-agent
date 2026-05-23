"""Dense embedding service with OpenTelemetry (traces, metrics, logs).

OTel providers are initialized inside the startup event (after uvicorn configures its logging).
Traces are created only when tracer is available; metrics use a no-op fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastembed import TextEmbedding
from pydantic import BaseModel

# ─── Configuration ─────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("OTEL_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("dense-embedder")

DENSE_MODEL_NAME: str = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
LOCAL_DENSE_MODEL_PATH: str | None = os.getenv("LOCAL_DENSE_MODEL_PATH") or (
    Path("/app/.resolved_model_path").read_text().strip()
    if Path("/app/.resolved_model_path").exists()
    else None
)
DENSE_DIM: int = int(os.getenv("DENSE_DIM", "384"))
DENSE_BATCH_SIZE: int = int(os.getenv("DENSE_BATCH_SIZE", "32"))
DENSE_NORMALIZE: bool = os.getenv("DENSE_NORMALIZE", "TRUE").upper() in ("1", "TRUE", "YES")
DENSE_CUDA: bool = os.getenv("DENSE_CUDA", "0").upper() in ("1", "TRUE", "YES")
PRELOAD_MODEL: bool = os.getenv("PRELOAD_MODEL", "0").upper() in ("1", "TRUE", "YES")

OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "dense-embedder")
OTEL_EXPORTER_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
OTEL_EXPORTER_INSECURE: bool = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
OTEL_METRIC_INTERVAL_MS: int = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "60000"))
OTEL_METRIC_TIMEOUT_MS: int = int(os.getenv("OTEL_METRIC_EXPORT_TIMEOUT_MS", "30000"))
DEPLOYMENT_ENV: str = os.getenv("DEPLOYMENT_ENVIRONMENT", os.getenv("ENV", "production"))
SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "0.1.0")

# ═══════════════════════════════════════════════════════════════════
# Global OTel handles — populated at startup
# ═══════════════════════════════════════════════════════════════════

_tracer_provider: Any = None
_meter_provider: Any = None
_logger_provider: Any = None
tracer: Any = None
meter: Any = None
request_counter: Any = None
request_duration: Any = None
requests_in_progress: Any = None
error_counter: Any = None


class _NoOpMetric:
    """Fallback for metrics when OTel is unavailable."""
    def add(self, *args, **kwargs): pass
    def record(self, *args, **kwargs): pass


def _init_otel() -> None:
    """Create OTel providers and attach log handler. Called in startup event."""
    global _tracer_provider, _meter_provider, _logger_provider
    global tracer, meter, request_counter, request_duration, requests_in_progress, error_counter

    from opentelemetry import _logs, metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.logging.handler import LoggingHandler

    resource = Resource.create({
        "service.name": OTEL_SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "deployment.environment": DEPLOYMENT_ENV,
    })

    # Traces
    try:
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=OTEL_EXPORTER_ENDPOINT, insecure=OTEL_EXPORTER_INSECURE),
            )
        )
        trace.set_tracer_provider(_tracer_provider)
        tracer = trace.get_tracer(__name__)
        log.info("Traces initialized")
    except Exception:
        log.exception("Traces initialization failed")
        tracer = None

    # Metrics
    try:
        _meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=OTEL_EXPORTER_ENDPOINT, insecure=OTEL_EXPORTER_INSECURE),
                    export_interval_millis=OTEL_METRIC_INTERVAL_MS,
                    export_timeout_millis=OTEL_METRIC_TIMEOUT_MS,
                )
            ],
        )
        metrics.set_meter_provider(_meter_provider)
        meter = metrics.get_meter(__name__)
        request_counter = meter.create_counter("dense.requests", "1", "Total embed requests")
        request_duration = meter.create_histogram("dense.request_duration", "s", "Embed request latency")
        requests_in_progress = meter.create_up_down_counter("dense.requests_in_progress", "1", "In-flight requests")
        error_counter = meter.create_counter("dense.errors", "1", "Total embed errors")
        log.info("Metrics initialized")
    except Exception:
        log.exception("Metrics initialization failed")
        meter = None
        noop = _NoOpMetric()
        request_counter = request_duration = requests_in_progress = error_counter = noop

    # Logs
    try:
        _logger_provider = LoggerProvider(resource=resource)
        _logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=OTEL_EXPORTER_ENDPOINT, insecure=OTEL_EXPORTER_INSECURE),
            )
        )
        _logs.set_logger_provider(_logger_provider)
        handler = LoggingHandler(level=logging.INFO, logger_provider=_logger_provider)
        logging.getLogger().addHandler(handler)
        log.info("Logs initialized")
    except Exception:
        log.exception("Logs initialization failed")


# ═══════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(title="dense-embedder", version=SERVICE_VERSION, docs_url=None, redoc_url=None)

_MAX_WORKERS: int = max(1, (os.cpu_count() or 4) // 2)
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


_MODEL_LOCK = threading.Lock()
_MODEL: TextEmbedding | None = None
_MODEL_ERROR: str | None = None
_READY_AT: float | None = None


def _l2_normalize(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).astype(float).tolist() if norm > 0 else arr.astype(float).tolist()


def _resolve_model_source() -> str:
    if LOCAL_DENSE_MODEL_PATH and Path(LOCAL_DENSE_MODEL_PATH).exists():
        return LOCAL_DENSE_MODEL_PATH
    if Path(DENSE_MODEL_NAME).exists():
        return DENSE_MODEL_NAME
    return DENSE_MODEL_NAME


def _load_model() -> None:
    global _MODEL, _MODEL_ERROR, _READY_AT
    if _MODEL is not None:
        return
    with _MODEL_LOCK:
        if _MODEL is not None:
            return
        source = _resolve_model_source()
        log.info("Loading model source=%s cuda=%s", source, DENSE_CUDA)
        try:
            kwargs = {"model_name": source}
            if DENSE_CUDA:
                try:
                    kwargs["providers"] = ["CUDAExecutionProvider"]
                except TypeError:
                    pass
            _MODEL = TextEmbedding(**kwargs)
            _ = list(_MODEL.embed(["_warmup_"]))
            _READY_AT = time.time()
            _MODEL_ERROR = None
            log.info("Model loaded dim=%d", DENSE_DIM)
        except Exception as exc:
            _MODEL = None
            _MODEL_ERROR = str(exc)
            log.exception("Model load failed")
            raise


def _embed_sync(texts: list[str]) -> list[list[float]]:
    _load_model()
    if _MODEL is None:
        raise RuntimeError(f"Model not loaded: {_MODEL_ERROR or 'unknown error'}")
    results: list[list[float]] = []
    for arr in _MODEL.embed(texts):
        vec = arr.tolist() if hasattr(arr, "tolist") else [float(x) for x in arr]
        if DENSE_NORMALIZE:
            vec = _l2_normalize(vec)
        if len(vec) != DENSE_DIM:
            raise RuntimeError(f"Dimension mismatch: expected {DENSE_DIM}, got {len(vec)}")
        results.append(vec)
    return results


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> dict[str, Any]:
    if not req.texts:
        raise HTTPException(400, "'texts' must be a non-empty list")
    if len(req.texts) > DENSE_BATCH_SIZE:
        raise HTTPException(400, f"Batch exceeds max ({DENSE_BATCH_SIZE})")

    labels = {"model": DENSE_MODEL_NAME, "cuda": str(DENSE_CUDA).lower()}
    requests_in_progress.add(1, labels)
    start = time.perf_counter()
    status = "success"

    # Use proper context manager if tracer is available, else just execute
    async def _run() -> list[list[float]]:
        return await asyncio.get_running_loop().run_in_executor(
            _EMBED_EXECUTOR, _embed_sync, req.texts
        )

    try:
        if tracer is not None:
            with tracer.start_as_current_span("POST /embed") as span:
                span.set_attributes({
                    "batch.size": len(req.texts),
                    "model.name": DENSE_MODEL_NAME,
                })
                vectors = await _run()
                span.set_attribute("vectors.count", len(vectors))
        else:
            vectors = await _run()

        log.info("Embed completed count=%d", len(vectors))
        return {"vectors": vectors}
    except HTTPException:
        status = "client_error"
        raise
    except RuntimeError as exc:
        status = "model_error"
        error_counter.add(1, {**labels, "error_type": "runtime"})
        log.error("Model error: %s", exc)
        raise HTTPException(503, str(exc))
    except Exception:
        status = "server_error"
        error_counter.add(1, {**labels, "error_type": "exception"})
        log.exception("Embed failed")
        raise HTTPException(500, "Internal server error")
    finally:
        elapsed = time.perf_counter() - start
        request_counter.add(1, {**labels, "status": status})
        request_duration.record(elapsed, labels)
        requests_in_progress.add(-1, labels)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok", "model": DENSE_MODEL_NAME, "dim": DENSE_DIM,
        "normalize": DENSE_NORMALIZE, "cuda": DENSE_CUDA, "model_error": _MODEL_ERROR,
    }


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    if _MODEL is None and _MODEL_ERROR is None:
        try:
            _load_model()
        except Exception:
            pass
    if _MODEL is not None and _READY_AT is not None:
        return {"status": "ready", "ready_at": _READY_AT, "model": DENSE_MODEL_NAME}
    raise HTTPException(503, {"status": "not_ready", "model_error": _MODEL_ERROR})


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    # OTel initialization must happen AFTER uvicorn has configured its logging
    _init_otel()

    # Instrument FastAPI now that the real TracerProvider is set
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app, tracer_provider=_tracer_provider)

    log.info("Starting dense-embedder model=%s dim=%d workers=%d",
             DENSE_MODEL_NAME, DENSE_DIM, _MAX_WORKERS)
    if PRELOAD_MODEL:
        try:
            await asyncio.get_running_loop().run_in_executor(_EMBED_EXECUTOR, _load_model)
        except Exception:
            log.exception("Preload failed")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    log.info("Shutting down — flushing telemetry")
    _EMBED_EXECUTOR.shutdown(wait=True)
    for name, p in [("traces", _tracer_provider), ("metrics", _meter_provider), ("logs", _logger_provider)]:
        if p is None:
            continue
        try:
            if hasattr(p, "force_flush"):
                p.force_flush(timeout_millis=10_000)
            p.shutdown()
        except Exception:
            log.exception("Shutdown error: %s", name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "host_dense:app",
        host=os.getenv("DENSE_HOST", "0.0.0.0"),
        port=int(os.getenv("DENSE_PORT", "8200")),
        log_level=LOG_LEVEL.lower(),
        loop="uvloop",
        http="httptools",
    )