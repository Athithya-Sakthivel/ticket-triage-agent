"""
Dense embedding service — leaf service, logs only.

Generates embeddings using fastembed. No traces, no custom metrics.
Its latency and error rate are captured by retriever-minimal's
CLIENT span ("dense /embed").

Logs are bridged to OpenTelemetry at module import time.
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
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("dense-embedder")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

# ─── Configuration ────────────────────────────────────────────────
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
PRELOAD_MODEL: bool = os.getenv("PRELOAD_MODEL", "1").upper() in ("1", "TRUE", "YES")

# ═══════════════════════════════════════════════════════════════════
# OpenTelemetry — Logs only (module import time, before uvicorn)
# ═══════════════════════════════════════════════════════════════════
from opentelemetry import _logs
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

_otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
_service_name = os.getenv("OTEL_SERVICE_NAME", "dense-embedder")

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
log.info("OTel logs initialised — endpoint=%s", _otel_endpoint)

# ─── FastAPI ──────────────────────────────────────────────────────
app = FastAPI(title="dense-embedder", version="0.1.0", docs_url=None, redoc_url=None)

_MAX_WORKERS: int = max(1, (os.cpu_count() or 4) // 2)
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
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
            from fastembed import TextEmbedding
            kwargs: dict[str, Any] = {"model_name": source}
            if DENSE_CUDA:
                kwargs["providers"] = ["CUDAExecutionProvider"]
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


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> dict[str, Any]:
    if not req.texts:
        raise HTTPException(400, "'texts' must be a non-empty list")
    if len(req.texts) > DENSE_BATCH_SIZE:
        raise HTTPException(400, f"Batch exceeds max ({DENSE_BATCH_SIZE})")

    log.info("Embed started batch_size=%d", len(req.texts))
    start = time.perf_counter()

    try:
        vectors = await asyncio.get_running_loop().run_in_executor(
            _EMBED_EXECUTOR, _embed_sync, req.texts
        )
        elapsed = time.perf_counter() - start
        log.info("Embed completed count=%d elapsed=%.3fs", len(vectors), elapsed)
        return {"vectors": vectors}
    except HTTPException:
        raise
    except RuntimeError as exc:
        log.error("Model error: %s", exc)
        raise HTTPException(503, str(exc))
    except Exception:
        log.exception("Embed failed")
        raise HTTPException(500, "Internal server error")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": DENSE_MODEL_NAME,
        "dim": DENSE_DIM,
        "normalize": DENSE_NORMALIZE,
        "cuda": DENSE_CUDA,
        "model_error": _MODEL_ERROR,
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


# ═══════════════════════════════════════════════════════════════════
# Startup / Shutdown
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup() -> None:
    log.info("Starting dense-embedder model=%s dim=%d workers=%d",
             DENSE_MODEL_NAME, DENSE_DIM, _MAX_WORKERS)
    if PRELOAD_MODEL:
        try:
            await asyncio.get_running_loop().run_in_executor(_EMBED_EXECUTOR, _load_model)
        except Exception:
            log.exception("Preload failed")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    log.info("Shutting down — flushing logs")
    _EMBED_EXECUTOR.shutdown(wait=True)
    if _logger_provider:
        try:
            _logger_provider.force_flush(timeout_millis=10_000)
            _logger_provider.shutdown()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "host_dense:app",
        host=os.getenv("DENSE_HOST", "0.0.0.0"),
        port=int(os.getenv("DENSE_PORT", "8200")),
        log_level=LOG_LEVEL.lower(),
        log_config=None,
        loop="uvloop",
        http="httptools",
    )