"""
OpenTelemetry setup for the MCP context server.

ARCHITECTURE (matches retriever-minimal pattern):
──────────────────────────────────────────────────────────────────
1. LOGS   – set up at *module import time* (before uvicorn).
            LoggingInstrumentor bridges Python logging → OTel.
            LoggingHandler attaches to root logger, injects trace
            context (trace_id, span_id) into every log record
            emitted while a span is active.

2. TRACES – set up via init_otel(), called BEFORE importing FastMCP.
            FastMCP 3.3.1 auto‑creates a span for every tool call,
            resource read, and prompt render once a TracerProvider
            is registered globally.  We don't create spans manually —
            FastMCP does it.  Our tool wrappers run INSIDE that span.

3. METRICS – set up inside init_otel() alongside traces.
            We create three instruments:
            - mcp_context.requests (counter, labels: tool, status)
            - mcp_context.duration  (histogram, seconds)
            - mcp_context.errors    (counter, labels: tool)
            Our record_metrics decorator records values during tool execution.

WHY LOGS NEED A LOG STATEMENT INSIDE AN ACTIVE SPAN:
──────────────────────────────────────────────────────────────────
- Startup logs ("Starting server...") run BEFORE any span exists.
  They are exported but have NO trace_id → collector shows no "Trace ID:".
- FastMCP creates a span ONLY during tool execution.
- If we never call log.info() while inside that span, no log record
  ever gets a Trace ID attached → the test grep for "Trace ID:" fails.
- The fix: emit log.info() inside our tool wrapper, which runs
  within FastMCP's tool span.  The LoggingInstrumentor then injects
  the current span context into that log record.
- This is EXACTLY what retriever-minimal does: it calls log.info()
  inside the FastAPIInstrumentor span in its /retrieve handler.

LOGGER LEVELS:
──────────────────────────────────────────────────────────────────
- Root logger level is controlled by basicConfig in main.py (INFO).
- We do NOT set root to WARNING here — that would silence our app logs.
- Noisy third‑party loggers (uvicorn, httpx, mcp.server, etc.) are
  suppressed individually in main.py.
- Our application logger ("mcp-context-server") inherits root level
  (INFO) and logs to stderr + OTel via the LoggingHandler.

ORDERING REQUIREMENT (FastMCP docs):
──────────────────────────────────────────────────────────────────
FastMCP docs state: "If you're using OpenTelemetry, you must configure
the SDK before importing FastMCP."  We follow this — init_otel() is
called in main.py BEFORE `from fastmcp import FastMCP`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("mcp-server")


# ═══════════════════════════════════════════════════════════════════
# SAFE FALLBACKS – populated before init_otel(), so endpoint code
# never hits AttributeError on .add() / .record().
# ═══════════════════════════════════════════════════════════════════

class _NoOp:
    """No-op instrument that silently accepts .add() and .record() calls."""
    def add(self, *args: Any, **kwargs: Any) -> None: pass
    def record(self, *args: Any, **kwargs: Any) -> None: pass


class _Metrics:
    """Container for metric instruments.  Holds no-ops until init_otel()
    replaces them with real instruments.  Same pattern as
    retriever-minimal/telemetry.py."""
    __slots__ = ("request_counter", "request_duration", "error_counter")

    def __init__(self) -> None:
        noop = _NoOp()
        self.request_counter = noop
        self.request_duration = noop
        self.error_counter = noop


# Module‑level public handles — populated by init_otel()
tracer: Any = None          # opentelemetry.trace.Tracer | None
meter: Any = None           # opentelemetry.metrics.Meter | None
metrics_ = _Metrics()       # Always safe to call .add() / .record()

# Internal references — exposed for shutdown() and testing
_tracer_provider: Any = None
_meter_provider: Any = None
_logger_provider: Any = None  # Kept for force_flush during shutdown


# ═══════════════════════════════════════════════════════════════════
# 1. LOGS — MUST run at module‑import time (before uvicorn touches
#    logging with its internal dictConfig).
#
#    We create a LoggerProvider with a BatchLogRecordProcessor that
#    exports via OTLP/gRPC.  A LoggingHandler is attached to the root
#    logger, so EVERY log.info() / log.error() call anywhere in the
#    process gets bridged to OTel.
#
#    LoggingInstrumentor().instrument() monkey‑patches the logging
#    module so that when a span is active, the span's trace_id and
#    span_id are automatically injected into the log record.
#
#    IMPORTANT: We do NOT set the root logger level here.  The root
#    level is controlled by basicConfig in main.py (default: INFO).
#    Setting root to WARNING here would silently suppress all our
#    application logs, including the log.info() calls inside tool
#    execution that carry the Trace ID.  Noisy third‑party loggers
#    are suppressed individually in main.py instead.
# ═══════════════════════════════════════════════════════════════════

def _init_logging() -> None:
    """Bridge Python logging → OTel Logs via OTLP/gRPC.

    Called at module import time (line below) — before uvicorn
    or FastMCP touch the logging module.  This is the only reliable
    way to keep the OTel handler alive, because uvicorn calls
    logging.config.dictConfig() during startup, which removes any
    previously‑attached handlers.
    """
    global _logger_provider

    from opentelemetry import _logs
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://signoz-otel-collector.signoz.svc.cluster.local:4317",
    )
    service_name = os.getenv("OTEL_SERVICE_NAME", "mcp-server")

    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
    })

    _logger_provider = LoggerProvider(resource=resource)
    _logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=endpoint, insecure=True)
        )
    )
    _logs.set_logger_provider(_logger_provider)

    # Attach OTel handler to root logger.  level=NOTSET means all log
    # levels pass through this handler — the root logger's own level
    # (set by basicConfig in main.py) still filters before this handler
    # sees the record.
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=_logger_provider)
    logging.getLogger().addHandler(handler)

    # Inject trace context (trace_id, span_id) into every log record
    # emitted while a span is active.
    LoggingInstrumentor().instrument(set_logging_format=False)
    log.info("OTel logs bridge initialised — endpoint=%s", endpoint)


# Execute at import time — MUST happen before uvicorn starts.
_init_logging()


# ═══════════════════════════════════════════════════════════════════
# 2. TRACES + METRICS — call BEFORE importing FastMCP
#
#    Creates a TracerProvider and MeterProvider, both exporting via
#    OTLP/gRPC to the same collector endpoint.
#
#    After this function returns:
#    - FastMCP's internal get_tracer() will see our TracerProvider
#      and auto‑create spans for every tool call.
#    - Our record_metrics decorator (in main.py) can safely call
#      telemetry.metrics_.request_counter.add() etc.
#
#    This function is idempotent — calling it multiple times is safe.
# ═══════════════════════════════════════════════════════════════════

def init_otel() -> None:
    """Create TracerProvider + MeterProvider with OTLP/gRPC exporters.

    MUST be called **before** ``from fastmcp import FastMCP`` so that
    FastMCP's internal ``get_tracer()`` sees a registered provider.
    If called after FastMCP is imported, spans will be no‑ops.

    Idempotent — safe to call multiple times.
    """
    global _tracer_provider, _meter_provider, tracer, meter

    if _tracer_provider is not None:
        return

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://signoz-otel-collector.signoz.svc.cluster.local:4317",
    )
    insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
    service_name = os.getenv("OTEL_SERVICE_NAME", "mcp-context-server")

    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
    })

    # ── Traces ───────────────────────────────────────────────────
    try:
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
            )
        )
        trace.set_tracer_provider(_tracer_provider)
        tracer = trace.get_tracer(__name__)
        log.info("OTel traces initialised — endpoint=%s", endpoint)
    except Exception:
        log.exception("OTel traces initialisation failed")

    # ── Metrics ──────────────────────────────────────────────────
    try:
        export_interval = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "60000"))
        export_timeout = int(os.getenv("OTEL_METRIC_EXPORT_TIMEOUT_MS", "30000"))

        _meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=endpoint, insecure=insecure),
                    export_interval_millis=export_interval,
                    export_timeout_millis=export_timeout,
                )
            ],
        )
        metrics.set_meter_provider(_meter_provider)
        meter = metrics.get_meter(__name__)

        metrics_.request_counter = meter.create_counter(
            "mcp_context.requests", "1", "Total tool calls")
        metrics_.request_duration = meter.create_histogram(
            "mcp_context.duration", "s", "Tool call latency")
        metrics_.error_counter = meter.create_counter(
            "mcp_context.errors", "1", "Total tool errors")
        log.info("OTel metrics initialised — interval=%sms", export_interval)
    except Exception:
        log.exception("OTel metrics initialisation failed")


# ═══════════════════════════════════════════════════════════════════
# 3. SHUTDOWN — force‑flush and shut down all providers
#
#    Called from the lifespan's finally block in main.py.
#    Each provider's force_flush() ensures any batched but unexported
#    telemetry is sent before the process exits.  This is critical for
#    logs and traces, which use batch processors that may hold data
#    for a few seconds.
# ═══════════════════════════════════════════════════════════════════

def shutdown() -> None:
    """Force‑flush and shut down ALL providers: traces, metrics, logs."""
    for name, p in [
        ("traces", _tracer_provider),
        ("metrics", _meter_provider),
        ("logs", _logger_provider),
    ]:
        if p is None:
            continue
        try:
            if hasattr(p, "force_flush"):
                p.force_flush(timeout_millis=10_000)
            p.shutdown()
        except Exception:
            log.exception("OTel shutdown error: %s", name)