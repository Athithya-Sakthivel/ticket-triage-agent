"""
OpenTelemetry traces and metrics for a FastAPI service.
======================================================================
LOGS are NOT set up here – they MUST be initialised in main.py at
module‑import time (before uvicorn touches the logging module).
Traces and metrics are initialised in init_otel(), which is called
from the FastAPI startup event (after uvicorn is stable).
======================================================================
WHY THIS SEPARATION?
- uvicorn calls logging.config.dictConfig() during startup, which
  removes any previously‑attached handlers.  Setting up the log
  handler at module‑import time (before uvicorn runs) is the only
  reliable way to keep it alive.
- The tracer provider must be set globally *before* any
  instrumentation library (FastAPIInstrumentor) calls get_tracer().
  We do that inside init_otel(), then instrument the app explicitly.
======================================================================
USAGE (in main.py):
    import telemetry

    @app.on_event("startup")
    async def startup():
        telemetry.init_otel()                     # traces + metrics
        FastAPIInstrumentor.instrument_app(app,
                        tracer_provider=telemetry._tracer_provider)

    @app.get("/work")
    async def work():
        telemetry.metrics_.my_counter.add(1, {"key": "val"})  # always safe
        with telemetry.tracer.start_as_current_span("work"): ...
======================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("retriever-minimal")


# ------------------------------------------------------------------
# Safe fallback – returned before OTel is initialised, so endpoint
# code never hits AttributeError on .add() / .record()
# ------------------------------------------------------------------
class _NoOp:
    def add(self, *args: Any, **kwargs: Any) -> None: pass
    def record(self, *args: Any, **kwargs: Any) -> None: pass


# ------------------------------------------------------------------
# Mutable namespace that always returns the latest real instrument.
# We never rebind the module‑level name, so `from telemetry import
# metrics_` would still see the old object.  Importers must do
# `import telemetry` and access `telemetry.metrics_.<name>`.
# ------------------------------------------------------------------
class _Metrics:
    __slots__ = (
        "request_counter",
        "request_duration",
        "requests_in_progress",
        "error_counter",
    )

    def __init__(self) -> None:
        noop = _NoOp()
        self.request_counter = noop
        self.request_duration = noop
        self.requests_in_progress = noop
        self.error_counter = noop


# ── Module‑level public handles (populated by init_otel) ───────────
tracer: Any = None            # opentelemetry.trace.Tracer | None
meter: Any = None             # opentelemetry.metrics.Meter | None
metrics_ = _Metrics()         # always returns usable instruments

# Internal references – exposed so FastAPIInstrumentor can be passed
# the real TracerProvider.
_tracer_provider: Any = None
_meter_provider: Any = None


# ------------------------------------------------------------------
def init_otel() -> None:
    """
    Create TracerProvider and MeterProvider with OTLP/gRPC exporters.
    MUST be called inside the FastAPI startup event (not at import
    time), because uvicorn must be fully stable before we touch the
    global trace provider.
    """
    global _tracer_provider, _meter_provider, tracer, meter

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
    service_name = os.getenv("OTEL_SERVICE_NAME", "retriever-minimal")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
            "deployment.environment": os.getenv(
                "DEPLOYMENT_ENVIRONMENT", "production"
            ),
        }
    )

    # ── Traces ───────────────────────────────────────────────────
    try:
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
            )
        )
        # Set the GLOBAL provider so that FastAPIInstrumentor (called
        # right after this) picks it up.
        trace.set_tracer_provider(_tracer_provider)
        tracer = trace.get_tracer(__name__)
        log.info("Traces initialized — endpoint=%s", endpoint)
    except Exception:
        log.exception("Traces initialization failed")
        tracer = None

    # ── Metrics ──────────────────────────────────────────────────
    try:
        _meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=endpoint, insecure=insecure),
                    export_interval_millis=int(
                        os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "60000")
                    ),
                    export_timeout_millis=int(
                        os.getenv("OTEL_METRIC_EXPORT_TIMEOUT_MS", "30000")
                    ),
                )
            ],
        )
        metrics.set_meter_provider(_meter_provider)
        meter = metrics.get_meter(__name__)

        # Populate the mutable namespace – all endpoint code that
        # accesses `telemetry.metrics_.<name>` now sees real counters.
        metrics_.request_counter = meter.create_counter(
            "retrieve.requests", "1", "Total retrieve requests"
        )
        metrics_.request_duration = meter.create_histogram(
            "retrieve.duration", "s", "Retrieve request latency"
        )
        metrics_.requests_in_progress = meter.create_up_down_counter(
            "retrieve.requests_in_progress", "1", "In-flight requests"
        )
        metrics_.error_counter = meter.create_counter(
            "retrieve.errors", "1", "Total retrieve errors"
        )
        log.info(
            "Metrics initialized — interval=%sms",
            os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "60000"),
        )
    except Exception:
        log.exception("Metrics initialization failed")
        meter = None


# ------------------------------------------------------------------
def shutdown() -> None:
    """Force‑flush and shut down trace & metric providers."""
    for name, p in [("traces", _tracer_provider), ("metrics", _meter_provider)]:
        if p is None:
            continue
        try:
            if hasattr(p, "force_flush"):
                p.force_flush(timeout_millis=10_000)
            p.shutdown()
        except Exception:
            log.exception("Shutdown error: %s", name)