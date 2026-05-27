"""
OpenTelemetry + OpenInference setup for the Agent Service.

Invariants enforced:
  1. Log bridge at module import time
  2. TracerProvider before framework imports
  6. Exactly 3 metric instruments (agent.requests, agent.duration, agent.errors)
  7. No sampling in application code
  9. Force-flush on shutdown
 10. Root logger never set to WARNING
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("agent-service")


# ── Safe fallbacks ────────────────────────────────────────────────
class _NoOp:
    def add(self, *args: Any, **kwargs: Any) -> None: pass
    def record(self, *args: Any, **kwargs: Any) -> None: pass


class _Metrics:
    __slots__ = ("request_counter", "request_duration", "error_counter")
    def __init__(self) -> None:
        noop = _NoOp()
        self.request_counter = noop
        self.request_duration = noop
        self.error_counter = noop


tracer: Any = None
metrics_ = _Metrics()
_tracer_provider: Any = None
_meter_provider: Any = None
_logger_provider: Any = None


# ═══════════════════════════════════════════════════════════════════
# 1. LOGS – at module import time (before uvicorn)
# ═══════════════════════════════════════════════════════════════════
def _init_logging() -> None:
    global _logger_provider
    from opentelemetry import _logs
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT",
                         "http://signoz-otel-collector.signoz.svc.cluster.local:4317")
    service_name = os.getenv("OTEL_SERVICE_NAME", "agent-service")

    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
    })

    _logger_provider = LoggerProvider(resource=resource)
    _logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=True))
    )
    _logs.set_logger_provider(_logger_provider)

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=_logger_provider)
    logging.getLogger().addHandler(handler)
    LoggingInstrumentor().instrument(set_logging_format=False)
    log.info("OTel logs bridge initialised – endpoint=%s", endpoint)

_init_logging()


# ═══════════════════════════════════════════════════════════════════
# 2. TRACES + METRICS + OpenInference – before LangChain/FastAPI
# ═══════════════════════════════════════════════════════════════════
def init_otel() -> None:
    global _tracer_provider, _meter_provider, tracer

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

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT",
                         "http://signoz-otel-collector.signoz.svc.cluster.local:4317")
    insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
    service_name = os.getenv("OTEL_SERVICE_NAME", "agent-service")

    resource = Resource.create({
        "service.name": service_name,
        "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
    })

    # Traces
    try:
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
        )
        trace.set_tracer_provider(_tracer_provider)
        tracer = trace.get_tracer(__name__)
        log.info("OTel traces initialised – endpoint=%s", endpoint)
    except Exception:
        log.exception("OTel traces initialisation failed")

    # Metrics
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
            "agent.requests", "1", "Total agent invocations")
        metrics_.request_duration = meter.create_histogram(
            "agent.duration", "s", "Agent invocation latency")
        metrics_.error_counter = meter.create_counter(
            "agent.errors", "1", "Total agent errors")
        log.info("OTel metrics initialised – interval=%sms", export_interval)
    except Exception:
        log.exception("OTel metrics initialisation failed")

    # OpenInference
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.dspy import DSPyInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
        DSPyInstrumentor().instrument(tracer_provider=_tracer_provider)
        log.info("OpenInference instrumentors registered")
    except Exception:
        log.exception("OpenInference instrumentation failed")


def shutdown() -> None:
    """Force-flush and shut down all providers."""
    for name, p in [("traces", _tracer_provider), ("metrics", _meter_provider), ("logs", _logger_provider)]:
        if p is None:
            continue
        try:
            if hasattr(p, "force_flush"):
                p.force_flush(timeout_millis=10_000)
            p.shutdown()
        except Exception:
            log.exception("OTel shutdown error: %s", name)