# =============================================================================
# MCP Server — Production Docker Image (Single-Stage)
# =============================================================================
FROM python:3.13.12-slim@sha256:f1927c75e81efd1e091dbd64b6c0ecaa5630b38635a3d1c04034ac636e1f94c8

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    OTEL_SERVICE_NAME=mcp-server \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://signoz-otel-collector.signoz.svc.cluster.local:4317 \
    OTEL_EXPORTER_OTLP_INSECURE=true \
    OTEL_LOG_LEVEL=INFO \
    LOG_LEVEL=INFO \
    DEPLOYMENT_ENVIRONMENT=production \
    SERVICE_VERSION=0.1.0 \
    HOST=0.0.0.0 \
    PORT=8001

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libssl3 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appgroup \
    && useradd -m -u 1000 -g appgroup -s /sbin/nologin appuser \
    && mkdir -p /app \
    && chown -R appuser:appgroup /app

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup src/ /app/src/

RUN find /app -type f -perm /6000 -exec chmod ug-s {} \; 2>/dev/null || true \
    && find /app -type d -perm /o+w -exec chmod o-w {} \; 2>/dev/null || true

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent --max-time 5 http://127.0.0.1:8001/healthz || exit 1

CMD ["python", "-u", "/app/src/main.py"]