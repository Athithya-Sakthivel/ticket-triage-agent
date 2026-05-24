# ─── Base image ────────────────────────────────────────────────────
# Python 3.13 slim with pinned digest for reproducible builds.
# apt-get upgrade patches CVEs (e.g. OpenSSL) at build time.
FROM python:3.13.12-slim@sha256:f1927c75e81efd1e091dbd64b6c0ecaa5630b38635a3d1c04034ac636e1f94c8

# ─── Environment ───────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    OTEL_SERVICE_NAME=retriever-minimal \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://signoz-otel-collector.signoz.svc.cluster.local:4317 \
    OTEL_LOG_LEVEL=INFO \
    PORT=8001

# ─── System packages ───────────────────────────────────────────────
# ca-certificates + curl = healthcheck probe support.
# openssl / libssl3 patched via upgrade to fix known CVEs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl openssl libssl3 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# ─── Non‑root user ─────────────────────────────────────────────────
# Required by Kubernetes securityContext.runAsNonRoot.
# Fixed uid/gid (1000) for consistency across environments.
RUN groupadd -g 1000 appgroup \
    && useradd -m -u 1000 -g appgroup -s /bin/sh appuser \
    && mkdir -p /app \
    && chown -R appuser:appgroup /app

# ─── Python dependencies ───────────────────────────────────────────
RUN python -m pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ─── Application code ──────────────────────────────────────────────
COPY main.py store.py /app/
RUN chown -R appuser:appgroup /app

# ─── Runtime ───────────────────────────────────────────────────────
USER appuser
EXPOSE 8001

# Kubernetes liveness probe – returns 200 as long as the process is alive.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8001/healthz || exit 1

# Start via python (not uvicorn CLI) so log_config=None in main.py works.
CMD ["python", "-u", "/app/main.py"]