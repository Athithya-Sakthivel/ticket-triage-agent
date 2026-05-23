FROM python:3.13.12-slim@sha256:f1927c75e81efd1e091dbd64b6c0ecaa5630b38635a3d1c04034ac636e1f94c8

ENV PYTHONUNBUFFERED=1 \
    OTEL_SERVICE_NAME=retriever-minimal \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
    OTEL_LOG_LEVEL=INFO \
    PORT=8001

RUN mkdir -p /app \
 && groupadd -g 1000 appgroup \
 && useradd -m -u 1000 -g appgroup -s /bin/sh appuser \
 && chown -R appuser:appgroup /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py telemetry.py store.py /app/
RUN chown -R appuser:appgroup /app

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8001/healthz || exit 1

CMD ["python", "-u", "/app/main.py"]