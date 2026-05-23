FROM python:3.13.12-slim@sha256:f1927c75e81efd1e091dbd64b6c0ecaa5630b38635a3d1c04034ac636e1f94c8

ARG FASTEMBED_GPU=0
ARG DENSE_MODEL_NAME
ARG DENSE_DIM

ENV FASTEMBED_CACHE=/models_cache \
    HF_HOME=/models_cache \
    HUGGINGFACE_HUB_CACHE=/models_cache \
    TRANSFORMERS_CACHE=/models_cache \
    PYTHONUNBUFFERED=1 \
    DENSE_MODEL_NAME=${DENSE_MODEL_NAME} \
    DENSE_DIM=${DENSE_DIM} \
    DENSE_BATCH_SIZE=32 \
    DENSE_NORMALIZE=TRUE \
    DENSE_CUDA=0 \
    DENSE_HOST=0.0.0.0 \
    DENSE_PORT=8200 \
    OTEL_SERVICE_NAME=dense-embedder \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
    OTEL_METRIC_EXPORT_INTERVAL_MS=60000 \
    OTEL_METRIC_EXPORT_TIMEOUT_MS=30000 \
    OTEL_LOG_LEVEL=INFO \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ORT_NUM_THREADS=1

RUN mkdir -p /models_cache /app /var/log/dense \
 && groupadd -g 1000 appgroup \
 && useradd -m -u 1000 -g appgroup -s /bin/sh appuser \
 && chown -R appuser:appgroup /models_cache /app /var/log/dense

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates openssl libssl3 curl \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip wheel

RUN if [ "$FASTEMBED_GPU" = "1" ]; then \
      python -m pip install --no-cache-dir fastembed-gpu==0.8.0 onnxruntime-gpu==1.25.0; \
    else \
      python -m pip install --no-cache-dir fastembed==0.8.0 onnxruntime==1.25.0; \
    fi

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY host_dense.py /app/host_dense.py
RUN chown -R appuser:appgroup /app

USER appuser
ENV HOME=/home/appuser

EXPOSE 8200

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8200/healthz || exit 1

CMD ["python", "-u", "/app/host_dense.py"]