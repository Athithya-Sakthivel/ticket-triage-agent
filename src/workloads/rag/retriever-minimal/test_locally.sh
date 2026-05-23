#!/usr/bin/env bash
set -euo pipefail

# ─── Config ─────────────────────────────────────────────────────
OTEL_COLLECTOR_IMAGE="otel/opentelemetry-collector:0.149.0"
OTEL_COLLECTOR_NAME="otel-retriever-test"
OTEL_CONFIG_DIR="$(mktemp -d /tmp/otel-retriever-XXXXXX)"
COLLECTOR_GRPC_PORT=4317

DENSE_PORT=8200
QDRANT_PORT=6333
RETRIEVER_PORT=8001

command -v docker >/dev/null 2>&1 || { echo "[ERROR] docker not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 not found" >&2; exit 1; }

cleanup() {
  set +e
  docker rm -f "${OTEL_COLLECTOR_NAME}" >/dev/null 2>&1 || true
  rm -rf "${OTEL_CONFIG_DIR}" >/dev/null 2>&1 || true
  kill ${PF_DENSE:-} ${PF_QDRANT:-} 2>/dev/null || true
  set -e
}
trap cleanup EXIT

# ─── OTel Collector (Docker, bound to localhost) ────────────────
cat >"${OTEL_CONFIG_DIR}/config.yaml" <<EOF
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:${COLLECTOR_GRPC_PORT}
processors:
  batch: {}
exporters:
  debug:
    verbosity: detailed
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
EOF

chmod 644 "${OTEL_CONFIG_DIR}/config.yaml"
chmod 755 "${OTEL_CONFIG_DIR}"

echo "[1/7] Starting OTel Collector..."
docker rm -f "${OTEL_COLLECTOR_NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${OTEL_COLLECTOR_NAME}" \
  -p "127.0.0.1:${COLLECTOR_GRPC_PORT}:4317" \
  -v "${OTEL_CONFIG_DIR}:/etc/otelcol:ro" \
  "${OTEL_COLLECTOR_IMAGE}" \
  --config=/etc/otelcol/config.yaml >/dev/null
sleep 2
docker logs --tail 4 "${OTEL_COLLECTOR_NAME}" 2>&1

# ─── Port-forward dense & qdrant ────────────────────────────────
echo "[2/7] Setting up port forwards..."
kubectl port-forward -n inference svc/dense-svc ${DENSE_PORT}:8200 >/dev/null 2>&1 &
PF_DENSE=$!
kubectl port-forward -n qdrant svc/qdrant ${QDRANT_PORT}:6333 >/dev/null 2>&1 &
PF_QDRANT=$!
sleep 2

# Verify forwards are up
if ! curl -fsS --max-time 2 "http://127.0.0.1:${DENSE_PORT}/healthz" >/dev/null 2>&1; then
  echo "[ERROR] dense-svc not reachable on port ${DENSE_PORT}" >&2
  exit 1
fi
if ! curl -fsS --max-time 2 "http://127.0.0.1:${QDRANT_PORT}/collections" >/dev/null 2>&1; then
  echo "[ERROR] qdrant not reachable on port ${QDRANT_PORT}" >&2
  exit 1
fi
echo "  dense-svc: OK, qdrant: OK"

# ─── Install dependencies (if needed) ───────────────────────────
echo "[3/7] Checking dependencies..."
python3 -c "import fastapi, httpx, qdrant_client, opentelemetry" 2>/dev/null || {
  echo "  Installing dependencies..."
  pip install -q fastapi uvicorn[standard] uvloop pydantic httpx numpy qdrant-client \
    "opentelemetry-api==1.42.1" "opentelemetry-sdk==1.42.1" \
    "opentelemetry-exporter-otlp-proto-grpc==1.42.1" \
    "opentelemetry-instrumentation-fastapi==0.63b1" \
    "opentelemetry-instrumentation-httpx==0.63b1" \
    "opentelemetry-instrumentation-logging==0.63b1"
}
echo "  Dependencies OK"

# ─── Start retriever locally ────────────────────────────────────
echo "[4/7] Starting retriever (local process)..."
export DENSE_URL="http://127.0.0.1:${DENSE_PORT}"
export QDRANT_URL="http://127.0.0.1:${QDRANT_PORT}"
export COLLECTION_NAME="${COLLECTION_NAME:-documents}"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:${COLLECTOR_GRPC_PORT}"
export OTEL_LOG_LEVEL=INFO
export OTEL_METRIC_EXPORT_INTERVAL_MS=5000
export OTEL_SERVICE_NAME=retriever-minimal
export PORT="${RETRIEVER_PORT}"
export OTEL_EXPORTER_OTLP_INSECURE=true

python3 main.py > /tmp/retriever.log 2>&1 &
RETRIEVER_PID=$!
trap "cleanup; kill ${RETRIEVER_PID} 2>/dev/null || true" EXIT

# ─── Wait for ready ─────────────────────────────────────────────
echo "[5/7] Waiting for readiness..."
for ((i=0; i<30; i++)); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${RETRIEVER_PORT}/readyz" >/dev/null 2>&1; then
    echo "  Ready"
    break
  fi
  sleep 1
done

# ─── Send test requests ─────────────────────────────────────────
echo "[6/7] Sending retrieve requests..."
QUERIES=(
  "how do I reset my password?"
  "what is the refund policy?"
  "shipping costs and delivery time"
)
for query in "${QUERIES[@]}"; do
  echo ""
  echo "  Query: ${query}"
  RESP=$(curl -fsS -X POST "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"${query}\", \"top_k\": 3}" 2>&1) || {
    echo "  [ERROR] Request failed — check /tmp/retriever.log"
    echo "  Last 20 lines of log:"
    tail -20 /tmp/retriever.log
    exit 1
  }
  echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "${RESP}"
done

# ─── Verify OTel signals ────────────────────────────────────────
echo ""
echo "Waiting for telemetry export (10s)..."
sleep 10

COLLECTOR_LOGS="$(docker logs --tail 500 "${OTEL_COLLECTOR_NAME}" 2>&1)"

check() {
  if echo "${COLLECTOR_LOGS}" | grep -q "$2"; then
    echo "  [PASS] $1"
  else
    echo "  [FAIL] $1"
  fi
}

echo ""
echo "─── OTel Signal Checks ───"
check "Traces (Span #)"      "Span #"
check "Metrics"              "retrieve.requests\|retrieve.duration\|retrieve.requests_in_progress"
check "Logs"                 "Trace ID:"

echo ""
echo "─── Collector Sample ───"
echo "${COLLECTOR_LOGS}" | grep -E "Span \#|Metric \#|Trace ID:" | head -30

echo ""
echo "─── Retriever Logs ───"
tail -15 /tmp/retriever.log

# ─── Cross-service trace verification ───────────────────────────
echo ""
echo "─── Cross-service trace verification ───"
TRACE_ID=$(echo "${COLLECTOR_LOGS}" | grep "Trace ID:" | head -1 | sed 's/.*Trace ID: //' | cut -d' ' -f1)
echo "  Sample Trace ID: ${TRACE_ID}"

if [ -n "${TRACE_ID}" ]; then
  echo "  All spans/logs for this trace:"
  # Show all collector lines related to this trace ID
  echo "${COLLECTOR_LOGS}" | grep "${TRACE_ID}" | head -20
  echo ""
  # Count unique services in this trace
  SERVICES=$(echo "${COLLECTOR_LOGS}" | grep "${TRACE_ID}" | grep -o '"service.name":"[^"]*"' | sort -u)
  echo "  Services in this trace:"
  echo "${SERVICES}" | sed 's/"service.name":"\(.*\)"/    \1/'
else
  echo "  [WARN] No Trace ID found — context propagation may not be working"
fi

echo ""
echo "[SUCCESS] Local test complete."