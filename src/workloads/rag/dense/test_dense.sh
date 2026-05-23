#!/usr/bin/env bash
set -euo pipefail

OTEL_COLLECTOR_IMAGE="${OTEL_COLLECTOR_IMAGE:-otel/opentelemetry-collector:0.149.0}"
OTEL_COLLECTOR_NAME="${OTEL_COLLECTOR_NAME:-otel-dense-test}"
OTEL_CONFIG_DIR="$(mktemp -d /tmp/otel-dense-XXXXXX)"
COLLECTOR_GRPC_PORT="${COLLECTOR_GRPC_PORT:-4317}"

TEST_MODE="${TEST_MODE:-cpu}"
FASTEMBED_GPU_ARG=0
case "${TEST_MODE,,}" in gpu) FASTEMBED_GPU_ARG=1 ;; *) FASTEMBED_GPU_ARG=0 ;; esac

DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-BAAI/bge-small-en-v1.5}"
DENSE_DIM="${DENSE_DIM:-384}"
IMAGE_LOCAL="dense:test"
CONTAINER_NAME="dense-test-${TEST_MODE}"
HOST_PORT="${HOST_PORT:-9021}"
DOCKER_NETWORK="${DOCKER_NETWORK:-dense-test-net}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-120}"

command -v docker >/dev/null 2>&1 || { echo "[ERROR] docker not found" >&2; exit 1; }

case "$(uname -m)" in
  x86_64|amd64) PLATFORM="linux/amd64" ;;
  aarch64|arm64) PLATFORM="linux/arm64" ;;
  *) PLATFORM="linux/amd64" ;;
esac

cleanup() {
  set +e
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm -f "${OTEL_COLLECTOR_NAME}" >/dev/null 2>&1 || true
  docker network rm "${DOCKER_NETWORK}" >/dev/null 2>&1 || true
  rm -rf "${OTEL_CONFIG_DIR}" >/dev/null 2>&1 || true
  set -e
}
trap cleanup EXIT

docker network create "${DOCKER_NETWORK}" >/dev/null 2>&1 || true

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
  --network "${DOCKER_NETWORK}" \
  -p "127.0.0.1:${COLLECTOR_GRPC_PORT}:4317" \
  -v "${OTEL_CONFIG_DIR}:/etc/otelcol:ro" \
  "${OTEL_COLLECTOR_IMAGE}" \
  --config=/etc/otelcol/config.yaml >/dev/null
sleep 2
docker logs --tail 4 "${OTEL_COLLECTOR_NAME}" 2>&1

echo "[2/7] Building image..."
docker build \
  --platform "${PLATFORM}" \
  --build-arg FASTEMBED_GPU="${FASTEMBED_GPU_ARG}" \
  --build-arg DENSE_MODEL_NAME="${DENSE_MODEL_NAME}" \
  --build-arg DENSE_DIM="${DENSE_DIM}" \
  -t "${IMAGE_LOCAL}" . || { echo "[ERROR] Build failed" >&2; exit 1; }

echo "[3/7] Starting dense service..."
docker run --name "${CONTAINER_NAME}" -d \
  -p "${HOST_PORT}:8200" \
  --network "${DOCKER_NETWORK}" \
  --shm-size=1.8g \
  -e OTEL_EXPORTER_OTLP_ENDPOINT="http://${OTEL_COLLECTOR_NAME}:${COLLECTOR_GRPC_PORT}" \
  -e OTEL_LOG_LEVEL=INFO \
  -e OTEL_METRIC_EXPORT_INTERVAL_MS=5000 \
  "${IMAGE_LOCAL}" >/dev/null

echo "[4/7] Waiting for readiness..."
for ((i=0; i<WAIT_TIMEOUT; i++)); do
  if body=$(curl -fsS --max-time 2 "http://127.0.0.1:${HOST_PORT}/readyz" 2>/dev/null); then
    echo "  ${body}"
    break
  fi
  sleep 1
done
curl -fsS --max-time 2 "http://127.0.0.1:${HOST_PORT}/readyz" >/dev/null 2>&1 || {
  echo "[ERROR] Service not ready" >&2
  docker logs --tail 50 "${CONTAINER_NAME}" || true
  exit 1
}

echo "[5/7] Sending embed requests..."
for i in 1 2 3 4 5; do
  resp=$(curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/embed" \
    -H "Content-Type: application/json" \
    -d "{\"texts\":[\"test request ${i}\"]}") || { echo "[ERROR] Request ${i} failed" >&2; docker logs --tail 20 "${CONTAINER_NAME}"; exit 1; }
  vec_len=$(echo "${resp}" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['vectors'][0]))" 2>/dev/null || echo 0)
  [ "${vec_len}" = "${DENSE_DIM}" ] || { echo "[ERROR] Dim mismatch: ${vec_len} != ${DENSE_DIM}" >&2; exit 1; }
  echo "  Request ${i}: ${vec_len}-dim ✓"
done

echo "[6/7] Waiting for batch export (10s)..."
sleep 10

echo "[7/7] Verifying signals..."
COLLECTOR_LOGS="$(docker logs --tail 300 "${OTEL_COLLECTOR_NAME}" 2>&1)"

check() {
  if echo "${COLLECTOR_LOGS}" | grep -q "$2"; then
    echo "  [PASS] $1"
  else
    echo "  [FAIL] $1"
  fi
}
check "Traces (Span #)"      "Span #"
check "Metrics"              "dense.requests\|dense.request_duration\|dense.requests_in_progress"
check "Logs (LogRecord)"     "LogRecord\|Body:"

echo ""
echo "─── Collector Sample ───"
echo "${COLLECTOR_LOGS}" | grep -E "Span \#|Metric \#|LogRecord|dense\.|Body:" | head -20
echo ""
echo "─── Service Logs ───"
docker logs --tail 10 "${CONTAINER_NAME}" 2>&1
echo ""
echo "[SUCCESS] E2E test complete."