#!/usr/bin/env bash
set -euo pipefail



# ═══════════════════════════════════════════════════════════════════
# Dense Embedder — Battle Test (Real SigNoz, Docker Host Network)
# ═══════════════════════════════════════════════════════════════════
# Verifies:
#   1. Logs  — MUST exist  (leaf service exports logs)
#   2. Traces — MUST NOT exist (leaf services don't create spans)
#   3. Metrics — MUST NOT exist (leaf services export no custom metrics)
#   4. Container stdout — MUST show "Embed completed"
#   5. No OTLP export errors
#
# Requirements:
#   kubectl, docker, python3, curl
#   SigNoz deployed in namespace signoz (or set SIGNOZ_NAMESPACE)
# ═══════════════════════════════════════════════════════════════════

# ─── Config ─────────────────────────────────────────────────────
SIGNOZ_NAMESPACE="${SIGNOZ_NAMESPACE:-signoz}"
COLLECTOR_SVC="${COLLECTOR_SVC:-signoz-otel-collector}"
COLLECTOR_PORT="${COLLECTOR_PORT:-4317}"
CLICKHOUSE_SVC="${CLICKHOUSE_SVC:-chi-signoz-clickhouse-cluster-0-0}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-8123}"

DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-BAAI/bge-small-en-v1.5}"
DENSE_DIM="${DENSE_DIM:-384}"
IMAGE_LOCAL="dense:test"
CONTAINER_NAME="dense-battle-test"
SERVICE_URL="http://127.0.0.1:8200"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"

command -v docker   >/dev/null 2>&1 || { echo "[ERROR] docker not found"   >&2; exit 1; }
command -v kubectl  >/dev/null 2>&1 || { echo "[ERROR] kubectl not found"  >&2; exit 1; }
command -v python3  >/dev/null 2>&1 || { echo "[ERROR] python3 not found"  >&2; exit 1; }
command -v curl     >/dev/null 2>&1 || { echo "[ERROR] curl not found"     >&2; exit 1; }

case "$(uname -m)" in
  x86_64|amd64) PLATFORM="linux/amd64" ;;
  aarch64|arm64) PLATFORM="linux/arm64" ;;
  *)            PLATFORM="linux/amd64" ;;
esac

# ─── Global state ───────────────────────────────────────────────
PF_COLLECTOR=""
PF_CLICKHOUSE=""
TEST_START_EPOCH_NS=""
declare -a RESULTS

cleanup() {
  set +e
  echo ""
  echo "[CLEANUP] Tearing down..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  kill "${PF_COLLECTOR}"  2>/dev/null || true
  kill "${PF_CLICKHOUSE}" 2>/dev/null || true
  set -e
}
trap cleanup EXIT

# ─── Helper: record a check result ──────────────────────────────
record() {
  local name="$1" passed="$2" detail="$3"
  local mark="[PASS]"
  [[ "${passed}" == "true" ]] || mark="[FAIL]"
  echo "  ${mark} ${name}"
  [[ -n "${detail}" ]] && echo "         ${detail}"
  RESULTS+=("${mark} ${name}")
}

# ─── Port-forwards ──────────────────────────────────────────────
echo "[1/9] Port-forwarding SigNoz services..."
kubectl port-forward -n "${SIGNOZ_NAMESPACE}" svc/"${COLLECTOR_SVC}" \
  "${COLLECTOR_PORT}:4317" >/tmp/pf-collector.log 2>&1 &
PF_COLLECTOR=$!

kubectl port-forward -n "${SIGNOZ_NAMESPACE}" svc/"${CLICKHOUSE_SVC}" \
  "${CLICKHOUSE_PORT}:8123" >/tmp/pf-clickhouse.log 2>&1 &
PF_CLICKHOUSE=$!

echo "  Waiting for ports..."
for ((i=0; i<30; i++)); do
  if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${COLLECTOR_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${CLICKHOUSE_PORT}" 2>/dev/null; then
    echo "  Both ports reachable (collector :${COLLECTOR_PORT}, clickhouse :${CLICKHOUSE_PORT})"
    break
  fi
  sleep 1
done

# ─── ClickHouse helper ──────────────────────────────────────────
CH_URL="http://127.0.0.1:${CLICKHOUSE_PORT}"

ch_query() {
  # Execute a ClickHouse query. Returns JSONEachRow on success,
  # or '{"c":"ERR"}' on failure. Never exits with non-zero.
  curl -fsS --max-time 15 -X POST "${CH_URL}/" \
    --data-binary "${1} FORMAT JSONEachRow" 2>/dev/null || echo '{"c":"ERR"}'
}


ch_count() {
  local result
  result=$(ch_query "$1" | python3 -c "
import sys, json
try:
    rows = [json.loads(l) for l in sys.stdin if l.strip()]
    print(rows[0]['c'] if rows else 0)
except Exception:
    print(0)
" 2>/dev/null) || result=0
  # Guarantee we always output a number, even if something went wrong
  if [[ "${result}" =~ ^[0-9]+$ ]]; then
    echo "${result}"
  else
    echo 0
  fi
}


# ─── Verify ClickHouse ──────────────────────────────────────────
echo "  Verifying ClickHouse..."
if ! curl -fsS --max-time 5 "${CH_URL}/ping" >/dev/null 2>&1; then
  echo "[ERROR] ClickHouse not responding on port ${CLICKHOUSE_PORT}" >&2
  exit 1
fi

LOGS_TABLE_EXISTS=$(ch_query "EXISTS TABLE signoz_logs.distributed_logs_v2" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
if [[ "${LOGS_TABLE_EXISTS}" != "1" ]]; then
  echo "[ERROR] Expected table signoz_logs.distributed_logs_v2 not found." >&2
  echo "  Available tables in signoz_logs:" >&2
  ch_query "SHOW TABLES FROM signoz_logs"
  exit 1
fi

TRACES_TABLE_EXISTS=$(ch_query "EXISTS TABLE signoz_traces.distributed_signoz_index_v3" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
if [[ "${TRACES_TABLE_EXISTS}" != "1" ]]; then
  echo "[ERROR] Expected table signoz_traces.distributed_signoz_index_v3 not found." >&2
  echo "  Available tables in signoz_traces:" >&2
  ch_query "SHOW TABLES FROM signoz_traces"
  exit 1
fi
echo "  ClickHouse: OK (logs + traces tables confirmed)"

# ─── Build image ────────────────────────────────────────────────
echo "[2/9] Building Docker image..."
docker build \
  --platform "${PLATFORM}" \
  --build-arg FASTEMBED_GPU=0 \
  --build-arg DENSE_MODEL_NAME="${DENSE_MODEL_NAME}" \
  --build-arg DENSE_DIM="${DENSE_DIM}" \
  -t "${IMAGE_LOCAL}" . >/dev/null 2>&1 || { echo "[ERROR] Build failed" >&2; exit 1; }
echo "  Image ready: ${IMAGE_LOCAL}"

# ─── Start container with host network ──────────────────────────
echo "[3/9] Starting dense service with host network..."
docker run --name "${CONTAINER_NAME}" -d \
  --network host \
  --shm-size=1.8g \
  -e DENSE_HOST="127.0.0.1" \
  -e DENSE_PORT="8200" \
  -e OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:${COLLECTOR_PORT}" \
  -e OTEL_SERVICE_NAME="dense-embedder" \
  -e OTEL_EXPORTER_OTLP_INSECURE="true" \
  -e LOG_LEVEL="INFO" \
  -e DEPLOYMENT_ENVIRONMENT="battle-test" \
  -e SERVICE_VERSION="0.1.0" \
  "${IMAGE_LOCAL}" >/dev/null
echo "  Container: ${CONTAINER_NAME} (host network, listening on ${SERVICE_URL})"

# ─── Wait for readiness ─────────────────────────────────────────
echo "[4/9] Waiting for model to load..."
READY=0
for ((i=0; i<WAIT_TIMEOUT; i++)); do
  if body=$(curl -fsS --max-time 2 "${SERVICE_URL}/readyz" 2>/dev/null); then
    echo "  ${body}"
    READY=1
    break
  fi
  (( i % 20 == 0 && i > 0 )) && echo "  Still waiting... (${i}s elapsed, model may be downloading)"
  sleep 1
done

if [[ ${READY} -eq 0 ]]; then
  echo "[ERROR] Service not ready after ${WAIT_TIMEOUT}s" >&2
  docker logs --tail 30 "${CONTAINER_NAME}" || true
  exit 1
fi

# Quick OTLP connectivity check
if docker exec "${CONTAINER_NAME}" timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/${COLLECTOR_PORT}" 2>/dev/null; then
  echo "  OTLP endpoint reachable from container"
else
  echo "  [WARN] OTLP endpoint NOT reachable — check port-forward"
fi

# Record the test window start (nanoseconds since epoch — ClickHouse timestamps are UInt64 nanoseconds)
TEST_START_EPOCH_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")
sleep 1

# ─── Battle-test: send varied requests ──────────────────────────
echo "[5/9] Sending ${NUM_REQUESTS} embed requests..."

SUCCESS=0
FAILED=0
TIMINGS=""

for ((i=1; i<=NUM_REQUESTS; i++)); do
  BATCH_SIZE=$(( (RANDOM % 8) + 1 ))
  TEXTS="["
  for ((j=1; j<=BATCH_SIZE; j++)); do
    TEXTS+="\"battle-test-$(date +%s)-${i}-${j}\""
    [[ ${j} -lt ${BATCH_SIZE} ]] && TEXTS+=", "
  done
  TEXTS+="]"

  START_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")

  if resp=$(curl -fsS -X POST "${SERVICE_URL}/embed" \
    -H "Content-Type: application/json" \
    -d "{\"texts\":${TEXTS}}" \
    -w "\n%{http_code}" 2>/dev/null); then
    HTTP_CODE=$(echo "${resp}" | tail -1)
    BODY=$(echo "${resp}" | sed '$d')

    if [[ "${HTTP_CODE}" == "200" ]]; then
      VEC_COUNT=$(echo "${BODY}" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['vectors']))" 2>/dev/null || echo 0)
      if [[ "${VEC_COUNT}" -eq "${BATCH_SIZE}" ]]; then
        SUCCESS=$((SUCCESS + 1))
        echo -n "✓"
      else
        FAILED=$((FAILED + 1))
        echo -n "✗"
      fi
    else
      FAILED=$((FAILED + 1))
      echo -n "✗"
    fi
  else
    FAILED=$((FAILED + 1))
    echo -n "✗"
  fi

  END_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")
  ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
  TIMINGS+="${ELAPSED_MS} "

  (( i % 10 == 0 )) && echo " ${i}/${NUM_REQUESTS}"
  sleep 0.08
done
echo ""

TEST_END_EPOCH_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")

# ─── Latency stats ──────────────────────────────────────────────
if [[ ${SUCCESS} -gt 0 ]]; then
  AVG_MS=$(echo "${TIMINGS}" | python3 -c "
import sys, statistics
vals = [float(x) for x in sys.stdin.read().split()]
print(f'{statistics.mean(vals):.1f}')" 2>/dev/null || echo "?")
  MAX_MS=$(echo "${TIMINGS}" | python3 -c "
import sys
vals = [float(x) for x in sys.stdin.read().split()]
print(f'{max(vals):.1f}')" 2>/dev/null || echo "?")
else
  AVG_MS="?"; MAX_MS="?"
fi

echo "  Requests: ${NUM_REQUESTS} | Success: ${SUCCESS} | Failed: ${FAILED}"
echo "  Latency:  avg=${AVG_MS}ms  max=${MAX_MS}ms"

[[ ${FAILED} -gt 0 ]] && { echo "[ERROR] ${FAILED} requests failed — aborting" >&2; exit 1; }

# ─── Wait for ClickHouse ingestion ──────────────────────────────
echo "[6/9] Waiting for SigNoz to ingest telemetry (15s)..."
sleep 15

# ═══════════════════════════════════════════════════════════════════
#  ClickHouse verification
# ═══════════════════════════════════════════════════════════════════
echo "[7/9] Querying ClickHouse for verification..."

TIMESTAMP_START="${TEST_START_EPOCH_NS}"
TIMESTAMP_END="$(( TEST_END_EPOCH_NS + 30000000000 ))"

# ── Check 1: Logs must exist ───────────────────────────────────
echo ""
echo "  Querying logs table..."
LOG_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'dense-embedder'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND body LIKE '%Embed completed%'
")

if [[ "${LOG_COUNT}" -gt 0 ]]; then
  record "Logs (Embed completed in dense-embedder)" "true" "Found ${LOG_COUNT} log lines"
else
  record "Logs (Embed completed in dense-embedder)" "false" "Zero log lines found — check collector connectivity"
fi

# ── Check 2: Traces must NOT exist (leaf service) ───────────────
echo ""
echo "  Querying traces table..."
TRACE_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.distributed_signoz_index_v3
WHERE resources_string['service.name'] = 'dense-embedder'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
")

if [[ "${TRACE_COUNT}" -eq 0 ]]; then
  record "Traces absent (leaf service)" "true" "Zero spans — correct for leaf service"
else
  record "Traces absent (leaf service)" "false" "Found ${TRACE_COUNT} spans — leaf service should NOT create traces"
fi

# ── Check 3: Metrics must NOT exist (leaf service) ──────────────
echo ""
echo "  Querying metrics table..."
METRIC_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_metrics.distributed_time_series_v2
WHERE metric_name LIKE 'dense.%'
  AND timestamp_ms >= toUnixTimestamp64Milli(now64()) - 300000
")

if [[ "${METRIC_COUNT}" -eq 0 ]]; then
  record "Metrics absent (leaf service)" "true" "Zero dense.* metrics — correct for leaf service"
else
  record "Metrics absent (leaf service)" "false" "Found ${METRIC_COUNT} dense.* metrics — leaf service should NOT export custom metrics"
fi

# ── Check 4: Container stdout ───────────────────────────────────
echo ""
echo "─── Container Logs (local) ───"
CONTAINER_LOG=$(docker logs --tail 20 "${CONTAINER_NAME}" 2>&1)
echo "${CONTAINER_LOG}" | tail -10

if echo "${CONTAINER_LOG}" | grep -q "Embed completed"; then
  record "Container stdout (Embed completed)" "true" ""
else
  record "Container stdout (Embed completed)" "false" "Application may have crashed or not logged"
fi

# ── Check 5: No OTLP export errors ──────────────────────────────
if echo "${CONTAINER_LOG}" | grep -q "Failed to export"; then
  record "OTLP export errors" "false" "gRPC export failed — check port-forward and network mode"
else
  record "OTLP export errors (none)" "true" ""
fi

# ── Check 6: Health endpoints ───────────────────────────────────
if curl -fsS --max-time 2 "${SERVICE_URL}/healthz" >/dev/null 2>&1; then
  record "Health endpoint /healthz" "true" ""
else
  record "Health endpoint /healthz" "false" "Service may have stopped"
fi

if curl -fsS --max-time 2 "${SERVICE_URL}/readyz" >/dev/null 2>&1; then
  record "Health endpoint /readyz" "true" ""
else
  record "Health endpoint /readyz" "false" "Service may have stopped"
fi

# ═══════════════════════════════════════════════════════════════════
#  Final report
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "[8/9] ═══════════════════════════════════════════════════════"
echo "       Battle Test Summary"
echo "       ═══════════════════════════════════════════════════════"
echo "       Requests:  ${SUCCESS}/${NUM_REQUESTS} succeeded (avg ${AVG_MS}ms, max ${MAX_MS}ms)"
echo ""
echo "       Verification:"
for result in "${RESULTS[@]}"; do
  echo "       ${result}"
done

FAIL_COUNT=$(printf '%s\n' "${RESULTS[@]}" | grep -c "\[FAIL\]" || true)
echo ""
if [[ "${FAIL_COUNT}" -eq 0 ]]; then
  echo "       ✅ All checks passed — dense-embedder is battle-ready"
  echo "       ═══════════════════════════════════════════════════════"
  exit 0
else
  echo "       ❌ ${FAIL_COUNT} check(s) failed — review details above"
  echo "       ═══════════════════════════════════════════════════════"
  exit 1
fi