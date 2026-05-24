#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# Retriever-Minimal — Battle Test (Real SigNoz, Local Process)
# ═══════════════════════════════════════════════════════════════════
# Every check prints explicit values so you never debug a binary yes/no.
#
# ClickHouse tables (SigNoz v0.118.0 / ClickHouse 25.5.6.14):
#   signoz_logs.distributed_logs_v2        — logs (resources_string, body, trace_id, timestamp)
#   signoz_traces.signoz_index_v3          — spans (name, trace_id, resource_string_service$$name)
#   signoz_metrics.distributed_samples_v4  — metrics (metric_name, unix_milli)
#
# Requirements:
#   kubectl, python3, curl
#   SigNoz, dense, Qdrant already deployed
# ═══════════════════════════════════════════════════════════════════

# ─── Config ─────────────────────────────────────────────────────
SIGNOZ_NAMESPACE="${SIGNOZ_NAMESPACE:-signoz}"
COLLECTOR_SVC="${COLLECTOR_SVC:-signoz-otel-collector}"
COLLECTOR_PORT="${COLLECTOR_PORT:-4317}"
CLICKHOUSE_SVC="${CLICKHOUSE_SVC:-chi-signoz-clickhouse-cluster-0-0}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-8123}"

DENSE_SVC="${DENSE_SVC:-dense-svc}"
DENSE_NAMESPACE="${DENSE_NAMESPACE:-inference}"
DENSE_PORT="${DENSE_PORT:-8200}"
DENSE_URL="http://127.0.0.1:${DENSE_PORT}"

QDRANT_SVC="${QDRANT_SVC:-qdrant}"
QDRANT_NAMESPACE="${QDRANT_NAMESPACE:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_URL="http://127.0.0.1:${QDRANT_PORT}"

COLLECTION_NAME="${COLLECTION_NAME:-documents}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8001}"
RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_PORT}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"

command -v kubectl  >/dev/null 2>&1 || { echo "[ERROR] kubectl not found"  >&2; exit 1; }
command -v python3  >/dev/null 2>&1 || { echo "[ERROR] python3 not found"  >&2; exit 1; }
command -v curl     >/dev/null 2>&1 || { echo "[ERROR] curl not found"     >&2; exit 1; }

# ─── Global state ───────────────────────────────────────────────
PF_COLLECTOR=""
PF_CLICKHOUSE=""
PF_DENSE=""
PF_QDRANT=""
RETRIEVER_PID=""
TEST_START_EPOCH_NS=""
declare -a RESULTS

cd src/workloads/rag/retriever-minimal
source .venv/bin/activate

cleanup() {
  set +e
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "[CLEANUP] Tearing down all port-forwards and processes..."
  kill -INT "${RETRIEVER_PID}" 2>/dev/null || true
  sleep 3  # Wait for graceful shutdown + force_flush of traces
  kill "${PF_COLLECTOR}"  2>/dev/null || true
  kill "${PF_CLICKHOUSE}" 2>/dev/null || true
  kill "${PF_DENSE}"      2>/dev/null || true
  kill "${PF_QDRANT}"     2>/dev/null || true
  echo "[CLEANUP] Done."
  set -e
}
trap cleanup EXIT

# ─── Helper: record a check result with explicit values ─────────
record_pass() {
  local name="$1" detail="$2"
  echo "  ✅ PASS | ${name}"
  [[ -n "${detail}" ]] && echo "          ${detail}"
  RESULTS+=("PASS: ${name}")
}

record_fail() {
  local name="$1" detail="$2"
  echo "  ❌ FAIL | ${name}"
  [[ -n "${detail}" ]] && echo "          ${detail}"
  RESULTS+=("FAIL: ${name}")
}

# ─── Port-forwards ──────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "[STEP 1/10] Port-forwarding SigNoz, dense, and Qdrant..."
echo "════════════════════════════════════════════════════════════"

kubectl port-forward -n "${SIGNOZ_NAMESPACE}" svc/"${COLLECTOR_SVC}" \
  "${COLLECTOR_PORT}:4317" >/tmp/pf-collector.log 2>&1 &
PF_COLLECTOR=$!
echo "  → SigNoz collector :${COLLECTOR_PORT} (PID ${PF_COLLECTOR})"

kubectl port-forward -n "${SIGNOZ_NAMESPACE}" svc/"${CLICKHOUSE_SVC}" \
  "${CLICKHOUSE_PORT}:8123" >/tmp/pf-clickhouse.log 2>&1 &
PF_CLICKHOUSE=$!
echo "  → ClickHouse :${CLICKHOUSE_PORT} (PID ${PF_CLICKHOUSE})"

kubectl port-forward -n "${DENSE_NAMESPACE}" svc/"${DENSE_SVC}" \
  "${DENSE_PORT}:8200" >/tmp/pf-dense.log 2>&1 &
PF_DENSE=$!
echo "  → Dense :${DENSE_PORT} (PID ${PF_DENSE})"

kubectl port-forward -n "${QDRANT_NAMESPACE}" svc/"${QDRANT_SVC}" \
  "${QDRANT_PORT}:6333" >/tmp/pf-qdrant.log 2>&1 &
PF_QDRANT=$!
echo "  → Qdrant :${QDRANT_PORT} (PID ${PF_QDRANT})"

echo "  Waiting for all ports to become reachable..."
for ((i=0; i<30; i++)); do
  if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${COLLECTOR_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${CLICKHOUSE_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${DENSE_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${QDRANT_PORT}" 2>/dev/null; then
    echo "  ✅ All 4 ports reachable"
    break
  fi
  sleep 1
done

# Verify critical dependencies
echo "  Verifying dependency health..."
DENSE_HEALTH=$(curl -fsS --max-time 5 "http://127.0.0.1:${DENSE_PORT}/healthz" 2>/dev/null || echo "UNREACHABLE")
echo "  → Dense /healthz: ${DENSE_HEALTH}"
if [[ "${DENSE_HEALTH}" != "{\"status\":\"ok\"}" ]]; then
  echo "[FATAL] Dense service not healthy — aborting"
  exit 1
fi

QDRANT_HTTP_CODE=$(curl -fsS --max-time 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${QDRANT_PORT}/collections" 2>/dev/null || echo "000")
if [[ "${QDRANT_HTTP_CODE}" == "200" ]]; then
  QDRANT_OK="ok"
else
  QDRANT_OK="fail (HTTP ${QDRANT_HTTP_CODE})"
fi
echo "  → Qdrant /collections: ${QDRANT_OK}"
if [[ "${QDRANT_OK}" != "ok" ]]; then
  echo "[FATAL] Qdrant not reachable — aborting"
  exit 1
fi

# ─── ClickHouse helpers ──────────────────────────────────────────
CH_URL="http://127.0.0.1:${CLICKHOUSE_PORT}"

ch_query() {
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
  if [[ "${result}" =~ ^[0-9]+$ ]]; then
    echo "${result}"
  else
    echo 0
  fi
}

# ─── Verify ClickHouse tables ───────────────────────────────────
echo ""
echo "  Verifying ClickHouse tables exist..."

LOGS_TABLE_EXISTS=$(ch_query "EXISTS TABLE signoz_logs.distributed_logs_v2" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
SPANS_TABLE_EXISTS=$(ch_query "EXISTS TABLE signoz_traces.signoz_index_v3" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
METRICS_TABLE_EXISTS=$(ch_query "EXISTS TABLE signoz_metrics.distributed_samples_v4" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")

echo "  → signoz_logs.distributed_logs_v2:   $([[ "${LOGS_TABLE_EXISTS}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"
echo "  → signoz_traces.signoz_index_v3:     $([[ "${SPANS_TABLE_EXISTS}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"
echo "  → signoz_metrics.distributed_samples_v4: $([[ "${METRICS_TABLE_EXISTS}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"

if [[ "${LOGS_TABLE_EXISTS}" != "1" || "${SPANS_TABLE_EXISTS}" != "1" || "${METRICS_TABLE_EXISTS}" != "1" ]]; then
  echo "[FATAL] Required ClickHouse tables missing — is the SigNoz migrator running?"
  echo "  Check: kubectl get jobs -n signoz | grep migrator"
  exit 1
fi

# ─── Start retriever locally ────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "[STEP 2/10] Starting retriever-minimal as local process..."
echo "════════════════════════════════════════════════════════════"

export DENSE_URL="${DENSE_URL}"
export QDRANT_URL="${QDRANT_URL}"
export COLLECTION_NAME="${COLLECTION_NAME}"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:${COLLECTOR_PORT}"
export OTEL_EXPORTER_OTLP_INSECURE="true"
export OTEL_SERVICE_NAME="retriever-minimal"
export OTEL_LOG_LEVEL="INFO"
export OTEL_METRIC_EXPORT_INTERVAL_MS="5000"
export LOG_LEVEL="INFO"
export DEPLOYMENT_ENVIRONMENT="battle-test"
export SERVICE_VERSION="0.1.0"
export PORT="${RETRIEVER_PORT}"

echo "  → OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT}"
echo "  → OTEL_SERVICE_NAME: ${OTEL_SERVICE_NAME}"
echo "  → DENSE_URL: ${DENSE_URL}"
echo "  → QDRANT_URL: ${QDRANT_URL}"
echo "  → COLLECTION_NAME: ${COLLECTION_NAME}"
echo "  → Log file: /tmp/retriever.log"

python3 main.py > /tmp/retriever.log 2>&1 &
RETRIEVER_PID=$!
echo "  → Process PID: ${RETRIEVER_PID}"

# ─── Wait for readiness ─────────────────────────────────────────
echo ""
echo "[STEP 3/10] Waiting for readiness..."
for ((i=0; i<30; i++)); do
  if curl -fsS --max-time 2 "${RETRIEVER_URL}/readyz" >/dev/null 2>&1; then
    READYZ_BODY=$(curl -fsS --max-time 2 "${RETRIEVER_URL}/readyz" 2>/dev/null || echo "{}")
    echo "  ✅ Ready — $(echo "${READYZ_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'status={d[\"status\"]} dense={d[\"dense\"]} qdrant={d[\"qdrant\"]}')" 2>/dev/null || echo "${READYZ_BODY}")"
    break
  fi
  sleep 1
done

# Check OTel init status from logs
sleep 1
if grep -q "OTel traces initialised" /tmp/retriever.log 2>/dev/null; then
  echo "  ✅ OTel traces initialized (confirmed from process logs)"
else
  echo "  ⚠️  OTel traces NOT initialized — check /tmp/retriever.log"
fi
if grep -q "OTel metrics initialised" /tmp/retriever.log 2>/dev/null; then
  echo "  ✅ OTel metrics initialized (confirmed from process logs)"
else
  echo "  ⚠️  OTel metrics NOT initialized — check /tmp/retriever.log"
fi

# Record test window start (nanoseconds)
TEST_START_EPOCH_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")
sleep 1

# ─── Battle-test: send varied queries ──────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "[STEP 4/10] Battle-test: sending ${NUM_REQUESTS} varied queries..."
echo "════════════════════════════════════════════════════════════"

SUCCESS=0
FAILED=0
TIMINGS=""
QUERIES=(
  "how do I reset my password?"
  "what is the refund policy for damaged items?"
  "shipping costs and delivery time to Mumbai"
  "I received a wrong product, want refund"
  "return policy for electronics"
  "how to cancel my order before shipment?"
  "my delivery is late by 5 days"
  "what is the warranty on mobile phones?"
  "can I return shoes after wearing them once?"
  "payment failed but money deducted from bank"
  "how to contact customer support?"
  "refund not received after 10 days"
  "is COD available for my pin code?"
  "damaged product on delivery"
  "how to track my order?"
)

for ((i=1; i<=NUM_REQUESTS; i++)); do
  QUERY="${QUERIES[$(( (i-1) % ${#QUERIES[@]} ))]}"
  TOPK=$(( (RANDOM % 5) + 1 ))

  START_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")

  if resp=$(curl -fsS -X POST "${RETRIEVER_URL}/retrieve" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"${QUERY}\", \"top_k\": ${TOPK}}" \
    -w "\n%{http_code}" 2>/dev/null); then
    HTTP_CODE=$(echo "${resp}" | tail -1)
    BODY=$(echo "${resp}" | sed '$d')

    if [[ "${HTTP_CODE}" == "200" ]]; then
      RESULT_COUNT=$(echo "${BODY}" | python3 -c "import sys,json;print(len(json.load(sys.stdin)['results']))" 2>/dev/null || echo 0)
      if [[ "${RESULT_COUNT}" -le "${TOPK}" ]]; then
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
  P95_MS=$(echo "${TIMINGS}" | python3 -c "
import sys, statistics
vals = sorted([float(x) for x in sys.stdin.read().split()])
idx = int(len(vals) * 0.95)
print(f'{vals[min(idx, len(vals)-1)]:.1f}')" 2>/dev/null || echo "?")
else
  AVG_MS="?"; MAX_MS="?"; P95_MS="?"
fi

echo ""
echo "  ═══════════════════════════════════════════"
echo "  Request Summary"
echo "  ═══════════════════════════════════════════"
echo "  Total:    ${NUM_REQUESTS}"
echo "  Success:  ${SUCCESS} (HTTP 200 + valid response)"
echo "  Failed:   ${FAILED}"
echo "  Latency:  avg=${AVG_MS}ms  p95=${P95_MS}ms  max=${MAX_MS}ms"

[[ ${FAILED} -gt 0 ]] && { echo "[FATAL] ${FAILED} requests failed — aborting" >&2; exit 1; }

# ─── Wait for ClickHouse ingestion ──────────────────────────────
echo ""
echo "[STEP 5/10] Waiting for SigNoz to ingest telemetry into ClickHouse (15s)..."
sleep 15

# ═══════════════════════════════════════════════════════════════════
#  ClickHouse verification
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "[STEP 6/10] Querying ClickHouse for traces, metrics, and logs..."
echo "════════════════════════════════════════════════════════════"

TIMESTAMP_START="${TEST_START_EPOCH_NS}"
TIMESTAMP_END="$(( TEST_END_EPOCH_NS + 30000000000 ))"

# Common subquery: trace_ids from our retriever-minimal logs in the test window
TRACE_IDS_SUBQUERY="
    SELECT DISTINCT trace_id
    FROM signoz_logs.distributed_logs_v2
    WHERE resources_string['service.name'] = 'retriever-minimal'
      AND timestamp >= ${TIMESTAMP_START}
      AND timestamp <= ${TIMESTAMP_END}
      AND trace_id != ''
"

# ── Check 1: Logs must exist ───────────────────────────────────
echo ""
echo "  ── Check 1: Application Logs ──"
LOG_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'retriever-minimal'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND body LIKE '%Retrieve completed%'
")
echo "  → Log lines matching 'Retrieve completed': ${LOG_COUNT}"
echo "  → Expected: >= ${NUM_REQUESTS} (one per request)"
if [[ "${LOG_COUNT}" -ge "${NUM_REQUESTS}" ]]; then
  record_pass "Application logs exported to SigNoz" "${LOG_COUNT} lines found (>=${NUM_REQUESTS} expected)"
else
  record_fail "Application logs exported to SigNoz" "Found ${LOG_COUNT}, expected >=${NUM_REQUESTS}"
fi

# ── Check 2: Traces must exist ─────────────────────────────────
echo ""
echo "  ── Check 2: Distributed Traces ──"
TRACE_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.signoz_index_v3
WHERE trace_id IN (${TRACE_IDS_SUBQUERY})
")
echo "  → Spans correlated with our log trace_ids: ${TRACE_COUNT}"
echo "  → Expected: 80 (20 POST /retrieve + 20 dense /embed + 20 qdrant search + 20 SERVER)"
if [[ "${TRACE_COUNT}" -ge 60 ]]; then
  record_pass "Traces exported to ClickHouse" "${TRACE_COUNT} spans found (>=60 expected)"
else
  record_fail "Traces exported to ClickHouse" "Found ${TRACE_COUNT}, expected >=60"
fi

# ── Check 3: Child spans ───────────────────────────────────────
echo ""
echo "  ── Check 3: Child Spans (invariant #4) ──"
DENSE_SPAN_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.signoz_index_v3
WHERE name = 'dense /embed'
  AND trace_id IN (${TRACE_IDS_SUBQUERY})
")
echo "  → 'dense /embed' child spans: ${DENSE_SPAN_COUNT}"
echo "  → Expected: 20 (one per request)"
if [[ "${DENSE_SPAN_COUNT}" -eq "${NUM_REQUESTS}" ]]; then
  record_pass "Child span: dense /embed" "${DENSE_SPAN_COUNT} spans (exactly ${NUM_REQUESTS})"
else
  record_fail "Child span: dense /embed" "Found ${DENSE_SPAN_COUNT}, expected ${NUM_REQUESTS}"
fi

QDRANT_SPAN_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.signoz_index_v3
WHERE name = 'qdrant search'
  AND trace_id IN (${TRACE_IDS_SUBQUERY})
")
echo "  → 'qdrant search' child spans: ${QDRANT_SPAN_COUNT}"
echo "  → Expected: 20 (one per request)"
if [[ "${QDRANT_SPAN_COUNT}" -eq "${NUM_REQUESTS}" ]]; then
  record_pass "Child span: qdrant search" "${QDRANT_SPAN_COUNT} spans (exactly ${NUM_REQUESTS})"
else
  record_fail "Child span: qdrant search" "Found ${QDRANT_SPAN_COUNT}, expected ${NUM_REQUESTS}"
fi

# ── Check 4: Metrics must exist ─────────────────────────────────
echo ""
echo "  ── Check 4: Custom Metrics (invariant #6) ──"
METRIC_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_metrics.distributed_samples_v4
WHERE metric_name = 'retrieve.requests'
  AND unix_milli >= toUnixTimestamp64Milli(now64()) - 300000
")
echo "  → 'retrieve.requests' metric samples: ${METRIC_COUNT}"
echo "  → Expected: > 0 (any positive count confirms export)"
if [[ "${METRIC_COUNT}" -gt 0 ]]; then
  record_pass "Metrics exported to ClickHouse" "${METRIC_COUNT} samples found"
else
  record_fail "Metrics exported to ClickHouse" "Zero samples — check signozclickhousemetrics exporter"
fi

# ── Check 5: Log correlation ───────────────────────────────────
echo ""
echo "  ── Check 5: Log-Trace Correlation (invariant #12) ──"
LOG_WITH_TRACE=$(ch_count "
SELECT count() AS c
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'retriever-minimal'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND trace_id != ''
  AND body LIKE '%Retrieve completed%'
")
echo "  → Log lines with trace_id: ${LOG_WITH_TRACE}"
echo "  → Expected: >= ${NUM_REQUESTS}"
if [[ "${LOG_WITH_TRACE}" -ge "${NUM_REQUESTS}" ]]; then
  record_pass "Log-trace correlation" "${LOG_WITH_TRACE} correlated lines (>=${NUM_REQUESTS})"
else
  record_fail "Log-trace correlation" "Found ${LOG_WITH_TRACE}, expected >=${NUM_REQUESTS}"
fi

# ─── Local checks ────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "[STEP 7/10] Local process checks..."
echo "════════════════════════════════════════════════════════════"

RETRIEVER_LOG=$(cat /tmp/retriever.log 2>/dev/null || echo "")

# Check 6: Local stdout
echo ""
echo "  ── Check 6: Process stdout ──"
COMPLETED_LINES=$(echo "${RETRIEVER_LOG}" | grep -c "Retrieve completed" || echo 0)
echo "  → 'Retrieve completed' in local log: ${COMPLETED_LINES} lines"
if [[ "${COMPLETED_LINES}" -ge "${NUM_REQUESTS}" ]]; then
  record_pass "Local stdout" "${COMPLETED_LINES} lines"
else
  record_fail "Local stdout" "Found ${COMPLETED_LINES}, expected >=${NUM_REQUESTS}"
fi

# Check 7: OTLP export errors
echo ""
echo "  ── Check 7: OTLP export errors ──"
OTLP_ERRORS=$(echo "${RETRIEVER_LOG}" | grep -cE "Failed to export|StatusCode\\.UNAVAILABLE|StatusCode\\.DEADLINE_EXCEEDED" 2>/dev/null || echo 0)
OTLP_ERRORS=$(echo "${OTLP_ERRORS}" | tr -d '\n')
echo "  → OTLP export errors: ${OTLP_ERRORS}"
if [[ "${OTLP_ERRORS}" -eq 0 ]]; then
  record_pass "OTLP export" "No errors detected"
else
  record_fail "OTLP export" "${OTLP_ERRORS} errors found — check collector connectivity"
fi

# Check 8: Health endpoints
echo ""
echo "  ── Check 8: Health endpoints ──"
HEALTHZ=$(curl -fsS --max-time 2 "${RETRIEVER_URL}/healthz" 2>/dev/null || echo "FAIL")
READYZ=$(curl -fsS --max-time 2 "${RETRIEVER_URL}/readyz" 2>/dev/null || echo "FAIL")
echo "  → /healthz: ${HEALTHZ}"
echo "  → /readyz:  ${READYZ}"
if [[ "${HEALTHZ}" == '{"status":"ok"}' ]]; then
  record_pass "Health /healthz" "200 OK"
else
  record_fail "Health /healthz" "Got: ${HEALTHZ}"
fi
if echo "${READYZ}" | grep -q '"ready"'; then
  record_pass "Health /readyz" "Ready"
else
  record_fail "Health /readyz" "Got: ${READYZ}"
fi

# ═══════════════════════════════════════════════════════════════════
#  Final report
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  BATTLE TEST COMPLETE — retriever-minimal"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Requests: ${SUCCESS}/${NUM_REQUESTS} succeeded"
echo "  Latency:  avg=${AVG_MS}ms  p95=${P95_MS}ms  max=${MAX_MS}ms"
echo ""
echo "  Results:"
for result in "${RESULTS[@]}"; do
  echo "    ${result}"
done

FAIL_COUNT=$(printf '%s\n' "${RESULTS[@]}" | grep -c "^FAIL:" || true)
PASS_COUNT=$(printf '%s\n' "${RESULTS[@]}" | grep -c "^PASS:" || true)
echo ""
echo "  ═══════════════════════════════════════════════"
echo "  TOTAL: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "  ═══════════════════════════════════════════════"

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
  echo ""
  echo "  ✅ retriever-minimal is BATTLE-READY"
  exit 0
else
  echo ""
  echo "  ❌ ${FAIL_COUNT} checks failed — review details above"
  echo ""
  echo "  Debugging tips:"
  echo "    - Check retriever logs:  cat /tmp/retriever.log"
  echo "    - Check collector logs:  kubectl logs -n signoz deployment/signoz-otel-collector --tail=50"
  echo "    - Check ClickHouse OOM: kubectl logs -n signoz chi-signoz-clickhouse-cluster-0-0-0 --tail=5 | grep MEMORY"
  echo "    - Manual ClickHouse query: kubectl port-forward -n signoz svc/chi-signoz-clickhouse-cluster-0-0 8123:8123"
  exit 1
fi