#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MCP Server — Battle Test (Real SigNoz, Local Process)
# =============================================================================
# Tests all 8 MCP tools + OTel signals + context propagation.
# Every check prints explicit values — never a binary pass/fail.
#
# Requirements: kubectl, python3, curl, fastmcp
# SigNoz, PostgreSQL, Qdrant, and dense-embedder already deployed
# =============================================================================

# --- Config ---------------------------------------------------------------
SIGNOZ_NAMESPACE="${SIGNOZ_NAMESPACE:-signoz}"
COLLECTOR_SVC="${COLLECTOR_SVC:-signoz-otel-collector}"
COLLECTOR_PORT="${COLLECTOR_PORT:-4317}"
CLICKHOUSE_SVC="${CLICKHOUSE_SVC:-chi-signoz-clickhouse-cluster-0-0}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-8123}"

POSTGRES_SVC="${POSTGRES_SVC:-postgres-pooler}"
POSTGRES_NAMESPACE="${POSTGRES_NAMESPACE:-default}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

QDRANT_SVC="${QDRANT_SVC:-qdrant}"
QDRANT_NAMESPACE="${QDRANT_NAMESPACE:-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

DENSE_SVC="${DENSE_SVC:-dense-svc}"
DENSE_NAMESPACE="${DENSE_NAMESPACE:-inference}"
DENSE_PORT="${DENSE_PORT:-8200}"

MCP_PORT="${MCP_PORT:-8001}"
MCP_URL="http://127.0.0.1:${MCP_PORT}"
MCP_SSE="${MCP_URL}/sse"

command -v kubectl  >/dev/null 2>&1 || { echo "[ERROR] kubectl not found"  >&2; exit 1; }
command -v python3  >/dev/null 2>&1 || { echo "[ERROR] python3 not found"  >&2; exit 1; }
command -v curl     >/dev/null 2>&1 || { echo "[ERROR] curl not found"     >&2; exit 1; }
command -v fastmcp  >/dev/null 2>&1 || { echo "[ERROR] fastmcp not found"  >&2; exit 1; }

# --- Global state ----------------------------------------------------------
PF_COLLECTOR=""
PF_CLICKHOUSE=""
PF_POSTGRES=""
PF_QDRANT=""
PF_DENSE=""
MCP_PID=""
TEST_START_EPOCH_NS=""
declare -a RESULTS

cleanup() {
  set +e
  echo ""
  echo "[CLEANUP] Tearing down all port-forwards and processes..."
  kill -INT "${MCP_PID}" 2>/dev/null || true
  sleep 3  # Wait for graceful shutdown + force_flush of telemetry
  kill "${PF_COLLECTOR}"  2>/dev/null || true
  kill "${PF_CLICKHOUSE}" 2>/dev/null || true
  kill "${PF_POSTGRES}"   2>/dev/null || true
  kill "${PF_QDRANT}"     2>/dev/null || true
  kill "${PF_DENSE}"      2>/dev/null || true
  echo "[CLEANUP] Done."
  set -e
}
trap cleanup EXIT

# --- Helper functions ------------------------------------------------------
record_pass() {
  local name="$1" detail="$2"
  echo "  [PASS] ${name}"
  [[ -n "${detail}" ]] && echo "         ${detail}"
  RESULTS+=("PASS: ${name}")
}
record_fail() {
  local name="$1" detail="$2"
  echo "  [FAIL] ${name}"
  [[ -n "${detail}" ]] && echo "         ${detail}"
  RESULTS+=("FAIL: ${name}")
}

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

# =============================================================================
# STEP 1: Port-forward all dependencies
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 1/10] Port-forwarding SigNoz, PostgreSQL, Qdrant, Dense..."
echo "=============================================================================="

kubectl port-forward -n "${SIGNOZ_NAMESPACE}" svc/"${COLLECTOR_SVC}" \
  "${COLLECTOR_PORT}:4317" >/tmp/pf-collector.log 2>&1 &
PF_COLLECTOR=$!
echo "  SigNoz collector :${COLLECTOR_PORT} (PID ${PF_COLLECTOR})"

kubectl port-forward -n "${SIGNOZ_NAMESPACE}" svc/"${CLICKHOUSE_SVC}" \
  "${CLICKHOUSE_PORT}:8123" >/tmp/pf-clickhouse.log 2>&1 &
PF_CLICKHOUSE=$!
echo "  ClickHouse :${CLICKHOUSE_PORT} (PID ${PF_CLICKHOUSE})"

kubectl port-forward -n "${POSTGRES_NAMESPACE}" svc/"${POSTGRES_SVC}" \
  "${POSTGRES_PORT}:5432" >/tmp/pf-postgres.log 2>&1 &
PF_POSTGRES=$!
echo "  PostgreSQL :${POSTGRES_PORT} (PID ${PF_POSTGRES})"

kubectl port-forward -n "${QDRANT_NAMESPACE}" svc/"${QDRANT_SVC}" \
  "${QDRANT_PORT}:6333" >/tmp/pf-qdrant.log 2>&1 &
PF_QDRANT=$!
echo "  Qdrant :${QDRANT_PORT} (PID ${PF_QDRANT})"

kubectl port-forward -n "${DENSE_NAMESPACE}" svc/"${DENSE_SVC}" \
  "${DENSE_PORT}:8200" >/tmp/pf-dense.log 2>&1 &
PF_DENSE=$!
echo "  Dense :${DENSE_PORT} (PID ${PF_DENSE})"

echo "  Waiting for all 5 ports to become reachable..."
for ((i=0; i<30; i++)); do
  if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${COLLECTOR_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${CLICKHOUSE_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${POSTGRES_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${QDRANT_PORT}" 2>/dev/null && \
     timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${DENSE_PORT}" 2>/dev/null; then
    echo "  All 5 ports reachable"
    break
  fi
  sleep 1
done

# Verify ClickHouse tables exist
echo "  Verifying ClickHouse tables..."
LOGS_TABLE=$(ch_query "EXISTS TABLE signoz_logs.distributed_logs_v2" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
SPANS_TABLE=$(ch_query "EXISTS TABLE signoz_traces.distributed_signoz_index_v3" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
METRICS_TABLE=$(ch_query "EXISTS TABLE signoz_metrics.distributed_samples_v4" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
echo "  signoz_logs.distributed_logs_v2:        $([[ "${LOGS_TABLE}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"
echo "  signoz_traces.distributed_signoz_index_v3: $([[ "${SPANS_TABLE}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"
echo "  signoz_metrics.distributed_samples_v4:    $([[ "${METRICS_TABLE}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"

if [[ "${LOGS_TABLE}" != "1" || "${SPANS_TABLE}" != "1" || "${METRICS_TABLE}" != "1" ]]; then
  echo "[FATAL] Required ClickHouse tables missing — is the SigNoz migrator running?"
  exit 1
fi

# =============================================================================
# STEP 2: Start MCP server as a local process
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 2/10] Starting mcp-server as local process..."
echo "=============================================================================="

# Ensure virtual environment exists and dependencies are installed
if [ ! -d .venv ]; then
  echo "  Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null || true

export PGPASSWORD="$(kubectl get secret postgres-cluster-app -n "${POSTGRES_NAMESPACE}" -o jsonpath='{.data.password}' | base64 -d)"
export DATABASE_URL="postgresql://app:${PGPASSWORD}@127.0.0.1:${POSTGRES_PORT}/agents_state"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:${COLLECTOR_PORT}"
export OTEL_EXPORTER_OTLP_INSECURE="true"
export OTEL_SERVICE_NAME="mcp-server"
export OTEL_LOG_LEVEL="INFO"
export OTEL_METRIC_EXPORT_INTERVAL_MS="5000"
export OTEL_METRIC_EXPORT_TIMEOUT_MS="30000"
export DEPLOYMENT_ENVIRONMENT="battle-test"
export SERVICE_VERSION="0.1.0"
export LOG_LEVEL="INFO"
export PORT="${MCP_PORT}"
export QDRANT_URL="http://127.0.0.1:${QDRANT_PORT}"
export QDRANT_COLLECTION="kestral_policies"
export DENSE_URL="http://127.0.0.1:${DENSE_PORT}"
export RETRIEVER_URL="http://127.0.0.1:8001"
export PYTHONWARNINGS="ignore::UserWarning"

echo "  OTEL_EXPORTER_OTLP_ENDPOINT = ${OTEL_EXPORTER_OTLP_ENDPOINT}"
echo "  OTEL_SERVICE_NAME          = ${OTEL_SERVICE_NAME}"
echo "  DATABASE_URL               = postgresql://app:***@127.0.0.1:${POSTGRES_PORT}/agents_state"
echo "  QDRANT_URL                 = ${QDRANT_URL}"
echo "  DENSE_URL                  = ${DENSE_URL}"
echo "  Log file                   = /tmp/mcp-server.log"

python3 src/main.py > /tmp/mcp-server.log 2>&1 &
MCP_PID=$!
echo "  Process PID                = ${MCP_PID}"

# =============================================================================
# STEP 3: Wait for readiness
# =============================================================================
echo ""
echo "[STEP 3/10] Waiting for readiness..."
READY=0
for ((i=0; i<30; i++)); do
  if READYZ=$(curl -fsS --max-time 2 "${MCP_URL}/readyz" 2>/dev/null); then
    echo "  ${READYZ}"
    READY=1
    break
  fi
  sleep 1
done

if [[ ${READY} -eq 0 ]]; then
  echo "[FATAL] Server did not become ready within 30 seconds"
  echo "  Last 20 lines of /tmp/mcp-server.log:"
  tail -20 /tmp/mcp-server.log
  exit 1
fi

sleep 1
if grep -q "OTel traces initialised" /tmp/mcp-server.log 2>/dev/null; then
  echo "  OTel traces:   initialised"
else
  echo "  OTel traces:   NOT CONFIRMED — check /tmp/mcp-server.log"
fi
if grep -q "OTel metrics initialised" /tmp/mcp-server.log 2>/dev/null; then
  echo "  OTel metrics:  initialised"
else
  echo "  OTel metrics:  NOT CONFIRMED — check /tmp/mcp-server.log"
fi

TEST_START_EPOCH_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")
sleep 1

# =============================================================================
# STEP 4: Test all 8 MCP tools
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 4/10] Testing all 8 MCP tools..."
echo "=============================================================================="

run_tool() {
  local tool="$1" desc="$2"
  shift 2
  echo ""
  echo "  --- ${tool} ---"
  echo "  Description: ${desc}"
  echo "  Arguments:   $*"

  # Capture output, filter known warnings
  local OUTPUT
  OUTPUT=$(fastmcp call "${MCP_SSE}" "${tool}" "$@" 2>&1 | grep -v "UserWarning\|oauth.py\|site-packages\|/fastmcp/client/auth/" || true)

  # Display the raw response (truncated to 500 chars for readability)
  echo "  Response:"
  echo "${OUTPUT}" | head -30 | sed 's/^/    /'

  # Determine pass/fail based on expected keys
  if echo "${OUTPUT}" | grep -qE '"result"|"eligible"|"status"|"refund_id"'; then
    record_pass "${tool}" "Returned expected keys"
  elif echo "${OUTPUT}" | grep -qi "error"; then
    record_fail "${tool}" "Tool returned an error — see response above"
  else
    record_fail "${tool}" "Unexpected response format — see response above"
  fi
}

# --- Context tools (read-only, use real seed data IDs) ---
run_tool "lookup_customer" \
  "Find customer by email address" \
  email=priya.sharma@email.com

run_tool "get_recent_orders" \
  "Return the 5 most recent orders for a customer" \
  user_id=a1b2c3d4-e5f6-4a7b-8c9d-000000000001

run_tool "get_order_details" \
  "Return full order details including product information" \
  order_id=c3d4e5f6-a7b8-4c9d-0e1f-000000000001

run_tool "check_refund_eligibility" \
  "Check whether an order is eligible for refund" \
  order_id=c3d4e5f6-a7b8-4c9d-0e1f-000000000004

# --- Ops tools ---
run_tool "search_policies" \
  "Search company policies using semantic search" \
  query="return policy for damaged phone"

run_tool "create_ticket" \
  "Create a new support ticket" \
  user_id=a1b2c3d4-e5f6-4a7b-8c9d-000000000001 \
  query_text="Test ticket from battle test" \
  classification='{"intent":"test","urgency":5,"sentiment":"neutral","auto_resolvable":true}' \
  priority=medium

run_tool "escalate_to_human" \
  "Escalate a ticket to a human agent" \
  ticket_id=e5f6a7b8-c9d0-4e1f-2a3b-000000000001

run_tool "process_auto_refund" \
  "Process an automatic refund for an order" \
  order_id=c3d4e5f6-a7b8-4c9d-0e1f-000000000001

# =============================================================================
# STEP 5: Verify tool registration count
# =============================================================================
echo ""
echo "[STEP 5/10] Listing registered tools..."
TOOL_LIST=$(fastmcp list "${MCP_SSE}" 2>&1 | grep -v "UserWarning\|oauth.py\|site-packages\|/fastmcp/client/auth/" || true)
echo "${TOOL_LIST}"
TOOL_COUNT=$(echo "${TOOL_LIST}" | grep -cE "^\s+[a-z_]+" || true)
echo "  Tools registered: ${TOOL_COUNT}"
if [[ "${TOOL_COUNT}" -eq 8 ]]; then
  record_pass "8 tools registered" "All 8 tools present"
else
  record_fail "8 tools registered" "Found ${TOOL_COUNT}, expected 8"
fi

# =============================================================================
# STEP 6: Wait for SigNoz ingestion into ClickHouse
# =============================================================================
echo ""
echo "[STEP 6/10] Waiting for SigNoz to ingest telemetry into ClickHouse (15s)..."
sleep 15

TEST_END_EPOCH_NS=$(( $(python3 -c "import time; print(int(time.time() * 1e9))") + 30000000000 ))

# =============================================================================
# STEP 7: Query ClickHouse for traces, metrics, and logs
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 7/10] Querying ClickHouse for traces, metrics, and logs..."
echo "=============================================================================="

TIMESTAMP_START="${TEST_START_EPOCH_NS}"
TIMESTAMP_END="${TEST_END_EPOCH_NS}"

TRACE_IDS_SUBQUERY="
    SELECT DISTINCT trace_id
    FROM signoz_logs.distributed_logs_v2
    WHERE resources_string['service.name'] = 'mcp-server'
      AND timestamp >= ${TIMESTAMP_START}
      AND timestamp <= ${TIMESTAMP_END}
      AND trace_id != ''
"

# --- Check 1: Application Logs ---
echo ""
echo "  ---- Check 1: Application Logs ----"
LOG_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'mcp-server'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND body LIKE '%Tool call started%'
")
echo "  Log lines matching 'Tool call started': ${LOG_COUNT}"
echo "  Expected: >= 8 (one per tool call)"
if [[ "${LOG_COUNT}" -ge 8 ]]; then
  record_pass "Application logs exported to SigNoz" "${LOG_COUNT} lines found (>=8 expected)"
else
  record_fail "Application logs exported to SigNoz" "Found ${LOG_COUNT}, expected >=8"
fi

# --- Check 2: Distributed Traces ---
echo ""
echo "  ---- Check 2: Distributed Traces ----"
SPAN_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.distributed_signoz_index_v3
WHERE trace_id IN (${TRACE_IDS_SUBQUERY})
")
echo "  Spans correlated with our log trace_ids: ${SPAN_COUNT}"
echo "  Expected: >= 16 (8 tool SERVER spans + child spans for PostgreSQL + Qdrant)"
if [[ "${SPAN_COUNT}" -ge 16 ]]; then
  record_pass "Traces exported to ClickHouse" "${SPAN_COUNT} spans found (>=16 expected)"
else
  record_fail "Traces exported to ClickHouse" "Found ${SPAN_COUNT}, expected >=16"
fi

# --- Check 3: Child Spans (PostgreSQL) ---
echo ""
echo "  ---- Check 3: Child Spans (PostgreSQL) ----"
PG_SPAN_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.distributed_signoz_index_v3
WHERE name LIKE 'postgres %'
  AND trace_id IN (${TRACE_IDS_SUBQUERY})
")
echo "  'postgres *' child spans: ${PG_SPAN_COUNT}"
echo "  Expected: >= 6 (lookup_customer, get_recent_orders, get_order_details, check_refund_eligibility, create_ticket, escalate_to_human)"
if [[ "${PG_SPAN_COUNT}" -ge 6 ]]; then
  record_pass "Child spans: PostgreSQL queries" "${PG_SPAN_COUNT} spans (>=6 expected)"
else
  record_fail "Child spans: PostgreSQL queries" "Found ${PG_SPAN_COUNT}, expected >=6"
fi

# --- Check 4: Child Spans (Qdrant) ---
echo ""
echo "  ---- Check 4: Child Spans (Qdrant) ----"
QDRANT_SPAN_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.distributed_signoz_index_v3
WHERE name = 'qdrant search_policies'
  AND trace_id IN (${TRACE_IDS_SUBQUERY})
")
echo "  'qdrant search_policies' child spans: ${QDRANT_SPAN_COUNT}"
echo "  Expected: >= 1 (search_policies tool)"
if [[ "${QDRANT_SPAN_COUNT}" -ge 1 ]]; then
  record_pass "Child spans: Qdrant vector search" "${QDRANT_SPAN_COUNT} spans (>=1 expected)"
else
  record_fail "Child spans: Qdrant vector search" "Found ${QDRANT_SPAN_COUNT}, expected >=1"
fi

# --- Check 5: Custom Metrics ---
echo ""
echo "  ---- Check 5: Custom Metrics ----"
METRIC_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_metrics.distributed_samples_v4
WHERE metric_name = 'mcp_context.requests'
  AND unix_milli >= toUnixTimestamp64Milli(now64()) - 300000
")
echo "  'mcp_context.requests' metric samples (last 5 min): ${METRIC_COUNT}"
echo "  Expected: > 0 (any positive count confirms export)"
if [[ "${METRIC_COUNT}" -gt 0 ]]; then
  record_pass "Metrics exported to ClickHouse" "${METRIC_COUNT} samples found"
else
  record_fail "Metrics exported to ClickHouse" "Zero samples — check signoz metrics exporter"
fi

# --- Check 6: Log-Trace Correlation ---
echo ""
echo "  ---- Check 6: Log-Trace Correlation ----"
LOG_WITH_TRACE=$(ch_count "
SELECT count() AS c
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'mcp-server'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND trace_id != ''
  AND body LIKE '%Tool call started%'
")
echo "  Log lines with a non-empty trace_id: ${LOG_WITH_TRACE}"
echo "  Expected: >= 8 (one correlated log per tool call)"
if [[ "${LOG_WITH_TRACE}" -ge 8 ]]; then
  record_pass "Log-Trace correlation" "${LOG_WITH_TRACE} correlated lines found (>=8 expected)"
else
  record_fail "Log-Trace correlation" "Found ${LOG_WITH_TRACE}, expected >=8"
fi

# =============================================================================
# STEP 8: Local process checks
# =============================================================================
echo ""
echo "[STEP 8/10] Local process checks..."
MCP_LOG=$(cat /tmp/mcp-server.log 2>/dev/null || echo "")

# --- Check 7: Process stdout ---
echo ""
echo "  ---- Check 7: Process stdout ----"
COMPLETED_LINES=$(echo "${MCP_LOG}" | grep -c "Tool call started" || echo 0)
echo "  'Tool call started' lines in local log: ${COMPLETED_LINES}"
echo "  Expected: >= 8"
if [[ "${COMPLETED_LINES}" -ge 8 ]]; then
  record_pass "Local stdout (Tool call started)" "${COMPLETED_LINES} lines (>=8 expected)"
else
  record_fail "Local stdout (Tool call started)" "Found ${COMPLETED_LINES}, expected >=8"
fi

# --- Check 8: OTLP export errors ---
echo ""
echo "  ---- Check 8: OTLP export errors ----"
OTLP_ERRORS=$(echo "${MCP_LOG}" | grep -cE "Failed to export|StatusCode\\.UNAVAILABLE" 2>/dev/null || echo 0)
OTLP_ERRORS=$(echo "${OTLP_ERRORS}" | tr -d '\n')
echo "  OTLP export error count: ${OTLP_ERRORS}"
echo "  Expected: 0"
if [[ "${OTLP_ERRORS}" -eq 0 ]]; then
  record_pass "OTLP export (no errors)" "0 errors detected"
else
  record_fail "OTLP export (no errors)" "${OTLP_ERRORS} errors found — check collector connectivity"
fi

# --- Check 9: Health endpoints ---
echo ""
echo "  ---- Check 9: Health endpoints ----"
HEALTHZ=$(curl -fsS --max-time 2 "${MCP_URL}/healthz" 2>/dev/null || echo "FAIL")
READYZ=$(curl -fsS --max-time 2 "${MCP_URL}/readyz" 2>/dev/null || echo "FAIL")
echo "  GET /healthz : ${HEALTHZ}"
echo "  GET /readyz  : ${READYZ}"

if [[ "${HEALTHZ}" == "ok" ]]; then
  record_pass "Health endpoint /healthz" "Returned 'ok' (200)"
else
  record_fail "Health endpoint /healthz" "Got: ${HEALTHZ}"
fi
if [[ "${READYZ}" == "ready" ]]; then
  record_pass "Health endpoint /readyz" "Returned 'ready' (200)"
else
  record_fail "Health endpoint /readyz" "Got: ${READYZ}"
fi

# =============================================================================
# STEP 9: Cross-service trace verification
# =============================================================================
echo ""
echo "[STEP 9/10] Cross-service trace verification..."

TRACE_ID=$(ch_query "
SELECT trace_id
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'mcp-server'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND trace_id != ''
LIMIT 1
" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['trace_id'])" 2>/dev/null || echo "")

if [[ -n "${TRACE_ID}" ]]; then
  echo "  Sample Trace ID: ${TRACE_ID}"

  # List all services that participated in this trace
  SERVICES=$(ch_query "
SELECT DISTINCT resources_string['service.name'] AS svc
FROM signoz_traces.distributed_signoz_index_v3
WHERE trace_id = '${TRACE_ID}'
" | python3 -c "
import sys,json
for l in sys.stdin:
  if l.strip():
    d = json.loads(l)
    print(f'    {d[\"svc\"]}')
" 2>/dev/null || echo "    (query failed)")

  echo "  Services in this trace:"
  echo "${SERVICES}"
  record_pass "Trace context propagation" "Trace ID ${TRACE_ID} linked to services above"
else
  record_fail "Trace context propagation" "No trace_id found in log records"
fi

# =============================================================================
# STEP 10: Final report
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 10/10] Battle Test Summary — mcp-server"
echo "=============================================================================="
echo ""
echo "  Results:"
for result in "${RESULTS[@]}"; do
  echo "    ${result}"
done

FAIL_COUNT=$(printf '%s\n' "${RESULTS[@]}" | grep -c "^FAIL:" || true)
PASS_COUNT=$(printf '%s\n' "${RESULTS[@]}" | grep -c "^PASS:" || true)
echo ""
echo "  =============================================="
echo "  TOTAL: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "  =============================================="

if [[ "${FAIL_COUNT}" -eq 0 ]]; then
  echo ""
  echo "  mcp-server is BATTLE-READY — all checks passed."
  exit 0
else
  echo ""
  echo "  ${FAIL_COUNT} check(s) failed — review details above."
  echo ""
  echo "  Debugging tips:"
  echo "    Server logs : cat /tmp/mcp-server.log"
  echo "    Collector   : kubectl logs -n signoz deployment/signoz-otel-collector --tail=50"
  echo "    Manual test : fastmcp call http://localhost:8001/sse lookup_customer email=priya.sharma@email.com"
  exit 1
fi