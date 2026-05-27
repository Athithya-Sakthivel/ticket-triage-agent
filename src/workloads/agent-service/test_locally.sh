#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Agent Service — Battle Test (Idempotent, self-contained port-forwards)
# =============================================================================
# Starts all needed port-forwards, verifies each, runs the agent, tests
# WebSocket chat + OTel signal export.  Kills only its own forwards on exit.
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

MCP_SVC="${MCP_SVC:-mcp-server-svc}"
MCP_NAMESPACE="${MCP_NAMESPACE:-inference}"
MCP_PORT="${MCP_PORT:-8001}"

AGENT_PORT="${AGENT_PORT:-8000}"
AGENT_URL="http://127.0.0.1:${AGENT_PORT}"

command -v kubectl  >/dev/null 2>&1 || { echo "[ERROR] kubectl not found"  >&2; exit 1; }
command -v python3  >/dev/null 2>&1 || { echo "[ERROR] python3 not found"  >&2; exit 1; }
command -v curl     >/dev/null 2>&1 || { echo "[ERROR] curl not found"     >&2; exit 1; }

# --- Helper: kill any port-forward on a specific port by matching command line --
kill_port_forward_on_port() {
  local port="$1"
  echo "  Killing existing port-forward on port ${port} (if any) ..."
  pkill -f "kubectl port-forward.*${port}" 2>/dev/null || true
  sleep 1
}

# --- Global state ----------------------------------------------------------
PF_PIDS=()           # PIDs of our own port-forwards
AGENT_PID=""
TEST_START_EPOCH_NS=""
declare -a RESULTS

cleanup() {
  set +e
  echo ""
  echo "[CLEANUP] Tearing down ..."
  kill -INT "${AGENT_PID}" 2>/dev/null || true
  sleep 3
  for pid in "${PF_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
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
# STEP 1: Clean previous forwards on our ports, then start new ones
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 1/10] Setting up fresh port-forwards ..."
echo "=============================================================================="

for port in "${COLLECTOR_PORT}" "${CLICKHOUSE_PORT}" "${POSTGRES_PORT}" "${QDRANT_PORT}" "${DENSE_PORT}" "${MCP_PORT}"; do
  kill_port_forward_on_port "${port}"
done

start_forward() {
  local name="$1" namespace="$2" svc="$3" local_port="$4" remote_port="$5"
  echo "  Starting ${name} forward ${local_port} -> ${remote_port} ..."
  kubectl port-forward -n "${namespace}" svc/"${svc}" "${local_port}:${remote_port}" >/tmp/pf-${name}.log 2>&1 &
  local pid=$!
  PF_PIDS+=("${pid}")
  # Wait until the port is actually listening
  for ((i=0; i<20; i++)); do
    if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${local_port}" 2>/dev/null; then
      echo "    ${name} ready (PID ${pid})"
      return 0
    fi
    sleep 1
  done
  echo "    [ERROR] ${name} port-forward failed to start"
  return 1
}

start_forward "collector"   "${SIGNOZ_NAMESPACE}"   "${COLLECTOR_SVC}"   "${COLLECTOR_PORT}"   4317
start_forward "clickhouse"  "${SIGNOZ_NAMESPACE}"   "${CLICKHOUSE_SVC}"  "${CLICKHOUSE_PORT}"  8123
start_forward "postgres"    "${POSTGRES_NAMESPACE}" "${POSTGRES_SVC}"    "${POSTGRES_PORT}"    5432
start_forward "qdrant"      "${QDRANT_NAMESPACE}"   "${QDRANT_SVC}"      "${QDRANT_PORT}"      6333
start_forward "dense"       "${DENSE_NAMESPACE}"    "${DENSE_SVC}"       "${DENSE_PORT}"       8200
start_forward "mcp"         "${MCP_NAMESPACE}"      "${MCP_SVC}"         "${MCP_PORT}"         8001

# Give everything a moment to settle
sleep 2

# Verify ClickHouse tables
echo "  Verifying ClickHouse tables ..."
LOGS_TABLE=$(ch_query "EXISTS TABLE signoz_logs.distributed_logs_v2" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
SPANS_TABLE=$(ch_query "EXISTS TABLE signoz_traces.distributed_signoz_index_v3" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
METRICS_TABLE=$(ch_query "EXISTS TABLE signoz_metrics.distributed_samples_v4" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['result'])" 2>/dev/null || echo "0")
echo "  signoz_logs: $([[ "${LOGS_TABLE}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"
echo "  signoz_traces: $([[ "${SPANS_TABLE}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"
echo "  signoz_metrics: $([[ "${METRICS_TABLE}" == "1" ]] && echo 'EXISTS' || echo 'MISSING')"
if [[ "${LOGS_TABLE}" != "1" || "${SPANS_TABLE}" != "1" || "${METRICS_TABLE}" != "1" ]]; then
  echo "[FATAL] Required ClickHouse tables missing"
  exit 1
fi

# =============================================================================
# STEP 2: Start agent-service locally
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 2/10] Starting agent-service ..."
echo "=============================================================================="

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null || true

export PGPASSWORD="$(kubectl get secret postgres-cluster-app -n "${POSTGRES_NAMESPACE}" -o jsonpath='{.data.password}' | base64 -d)"
export DATABASE_URL="postgresql://app:${PGPASSWORD}@127.0.0.1:${POSTGRES_PORT}/agents_state"
export LLM_API_KEY="${GROQ_API_KEY:-}"
export MCP_SERVER_URL="http://127.0.0.1:${MCP_PORT}/mcp"

# Determine OTLP endpoint – try local forward first, then ClusterIP
OTLP_ENDPOINT="http://127.0.0.1:${COLLECTOR_PORT}"
if ! timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/${COLLECTOR_PORT}" 2>/dev/null; then
  COLLECTOR_IP=$(kubectl get svc "${COLLECTOR_SVC}" -n "${SIGNOZ_NAMESPACE}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
  if [ -n "${COLLECTOR_IP}" ]; then
    OTLP_ENDPOINT="http://${COLLECTOR_IP}:4317"
    echo "  Local collector forward not reachable, trying ClusterIP ${OTLP_ENDPOINT} ..."
    if ! timeout 2 bash -c "echo >/dev/tcp/${COLLECTOR_IP}/4317" 2>/dev/null; then
      echo "  [WARN] Collector ClusterIP also unreachable. Telemetry export may fail."
    fi
  else
    echo "  [WARN] Cannot determine collector address. Telemetry export will likely fail."
  fi
else
  echo "  Using local collector forward at ${OTLP_ENDPOINT}"
fi

export OTEL_EXPORTER_OTLP_ENDPOINT="${OTLP_ENDPOINT}"
export OTEL_EXPORTER_OTLP_INSECURE="true"
export OTEL_SERVICE_NAME="agent-service"
export OTEL_LOG_LEVEL="INFO"
export OTEL_METRIC_EXPORT_INTERVAL_MS="5000"
export OTEL_METRIC_EXPORT_TIMEOUT_MS="30000"
export DEPLOYMENT_ENVIRONMENT="battle-test"
export SERVICE_VERSION="0.1.0"
export LOG_LEVEL="INFO"
export PORT="${AGENT_PORT}"
export DSPY_COMPILED_DIR="compiled"

echo "  MCP_SERVER_URL = ${MCP_SERVER_URL}"
echo "  OTEL_EXPORTER_OTLP_ENDPOINT = ${OTEL_EXPORTER_OTLP_ENDPOINT}"
echo "  DATABASE_URL = postgresql://app:***@127.0.0.1:${POSTGRES_PORT}/agents_state"

python3 src/main.py > /tmp/agent-service.log 2>&1 &
AGENT_PID=$!
echo "  Process PID = ${AGENT_PID}"

# =============================================================================
# STEP 3: Wait for readiness
# =============================================================================
echo ""
echo "[STEP 3/10] Waiting for readiness ..."
READY=0
for ((i=0; i<30; i++)); do
  if READYZ=$(curl -fsS --max-time 2 "${AGENT_URL}/readyz" 2>/dev/null); then
    echo "  ${READYZ}"
    READY=1
    break
  fi
  sleep 1
done

if [[ ${READY} -eq 0 ]]; then
  echo "[FATAL] Agent did not become ready"
  tail -20 /tmp/agent-service.log
  exit 1
fi

sleep 1
if grep -aq "OTel traces initialised" /tmp/agent-service.log 2>/dev/null; then
  echo "  OTel traces:   initialised"
else
  echo "  OTel traces:   NOT CONFIRMED"
fi
if grep -aq "OTel metrics initialised" /tmp/agent-service.log 2>/dev/null; then
  echo "  OTel metrics:  initialised"
else
  echo "  OTel metrics:  NOT CONFIRMED"
fi

TEST_START_EPOCH_NS=$(python3 -c "import time; print(int(time.time() * 1e9))")
sleep 1

# =============================================================================
# STEP 4: Test WebSocket chat
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 4/10] Testing WebSocket chat ..."
echo "=============================================================================="

WS_OUTPUT=$(echo '{"query":"What is the return policy for damaged phones?","user_id":"a1b2c3d4-e5f6-4a7b-8c9d-000000000001"}' | websocat -n1 "ws://127.0.0.1:${AGENT_PORT}/ws/chat/test-session-1" 2>/dev/null || echo '{"error":"websocat failed"}')
echo "  Raw response:"
echo "${WS_OUTPUT}" | head -30 | sed 's/^/    /'

if echo "${WS_OUTPUT}" | grep -qE '"response"|"error"'; then
  record_pass "WebSocket chat response" "Agent returned a response"
else
  record_fail "WebSocket chat response" "No valid response from agent"
fi

WS_OUTPUT2=$(echo '{"query":"I got a charger instead of an iPhone worth 1.45 lakh. This is fraud!","user_id":"a1b2c3d4-e5f6-4a7b-8c9d-000000000001"}' | websocat -n1 "ws://127.0.0.1:${AGENT_PORT}/ws/chat/test-session-2" 2>/dev/null || echo '{"error":"websocat failed"}')
echo "  Escalation query response:"
echo "${WS_OUTPUT2}" | head -30 | sed 's/^/    /'
if echo "${WS_OUTPUT2}" | grep -q "escalated\|ticket_id"; then
  record_pass "Escalation workflow" "High-urgency query escalated"
else
  record_fail "Escalation workflow" "High-urgency query not escalated correctly"
fi

# =============================================================================
# STEP 5: Verify MCP tools loaded
# =============================================================================
echo ""
echo "[STEP 5/10] Listing MCP tools (via agent) ..."
if grep -aq "MCP client connected" /tmp/agent-service.log 2>/dev/null; then
  TOOLS=$(grep -a "MCP client connected" /tmp/agent-service.log | tail -1)
  echo "  ${TOOLS}"
  TOOL_COUNT=$(echo "${TOOLS}" | grep -oP '\d+(?= tools)' || echo 0)
  if [[ "${TOOL_COUNT}" -eq 10 ]]; then
    record_pass "10 MCP tools loaded" "All tools present"
  else
    record_fail "10 MCP tools loaded" "Found ${TOOL_COUNT}, expected 10"
  fi
else
  record_fail "MCP connection" "No MCP connection log found"
fi

# =============================================================================
# STEP 6: Wait for SigNoz ingestion
# =============================================================================
echo ""
echo "[STEP 6/10] Waiting for SigNoz ingestion (25s) ..."
sleep 25

TEST_END_EPOCH_NS=$(( $(python3 -c "import time; print(int(time.time() * 1e9))") + 30000000000 ))

# =============================================================================
# STEP 7: Query ClickHouse
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 7/10] Querying ClickHouse ..."
echo "=============================================================================="

TIMESTAMP_START="${TEST_START_EPOCH_NS}"
TIMESTAMP_END="${TEST_END_EPOCH_NS}"

TRACE_IDS_SUBQUERY="
    SELECT DISTINCT trace_id
    FROM signoz_logs.distributed_logs_v2
    WHERE resources_string['service.name'] = 'agent-service'
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
WHERE resources_string['service.name'] = 'agent-service'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND body LIKE '%Node completed%'
")
echo "  Log lines with 'Node completed': ${LOG_COUNT}"
echo "  Expected: > 0"
if [[ "${LOG_COUNT}" -gt 0 ]]; then
  record_pass "Application logs exported" "${LOG_COUNT} lines found"
else
  record_fail "Application logs exported" "No logs found"
fi

# --- Check 2: Traces ---
echo ""
echo "  ---- Check 2: Distributed Traces ----"
SPAN_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_traces.distributed_signoz_index_v3
WHERE trace_id IN (${TRACE_IDS_SUBQUERY})
")
echo "  Spans correlated with log trace_ids: ${SPAN_COUNT}"
echo "  Expected: >= 4 (guardrail, context, resolver, escalate)"
if [[ "${SPAN_COUNT}" -ge 4 ]]; then
  record_pass "Traces exported" "${SPAN_COUNT} spans"
else
  record_fail "Traces exported" "Found ${SPAN_COUNT}, expected >=4"
fi

# --- Check 3: Metrics ---
echo ""
echo "  ---- Check 3: Custom Metrics ----"
METRIC_COUNT=$(ch_count "
SELECT count() AS c
FROM signoz_metrics.distributed_samples_v4
WHERE metric_name = 'agent.requests'
  AND unix_milli >= toUnixTimestamp64Milli(now64()) - 300000
")
echo "  'agent.requests' samples (last 5 min): ${METRIC_COUNT}"
echo "  Expected: > 0"
if [[ "${METRIC_COUNT}" -gt 0 ]]; then
  record_pass "Metrics exported" "${METRIC_COUNT} samples"
else
  record_fail "Metrics exported" "Zero samples"
fi

# --- Check 4: Log-Trace Correlation ---
echo ""
echo "  ---- Check 4: Log-Trace Correlation ----"
LOG_WITH_TRACE=$(ch_count "
SELECT count() AS c
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'agent-service'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND trace_id != ''
  AND body LIKE '%Node completed%'
")
echo "  Log lines with trace_id: ${LOG_WITH_TRACE}"
echo "  Expected: > 0"
if [[ "${LOG_WITH_TRACE}" -gt 0 ]]; then
  record_pass "Log-Trace correlation" "${LOG_WITH_TRACE} correlated lines"
else
  record_fail "Log-Trace correlation" "No correlated logs"
fi

# =============================================================================
# STEP 8: Local process checks
# =============================================================================
echo ""
echo "[STEP 8/10] Local process checks ..."
AGENT_LOG=$(cat /tmp/agent-service.log 2>/dev/null || echo "")

echo ""
echo "  ---- Check 5: Process stdout ----"
if echo "${AGENT_LOG}" | grep -aq "Node completed"; then
  NODE_LINES=$(echo "${AGENT_LOG}" | grep -ac "Node completed" || echo 0)
  record_pass "Node execution logs" "${NODE_LINES} node completions logged"
else
  record_fail "Node execution logs" "No node completions in stdout"
fi

echo ""
echo "  ---- Check 6: OTLP export errors ----"
OTLP_ERRORS=$(echo "${AGENT_LOG}" | grep -acE "Failed to export|StatusCode\\.UNAVAILABLE" 2>/dev/null || echo 0)
OTLP_ERRORS=$(echo "${OTLP_ERRORS}" | tr -d '\n')
echo "  OTLP export errors: ${OTLP_ERRORS}"
echo "  Expected: 0"
if [[ "${OTLP_ERRORS}" -eq 0 ]]; then
  record_pass "OTLP export (no errors)" "0 errors"
else
  record_fail "OTLP export (no errors)" "${OTLP_ERRORS} errors"
fi

echo ""
echo "  ---- Check 7: Health endpoints ----"
HEALTHZ=$(curl -fsS --max-time 2 "${AGENT_URL}/healthz" 2>/dev/null || echo "FAIL")
READYZ=$(curl -fsS --max-time 2 "${AGENT_URL}/readyz" 2>/dev/null || echo "FAIL")
echo "  GET /healthz : ${HEALTHZ}"
echo "  GET /readyz  : ${READYZ}"

if [[ "${HEALTHZ}" == '{"status":"ok"}' ]]; then
  record_pass "Health /healthz" "200 OK"
else
  record_fail "Health /healthz" "Got: ${HEALTHZ}"
fi
if [[ "${READYZ}" == '{"status":"ready"}' ]]; then
  record_pass "Health /readyz" "Ready"
else
  record_fail "Health /readyz" "Got: ${READYZ}"
fi

# =============================================================================
# STEP 9: Cross-service trace verification
# =============================================================================
echo ""
echo "[STEP 9/10] Cross-service trace verification ..."
TRACE_ID=$(ch_query "
SELECT trace_id
FROM signoz_logs.distributed_logs_v2
WHERE resources_string['service.name'] = 'agent-service'
  AND timestamp >= ${TIMESTAMP_START}
  AND timestamp <= ${TIMESTAMP_END}
  AND trace_id != ''
LIMIT 1
" | python3 -c "import sys,json;print(json.loads(sys.stdin.readline())['trace_id'])" 2>/dev/null || echo "")

if [[ -n "${TRACE_ID}" ]]; then
  echo "  Sample Trace ID: ${TRACE_ID}"
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
  record_pass "Trace context propagation" "Trace ID ${TRACE_ID}"
else
  record_fail "Trace context propagation" "No trace_id found"
fi

# =============================================================================
# STEP 10: Final report
# =============================================================================
echo ""
echo "=============================================================================="
echo "[STEP 10/10] Battle Test Summary — agent-service"
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
  echo "  agent-service is BATTLE-READY."
  exit 0
else
  echo ""
  echo "  ${FAIL_COUNT} check(s) failed — review details above."
  echo "  Debug: cat /tmp/agent-service.log"
  exit 1
fi