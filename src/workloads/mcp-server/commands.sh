


cd src/workloads/mcp-context-server
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt -q


kubectl port-forward -n inference svc/retriever-minimal-svc 8001:8001 &
sleep 2
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "return policy for damaged phone", "top_k": 3}'


# Terminal 1: PostgreSQL pooler
kubectl port-forward -n default svc/postgres-pooler 5432:5432 &

# Terminal 2: OTel collector (port 4317)
kubectl port-forward -n signoz svc/signoz-otel-collector 4317:4317 &

# Then override env vars and run
export DATABASE_URL="postgresql://app:$(kubectl get secret postgres-cluster-app -o jsonpath='{.data.password}' | base64 -d)@localhost:5432/agents_state"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export RETRIEVER_URL="http://localhost:8001"
export PYTHONWARNINGS="ignore::UserWarning"
python3 src/main.py


# new terminal
cd src/workloads/mcp-context-server
source .venv/bin/activate
curl http://localhost:8001/healthz   # → "ok"
curl http://localhost:8001/readyz    # → "ready" (if DB is up)


# List tools
fastmcp list http://localhost:8001/sse

# Call tools
fastmcp call http://localhost:8001/sse lookup_customer email=priya.sharma@email.com
fastmcp call http://localhost:8001/sse get_recent_orders user_id=a1b2c3d4-e5f6-4a7b-8c9d-000000000001
fastmcp call http://localhost:8001/sse get_order_details order_id=c3d4e5f6-a7b8-4c9d-0e1f-000000000001
fastmcp call http://localhost:8001/sse check_refund_eligibility order_id=c3d4e5f6-a7b8-4c9d-0e1f-000000000004
