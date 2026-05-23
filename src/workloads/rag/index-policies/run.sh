pip install qdrant-client httpx --break-system-packages

kubectl port-forward -n inference svc/dense-svc 8200:8200 & \
kubectl port-forward -n qdrant svc/qdrant 6333:6333 & \
sleep 2

python3 src/workloads/rag/index-policies/index.py src/workloads/rag/policies --recreate


# Port-forward Qdrant
kubectl port-forward -n inference svc/qdrant 6333:6333 &
sleep 3


# Port-forward again (the old one might still be running, kill it first)
kill %3 2>/dev/null
kubectl port-forward -n inference svc/retriever-minimal-svc 8001:8001 &
sleep 3
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I ordered an iPhone with COD and it arrived damaged. How do I get a cash refund?",
    "top_k": 5
  }'
