pip install qdrant-client httpx --break-system-packages

kubectl port-forward -n inference svc/dense-svc 8200:8200 & \
kubectl port-forward -n qdrant svc/qdrant 6333:6333 & \
sleep 2

python3 src/workloads/rag/index-policies/index.py src/workloads/rag/policies --recreate