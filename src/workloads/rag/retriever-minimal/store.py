"""Minimal Qdrant vector store — dense search only, no cache, no sparse."""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient


class QdrantStore:
    def __init__(self, url: str, api_key: str | None, collection_name: str, dense_dim: int):
        self.url = url
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.client = AsyncQdrantClient(url=url, api_key=api_key or None)

    async def ping(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def search(self, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Dense vector search only. Returns [{id, score, text}]."""
        resp = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: list[dict[str, Any]] = []
        for point in resp.points if hasattr(resp, 'points') else []:
            payload = point.payload or {}
            text = payload.get("text") or payload.get("content") or ""
            results.append({
                "id": point.id,
                "score": float(point.score or 0.0),
                "text": text,
            })

        return results

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass