"""
Centralised configuration from environment variables.
No .env file – strong defaults point to Kubernetes services.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables only."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL (via CloudNative PG PgBouncer pooler) ──────────
    database_url: str = (
        "postgresql://"
    )
    pool_min_size: int = 5
    pool_max_size: int = 25
    pool_command_timeout: float = 10.0

    # ── Retriever service ────────────────────────────────────────
    retriever_url: str = (
        "http://retriever-minimal-svc.inference.svc.cluster.local:8001"
    )

    # ── Dense embedding service ──────────────────────────────────
    dense_url: str = "http://dense-svc.inference.svc.cluster.local:8200"

    # ── Qdrant (direct access for policy search) ─────────────────
    qdrant_url: str = "http://qdrant.qdrant.svc.cluster.local:6333"
    qdrant_collection: str = "kestral_policies"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ── OpenTelemetry ─────────────────────────────────────────────
    otel_service_name: str = "mcp-server"
    otel_exporter_otlp_endpoint: str = "http://signoz-otel-collector.signoz.svc.cluster.local:4317"
    otel_exporter_otlp_insecure: bool = True
    otel_metric_export_interval_ms: int = 60_000
    otel_metric_export_timeout_ms: int = 30_000
    deployment_environment: str = "production"
    service_version: str = "0.1.0"

    # ── Server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"


settings = Settings()