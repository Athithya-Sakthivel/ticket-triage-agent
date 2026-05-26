"""
Centralised configuration for the Agent Service.
All settings are read from environment variables — no .env file dependency.

Usage:
    from config import settings, create_safeguard_lm, create_resolver_lm

    guard_lm = create_safeguard_lm()
    resolve_lm = create_resolver_lm()
"""

from __future__ import annotations

import os

import dspy
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables only."""

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ────────────────────────────────────────────────────────
    llm_api_key: str = ""
    llm_safeguard_model: str = "groq/openai/gpt-oss-safeguard-20b"
    llm_resolver_model: str = "groq/gpt-oss-120b"

    llm_safeguard_temperature: float = 0.0
    llm_safeguard_max_tokens: int = 1024

    llm_resolver_temperature: float = 0.2
    llm_resolver_max_tokens: int = 4096

    # ── MCP Server ─────────────────────────────────────────────────
    mcp_server_url: str = "http://mcp-server-svc.inference.svc.cluster.local:8001/mcp"

    # ── PostgreSQL ─────────────────────────────────────────────────
    database_url: str = (
        "postgresql://app:password@postgres-pooler.default.svc.cluster.local:5432/agents_state"
    )
    pool_min_size: int = 5
    pool_max_size: int = 25

    # ── OpenTelemetry ──────────────────────────────────────────────
    otel_service_name: str = "agent-service"
    otel_exporter_otlp_endpoint: str = (
        "http://signoz-otel-collector.signoz.svc.cluster.local:4317"
    )
    otel_exporter_otlp_insecure: bool = True
    otel_metric_export_interval_ms: int = 60_000
    otel_metric_export_timeout_ms: int = 30_000
    deployment_environment: str = "production"
    service_version: str = "0.1.0"

    # ── Server ─────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


settings = Settings()


# ── DSPy LM Factory Functions ──────────────────────────────────────

def create_safeguard_lm() -> dspy.LM:
    """LM for combined guardrail + classification (GPT-OSS-Safeguard 20B).

    Uses temperature 0 for deterministic safety + structured output.
    The safeguard model is a 21B MoE (3.6B active) built for policy-driven
    content classification — ideal for pre-flight safety checks and
    ticket triage in one call.[reference:0]
    """
    return dspy.LM(
        model=settings.llm_safeguard_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_safeguard_temperature,
        max_tokens=settings.llm_safeguard_max_tokens,
    )


def create_resolver_lm() -> dspy.LM:
    """LM for the agentic resolver node (GPT-OSS 120B).

    Uses low temperature for coherent, policy-compliant responses
    while retaining enough variability for natural conversation.
    """
    return dspy.LM(
        model=settings.llm_resolver_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_resolver_temperature,
        max_tokens=settings.llm_resolver_max_tokens,
    )