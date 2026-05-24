"""
asyncpg connection pool factory and all SQL query functions.

Designed for PgBouncer transaction mode:
- No session-level features (SET, LISTEN, temporary tables).
- Every query runs in its own transaction.

Includes both read queries (context tools) and write queries (ops tools).
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from config import settings


async def create_pool() -> asyncpg.Pool:
    """Create an asyncpg pool tuned for PgBouncer transaction pooling."""
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        command_timeout=settings.pool_command_timeout,
    )


# ═══════════════════════════════════════════════════════════════════
# READ QUERIES — Context Tools
# ═══════════════════════════════════════════════════════════════════

async def get_user_by_email(pool: asyncpg.Pool, email: str) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT id, full_name, email, phone, language_pref, segment, created_at "
                "FROM users WHERE email = $1",
                email,
            )
    return dict(row) if row else None


async def get_user_by_phone(pool: asyncpg.Pool, phone: str) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT id, full_name, email, phone, language_pref, segment, created_at "
                "FROM users WHERE phone = $1",
                phone,
            )
    return dict(row) if row else None


async def get_recent_orders(
    pool: asyncpg.Pool, user_id: str, limit: int = 5
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT id, user_id, product_id, status, quantity, amount, "
                "discount_amount, shipping_amount, cod_fee, payment_method, "
                "shipping_address, pincode, city, order_date, delivery_date, "
                "promised_delivery_date, is_delayed, delivery_attempts, "
                "tracking_number, notes "
                "FROM orders WHERE user_id = $1 "
                "ORDER BY order_date DESC LIMIT $2",
                user_id,
                limit,
            )
    return [dict(r) for r in rows]


async def get_order_with_product(
    pool: asyncpg.Pool, order_id: str
) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT o.id, o.user_id, o.product_id, o.status, o.quantity, "
                "o.amount, o.discount_amount, o.shipping_amount, o.cod_fee, "
                "o.payment_method, o.shipping_address, o.pincode, o.city, "
                "o.order_date, o.delivery_date, o.promised_delivery_date, "
                "o.is_delayed, o.delivery_attempts, o.tracking_number, o.notes, "
                "p.name AS product_name, p.category, p.subcategory, p.price, "
                "p.return_window_days, p.warranty_months, p.is_returnable, "
                "p.is_express_eligible "
                "FROM orders o JOIN products p ON o.product_id = p.id "
                "WHERE o.id = $1",
                order_id,
            )
    return dict(row) if row else None


async def get_billing_by_order(
    pool: asyncpg.Pool, order_id: str
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT id, order_id, user_id, transaction_type, amount, status, "
                "refund_eligible, refund_reason, payment_gateway, "
                "gateway_transaction_id, transaction_date, completed_date "
                "FROM billing WHERE order_id = $1 "
                "ORDER BY transaction_date DESC",
                order_id,
            )
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════
# WRITE QUERIES — Ops Tools
# ═══════════════════════════════════════════════════════════════════
async def insert_ticket(
    pool: asyncpg.Pool,
    user_id: str,
    query_text: str,
    classification: dict,
    priority: str,
) -> str:
    """Create a new ticket."""

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO tickets (
                    id,
                    user_id,
                    query_text,
                    classification,
                    resolution_type,
                    status,
                    priority,
                    source,
                    created_at,
                    updated_at
                )
                VALUES (
                    gen_random_uuid(),
                    $1,
                    $2,
                    $3::jsonb,
                    'escalated',
                    'pending_human',
                    $4,
                    'chat',
                    NOW(),
                    NOW()
                )
                RETURNING id
                """,
                user_id,
                query_text,
                json.dumps(classification),
                priority,
            )

    return str(row["id"])

async def update_ticket_status(
    pool: asyncpg.Pool, ticket_id: str, status: str
) -> None:
    """Update ticket status."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE tickets SET status = $1, updated_at = NOW() WHERE id = $2",
                status,
                ticket_id,
            )


async def set_ticket_priority(
    pool: asyncpg.Pool, ticket_id: str, priority: str
) -> None:
    """Update ticket priority."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE tickets SET priority = $1, updated_at = NOW() WHERE id = $2",
                priority,
                ticket_id,
            )


async def process_refund(
    pool: asyncpg.Pool, order_id: str, amount: float
) -> str:
    """Record a refund in the billing table. Returns the refund transaction ID."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """INSERT INTO billing
                   (id, order_id, user_id, transaction_type, amount, status,
                    refund_eligible, refund_reason, payment_gateway,
                    gateway_transaction_id, transaction_date, completed_date)
                   SELECT gen_random_uuid(), id, user_id, 'refund', $1, 'completed',
                          false, 'auto_refund', 'kestral_wallet',
                          'R-' || substr(gen_random_uuid()::text, 1, 8), NOW(), NOW()
                   FROM orders WHERE id = $2
                   RETURNING gateway_transaction_id""",
                amount,
                order_id,
            )
    return str(row["gateway_transaction_id"])