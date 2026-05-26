"""create tickets table

Revision ID: 001
Create Date: 2026-05-25
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None


def upgrade():
    op.create_table(
        "tickets",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("order_id", sa.UUID(), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("classification", sa.JSON(), nullable=True),
        sa.Column("resolution_type", sa.String(50), nullable=True),
        sa.Column("status", sa.String(50), server_default="open"),
        sa.Column("priority", sa.String(50), nullable=True),
        sa.Column("assigned_team", sa.String(100), nullable=True),
        sa.Column("assigned_agent", sa.String(255), nullable=True),
        sa.Column("resolution_summary", sa.Text(), nullable=True),
        sa.Column("source", sa.String(50), server_default="chat"),
        sa.Column("language", sa.String(5), server_default="en"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("resolved_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("idx_tickets_user_id", "tickets", ["user_id"])
    op.create_index("idx_tickets_order_id", "tickets", ["order_id"])
    op.create_index("idx_tickets_status", "tickets", ["status"])
    op.create_index("idx_tickets_priority", "tickets", ["priority"])
    op.create_index("idx_tickets_assigned_team", "tickets", ["assigned_team"])
    op.create_index("idx_tickets_created_at", "tickets", ["created_at"])


def downgrade():
    op.drop_index("idx_tickets_created_at")
    op.drop_index("idx_tickets_assigned_team")
    op.drop_index("idx_tickets_priority")
    op.drop_index("idx_tickets_status")
    op.drop_index("idx_tickets_order_id")
    op.drop_index("idx_tickets_user_id")
    op.drop_table("tickets")