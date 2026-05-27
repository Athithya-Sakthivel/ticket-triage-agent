"""create human_overrides table

Revision ID: 001
Create Date: 2026-05-27
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None


def upgrade():
    op.create_table(
        "human_overrides",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ticket_id", sa.UUID(), nullable=False),
        sa.Column(
            "original_classification", sa.JSON(), nullable=False
        ),
        sa.Column(
            "corrected_classification", sa.JSON(), nullable=False
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("overridden_by", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "idx_human_overrides_ticket_id", "human_overrides", ["ticket_id"]
    )
    op.create_index(
        "idx_human_overrides_created_at", "human_overrides", ["created_at"]
    )


def downgrade():
    op.drop_index("idx_human_overrides_created_at")
    op.drop_index("idx_human_overrides_ticket_id")
    op.drop_table("human_overrides")