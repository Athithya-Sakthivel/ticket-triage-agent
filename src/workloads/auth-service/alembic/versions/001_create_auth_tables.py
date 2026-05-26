"""create auth_users and auth_admins tables

Revision ID: 001
Create Date: 2026-05-25
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None


def upgrade():
    op.create_table(
        "auth_users",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("oidc_sub", sa.String(255), unique=True, nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("avatar_url", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("last_login", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "auth_admins",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("oidc_sub", sa.String(255), unique=True, nullable=False),
        sa.Column("display_name", sa.String(255), nullable=True),
        sa.Column("domain", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), server_default="admin"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("last_login", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
    )
    # Indexes are already created by the UNIQUE constraints on email and oidc_sub.


def downgrade():
    op.drop_table("auth_admins")
    op.drop_table("auth_users")