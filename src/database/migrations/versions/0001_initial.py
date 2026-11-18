"""initial tts_jobs claim-queue + claim_next_tts_job resolver

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

# FROZEN SNAPSHOT — the claim fn EXACTLY as it existed at this revision. Do not edit,
# and do NOT re-import it from src.database.queue_resolvers.
#
# A migration is a HISTORICAL SNAPSHOT. This file used to `from src.database.queue_resolvers
# import CLAIM_FN_SQL`, which means it replayed whatever that module says TODAY rather than
# what this revision actually created: once 0002 changed CLAIM_FN_SQL to the 4-arg form, a
# fresh DB running 0001 would have created the 4-arg fn (plus the shim and release_tts_claim)
# BEFORE 0002 ran at all. That happened to not explode only because every column it touches
# already exists in 0001 — luck, not design. The moment a later revision's function body
# references a column added in a later revision, a fresh DB fails on 0001 while every
# existing DB stays green (the sibling ai_content_generation_service hit exactly this).
#
# The live function lives in src.database.queue_resolvers and is owned by the LATEST
# migration; tests build it via register_claim_fn (create_all), which is correct — the
# suite should always exercise the current function.
_CLAIM_FN_SQL_AT_0001 = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id varchar, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_attempts int;
  BEGIN
    SELECT j.id, j.attempts INTO v_id, v_attempts
    FROM tts_jobs j
    WHERE j.status = 'queued'
       OR (j.status = 'claimed'
           AND (j.lease_expires_at IS NULL OR j.lease_expires_at < now()))
    ORDER BY
      -- fairness: fewest in-flight leases for this job's tenant first
      (SELECT count(*) FROM tts_jobs c
         WHERE c.tenant_id = j.tenant_id
           AND c.status = 'claimed'
           AND c.lease_expires_at > now()),
      j.priority DESC,
      j.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;  -- empty queue
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE tts_jobs
        SET status = 'failed',
            error_message = 'exceeded max claim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now()
      WHERE id = v_id;
      RETURN;  -- poison job skipped
    END IF;

    RETURN QUERY
    UPDATE tts_jobs
      SET status = 'claimed',
          claimed_by = p_worker_id,
          started_at = COALESCE(started_at, now()),
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0) + 1
    WHERE id = v_id
    RETURNING tts_jobs.job_id, tts_jobs.tenant_id, tts_jobs.voice,
              tts_jobs.speed, tts_jobs.provider, tts_jobs.text;
  END
  $fn$
"""


def upgrade() -> None:
    op.create_table(
        "tts_jobs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("batch_id", sa.String(64), nullable=False),
        sa.Column("voice", sa.String(64), nullable=True),
        sa.Column("speed", sa.Numeric(4, 3), server_default="1.0", nullable=False),
        sa.Column("provider", sa.String(32), server_default="kokoro", nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), server_default="queued", nullable=False),
        sa.Column("claimed_by", sa.String(64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("priority", sa.Integer(), server_default="0", nullable=False),
        sa.Column("result_key", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Numeric(9, 3), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_tts_jobs_job_id", "tts_jobs", ["job_id"], unique=True)
    op.create_index("ix_tts_jobs_tenant_id", "tts_jobs", ["tenant_id"])
    op.create_index("ix_tts_jobs_batch_id", "tts_jobs", ["batch_id"])
    op.create_index("ix_tts_jobs_status", "tts_jobs", ["status"])
    op.create_index("uq_tts_jobs_idem", "tts_jobs", ["tenant_id", "idempotency_key"], unique=True)
    op.create_index(
        "ix_tts_jobs_claimable",
        "tts_jobs",
        ["priority", "created_at"],
        postgresql_where=sa.text("status IN ('queued','claimed')"),
    )
    op.create_index("ix_tts_jobs_batch_lookup", "tts_jobs", ["batch_id", "status"])
    op.execute(_CLAIM_FN_SQL_AT_0001)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS claim_next_tts_job(text, int, int)")
    op.drop_table("tts_jobs")
