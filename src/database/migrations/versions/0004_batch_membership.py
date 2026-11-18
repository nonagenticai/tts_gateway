"""Batch membership becomes a RELATION: ``tts_batch_jobs``.

**The live defect.** ``submit_batch`` upserted with
``ON CONFLICT DO UPDATE SET batch_id = excluded.batch_id`` — a dedupe hit RE-TAGGED
the existing job row onto the newest batch — while ``get_batch`` read
``WHERE batch_id = :b``. Dedupe is global by ``(tenant_id, idempotency_key)``, so
any two runs that narrate an overlapping chunk share one row. When run B submitted
while run A was still polling, A's rows MOVED to B: ``get_batch(A)`` went empty, the
orchestrator's narrate stage spun to its poll budget and raised
``StagePreconditionError`` — a whole audiobook failed for synthesis that had
actually succeeded, visible under someone else's batch.

**Why a table and not a column.** The relation is genuinely many-to-many. Any
column on ``tts_jobs`` can hold exactly one batch and therefore has to CHOOSE
between run A and run B; both choices strand one of them (keeping the first hides
the clip from the re-attempt — the bug the re-tag was introduced to fix; keeping
the last is the bug above). Making membership its own append-only rows lets every
batch observe the same completed work, and lets dedupe stay exactly as strict as it
is: still one job row and one paid synthesis per ``(tenant, idempotency_key)``.

``tts_jobs.batch_id`` is DELIBERATELY kept and DELIBERATELY not dropped: it is
first-write provenance (and 0002/0003's ``ix_tts_jobs_batch_lookup`` still exists),
and this migration must be safe under a RollingUpdate where OLD pods are still
serving ``get_batch`` off that column for the length of the rollout. This migration
only ADDs; the backfill below makes every existing batch answerable through the new
table from the instant it commits, so the new pods see history and the old pods keep
working unchanged.

Backfill is ``SELECT tenant_id, batch_id, job_id FROM tts_jobs`` — one membership
row per existing job, under whatever batch it currently carries. That is the best
truth available: the earlier batches a re-tagged row once belonged to were
overwritten in place and are unrecoverable. It restores the invariant going
forward, it does not invent history.

Revision ID: 0004_batch_membership
Revises: 0003_tenant_id_uuid
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811

from alembic import op

revision = "0004_batch_membership"
down_revision = "0003_tenant_id_uuid"
branch_labels = None
depends_on = None

# Same role constant 0002/0003 use; the GRANT is guarded on its existence so CI
# (where the role does not exist) skips it and prod is a no-op for the owner.
APP_ROLE = "w2v_service"

_GRANT_SQL = f"""
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE tts_batch_jobs TO {APP_ROLE};
  END IF;
END $$
"""  # noqa: S608 -- APP_ROLE is a hardcoded module constant, not request data


def upgrade() -> None:
    op.create_table(
        "tts_batch_jobs",
        sa.Column("tenant_id", PgUUID(as_uuid=False), nullable=False),
        sa.Column("batch_id", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["job_id"], ["tts_jobs.job_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id", "batch_id", "job_id"),
    )
    op.create_index("ix_tts_batch_jobs_job", "tts_batch_jobs", ["job_id"])

    # Backfill every existing job under the batch it currently carries, so a poll of
    # an in-flight batch keeps answering across the rollout. ON CONFLICT DO NOTHING
    # makes the migration re-runnable against a partially-populated table.
    op.execute(
        "INSERT INTO tts_batch_jobs (tenant_id, batch_id, job_id) "
        "SELECT tenant_id, batch_id, job_id FROM tts_jobs "
        "WHERE batch_id IS NOT NULL "
        "ON CONFLICT DO NOTHING"
    )

    op.execute(_GRANT_SQL)


def downgrade() -> None:
    # ``tts_jobs.batch_id`` was never dropped, so reverting is a plain drop: the
    # column-scoped read path is still intact underneath.
    op.drop_index("ix_tts_batch_jobs_job", table_name="tts_batch_jobs")
    op.drop_table("tts_batch_jobs")
