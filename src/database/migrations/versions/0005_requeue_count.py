"""``tts_jobs.requeue_count`` — the monotonic bound on re-opening a ``failed`` job.

**The live defect.** A terminally-``failed`` row permanently blacklists its content.
Submit dedupes on ``(tenant_id, idempotency_key)`` with ``ON CONFLICT DO NOTHING``,
so a ``failed`` row wins the insert, the read-back publishes its stored
``error_message`` verbatim, nothing is enqueued and no worker is ever called. The
only sibling reconciliation that existed — ``_REQUEUE_MISSING_SQL`` — hard-codes
``status = 'done'``, so ``failed`` was never examined. Measured in production: four
orchestrator runs across 23 minutes all resolved to ONE ``job_id`` and replayed one
stored 503. Census at the time: ``done 537, failed 1`` — and that 1 was unrecoverable
without a manual UPDATE (there is no requeue route, sweeper or script).

**Why a new column and not ``attempts``.** The sibling reconciliation added with
this migration (``job_manager._REOPEN_FAILED_SQL``) resets ``attempts`` to 0, because
a re-open is a NEW synthesis that needs a full claim budget — exactly as
``_REQUEUE_MISSING_SQL`` already does for a vanished clip. That reset is precisely
what makes ``attempts`` unable to bound the re-open itself. Without a second,
MONOTONIC counter the fix trades a permanent blacklist for an infinite treadmill: an
orchestrator retrying a book every 20 minutes would re-burn ``max_attempts`` GPU
claims forever on text kokoro will never accept.

``requeue_count`` is never reset by any path. Poison therefore converges: after
``QUEUE_MAX_REQUEUES`` re-opens the row stays ``failed`` and its stored error is
replayed — today's behaviour — but only after a bounded number of genuine retries.

ADD-only and backfilled to 0 by the server default, so it is safe under a
RollingUpdate: old pods neither read nor write it, and every pre-existing row starts
with its full re-open budget (which is the intent — they have never been re-opened).

Revision ID: 0005_requeue_count
Revises: 0004_batch_membership
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005_requeue_count"
down_revision = "0004_batch_membership"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tts_jobs",
        sa.Column("requeue_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("tts_jobs", "requeue_count")
