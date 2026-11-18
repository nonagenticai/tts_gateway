"""GPU-busy release + absolute busy backstop (tts_gateway#45).

**The defect.** ``release_tts_claim`` refunds the attempt but keeps ``started_at``,
and ``claim_next_tts_job``'s lifetime arm fails a job once ``started_at`` is older
than ``max_job_lifetime_seconds`` (3600 s). A worker that releases on a sustained
GPU-busy signal (503/429) therefore hit a terminal "exceeded max job lifetime"
after ~4 busy cycles — a busy signal failing the job, which #45 forbids.

**The fix, ADDITIVE and rollout-safe:**

* ``tts_jobs.busy_since`` (nullable) — first busy release.
* ``release_tts_claim_busy(job, worker)`` — the refund PLUS ``started_at = NULL``
  (busy time does not count toward lifetime) and ``busy_since = COALESCE(...)``.
  The SIGTERM ``release_tts_claim`` is unchanged: a handoff keeps its clock.
* ``claim_next_tts_job(..., p_max_busy_age_seconds)`` — a 5-arg overload with an
  absolute backstop: past ``busy_since + max_busy_age`` the job fails with the
  stable reason ``gpu_busy_exhausted`` (re-drivable: a re-open clears busy_since).
* the 4-arg signature becomes a SQL shim delegating with the backstop disabled (0),
  and the 3-arg shim delegates straight to the 5-arg — old pods mid-rollout keep
  their exact semantics.

Drift-tolerant: ``ADD COLUMN IF NOT EXISTS``, ``CREATE OR REPLACE``, ``DROP ... IF
EXISTS``. The SQL is a FROZEN snapshot (a migration must never import live code);
``tests/test_queue.py`` pins the create_all set so drift between this file and
``queue_resolvers.py`` fails loudly.

Revision ID: 0006_busy_release_backstop
Revises: 0005_requeue_count
"""

from __future__ import annotations

from alembic import op

revision = "0006_busy_release_backstop"
down_revision = "0005_requeue_count"
branch_labels = None
depends_on = None

APP_ROLE = "w2v_service"

_CLAIM_FN_5ARG_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int,
    p_max_lifetime_seconds int, p_max_busy_age_seconds int
) RETURNS TABLE(job_id varchar, tenant_id uuid, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_attempts int; v_started timestamptz; v_busy_since timestamptz;
  BEGIN
    SELECT j.id, j.attempts, j.started_at, j.busy_since
      INTO v_id, v_attempts, v_started, v_busy_since
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

    -- ABSOLUTE busy backstop (tts_gateway#45). A GPU-busy release resets started_at
    -- (see release_tts_claim_busy) so time spent waiting on a saturated GPU never
    -- counts toward the lifetime arm below — but that removes the lifetime bound for
    -- a job that is busy FOREVER. This is the bound that replaces it, measured from
    -- the FIRST busy release (busy_since). The message starts with the stable token
    -- `gpu_busy_exhausted` so a caller can recognise it and re-drive (a re-open
    -- clears busy_since).
    IF p_max_busy_age_seconds > 0 AND v_busy_since IS NOT NULL
       AND v_busy_since < now() - make_interval(secs => p_max_busy_age_seconds) THEN
      UPDATE tts_jobs
        SET status = 'failed',
            completed_at = now(),
            claimed_by = NULL,
            lease_expires_at = NULL,
            error_message = 'gpu_busy_exhausted: GPU busy for over '
                            || p_max_busy_age_seconds || 's since the first busy release'
      WHERE id = v_id;
      RETURN;
    END IF;

    -- Wall-clock ceiling on a job's total RUNNING life. The attempt refund
    -- (release_tts_claim) deliberately removes the bound that `attempts` used to
    -- provide, so under sustained churn a job could otherwise be re-claimed
    -- forever and replay its synthesis every time. This is the bound that
    -- replaces it.
    --
    -- Measured from started_at (FIRST claim), never created_at, and `v_started IS
    -- NOT NULL` is load-bearing: a never-claimed row sitting in a backlog has
    -- consumed nothing, and ageing it out unstarted would mass-fail a queue that
    -- is merely behind. A busy release clears started_at, so busy time is exempt.
    IF p_max_lifetime_seconds > 0 AND v_started IS NOT NULL
       AND v_started < now() - make_interval(secs => p_max_lifetime_seconds) THEN
      UPDATE tts_jobs
        SET status = 'failed',
            completed_at = now(),
            lease_expires_at = NULL,
            error_message = 'exceeded max job lifetime (' || p_max_lifetime_seconds
                            || 's) — never completed across '
                            || COALESCE(v_attempts, 0) || ' attempt(s)'
      WHERE id = v_id;
      RETURN;
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

_SHIM_4ARG_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int,
    p_max_lifetime_seconds int
) RETURNS TABLE(job_id varchar, tenant_id uuid, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM claim_next_tts_job(
        p_worker_id, p_lease_seconds, p_max_attempts, p_max_lifetime_seconds, 0);
  $fn$
"""

_SHIM_3ARG_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id uuid, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM claim_next_tts_job(p_worker_id, p_lease_seconds, p_max_attempts, 0, 0);
  $fn$
"""

_RELEASE_BUSY_FN_SQL = """
CREATE OR REPLACE FUNCTION release_tts_claim_busy(
    p_job_id text, p_worker_id text
) RETURNS int
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_n int;
  BEGIN
    UPDATE tts_jobs
      SET status = 'queued',
          claimed_by = NULL,
          lease_expires_at = NULL,
          attempts = GREATEST(COALESCE(attempts, 1) - 1, 0),
          started_at = NULL,
          busy_since = COALESCE(busy_since, now())
    WHERE job_id = p_job_id
      AND status = 'claimed'
      AND claimed_by = p_worker_id;

    GET DIAGNOSTICS v_n = ROW_COUNT;
    RETURN v_n;
  END
  $fn$
"""

_GRANT_SQL = f"""
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
    GRANT EXECUTE ON FUNCTION claim_next_tts_job(text, int, int, int, int) TO {APP_ROLE};
    GRANT EXECUTE ON FUNCTION release_tts_claim_busy(text, text) TO {APP_ROLE};
  END IF;
END $$
"""  # noqa: S608 -- APP_ROLE is a hardcoded constant, not request data

# ── what downgrade restores: 0003's bodies, byte-for-byte ────────────────────────
_OLD_CLAIM_FN_4ARG_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int,
    p_max_lifetime_seconds int
) RETURNS TABLE(job_id varchar, tenant_id uuid, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_attempts int; v_started timestamptz;
  BEGIN
    SELECT j.id, j.attempts, j.started_at INTO v_id, v_attempts, v_started
    FROM tts_jobs j
    WHERE j.status = 'queued'
       OR (j.status = 'claimed'
           AND (j.lease_expires_at IS NULL OR j.lease_expires_at < now()))
    ORDER BY
      (SELECT count(*) FROM tts_jobs c
         WHERE c.tenant_id = j.tenant_id
           AND c.status = 'claimed'
           AND c.lease_expires_at > now()),
      j.priority DESC,
      j.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;
    END IF;

    IF p_max_lifetime_seconds > 0 AND v_started IS NOT NULL
       AND v_started < now() - make_interval(secs => p_max_lifetime_seconds) THEN
      UPDATE tts_jobs
        SET status = 'failed',
            completed_at = now(),
            lease_expires_at = NULL,
            error_message = 'exceeded max job lifetime (' || p_max_lifetime_seconds
                            || 's) — never completed across '
                            || COALESCE(v_attempts, 0) || ' attempt(s)'
      WHERE id = v_id;
      RETURN;
    END IF;

    IF COALESCE(v_attempts, 0) >= p_max_attempts THEN
      UPDATE tts_jobs
        SET status = 'failed',
            error_message = 'exceeded max claim attempts (' || p_max_attempts
                            || ') — failed as poison job',
            completed_at = now()
      WHERE id = v_id;
      RETURN;
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

_OLD_SHIM_3ARG_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id uuid, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM claim_next_tts_job(p_worker_id, p_lease_seconds, p_max_attempts, 0);
  $fn$
"""


def upgrade() -> None:
    op.execute("ALTER TABLE tts_jobs ADD COLUMN IF NOT EXISTS busy_since timestamptz")
    # Order: the real 5-arg fn first, then the shims that delegate to it.
    op.execute(_CLAIM_FN_5ARG_SQL)
    op.execute(_SHIM_4ARG_SQL)
    op.execute(_SHIM_3ARG_SQL)
    op.execute(_RELEASE_BUSY_FN_SQL)
    op.execute(_GRANT_SQL)


def downgrade() -> None:
    # Put the real 4-arg body back FIRST, then repoint the 3-arg shim at it, so no
    # surviving signature ever delegates to the 5-arg fn about to be dropped.
    op.execute(_OLD_CLAIM_FN_4ARG_SQL)
    op.execute(_OLD_SHIM_3ARG_SQL)
    op.execute("DROP FUNCTION IF EXISTS claim_next_tts_job(text, int, int, int, int)")
    op.execute("DROP FUNCTION IF EXISTS release_tts_claim_busy(text, text)")
    op.execute("ALTER TABLE tts_jobs DROP COLUMN IF EXISTS busy_since")
