"""release_tts_claim (attempt refund) + max-lifetime bound on claim_next_tts_job

Fixes the live poison bug: ``claim_next_tts_job`` charges ``attempts += 1`` on
every claim and fails the job at ``max_attempts``, but nothing ever refunded the
attempt when a worker was interrupted. ``tts-gateway-worker`` is a k8s
RollingUpdate deployment (plus remote kokoro workers on Macs that reboot), so
every rollout silently burned one of a HEALTHY chunk's four lives — four
interrupted claims poisoned it permanently.

This migration is ZERO-DOWNTIME: it only ADDS. Nothing live is dropped.

``tts-gateway`` is a 2-replica RollingUpdate and this migration runs in an
initContainer, BEFORE the new pods take traffic — so between "migration applied"
and "last old pod terminated", the OLD image is still serving and still calling
the 3-arg ``claim_next_tts_job``. Dropping that signature to "replace" it would
500 every claim for the length of the rollout: a claim outage caused by a
reliability fix. Postgres overloads on the argument list, so instead:

* ADDs ``claim_next_tts_job(worker, lease, max_attempts, max_lifetime)`` — the
  real function, with a wall-clock bound on a job's total running life measured
  from its FIRST claim.
* REPLACEs the 3-arg signature IN PLACE (no DROP) with a thin SHIM that delegates
  to the 4-arg one with a lifetime of ``0`` — bound disabled, i.e. byte-for-byte
  today's behaviour. Old pods keep working identically; new pods get the bound.
* ADDs ``release_tts_claim(job_id, worker_id)`` — the refund, guarded by
  ``(job_id, worker_id, status='claimed')`` so every race is a safe no-op. Pure
  addition, no compat concern.

⚠️ FOLLOW-UP: the 3-arg shim exists ONLY for rollout compatibility. Once the whole
fleet is on the new image it should be dropped in a follow-up migration — but not
before. ``test_queue.py`` pins BOTH signatures, so a premature "cleanup" of the
shim fails the suite instead of silently breaking a future deploy.

Grants — verified against the LIVE prod DB, not assumed: ``claim_next_tts_job``
and ``tts_jobs`` are both owned by ``w2v_service``, the app connects as
``w2v_service``, and ``proacl`` is NULL (DEFAULT privileges — the "Stage A"
REVOKE/GRANT was never actually applied here; EXECUTE works via the PUBLIC
default). So DROP+CREATE as the owner cannot lose the app its EXECUTE. The
``pg_roles``-guarded GRANT below is belt-and-braces: a no-op in prod (owner
already has EXECUTE) and skipped in CI (no such role). We deliberately do NOT
``REVOKE ... FROM PUBLIC`` — prod relies on that default today, and revoking it
would be a live behaviour change that could lock the service out.

Revision ID: 0002_release_claim_and_lifetime
Revises: 0001_initial
Create Date: 2026-07-13
"""

from __future__ import annotations

from alembic import op

revision = "0002_release_claim_and_lifetime"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

# The role the service actually connects as (== the function/table owner).
APP_ROLE = "w2v_service"

# FROZEN SNAPSHOT of the DDL this revision creates. Deliberately INLINED rather than
# imported from src.database.queue_resolvers, even though this file is currently the head.
#
# A migration is a historical snapshot: importing live code makes it replay whatever that
# module says TODAY. 0001 had exactly that bug (it imported the claim fn, so changing the
# module silently changed what 0001 creates on a fresh DB). Keeping the import here would
# just re-arm the same trap for the NEXT migration — 0003 is expected to drop the 3-arg
# shim, which would retroactively make THIS revision stop creating it.
#
# The single-sourcing that actually matters is preserved by test, not by import:
# tests/conftest.py builds the live CLAIM_FN_DDL via register_claim_fn(create_all), and
# test_resolver_set_matches_the_migrated_schema pins the resulting function signatures to
# the exact set this migration creates — so if the two paths ever drift, the suite goes red
# instead of prod.
_CLAIM_FN_4ARG_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int,
    p_max_lifetime_seconds int
) RETURNS TABLE(job_id varchar, tenant_id varchar, voice varchar,
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

_SHIM_FN_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id varchar, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM claim_next_tts_job(p_worker_id, p_lease_seconds, p_max_attempts, 0);
  $fn$
"""

_RELEASE_FN_SQL = """
CREATE OR REPLACE FUNCTION release_tts_claim(
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
          attempts = GREATEST(COALESCE(attempts, 1) - 1, 0)
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
    GRANT EXECUTE ON FUNCTION claim_next_tts_job(text, int, int, int) TO {APP_ROLE};
    GRANT EXECUTE ON FUNCTION claim_next_tts_job(text, int, int) TO {APP_ROLE};
    GRANT EXECUTE ON FUNCTION release_tts_claim(text, text) TO {APP_ROLE};
  END IF;
END $$
"""  # noqa: S608 -- APP_ROLE is a hardcoded constant, not request data

# The pre-0002 3-arg claim fn, inlined as a literal so downgrade does not depend
# on the current source of queue_resolvers.py.
_OLD_CLAIM_FN_SQL = """
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

_OLD_GRANT_SQL = f"""
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
    GRANT EXECUTE ON FUNCTION claim_next_tts_job(text, int, int) TO {APP_ROLE};
  END IF;
END $$
"""  # noqa: S608 -- APP_ROLE is a hardcoded constant, not request data


def upgrade() -> None:
    # Purely ADDITIVE, in this order: the real 4-arg fn, then the 3-arg rollout shim that
    # delegates to it, then the release/refund fn. Nothing live is dropped.
    op.execute(_CLAIM_FN_4ARG_SQL)
    op.execute(_SHIM_FN_SQL)
    op.execute(_RELEASE_FN_SQL)
    op.execute(_GRANT_SQL)


def downgrade() -> None:
    # Order matters: put the ORIGINAL 3-arg body back FIRST (CREATE OR REPLACE
    # overwrites the shim in place), so the 3-arg signature is never delegating to a
    # 4-arg function that is about to disappear. Only then remove the additions.
    op.execute(_OLD_CLAIM_FN_SQL)
    op.execute(_OLD_GRANT_SQL)
    op.execute("DROP FUNCTION IF EXISTS claim_next_tts_job(text, int, int, int)")
    op.execute("DROP FUNCTION IF EXISTS release_tts_claim(text, text)")
