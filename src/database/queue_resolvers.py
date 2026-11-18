"""SECURITY DEFINER queue resolvers for the cross-tenant TTS lease queue.

Ported from ``html_render_service.database.rls_resolvers.claim_next_render_job``.
Remote kokoro workers claim the next runnable job across ALL tenants via the
gateway API; under RLS enforce the gateway's runtime role binds no
``app.current_tenant``, so a plain claim SELECT would fail-closed to 0 rows. The
claim therefore runs through a SECURITY DEFINER function (owned by the table
owner → bypasses ENABLE), which atomically: picks one claimable row
(``FOR UPDATE SKIP LOCKED``), fails jobs past their wall-clock lifetime, fails
poison jobs past ``max_attempts``, else moves it to ``claimed`` with a fresh
lease + incremented attempt, and returns the synthesis inputs.

Fairness: unlike html_render's pure FIFO, the ORDER BY prefers the tenant
currently holding the FEWEST live leases, so N books drain concurrently and a
short lesson job is never stuck behind a whole 6-hour audiobook.

The functions below MUST be created together (``CLAIM_FN_DDL``; 0006 added the 5-arg
claim with the busy backstop and ``release_tts_claim_busy``):

* ``claim_next_tts_job(worker, lease, max_attempts, max_lifetime)`` — the claim,
  with a wall-clock lifetime bound.
* ``claim_next_tts_job(worker, lease, max_attempts)`` — a 3-arg SHIM kept only so
  that an old gateway pod still mid-rollout keeps claiming (see CLAIM_FN_DDL).
* ``release_tts_claim`` — the REFUND. ``claim_next_tts_job`` charges
  ``attempts += 1`` on *every* claim and poisons the job at ``max_attempts``.
  That counter is a CRASH counter. A graceful handoff (SIGTERM from a k8s
  RollingUpdate, a Mac worker reboot) is not a crash: the claim is surrendered
  without a verdict, so the job never got its fair try and must not be charged.
  Without the refund every rollout silently burns one of a HEALTHY chunk's four
  lives, and four interrupted claims poison it permanently — the live prod bug.
  A genuine crash (OOMKill/SIGKILL) never reaches the release path, so the lease
  just expires unrefunded and real poison detection is preserved exactly.
  ``release_tts_claim`` is SECURITY DEFINER for the same reason the claim is:
  the worker session binds no tenant, so under RLS enforce a plain UPDATE would
  fail closed.

⚠️ ``tenant_id`` in both ``claim_next_tts_job`` RETURNS TABLE clauses is ``uuid``,
matching the column since migration ``0003_tenant_id_uuid``. It must track the column
type exactly: ``ALTER COLUMN ... TYPE`` does not revalidate a function body, so a
mismatch is not a DDL error but a RUNTIME one — every claim dies with "structure of
query does not match function result type" and the queue silently stops draining.
``CREATE OR REPLACE`` cannot change a return type either, so changing it always means
DROP-then-CREATE of BOTH overloads, in a migration.

Privileges — verified against the LIVE prod DB, not assumed: both ``tts_jobs``
and ``claim_next_tts_job`` are owned by ``w2v_service``, the app connects as
``w2v_service``, and ``proacl`` is NULL — i.e. DEFAULT privileges. The
"Stage A" REVOKE-from-PUBLIC / GRANT-to-a-restricted-role that an earlier
version of this docstring described was **never actually applied** to this
database; EXECUTE works today via the PUBLIC default. So the migration that
drops+recreates these functions re-GRANTs (belt-and-braces, role-guarded) but
deliberately does NOT revoke from PUBLIC — revoking would be a live behaviour
change that could lock the service out. Single-sourced so the migration (prod)
and the ``create_all`` ``after_create`` event (tests) build identical functions.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

CLAIM_FN_SQL = """
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

# The pre-#45 4-arg signature, kept as a SQL shim for old pods mid-rollout: it
# delegates with the busy backstop DISABLED (0), which is exactly its old semantics.
SHIM_4ARG_FN_SQL = """
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

RELEASE_FN_SQL = """
CREATE OR REPLACE FUNCTION release_tts_claim(
    p_job_id text, p_worker_id text
) RETURNS int
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_n int;
  BEGIN
    -- Surrender the claim AND refund the attempt it consumed. GREATEST(...,0)
    -- keeps the counter non-negative even if a job is somehow released twice.
    --
    -- The (job_id, worker_id, status='claimed') predicate is what makes every
    -- race safe: a release that loses to a completion, to a lease-expiry
    -- reclaim by another worker, or to a duplicate release simply matches 0 rows
    -- and refunds nothing. Do not weaken it.
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

# A GPU-BUSY release (tts_gateway#45): the same refund as release_tts_claim, plus
#   * started_at = NULL — busy time must not count toward max_job_lifetime_seconds,
#     otherwise ~4 busy cycles end in a terminal "exceeded max job lifetime";
#   * busy_since = COALESCE(busy_since, now()) — arms the absolute busy backstop in
#     claim_next_tts_job, measured from the FIRST busy release.
# Separate from release_tts_claim on purpose: a SIGTERM handoff is not a busy signal
# and must keep its lifetime clock.
RELEASE_BUSY_FN_SQL = """
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

SHIM_FN_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id uuid, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM claim_next_tts_job(p_worker_id, p_lease_seconds, p_max_attempts, 0, 0);
  $fn$
"""

# Ordered DDL — all three statements are ADDITIVE. Nothing is ever dropped, which is
# what makes the deploy safe (see below).
#
# ⚠️ ROLLOUT SAFETY. tts-gateway is a 2-replica RollingUpdate and the migration runs in
# an initContainer, BEFORE the new pods take traffic. So there is a window where the
# migration has already applied and the OLD image is still serving. If this DDL DROPped
# the 3-arg claim_next_tts_job to "replace" it with the 4-arg one, every claim from an
# old pod would 500 for the length of the rollout — a claim outage caused by a
# reliability fix.
#
# Postgres overloads on the argument list, so the 4-arg function is a pure ADDITION and
# the 3-arg signature survives as a thin SHIM that delegates with a lifetime of 0
# (= bound disabled = exactly today's semantics). Old pods pass 3 args and behave
# identically; new pods pass 4 and get the bound. Neither image is ever broken.
#
# The shim exists ONLY for rollout compatibility. Once the whole fleet is on the new
# image, a follow-up migration should drop it — but not before, and `test_queue.py`
# pins both signatures so a premature "cleanup" fails loudly instead of breaking a
# deploy months from now.
CLAIM_FN_DDL: tuple[str, ...] = (
    CLAIM_FN_SQL,
    SHIM_4ARG_FN_SQL,
    SHIM_FN_SQL,
    RELEASE_FN_SQL,
    RELEASE_BUSY_FN_SQL,
)


def register_claim_fn(metadata: MetaData) -> None:
    """Create the queue resolvers after the tables exist (test create_all path)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for stmt in CLAIM_FN_DDL:
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)
