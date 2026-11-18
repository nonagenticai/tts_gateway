"""``tts_jobs.tenant_id`` becomes a real ``uuid``, and the claim fns are rebuilt with it.

Fleet ruling: ``tenant_id`` is a real ``uuid``, never text, never a sentinel.

Measured live 2026-08-27 against the prod ``tts_gateway`` database:

* Exactly ONE tenant-scoped column exists — ``tts_jobs.tenant_id character varying(255)``.
* 536 rows, **0 dirty** — every stored value already parses as a uuid. Unlike the
  sibling services there is no ``n8n-dev`` name to backfill here, so this migration
  carries NO ``UPDATE``. The fail-loud guard is kept anyway: "clean today" is a
  measurement, not an invariant, and this migration may first run on a database that
  has taken writes since. If someone wrote a tenant NAME in the meantime, the guard
  must stop the migration, not let the ``USING tenant_id::uuid`` blow up mid-ALTER
  with a message that names no table and no offender.
* ``pg_policies`` returns ZERO rows for this database, and ``git grep POLICY`` over
  ``src/`` finds none: this service has never created RLS policies (the claim path is
  SECURITY DEFINER instead — see ``queue_resolvers``). So there is deliberately no
  policy drop/recreate step. Adding a speculative one would CREATE isolation that
  neither prod nor ``create_all`` has, which is a behaviour change smuggled into a
  type migration.

**The trap this migration exists to defuse.** ``ALTER COLUMN ... TYPE`` does NOT
revalidate PL/pgSQL function bodies. ``claim_next_tts_job`` exists in TWO overloads
(the 4-arg real one and the 3-arg rollout shim 0002 added), and BOTH declare
``RETURNS TABLE(..., tenant_id varchar, ...)``. After the flip the body returns
``uuid`` in that position and every call dies at RUNTIME with

    structure of query does not match function result type

which silently kills the entire job-claim loop — a green migration, a Ready pod, and
a queue that never drains. ``CREATE OR REPLACE FUNCTION`` cannot change a return
type ("cannot change return type of existing function"), so the only correct order is:

1. ``DROP`` BOTH overloads by their full argument signature. Dropping by name alone
   is ambiguous and errors; dropping only one leaves the other broken.
2. ``ALTER`` the column.
3. ``CREATE`` both overloads again with ``tenant_id uuid`` in the RETURNS TABLE.
4. Re-apply the grants. DROP takes the ACL with it, so the ``pg_roles``-guarded
   GRANT from 0002 is re-run. As in 0002 we do NOT ``REVOKE ... FROM PUBLIC``:
   ``proacl`` is NULL in prod (DEFAULT privileges) and the app relies on the PUBLIC
   default today — revoking here would be a live behaviour change that could lock
   the service out of its own claim path.

``release_tts_claim`` is untouched on purpose: it returns ``int`` and its body never
projects ``tenant_id``, so the column type is invisible to it.

The function DDL below is a FROZEN LITERAL, copied from 0002, not imported from
``src.database.queue_resolvers``. Same reason 0002 spells it out: a migration is
immutable history, and importing live code makes an old revision replay whatever the
module says today. In particular ``downgrade()`` MUST recreate the era-correct
``tenant_id varchar`` signatures, which the live module (now uuid) can no longer
produce. ``tests/conftest.py`` + ``test_resolver_set_matches_the_migrated_schema``
keep the migration path and the ``create_all`` path pinned to each other.

Revision ID: 0003_tenant_id_uuid
Revises: 0002_release_claim_and_lifetime
"""

from __future__ import annotations

from alembic import op

revision = "0003_tenant_id_uuid"
down_revision = "0002_release_claim_and_lifetime"
branch_labels = None
depends_on = None

# The role the service actually connects as (== the table/function owner). Same
# constant 0002 uses; the GRANT is guarded on its existence so CI (no such role)
# skips it.
APP_ROLE = "w2v_service"

_UUID_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

# Both overloads, by FULL argument signature. Dropping by bare name raises
# "function name is not unique"; dropping only one leaves the other returning a
# varchar it can no longer produce.
_DROP_CLAIM_FNS = (
    "DROP FUNCTION IF EXISTS claim_next_tts_job(text, int, int, int)",
    "DROP FUNCTION IF EXISTS claim_next_tts_job(text, int, int)",
)

# FAIL LOUD. Not a `WHERE tenant_id ~ ...` filter on the ALTER: a filter would
# quietly leave unconvertible rows behind or drop them, and silent data loss on a
# tenant key is far worse than a refused migration.
_GUARD_SQL = f"""
DO $$
DECLARE
  bad_count bigint;
  sample text;
BEGIN
  SELECT count(*), min(tenant_id) INTO bad_count, sample
    FROM tts_jobs
   WHERE tenant_id IS NOT NULL AND tenant_id !~ '{_UUID_RE}';
  IF bad_count > 0 THEN
    RAISE EXCEPTION
      'tts_jobs.tenant_id: % row(s) are not a parseable uuid (e.g. %). Refusing to '
      'ALTER to uuid. Map each value to its tenant uuid in licenses.tenants and add '
      'an explicit backfill above, then re-run. Do NOT skip them with a WHERE.',
      bad_count, sample;
  END IF;
END $$
"""  # noqa: S608 -- the only interpolation is _UUID_RE, a module constant

# ── uuid-era function DDL (what this revision installs) ──────────────────────────
# Byte-identical to 0002's bodies except the declared `tenant_id` return type.
_CLAIM_FN_4ARG_UUID_SQL = """
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

_SHIM_FN_UUID_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id uuid, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM claim_next_tts_job(p_worker_id, p_lease_seconds, p_max_attempts, 0);
  $fn$
"""

# ── varchar-era function DDL (what downgrade must restore) ───────────────────────
# Spelled out in full as a LITERAL — deliberately NOT derived from the uuid constants
# above by string surgery, and never rendered from queue_resolvers.py (that module now
# declares uuid, so rendering it here would recreate exactly the broken function this
# migration exists to prevent). This is a byte-for-byte copy of what 0002 installed.
_CLAIM_FN_4ARG_VARCHAR_SQL = """
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

_SHIM_FN_VARCHAR_SQL = """
CREATE OR REPLACE FUNCTION claim_next_tts_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int
) RETURNS TABLE(job_id varchar, tenant_id varchar, voice varchar,
                speed numeric, provider varchar, txt text)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM claim_next_tts_job(p_worker_id, p_lease_seconds, p_max_attempts, 0);
  $fn$
"""

# DROP removes the ACL with the function, so the grants must be re-applied after
# every recreate. Role-guarded: a no-op in prod (owner already has EXECUTE via the
# NULL/default proacl) and skipped entirely in CI, where the role does not exist.
_GRANT_SQL = f"""
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
    GRANT EXECUTE ON FUNCTION claim_next_tts_job(text, int, int, int) TO {APP_ROLE};
    GRANT EXECUTE ON FUNCTION claim_next_tts_job(text, int, int) TO {APP_ROLE};
  END IF;
END $$
"""  # noqa: S608 -- APP_ROLE is a hardcoded module constant, not request data


def upgrade() -> None:
    # 1. Refuse loudly rather than half-apply if a non-uuid tenant value appeared
    #    after the 2026-08-27 measurement.
    op.execute(_GUARD_SQL)

    # 2. Both overloads must go BEFORE the ALTER: their RETURNS TABLE pins
    #    `tenant_id varchar`, CREATE OR REPLACE cannot change a return type, and the
    #    ALTER does not revalidate the bodies — so leaving them would produce a
    #    "successful" migration whose claim loop 500s on every call.
    for stmt in _DROP_CLAIM_FNS:
        op.execute(stmt)

    # 3. The type flip. Nullability, the index, and uq_tts_jobs_idem are rebuilt
    #    automatically by ALTER COLUMN TYPE.
    op.execute("ALTER TABLE tts_jobs ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")

    # 4. Recreate both overloads, now declaring `tenant_id uuid`, then restore the
    #    grants the DROP took with it.
    op.execute(_CLAIM_FN_4ARG_UUID_SQL)
    op.execute(_SHIM_FN_UUID_SQL)
    op.execute(_GRANT_SQL)


def downgrade() -> None:
    # Mirror image. The functions must go first for the same reason as above: their
    # uuid return type cannot survive the column becoming varchar.
    for stmt in _DROP_CLAIM_FNS:
        op.execute(stmt)

    op.execute(
        "ALTER TABLE tts_jobs ALTER COLUMN tenant_id TYPE character varying(255) "
        "USING tenant_id::text"
    )

    op.execute(_CLAIM_FN_4ARG_VARCHAR_SQL)
    op.execute(_SHIM_FN_VARCHAR_SQL)
    op.execute(_GRANT_SQL)
