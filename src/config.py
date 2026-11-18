"""tts_gateway settings.

Port 6320 (VERIFY free in-cluster + on the Mac hosts before deploy). Rejects
API_PORT=8000 via the w2v_common base. DB is the shared bare-metal Postgres;
storage is the gateway's own MinIO bucket for synthesized clips.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import StorageSettings as BaseStorageSettings


class StorageSettings(BaseStorageSettings):
    minio_bucket: str = Field(default="tts-gateway", description="Clip bucket")
    minio_external_endpoint: str | None = Field(default=None)


class QueueSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUEUE_")

    # Lease TTL; a worker heartbeats within this or its job is reclaimed. Kept
    # comfortably above a single kokoro synth (~20-40s for a full chunk).
    lease_seconds: int = Field(default=180)
    # Claim attempts before a chunk is failed as poison (a chunk that crashes
    # every worker must not starve the queue forever). Stays at 4 even though a
    # graceful release now REFUNDS the attempt: the refund only fires on an
    # orderly shutdown, so a genuine crash-loop (OOMKill/segfault) never reaches
    # it and 4 remains the correct fast crash guard.
    max_attempts: int = Field(default=4)
    # How many times a TERMINALLY-``failed`` job may be re-opened by a later submit
    # (``job_manager.reopen_failed_jobs``). Counted by the MONOTONIC
    # ``tts_jobs.requeue_count``, which nothing resets — unlike ``attempts``, which
    # the re-open itself resets to 0 and which therefore cannot carry this bound.
    #
    # 3 is a deliberate ceiling on wasted GPU: the true worst case is
    # ``max_requeues * max_attempts`` = 12 claims on one poison chunk, after which
    # the row stays ``failed`` for good and its stored error is replayed (exactly
    # today's behaviour). Anything larger turns a transient upstream 503 — the case
    # this exists for — into an unbounded re-synthesis treadmill.
    max_requeues: int = Field(default=3)
    # Wall-clock ceiling on a job's total RUNNING life, measured from its FIRST
    # claim (never from creation — a backlogged, never-claimed row has consumed
    # nothing and must not age out). This is the backstop that replaces the bound
    # the attempt-refund removes: without it, a job caught by endless rollout
    # churn could be re-claimed and re-synthesized forever. 0 disables it.
    max_job_lifetime_seconds: int = Field(default=3600)
    # Absolute backstop for GPU-busy churn (tts_gateway#45). A busy release clears
    # ``started_at`` so busy time never trips the lifetime bound above; this bounds
    # it instead, measured from the job's FIRST busy release. Past it the job fails
    # with the stable reason ``gpu_busy_exhausted`` (re-drivable). 0 disables it.
    max_busy_age_seconds: int = Field(default=86400)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    api_port: int = Field(default=6320, description="Service port (never 8000)")
    database_url: str = Field(..., description="Owner DB URL (alembic/initContainer)")
    runtime_database_url: str | None = Field(
        default=None,
        description="Restricted app-role URL (RLS enforce); falls back to database_url",
    )

    # Cap on a worker's clip upload (POST /api/tts/jobs/{id}/result). The route reads
    # at most this many bytes +1 and 413s on the extra one, so an oversized clip is
    # never written to storage and never held whole as a `bytes` object in the handler.
    #
    # It is ALSO the source of the transport-layer ceiling: src/asgi.py wires
    # `BodySizeLimitMiddleware` (src/api/body_limit.py) with this value plus
    # MULTIPART_FRAMING_HEADROOM_BYTES as the whole-REQUEST limit. That middleware is
    # what actually prevents resource exhaustion — it rejects with 413 before
    # Starlette's multipart parser runs, and that parser spools each file part into a
    # SpooledTemporaryFile (max_size 1 MiB) with NO size bound, having consumed the
    # whole body before the endpoint is entered. Without the middleware a multi-GB
    # POST still landed, on DISK, as node disk-pressure/eviction. Raising this value
    # therefore raises BOTH the per-file cap and the request ceiling.
    #
    # 100 MiB is ~35 minutes of 24 kHz 16-bit mono WAV — orders of magnitude above a
    # narration chunk, so it only ever catches abuse, never real audio. Tunable via
    # MAX_RESULT_BYTES.
    max_result_bytes: int = Field(
        default=100 * 1024 * 1024, gt=0, description="Max accepted clip upload size in bytes"
    )

    # Identity allowlist for the WORKER-facing surface (src/api/routes/workers.py:
    # claim / heartbeat / result / release / fail).
    #
    # The license middleware that guards /api only checks that the token is ACTIVE and
    # carries a non-empty tenant_id — it enforces no role, scope or service-account
    # identity. That is sufficient for the tenant-scoped submit/poll surface, and wholly
    # insufficient here: POST /api/tts/claim pops from the GLOBAL queue through a
    # SECURITY DEFINER function that binds no tenant, so any tenant holding any valid
    # license token could read another tenant's full narration text plus its job_id —
    # and that job_id then unlocks result/heartbeat/release/fail against the victim's
    # job. Tenant-scoping the by-id routes cannot fix claim: claim is cross-tenant by
    # design. Only an IDENTITY check can — the caller must be the worker service
    # account, not merely some authenticated tenant.
    #
    # Matched against the introspected subject_id (w2v_common's identity ContextVar),
    # so it is the license's SUBJECT, not its tenant. Kept a plain CSV string rather
    # than list[str] for the same reason as AUTH_DELEGATING_SUBJECTS: pydantic-settings
    # 2.14's complex-type env decode is broken. Split at read time below.
    #
    # FAIL-CLOSED: the default is EMPTY and empty means DENY EVERYONE, not "allow all".
    # An unset allowlist is a misconfiguration, and a misconfiguration must not silently
    # restore the very cross-tenant hole this setting exists to close. Operators must
    # set WORKER_SUBJECT_IDS to the worker service account's subject id(s) — plural so a
    # credential rotation can list old+new and roll without a queue outage.
    worker_subject_ids: str = Field(
        default="",
        description=(
            "Comma-separated subject IDs of the service account(s) allowed to call the "
            "worker surface (/api/tts/claim, /jobs/{id}/heartbeat|result|release|fail), "
            "env WORKER_SUBJECT_IDS. Matched against the introspected license subject_id. "
            "EMPTY (the default) DENIES EVERY caller with 403 — it never means 'allow "
            "all' — because these routes read and mutate jobs across ALL tenants and an "
            "unconfigured gateway must not fall open. List multiple ids to roll a "
            "credential without a queue outage."
        ),
    )

    storage: StorageSettings = Field(default_factory=StorageSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)

    def worker_subject_id_set(self) -> frozenset[str]:
        """Parse ``worker_subject_ids``; empty frozenset means deny-all (see the field)."""
        return frozenset(s.strip() for s in self.worker_subject_ids.split(",") if s.strip())

    def effective_db_url(self) -> str:
        return self.runtime_database_url or self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
