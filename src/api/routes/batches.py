"""Submit + poll — the orchestrator-facing surface (tenant-scoped)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import LICENSE_GATE

from src.api.schemas import (
    BatchStatusResponse,
    JobRef,
    JobStatus,
    SubmitBatchRequest,
    SubmitBatchResponse,
)
from src.config import get_settings
from src.database.session import get_tenant_session
from src.services import job_manager as jm
from src.services.storage import ClipStorageUnavailableError, get_storage

# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.75.0), mounted router-wide.
# Until now this service authenticated the licence token for tenant IDENTITY
# only (AuthMiddleware answers "who are you") and never asked "may you" — a
# tenant whose licence is expired or revoked still carries a syntactically
# valid token and was served. LICENSE_GATE closes that: none/expired/revoked
# -> 402 license_<state>, an unestablishable verdict -> 503 license_unavailable
# + Retry-After, active/expiring_soon -> pass.
#
# This is the ENTIRE tenant-scoped surface of the service (submit + poll), so
# both routes below are correctly gated. The worker surface (src/api/routes/
# workers.py, mounted separately) is DELIBERATELY NOT gated by this change: it
# is an internal service-account callback (claim/heartbeat/result/release/fail)
# authorized by `require_worker_subject` against WORKER_SUBJECT_IDS, not by a
# tenant's own licence — the worker acts across every tenant's queue by design
# (see that module's docstring), so a tenant-scoped licence check does not apply
# to it. The health router (health.py: /health, /readyz, /livez) is likewise
# not gated: a 402/503 on a probe CrashLoops the pod, and an unlicensed tenant
# cannot be billed by killing our own deployment. The gate also refuses these
# paths by path (`is_license_exempt_path`), so this is belt and braces, not a
# single point of failure.
#
# DORMANT ON MERGE: require_license is a no-op unless W2V_LICENSE_GATE_ENABLED
# is true, read per-call from os.environ. It defaults to False, so this change
# alters no behaviour until gitops arms the flag. Preserve that default.
router = APIRouter(prefix="/api/tts", tags=["batches"], dependencies=list(LICENSE_GATE))

# Reported instead of ``done`` when the job HAS an artifact that storage cannot
# presign right now. Chosen to be outside the orchestrator's terminal set
# (``{"done", "failed"}``) so a poll re-tries rather than concluding; see the
# handler in ``get_batch``. It is a REPORTING status only — never written to
# ``tts_jobs.status``.
STORAGE_UNAVAILABLE_STATUS = "storage_unavailable"


def _tenant(request: Request) -> str:
    return str(getattr(request.state, "tenant_id", "") or "")


@router.post(
    "/batches",
    response_model=SubmitBatchResponse,
    status_code=202,
    operation_id="tts.batches.submit",
)
async def submit_batch(
    body: SubmitBatchRequest,
    request: Request,
    session: AsyncSession = Depends(get_tenant_session),
) -> SubmitBatchResponse:
    """Enqueue a book idempotently, confirming every dedupe HIT still has its clip.

    Submit deduplicates on ``sha256(tenant|voice|speed|preproc|text)`` and that key
    IS the object name, so a ``done`` row can outlive its audio (lifecycle expiry, a
    bucket sweep, a half-failed upload). Returning ``done`` for such a row is a dead
    reference: the poll presigns a URL for nothing and the downstream compile 404s,
    failing the whole run. So every hit that comes back ``done`` is confirmed
    against storage first, and a definitively-absent clip is re-queued rather than
    published. See ``jm.requeue_jobs_with_missing_clips`` for cost + race handling.
    """
    tenant_id = _tenant(request)
    jobs = await jm.submit_batch(
        session,
        tenant_id=tenant_id,
        batch_id=body.batch_id,
        chunks=[c.model_dump() for c in body.chunks],
    )
    # BEFORE the storage reconciliation and before anything is published: a dedupe
    # hit that came back ``failed`` is otherwise permanent. ``ON CONFLICT DO NOTHING``
    # lets the dead row win the insert, so submit replays its stored error, enqueues
    # nothing and never calls a worker — that (tenant, voice, speed, text) can never
    # be synthesized again without a manual UPDATE. Bounded by the monotonic
    # ``requeue_count`` so a genuinely poison chunk still converges to failed.
    jobs = await jm.reopen_failed_jobs(
        session,
        tenant_id=tenant_id,
        jobs=jobs,
        max_requeues=get_settings().queue.max_requeues,
    )
    jobs = await jm.requeue_jobs_with_missing_clips(
        session, tenant_id=tenant_id, jobs=jobs, clip_exists=get_storage().exists
    )
    return SubmitBatchResponse(batch_id=body.batch_id, jobs=[JobRef(**j) for j in jobs])


@router.get(
    "/batches/{batch_id}", response_model=BatchStatusResponse, operation_id="tts.batches.get"
)
async def get_batch(
    batch_id: str,
    request: Request,
    session: AsyncSession = Depends(get_tenant_session),
) -> BatchStatusResponse:
    rows = await jm.get_batch(session, tenant_id=_tenant(request), batch_id=batch_id)
    storage = get_storage()
    jobs: list[JobStatus] = []
    for r in rows:
        # Presign at READ time so the URL is always fresh when narrate assembles.
        url: str | None = None
        status = r["status"]
        error = r["error"]
        if status == "done" and r["result_key"]:
            try:
                url = await storage.presign(r["result_key"])
            except ClipStorageUnavailableError as exc:
                # ⚠️ A batch whose artifact cannot be presigned is NOT done.
                #
                # The job really did synthesize and its bytes really are in the
                # bucket — what failed is our ability to hand out a URL right now.
                # Reporting ``done`` anyway published a terminal status with
                # ``result_url=None``: the orchestrator stops polling on ``done`` and
                # the run dies (or worse, a consumer treats the empty URL as
                # "no audio needed"). ``storage_unavailable`` is deliberately NEITHER
                # ``done`` NOR ``failed``, which is precisely the set the narrate
                # stage treats as terminal — so the poll simply RE-TRIES and the
                # batch completes by itself the moment MinIO answers again, within
                # the same poll budget. Nothing in the queue is mutated: the row
                # stays ``done`` in the database, because the job is done.
                status = STORAGE_UNAVAILABLE_STATUS
                error = str(exc)
        jobs.append(
            JobStatus(
                job_id=r["job_id"],
                idempotency_key=r["idempotency_key"],
                status=status,
                result_url=url,
                duration_seconds=r["duration_seconds"],
                # Derived from the DB status, NOT the reporting `status` above: a job
                # reported ``storage_unavailable`` is still ``done`` in the queue, and
                # its measurement gap is just as real.
                duration_unmeasured=r["duration_unmeasured"],
                error=error,
            )
        )
    return BatchStatusResponse(batch_id=batch_id, jobs=jobs)
