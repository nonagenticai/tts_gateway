"""Worker-facing surface — remote kokoro workers claim/heartbeat/report here.

Cross-tenant: claim runs through the SECURITY DEFINER fn (binds no tenant), so it
hands back another tenant's narration text and job_id by design, and that job_id
then addresses every other route in this module. The license middleware guarding
``/api`` is therefore NOT sufficient authorization here: it only proves the token is
active and carries some tenant, which every customer token also does. Under that
check alone any licensed tenant could drain the queue, read other tenants' text, and
inject audio into their jobs.

What is ACTUALLY enforced: ``require_worker_subject`` is attached to the router
itself (not to individual routes, so a route added later cannot be born unguarded)
and requires the introspected license SUBJECT to be listed in ``WORKER_SUBJECT_IDS``.
An unset allowlist denies everyone.

P7 (w2v_common 0.92.0): ``authorize_worker`` — the router-level guard — admits a
worker by EITHER of two identities:

* a verified ``scope=service`` cell token whose issuer is one of
  ``WORKER_CELL_CALLERS`` (each worker Deployment has its own Ed25519 key), or
* the legacy licence lane: ``require_worker_subject`` against ``WORKER_SUBJECT_IDS``.

The cell lane only counts when the middleware's SERVICE lane authenticated the
request (``enforce``, or ``dual`` with no licence). In ``dual`` with a licence the
token is merely recorded and the licence check decides exactly as before. Any other
cell caller (the orchestrator, a tenant-scope token) gets nothing from this route:
the service lane refuses non-opted issuers and the legacy check still applies.

This is an identity check — "are you the worker
service account" — not a tenant scope; a tenant scope cannot express ``claim``, which
is inherently global.

The worker still never touches Postgres or MinIO directly — the gateway owns both.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth.cell_jwt import CellClaims
from w2v_common.auth.cell_service import require_cell_service_caller
from w2v_common.auth.identity_context import (
    get_subject_id_from_context,
    get_tenant_source_from_context,
)
from w2v_common.tenant import TenantSource

from src.api.schemas import MAX_DURATION_SECONDS, ClaimedJob, ClaimRequest
from src.config import get_settings
from src.database.session import get_raw_session
from src.services import job_manager as jm
from src.services.storage import get_storage

logger = logging.getLogger(__name__)


def require_worker_subject() -> None:
    """Allow only the allowlisted service account onto this router.

    ``get_settings`` is ``lru_cache``d, so changing ``WORKER_SUBJECT_IDS`` needs a
    POD RESTART. Do not read this as hot-reloadable during an incident.

    403, never 404: this is an identity failure, and there is no enumeration oracle
    to hide — the caller already knows the route exists.

    Empty allowlist ⇒ deny. The alternative (empty means "unconfigured, allow all")
    would make every misconfigured or newly-bootstrapped deploy fall open on exactly
    the cross-tenant surface this guard protects, and would do so silently.

    SCOPE, stated honestly: today both worker deployments authenticate with the
    fleet-wide ``DOWNSTREAM_LICENSE_TOKEN``, so its subject id is shared by every
    w2v service. Allowlisting it closes the cross-TENANT hole (any customer's
    license used to be accepted here) but not cross-SERVICE. Narrowing that
    further needs a dedicated ``svc-tts-worker`` license, not a code change.
    """
    allowed = get_settings().worker_subject_id_set()
    if not allowed:
        logger.error(
            "worker surface refused: WORKER_SUBJECT_IDS is empty, so no caller can be "
            "recognised as the worker service account. Set it to the worker license's "
            "subject id — an empty allowlist denies, it does not allow"
        )
        raise HTTPException(403, "worker surface is not configured")

    subject = get_subject_id_from_context().strip()
    # Refuse an ABSENT identity explicitly rather than relying on the allowlist
    # parser having filtered blanks out. A licence with no subject introspects to
    # "" (w2v_common auth middleware), and if a future edit ever let "" into the
    # allowlist, `"" in allowed` would admit every subjectless caller. Keeping the
    # invariant here makes it local and unmissable instead of action-at-a-distance.
    if not subject:
        logger.warning("worker surface refused: request carries no authenticated subject")
        raise HTTPException(403, "caller is not an authorized TTS worker")

    if subject not in allowed:
        # Subject ids are opaque UUIDs, not credentials — safe to log, and the only
        # way an operator can tell a rotation miss from an actual intrusion attempt.
        logger.warning(
            "worker surface refused: subject %s is not in WORKER_SUBJECT_IDS",
            subject or "<none>",
        )
        raise HTTPException(403, "caller is not an authorized TTS worker")


#: Cell identities (k8s Service names) of the two worker Deployments. Each signs with
#: its OWN key (gitops: tts-gateway-worker[-mini1]-cell-secret), and the gateway's
#: CELL_ALLOWED_CALLERS must list them too — the library applies both.
WORKER_CELL_CALLERS: frozenset[str] = frozenset({"tts-gateway-worker", "tts-gateway-worker-mini1"})

# allow_other_lanes=True: a request that did NOT arrive on the service lane returns
# None instead of 403, so the legacy licence check below still runs (dual migration).
# Being a sub-dependency of the router guard is also what makes the middleware see
# the opt-in: it walks the matched route's dependency tree for it.
_worker_service_caller = require_cell_service_caller(WORKER_CELL_CALLERS, allow_other_lanes=True)


def authorize_worker(
    service_caller: CellClaims | None = Depends(_worker_service_caller),
) -> None:
    """Router guard: a verified worker cell identity, else the legacy subject allowlist."""
    if service_caller is not None:
        return
    # The cell TENANT lane binds its subject from the caller's forwarded
    # ``X-Subject-Id`` header — run attribution, not an authenticated identity.
    # Letting the allowlist read it would admit any allowed cell caller (the
    # orchestrator) to the cross-tenant queue just by naming the worker's subject,
    # with no licence at all (enforce, or dual without one). In dual WITH a licence
    # the licence lane authenticated and set its own provenance, so this never fires.
    if get_tenant_source_from_context() is TenantSource.CELL_JWT:
        logger.warning("worker surface refused: a tenant-scope cell token is not a worker")
        raise HTTPException(403, "caller is not an authorized TTS worker")
    require_worker_subject()


# The dependency hangs off the ROUTER, so it covers every route below and every route
# a future change adds — there is no per-route opt-in to forget.
router = APIRouter(prefix="/api/tts", tags=["workers"], dependencies=[Depends(authorize_worker)])


@router.post("/claim", operation_id="tts.claim")
async def claim(body: ClaimRequest, session: AsyncSession = Depends(get_raw_session)) -> Response:
    q = get_settings().queue
    job = await jm.claim_next(
        session,
        worker_id=body.worker_id,
        lease_seconds=q.lease_seconds,
        max_attempts=q.max_attempts,
        max_lifetime_seconds=q.max_job_lifetime_seconds,
        max_busy_age_seconds=q.max_busy_age_seconds,
    )
    if job is None:
        return Response(status_code=204)  # empty queue
    payload = ClaimedJob(
        job_id=job["job_id"],
        voice=job["voice"],
        speed=job["speed"],
        provider=job["provider"],
        text=job["text"],
        # ``claim_next`` returns the row's tenant_id (a uuid.UUID from the resolver).
        # Serialised as a plain string so the worker can stamp it on the synth hop
        # without importing the gateway's models. See ClaimedJob.tenant_id.
        tenant_id=str(job["tenant_id"]) if job.get("tenant_id") else None,
        lease_seconds=q.lease_seconds,
    )
    return Response(content=payload.model_dump_json(), media_type="application/json")


@router.post("/jobs/{job_id}/heartbeat", status_code=204, operation_id="tts.heartbeat")
async def heartbeat(
    job_id: str,
    worker_id: str | None = Form(None),
    session: AsyncSession = Depends(get_raw_session),
) -> None:
    # worker_id → owner-guarded renewal; a beat for a claim this worker no longer
    # holds is a 409, which the worker reads as "lease lost".
    ok = await jm.renew_lease(
        session,
        job_id=job_id,
        lease_seconds=get_settings().queue.lease_seconds,
        worker_id=worker_id,
    )
    if not ok:
        raise HTTPException(409, "job not in claimed state")


@router.post("/jobs/{job_id}/result", status_code=204, operation_id="tts.result")
async def report_result(
    job_id: str,
    audio: UploadFile = File(...),
    duration_seconds: float | None = Form(None, le=MAX_DURATION_SECONDS, allow_inf_nan=False),
    worker_id: str | None = Form(None),
    session: AsyncSession = Depends(get_raw_session),
) -> None:
    """Store the clip and finish the job.

    **Size.** Two gates protect this endpoint. ``BodySizeLimitMiddleware`` refuses an
    oversized REQUEST at the ASGI layer — ahead of the multipart parser, so the body
    is never spooled to a temp file and resource exhaustion is genuinely prevented,
    not merely converted into node disk pressure. The ``read(max + 1)`` below is the
    per-FILE bound behind it, keeping an oversized clip out of the object store.

    **Duration bound.** ``duration_seconds`` is persisted as ``Numeric(9, 3)``, whose
    largest representable value is ``999999.999``. Before this bound existed, a worker
    reporting anything above it -- or ``inf``/``nan``, which ``float()`` accepts from a
    form field -- reached the UPDATE in ``job_manager`` and raised a DB ``DataError``.
    That surfaced as a 500, the worker's ``raise_for_status()`` fired, the worker POSTed
    ``/fail``, and a perfectly good job burned a retry attempt until it died. ``le`` is
    anchored on the exact column ceiling rather than ``lt=1e6`` because Postgres rounds
    to scale BEFORE checking precision, so ``999999.9995`` would round to ``1000000.000``
    and overflow; ``allow_inf_nan=False`` kills the non-finite cases outright instead of
    relying on NaN comparison semantics, which never reject.

    The bound is deliberately ONE-SIDED: there is no lower bound. A non-positive duration
    is logged below but still accepted -- see that block for why.

    ``worker_id`` is OPTIONAL only for rollout compatibility with the manually-deployed
    Mac kokoro workers (see job_manager). When present, the completion is guarded on
    ``claimed_by`` so a worker whose lease already expired cannot finish the claim a
    different worker now holds.

    **A missing measurement is not a duration of zero.** ``duration_seconds`` is
    SELF-REPORTED and is republished by ``GET /api/tts/batches/{batch_id}``, from
    which the orchestrator builds its narration timeline. ``tts_backend`` emits
    ``duration_sec = frames / sr if sr else 0.0``, so real, good audio legitimately
    arrives with NO length — a normal production event, not an attack.

    It used to arrive as the number ``0.0`` and be stored as such, which made the
    gap unexpressible: the timeline builder allocated zero seconds to a chunk full
    of speech, every later chunk drifted against the picture, and the batch and the
    run both reported success. Nothing downstream could tell that fabricated zero
    from a measured one.

    So the field is OPTIONAL, and missing / null / non-positive are all normalised
    to ``None`` → persisted as SQL NULL (``tts_jobs.duration_seconds`` is already
    nullable) → surfaced to the consumer as ``duration_unmeasured: true`` on a job
    that is otherwise plainly ``done``. NULL is a value the timeline builder cannot
    silently add up, and the flag says which of the two zeroes this is.

    The job is still COMPLETED — the audio is real and losing it is worse than the
    drift. Rejecting (e.g. a ``gt=0`` bound → 422) is the trap: ``worker.py`` calls
    ``raise_for_status()``, converts the 422 into a POST ``/fail``, and the queue
    requeues a deterministically-failing job until ``max_attempts`` burns out and it
    lands in ``status='failed'`` — and the orchestrator's audiobook narrate stage
    raises ``StagePreconditionError`` on ANY failed job, aborting the whole book for
    one chunk that synthesized fine.

    The upper bound stays one-sided and unchanged: it exists only to keep an
    unstorable value out of ``Numeric(9, 3)``.

    Validating the value for real would mean decoding the clip, which is
    deliberately out of scope here.
    """
    info = await jm.get_job_keyinfo(session, job_id=job_id)
    if info is None:
        raise HTTPException(404, "job not found")

    # ── Ownership is checked BEFORE anything is written ──────────────────────
    # The clip key below is `{tenant_id}/{idempotency_key}.wav` — DETERMINISTIC, so
    # every worker that ever holds this job addresses the SAME object. `complete()`
    # is guarded on `claimed_by` and correctly refuses a worker whose lease expired
    # and whose job another worker reclaimed — but that guard used to run AFTER
    # `storage.put`, so the stale worker was refused the DB row and granted the
    # bytes. It overwrote the live owner's clip in MinIO while the owner's
    # `duration_seconds` (already committed, or about to be) described the audio it
    # had replaced. The two deployed workers do not even share a synth backend
    # (in-cluster modality-gateway vs. off-cluster kokoro on mini1), so the
    # durations genuinely diverge and the orchestrator's narration timeline drifts
    # against the picture — for a job nothing ever reported as failed.
    #
    # Checking here rather than staging-and-promoting: promotion needs a
    # copy/rename primitive `ClipStorage` does not have, would move every clip's
    # bytes twice, and — worse — opens a window in which `complete()` has committed
    # `result_key` while the canonical object is still absent, so a fast batch poll
    # presigns a URL to nothing. A pre-write owner check adds no such window and
    # closes the case that actually happens in production: a worker that lost its
    # claim MINUTES ago (lease expiry + reclaim) and returns with audio. The
    # residual — losing the lease in the microseconds between this SELECT and the
    # put — is unavoidable at any check point and is still caught by `complete()`,
    # which keeps the DB truthful.
    #
    # `worker_id is None` is the manually-deployed Mac launchd workers (see
    # job_manager): no identity to compare, so behaviour is exactly as before.
    # 204, not 409: 409-ing a worker that has already lost its job only buys a
    # retry storm — same reasoning as the `complete()` miss below.
    if worker_id is not None and info["claimed_by"] != worker_id:
        logger.warning(
            "job %s: worker %r reported a result but the claim is held by %r — "
            "discarding the upload; a stale writer must not overwrite the owner's clip",
            job_id,
            worker_id,
            info["claimed_by"],
        )
        return

    # Bound the read to one byte past the limit. This is now the SECOND of two
    # gates, not the only one: `BodySizeLimitMiddleware` (src/api/body_limit.py,
    # wired in src/asgi.py from this same `max_result_bytes` plus multipart framing
    # headroom) refuses an oversized request at the ASGI layer, before Starlette's
    # multipart parser spools the file part into its unbounded
    # SpooledTemporaryFile — so an abusive upload no longer reaches disk at all.
    # This check still earns its place: it is the per-FILE bound (the middleware
    # bounds the whole request), and it holds even when the router is mounted
    # without the middleware (e.g. the unit tests below).
    max_bytes = get_settings().max_result_bytes
    data = await audio.read(max_bytes + 1)
    if not data:
        raise HTTPException(400, "empty audio payload")
    if len(data) > max_bytes:
        raise HTTPException(413, "audio payload exceeds size limit")

    # Normalise the measurement gap to NULL. A clip in this queue is never genuinely
    # zero seconds long (the queue only holds non-empty text), so a non-positive value
    # is the synth backend's "no sample rate" sentinel, not a length. Storing NULL is
    # what lets `get_batch` publish `duration_unmeasured` instead of a number the
    # consumer would trust.
    measured_duration = duration_seconds if (duration_seconds or 0.0) > 0 else None
    if measured_duration is None:
        logger.warning(
            "job %s: worker reported no usable duration (duration_seconds=%r; likely a "
            "synth response with duration_sec absent or 0.0) — keeping the clip and "
            "completing the job, but persisting the duration as NULL so the batch poll "
            "reports duration_unmeasured instead of a fabricated zero-length chunk",
            job_id,
            duration_seconds,
        )

    storage = get_storage()
    key = storage.key_for(info["tenant_id"], info["idempotency_key"])
    await storage.put(key, data)
    ok = await jm.complete(
        session,
        job_id=job_id,
        result_key=key,
        duration_seconds=measured_duration,
        worker_id=worker_id,
    )
    if not ok:
        # A guarded miss here now means the claim was lost in the narrow window between
        # the ownership check above and this UPDATE. Benign no-op, NOT a fault: 409-ing
        # a stale worker only invites a retry storm on a job it has already lost, and
        # the DB row still names the live owner, which is what `get_batch` publishes.
        #
        # ⚠️ It does NOT mean the audio is harmless. The key is
        # `{tenant}/{idempotency_key}.wav` — deterministic, NOT content-addressed — so
        # this worker's bytes have replaced whatever was there. That is precisely why
        # the ownership check above exists and why this window must stay as small as
        # it is; do not relax the pre-write check on the strength of this one.
        #
        # Legacy callers (no worker_id) keep today's 409.
        if worker_id is not None:
            return
        raise HTTPException(409, "job not in claimed state")


@router.post("/jobs/{job_id}/release", status_code=200, operation_id="tts.release")
async def release_claim(
    job_id: str,
    worker_id: str = Form(...),
    reason: str | None = Form(None),
    session: AsyncSession = Depends(get_raw_session),
) -> dict[str, bool]:
    """Surrender a claim at graceful shutdown; the consumed attempt is REFUNDED.

    Called by a worker's SIGTERM handler (k8s RollingUpdate, Mac reboot). Without
    it every rollout burns one of a HEALTHY chunk's ``max_attempts`` lives and
    four interrupted claims poison it permanently.

    Always 200, never an error status: this is a shutdown path, and a worker that
    is already dying must not see something it might retry. A release that no
    longer owns the claim (completed / reclaimed / already released) is a truthful
    ``{"released": false}``, not a fault. If the call never lands at all, the
    server self-heals via lease expiry.
    """
    # ``reason=gpu_busy`` (tts_gateway#45): the synth GPU stayed saturated. Busy time
    # must not count toward the lifetime bound — see job_manager.release_claim.
    released = await jm.release_claim(
        session, job_id=job_id, worker_id=worker_id, busy=reason == "gpu_busy"
    )
    return {"released": released}


@router.post("/jobs/{job_id}/fail", status_code=200, operation_id="tts.fail")
async def report_failure(
    job_id: str,
    error: str = Form(...),
    worker_id: str | None = Form(None),
    terminal: bool = Form(False),
    session: AsyncSession = Depends(get_raw_session),
) -> dict[str, str]:
    """Report a synth failure.

    ``worker_id`` is OPTIONAL only for rollout compatibility with the manually-deployed
    Mac kokoro workers (see job_manager). When present, the failure is guarded on
    ``claimed_by``, so a stale worker's late error cannot requeue — or, at the cap,
    POISON — the claim another worker is actively working. A guarded miss returns
    ``{"status": "unknown"}``: same shape, same 200, no retry bait.

    ``terminal=true`` fails the job outright instead of requeueing it — for a failure
    no retry can fix (a 402 customer-config refusal from the synth target). It
    REQUIRES ``worker_id``: an unguarded terminal fail would let any holder of the
    shared legacy subject kill any claimed job outright.
    """
    if terminal and worker_id is None:
        raise HTTPException(422, "terminal=true requires worker_id")
    status = await jm.fail(
        session,
        job_id=job_id,
        error=error,
        max_attempts=get_settings().queue.max_attempts,
        worker_id=worker_id,
        terminal=terminal,
    )
    return {"status": status}
