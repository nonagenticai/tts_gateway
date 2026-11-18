"""Queue operations over ``tts_jobs`` — the gateway's whole job lifecycle.

Submit is idempotent (``ON CONFLICT DO NOTHING`` on the dedupe key), so a resumed
orchestrator narrate that re-submits an identical book collapses onto existing
rows — no synthesis repeats. Which batches a (shared) job belongs to is a separate
many-to-many fact in ``tts_batch_jobs``, so a second submit ADDS a batch rather than
stealing the row from an in-flight one. Claim goes through the SECURITY DEFINER
``claim_next_tts_job`` (cross-tenant, RLS-safe); complete/fail/renew are plain
guarded updates. Nothing here is worker-pool code: the workers are REMOTE (kokoro
on the Macs) and drive claim/heartbeat/result over the HTTP API.
"""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from hashlib import sha256
from typing import Any, TypedDict, cast
from uuid import uuid7

from sqlalchemy import CursorResult, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TtsJob

logger = logging.getLogger(__name__)


class JobRow(TypedDict):
    """One entry of a submit response: the caller's ref plus the authoritative job.

    A real type rather than ``dict[str, Any]``: this shape crosses two seams — it is
    what ``submit_batch`` hands to ``requeue_jobs_with_missing_clips`` (which REWRITES
    ``status`` in place when a dedupe hit turns out to have no clip), and what the
    route unpacks into the ``JobRef`` pydantic model. Under ``dict[str, Any]`` a typo
    in either place was a silent runtime KeyError, and the ``status`` rewrite — the
    whole point of the re-queue — was unchecked.
    """

    ref: str | None
    job_id: str | None
    idempotency_key: str
    status: str


# Version of the tts_backend synthesis pre-processing (normalize + split + gaps +
# misaki G2P). It is folded into the dedupe key so that improving the
# pre-processing forces a clean re-synth instead of replaying stale cached clips —
# the tts_backend change does NOT alter the request `text`, so without this token the
# key is unchanged and ON CONFLICT DO NOTHING would return the old audio.
# ⚠️ Keep in lockstep with w2v_orchestrator's tts_storage.PREPROC_VERSION and bump on
# every tts_backend pre-processing change. Old objects stay cacheable under old keys.
# "1" → "2" (2026-07-04): misaki G2P activated on the tts_backend live path
# (TTS_G2P=misaki on both minis) — phonemization changed without changing request text.
PREPROC_VERSION = "2"


def idempotency_key(tenant_id: str, voice: str | None, speed: float, txt: str) -> str:
    """Dedupe key: identical (tenant, voice, speed, text, preproc-version) → one job/one synth."""
    raw = f"{tenant_id}|{voice or ''}|{float(speed):.3f}|{PREPROC_VERSION}|{txt}"
    return sha256(raw.encode("utf-8")).hexdigest()


def _tenant_uuid(tenant_id: str) -> _uuid.UUID:
    """The tenant as a real ``uuid`` for the DB layer; the caller keeps its string.

    ``tts_jobs.tenant_id`` is a native ``uuid`` column since 0003. The raw ``text()``
    queries below carry no SQLAlchemy type information, so their bind must be a real
    ``uuid.UUID`` — a bare string reaches Postgres as ``text`` and raises
    ``operator does not exist: uuid = text``. (The ORM insert goes the other way: the
    column is declared ``as_uuid=False``, so its bind processor wants the ``str``.)
    Either way this is also the validation point.

    This deliberately does NOT touch ``idempotency_key`` above: that hash is over the
    tenant STRING exactly as the caller supplied it, and re-normalising it here would
    change every dedupe key in existence, orphaning 500+ already-synthesized clips and
    re-rendering whole books. Parsing is one-way and local to the query layer.

    Raises ``ValueError`` on a non-uuid tenant. That is intentional and fail-loud: a
    caller reaching this service with a tenant NAME is a licence/middleware defect,
    and silently returning "no rows" would present it as an empty queue.
    """
    return _uuid.UUID(tenant_id)


async def submit_batch(
    session: AsyncSession,
    *,
    tenant_id: str,
    batch_id: str,
    chunks: list[dict[str, Any]],
) -> list[JobRow]:
    """Enqueue a book's chunks idempotently; return one row per INPUT chunk.

    ``chunks`` items: ``{ref, voice, speed, text}``. Returns, per input chunk,
    ``{ref, job_id, idempotency_key, status}`` — a re-submit of the identical
    chunk returns the ORIGINAL job_id (many already ``done``), so the caller's
    poll finds finished work without re-synthesizing.
    """
    rows = []
    keyed: list[tuple[dict[str, Any], str]] = []
    seen_keys: set[str] = set()
    for chunk in chunks:
        voice = chunk.get("voice")
        speed = float(chunk.get("speed") or 1.0)
        txt = str(chunk["text"])
        key = idempotency_key(tenant_id, voice, speed, txt)
        keyed.append((chunk, key))
        # Dedupe rows WITHIN this submit by conflict key: two identical chunks in
        # one book map to one job. Required for the ON CONFLICT DO UPDATE below —
        # Postgres rejects a statement that touches the same conflict target row
        # twice ("cannot affect row a second time"). Every input chunk (dupes
        # included) is still mapped back to its job in the read-back loop.
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(
            {
                "job_id": uuid7().hex,
                "tenant_id": str(_tenant_uuid(tenant_id)),
                "idempotency_key": key,
                "batch_id": batch_id,
                "voice": voice,
                "speed": speed,
                "text": txt,
                "status": "queued",
            }
        )

    if rows:
        stmt = pg_insert(TtsJob).values(rows)
        # A dedupe hit is a NO-OP on the job row. ``batch_id`` stays at first-write.
        #
        # ⚠️ This used to be ``DO UPDATE SET batch_id = excluded.batch_id`` — re-tagging
        # the existing row onto the newest batch. The intent was right (a re-attempt
        # under a fresh ``run_id`` must SEE its own already-``done`` clips) but the
        # mechanism MOVED the row instead of sharing it: while run A polls
        # ``get_batch(A)``, a concurrent run B submitting an overlapping chunk pulled
        # A's row into B, A's batch went empty, and A burned its whole poll budget and
        # then raised — for synthesis that had in fact succeeded, under B's batch.
        #
        # Both requirements are met by making membership a RELATION instead of a
        # column: the insert below adds ``(tenant, batch, job)`` to ``tts_batch_jobs``
        # for EVERY submit, hit or miss, so the re-attempt sees the done clip AND the
        # first batch keeps it. Dedupe is untouched — still exactly one job row, one
        # synthesis, per (tenant, idempotency_key).
        stmt = stmt.on_conflict_do_nothing(index_elements=["tenant_id", "idempotency_key"])
        await session.execute(stmt)

    # Read back the authoritative rows (existing OR just-inserted) by dedupe key.
    keys = [k for _, k in keyed]
    result = await session.execute(
        text(
            "SELECT idempotency_key, job_id, status FROM tts_jobs "
            "WHERE tenant_id = :t AND idempotency_key = ANY(:keys)"
        ),
        {"t": _tenant_uuid(tenant_id), "keys": keys},
    )
    by_key = {r.idempotency_key: (r.job_id, r.status) for r in result}

    # Record THIS batch's membership over the authoritative job ids — the read-back
    # ids, not the ones generated above, because a dedupe hit kept the ORIGINAL job_id
    # and the freshly generated one was discarded by ON CONFLICT DO NOTHING.
    # Append-only and idempotent: a re-submit of the same book under the same batch
    # conflicts on the PK and does nothing.
    member_ids = sorted({job_id for job_id, _ in by_key.values() if job_id})
    if member_ids:
        await session.execute(
            text(
                "INSERT INTO tts_batch_jobs (tenant_id, batch_id, job_id) "
                "SELECT :t, :b, unnest(CAST(:ids AS varchar[])) "
                "ON CONFLICT DO NOTHING"
            ),
            {"t": _tenant_uuid(tenant_id), "b": batch_id, "ids": member_ids},
        )
    await session.commit()

    out: list[JobRow] = []
    for chunk, key in keyed:
        job_id, status = by_key.get(key, (None, "queued"))
        out.append(
            {"ref": chunk.get("ref"), "job_id": job_id, "idempotency_key": key, "status": status}
        )
    return out


# Probe signature: key -> True present / False definitively absent / None unknown.
# See ``ClipStorage.exists``; injected rather than imported so this module stays
# storage-free and the probe is substitutable in tests.
ClipProbe = Callable[[str], Awaitable[bool | None]]

# Bounds the concurrent stat_object fan-out. A submit whose dedupe hits are ALL
# ``done`` (a full re-run of a book) probes one key per chunk; a book is O(100s) of
# chunks and an unbounded gather would open that many MinIO connections at once.
_PROBE_CONCURRENCY = 16

# Re-open a ``done`` job whose stored clip has vanished.
#
# ⚠️ ``attempts`` and ``started_at`` are RESET, and that is load-bearing, not
# tidiness. ``claim_next_tts_job`` fails a job outright when
# ``started_at < now() - max_job_lifetime_seconds`` (3600s by default) or when
# ``attempts >= max_attempts``. A job being re-queued here succeeded once, days or
# weeks ago: it carries a spent attempt and an ancient ``started_at``. Left alone,
# the very next claim would age it out into ``status='failed'`` instead of
# synthesizing it — and the orchestrator's narrate stage raises
# ``StagePreconditionError`` on ANY failed job, so we would have converted a
# recoverable 404 into a hard whole-book failure. This is a NEW synthesis of a lost
# artifact, not a retry of a misbehaving chunk, so it starts with a clean sheet.
#
# ⚠️ The row is RE-OPENED, never deleted. Deleting it would drop the batch tag the
# orchestrator polls by and race every other holder of the dedupe key; the queue
# already owns ``status`` as its lifecycle, so we move it within that lifecycle.
#
# Race safety is the same compare-and-swap the rest of this module uses. The
# predicate is ``status='done' AND completed_at = :c``, where ``:c`` is the value we
# READ before probing:
#   * two concurrent submits both see the object missing → both issue this UPDATE →
#     Postgres serializes them on the row lock; the first matches, the second sees
#     ``status='queued'`` and matches 0 rows. Exactly one re-queue, no lost update,
#     and no second job row (the dedupe unique index is untouched — we reuse the row).
#   * a worker re-completes the job in the window between our probe and this UPDATE
#     (so the object exists again, under a NEW ``completed_at``) → the CAS fails and
#     we leave the fresh, valid clip alone. Without the ``completed_at`` arm the
#     ``status='done'`` test alone would still match and we would discard good audio.
# A 0-row outcome is therefore never an error; the caller re-reads the truth.
_REQUEUE_MISSING_SQL = (
    "UPDATE tts_jobs SET status = 'queued', result_key = NULL, duration_seconds = NULL, "
    "error_message = NULL, claimed_by = NULL, lease_expires_at = NULL, "
    "completed_at = NULL, started_at = NULL, busy_since = NULL, attempts = 0 "
    "WHERE job_id = :j AND status = 'done' AND completed_at = :c "
    "RETURNING status"
)


# Re-open a TERMINALLY-``failed`` job so a later submit synthesizes it again.
#
# **The defect this closes.** Submit dedupes with ``ON CONFLICT DO NOTHING`` on
# ``(tenant_id, idempotency_key)``. A ``failed`` row wins that conflict exactly as a
# ``done`` one does, so the insert is a no-op, the read-back publishes the row's
# stored ``error_message`` verbatim, nothing is enqueued and NO worker is ever
# called. The failure is therefore permanent and content-scoped: that text can never
# be synthesized again for that tenant, at any voice/speed it was failed at, by any
# future run. Measured in production: four orchestrator runs across 23 minutes all
# resolved to one ``job_id`` and replayed one stored 503 — an error naming a URL
# nobody had called, which reads exactly like a live outage. There is no requeue
# route, sweeper or script; only a manual UPDATE recovered it.
#
# This is a SIBLING of ``_REQUEUE_MISSING_SQL``, deliberately not a widening of it.
# The two reconcile different disagreements (a ``done`` row whose artifact is gone
# vs. a ``failed`` row whose failure was transient), they have different guards, and
# only this one is bounded. Folding ``failed`` into that statement would also make
# every ``failed`` job pay a storage probe for a ``result_key`` it does not have.
#
# ⚠️ ``attempts`` is reset and ``started_at`` cleared for the same load-bearing
# reason documented on ``_REQUEUE_MISSING_SQL``: a re-opened row carries a spent
# attempt budget and an ancient ``started_at``, and ``claim_next_tts_job`` would
# fail it straight back out on the very next claim (``attempts >= max_attempts``, or
# ``started_at`` past ``max_job_lifetime_seconds``) without ever synthesizing. This
# is a new synthesis, so it starts on a clean sheet.
#
# ⚠️ ``COALESCE(requeue_count,0) < :cap`` is NOT optional. Because ``attempts`` is
# reset above, it cannot bound the re-open; without the monotonic counter this
# trades a permanent blacklist for an infinite treadmill — an orchestrator retrying
# a book every 20 minutes re-burning ``max_attempts`` GPU claims forever on text
# kokoro will never accept. ``requeue_count`` is reset by nothing, so poison
# converges: after ``:cap`` re-opens the row stays ``failed`` and its stored error is
# replayed, i.e. exactly today's behaviour, but only after bounded genuine retries.
#
# ⚠️ Re-synth is safe to double-run and needs no artifact dedupe: the storage key is
# ``{tenant}/{idempotency_key}.wav``, fully determined by the dedupe key, so a second
# synthesis OVERWRITES and cannot double-produce. A ``failed`` row has no
# ``result_key`` to discard either — and it is cleared here regardless, so a row that
# somehow carries one cannot publish a stale reference after re-opening.
#
# Race safety is the same compare-and-swap the rest of this module uses, on the same
# column: ``status='failed' AND completed_at = :c`` where ``:c`` is the value READ
# before the update. Two concurrent submits both see ``failed`` → both issue this
# UPDATE → Postgres serializes them on the row lock, the first matches and the second
# sees ``status='queued'`` and matches 0 rows. Exactly one re-open, one increment of
# ``requeue_count``, no lost update and no second job row. A 0-row outcome is never
# an error; the caller re-reads the truth rather than assuming which arm rejected it
# (a cap rejection and a lost CAS are indistinguishable from the row count alone).
#
# No claimed-worker race exists on a ``failed`` row: both paths that write ``failed``
# — ``_FAIL_SQL`` and ``claim_next_tts_job``'s poison/lifetime arms — clear
# ``claimed_by`` and ``lease_expires_at``, so nothing holds it. No extra locking.
#
# ⚠️ ``IS NOT DISTINCT FROM`` here is DEFENCE IN DEPTH, not a live requirement, and
# the earlier note claiming a ``failed`` row may carry a NULL ``completed_at`` was
# simply wrong. Every writer of ``failed`` sets it: ``_FAIL_SQL``'s CASE arm, and
# both the poison and lifetime arms of ``claim_next_tts_job`` (migration
# ``0003_tenant_id_uuid.py``). So no row this CAS can meet today has ``:c`` NULL, and
# a bare ``=`` leaves every test green — measured, including the NULL path probed
# directly against Postgres. It is kept null-safe anyway so that a FUTURE writer that
# forgets ``completed_at`` cannot make one row silently unrecoverable forever (a bare
# ``completed_at = NULL`` is NULL, never true, and the 0-row result is
# indistinguishable from a lost race — the exact failure mode this function deletes).
# For every non-NULL ``:c`` the two operators are identical.
#
# The concurrency safety above is therefore carried by the ``status = 'failed'`` arm,
# not by this one: the winner flips the row to ``queued``, the loser matches 0 rows on
# that predicate and re-reads the truth in the ``else`` branch of ``reopen_failed_jobs``
# rather than guessing which arm rejected it.
_REOPEN_FAILED_SQL = (
    "UPDATE tts_jobs SET status = 'queued', attempts = 0, "
    "requeue_count = COALESCE(requeue_count, 0) + 1, "
    "error_message = NULL, claimed_by = NULL, lease_expires_at = NULL, "
    "completed_at = NULL, started_at = NULL, busy_since = NULL, result_key = NULL, "
    "duration_seconds = NULL "
    "WHERE job_id = :j AND status = 'failed' "
    "AND completed_at IS NOT DISTINCT FROM :c "
    "AND COALESCE(requeue_count, 0) < :cap "
    "RETURNING status"
)


def _apply_statuses(jobs: list[JobRow], statuses: dict[str, str]) -> list[JobRow]:
    """Overwrite each job's ``status`` with the truth the DB just reported."""
    for j in jobs:
        new_status = statuses.get(str(j.get("job_id")))
        if new_status is not None:
            j["status"] = new_status
    return jobs


async def reopen_failed_jobs(
    session: AsyncSession,
    *,
    tenant_id: str,
    jobs: list[JobRow],
    max_requeues: int,
) -> list[JobRow]:
    """Re-open the dedupe HITS that came back ``failed``, up to a monotonic cap.

    Without this, ``ON CONFLICT DO NOTHING`` makes a failed synthesis PERMANENT for
    that (tenant, voice, speed, text): every later submit collapses onto the dead row
    and replays its stored ``error_message`` without enqueueing anything or calling a
    worker. Almost every failure the queue actually produces is transient — an
    upstream 503, a pod rolling mid-claim, a lease lost to a Tailnet blip — and none
    of them justify blacklisting the content.

    Returns ``jobs`` with the affected entries' ``status`` corrected to ``queued``, so
    the caller publishes the re-opened truth rather than a stale error. Must run
    BEFORE that read-back is published — it is called from the submit route for
    exactly that reason.

    Bounded by ``max_requeues`` against the monotonic ``requeue_count``: a genuinely
    poison chunk is re-opened at most that many times and then stays ``failed``,
    replaying its error exactly as today. See ``_REOPEN_FAILED_SQL`` for the guard,
    the CAS and why ``attempts`` cannot carry this bound.

    **Cost.** One SELECT, only when this submit actually has a ``failed`` hit — which
    is the rare case (production census at discovery: ``done 537, failed 1``). Zero
    storage round trips: a ``failed`` row has no artifact to probe for.
    """
    failed = [j for j in jobs if j.get("status") == "failed" and j.get("job_id")]
    if not failed:
        return jobs

    ids = [str(j["job_id"]) for j in failed]
    result = await session.execute(
        text(
            "SELECT job_id, completed_at, status FROM tts_jobs "
            "WHERE tenant_id = :t AND job_id = ANY(:ids)"
        ),
        {"t": _tenant_uuid(tenant_id), "ids": ids},
    )
    rows = list(result)
    stored: dict[str, datetime | None] = {
        r.job_id: r.completed_at for r in rows if r.status == "failed"
    }
    # A row the caller read back as ``failed`` that is no longer ``failed``: a
    # concurrent submit re-opened it between that read-back and this SELECT. Publish
    # what it says NOW — otherwise the loser of that race still reports the very
    # stale error this function exists to suppress.
    reopened: dict[str, str] = {r.job_id: r.status for r in rows if r.status != "failed"}

    for job_id, completed_at in stored.items():
        row = (
            await session.execute(
                text(_REOPEN_FAILED_SQL),
                {"j": job_id, "c": completed_at, "cap": max_requeues},
            )
        ).first()
        if row is not None:
            logger.warning(
                "job %s: dedupe hit on a FAILED row — re-opened for synthesis "
                "instead of replaying its stored error (cap=%s)",
                job_id,
                max_requeues,
            )
            reopened[job_id] = row.status
        else:
            # 0 rows. Either the re-queue cap is spent (a genuinely poison chunk,
            # which must keep behaving exactly as it does today) or another submit
            # won the CAS and already re-opened it. The row count cannot tell those
            # apart, so do not guess — re-read and report the truth.
            current = (
                await session.execute(
                    text("SELECT status FROM tts_jobs WHERE job_id = :j"), {"j": job_id}
                )
            ).first()
            if current is not None:
                reopened[job_id] = current.status
    await session.commit()

    return _apply_statuses(jobs, reopened)


async def requeue_jobs_with_missing_clips(
    session: AsyncSession,
    *,
    tenant_id: str,
    jobs: list[JobRow],
    clip_exists: ClipProbe,
) -> list[JobRow]:
    """Verify each dedupe HIT still has its clip in storage; re-queue the ones that don't.

    The dedupe key IS the object name (``{tenant}/{idempotency_key}.wav``), so a
    ``done`` row and its clip are two independent facts that can disagree: a
    lifecycle expiry, a bucket sweep or a half-failed upload removes the object and
    leaves the row. Submit then answered ``done`` in seconds, the poll presigned a
    URL for an object that is gone, and the downstream compile died on a 404 —
    aborting a whole audiobook run for audio that simply needs re-synthesizing.

    Returns ``jobs`` with the affected entries' ``status`` corrected, so the caller
    never publishes a reference it has not confirmed.

    **Cost.** One ``stat_object`` per *already-``done``* job in this submit, once
    per submit — NOT per poll. A first-time submit does zero extra work (nothing is
    ``done`` yet) and pays neither the SELECT nor any probe. The worst case is a
    full re-run of a fully-cached book: one in-cluster HEAD per chunk, fanned out
    ``_PROBE_CONCURRENCY``-wide, i.e. sub-second for a few hundred chunks against a
    submit that is already doing a multi-row INSERT and a read-back. It replaces a
    404 that costs the entire run.

    Only a ``False`` (definitively absent) probe re-queues. ``None`` — MinIO down,
    unreachable, permission error — keeps today's behaviour, because re-queueing a
    whole book on a read-side blip would be far more expensive than the bug.
    """
    done = [j for j in jobs if j.get("status") == "done" and j.get("job_id")]
    if not done:
        return jobs

    ids = [str(j["job_id"]) for j in done]
    result = await session.execute(
        text(
            "SELECT job_id, result_key, completed_at, status FROM tts_jobs "
            "WHERE tenant_id = :t AND job_id = ANY(:ids)"
        ),
        {"t": _tenant_uuid(tenant_id), "ids": ids},
    )
    rows = list(result)
    stored: dict[str, tuple[str | None, datetime | None]] = {
        r.job_id: (r.result_key, r.completed_at) for r in rows if r.status == "done"
    }
    # A row the caller read back as ``done`` that is NOT ``done`` any more: a
    # concurrent submit already re-queued it (or a worker re-opened it) between that
    # read-back and this SELECT. Correcting it here matters as much as the CAS-loser
    # branch below — otherwise the loser of that race publishes the very dead
    # reference this function exists to suppress.
    corrected: dict[str, str] = {r.job_id: r.status for r in rows if r.status != "done"}

    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)

    async def _probe(key: str) -> bool | None:
        async with sem:
            return await clip_exists(key)

    # De-duplicate by KEY, not by job: distinct jobs of one tenant never share a key
    # (the key IS the dedupe key), but the same key can appear twice in `done` if the
    # caller passes overlapping batches. One probe per distinct key either way.
    keys = sorted({k for k, _ in stored.values() if k})
    probes = await asyncio.gather(*(_probe(k) for k in keys))
    missing = {k for k, present in zip(keys, probes, strict=True) if present is False}
    if not missing:
        return _apply_statuses(jobs, corrected)

    requeued: dict[str, str] = dict(corrected)
    for job_id, (key, completed_at) in stored.items():
        if key not in missing:
            continue
        row = (
            await session.execute(text(_REQUEUE_MISSING_SQL), {"j": job_id, "c": completed_at})
        ).first()
        if row is not None:
            logger.warning(
                "job %s: dedupe hit on a clip that is GONE from storage (key=%s) — "
                "re-queued for synthesis instead of returning a dead reference",
                job_id,
                key,
            )
            requeued[job_id] = row.status
        else:
            # Lost the CAS: another submit re-queued it, or a worker re-completed it
            # with fresh bytes. Report whatever the row actually says now.
            current = (
                await session.execute(
                    text("SELECT status FROM tts_jobs WHERE job_id = :j"), {"j": job_id}
                )
            ).first()
            if current is not None:
                requeued[job_id] = current.status
    await session.commit()

    return _apply_statuses(jobs, requeued)


async def get_batch(
    session: AsyncSession, *, tenant_id: str, batch_id: str
) -> list[dict[str, Any]]:
    """All jobs for a batch with current status + result key/duration.

    Membership is resolved through ``tts_batch_jobs``, NOT through
    ``tts_jobs.batch_id``. A deduped job legitimately belongs to several batches at
    once, and the old column-scoped read meant the newest submit silently emptied
    every earlier batch that shared a chunk.
    """
    result = await session.execute(
        text(
            "SELECT j.job_id, j.idempotency_key, j.status, j.result_key, "
            "       j.duration_seconds, j.error_message "
            "FROM tts_batch_jobs m "
            "JOIN tts_jobs j ON j.job_id = m.job_id "
            "WHERE m.tenant_id = :t AND m.batch_id = :b"
        ),
        {"t": _tenant_uuid(tenant_id), "b": batch_id},
    )
    return [
        {
            "job_id": r.job_id,
            "idempotency_key": r.idempotency_key,
            "status": r.status,
            "result_key": r.result_key,
            "duration_seconds": float(r.duration_seconds)
            if r.duration_seconds is not None
            else None,
            # A FINISHED job with no stored duration is a measurement gap, not a
            # pending one: the clip exists, its length was never measured. Only
            # ``done`` qualifies — a queued/claimed row has no duration YET, and
            # flagging that would make the signal meaningless.
            "duration_unmeasured": r.status == "done" and r.duration_seconds is None,
            "error": r.error_message,
        }
        for r in result
    ]


async def get_job_keyinfo(session: AsyncSession, *, job_id: str) -> dict[str, Any] | None:
    """(tenant_id, idempotency_key, status, claimed_by) for a clip's storage key.

    ``claimed_by`` is part of this row because the storage key it builds —
    ``{tenant_id}/{idempotency_key}.wav`` — is DETERMINISTIC, not content-addressed.
    Every worker that ever holds this job writes the same key, so the result route
    has to know who owns the claim BEFORE it puts any bytes there; the
    ``claimed_by`` guard on ``complete()`` only protects the DB row, and by the time
    it says "no" the object store has already been overwritten.
    """
    result = await session.execute(
        text(
            "SELECT tenant_id, idempotency_key, status, claimed_by FROM tts_jobs WHERE job_id = :j"
        ),
        {"j": job_id},
    )
    row = result.first()
    if row is None:
        return None
    return {
        # Back to a string at the boundary: the storage key
        # ``{tenant_id}/{idempotency_key}.wav`` is a deterministic address for
        # objects already written under the canonical lowercase uuid text, and every
        # consumer of this dict treats tenant_id as text.
        "tenant_id": str(row.tenant_id),
        "idempotency_key": row.idempotency_key,
        "status": row.status,
        "claimed_by": row.claimed_by,
    }


async def claim_next(
    session: AsyncSession,
    *,
    worker_id: str,
    lease_seconds: int,
    max_attempts: int,
    max_lifetime_seconds: int,
    max_busy_age_seconds: int = 0,
) -> dict[str, Any] | None:
    """Claim one job via the SECURITY DEFINER resolver; ``None`` if queue empty.

    ``None`` also covers the two terminal verdicts the resolver reaches instead of
    claiming: a poison job past ``max_attempts``, and a job past
    ``max_lifetime_seconds`` of wall-clock running life (measured from its FIRST
    claim, so a backlogged never-claimed row is exempt). Both are failed in-place.
    A third: ``max_busy_age_seconds`` past the job's FIRST GPU-busy release fails it
    as ``gpu_busy_exhausted`` (0 disables; see ``release_claim(busy=True)``).
    """
    result = await session.execute(
        text("SELECT * FROM claim_next_tts_job(:w, :lease, :max, :life, :busy)"),
        {
            "w": worker_id,
            "lease": lease_seconds,
            "max": max_attempts,
            "life": max_lifetime_seconds,
            "busy": max_busy_age_seconds,
        },
    )
    row = result.first()
    await session.commit()
    if row is None:
        return None
    return {
        "job_id": row.job_id,
        # The resolver now projects a real uuid; the worker-facing payload stays text.
        "tenant_id": str(row.tenant_id),
        "voice": row.voice,
        "speed": float(row.speed) if row.speed is not None else 1.0,
        "provider": row.provider,
        "text": row.txt,
    }


async def release_claim(
    session: AsyncSession, *, job_id: str, worker_id: str, busy: bool = False
) -> bool:
    """Surrender a claim at graceful shutdown and REFUND the attempt it consumed.

    Returns True iff this worker still held the claim (so the job went back to
    ``queued`` with ``attempts`` decremented). False — a no-op — when the job
    already completed, was reclaimed by someone else after a lease expiry, or was
    already released: the resolver's ``(job_id, worker_id, status='claimed')``
    predicate makes all of those races safe.

    Why the refund: ``claim_next_tts_job`` charges an attempt on EVERY claim and
    poisons at ``max_attempts``. A rollout/reboot that interrupts a healthy job is
    not the job misbehaving, so it must not be charged. A real crash never reaches
    this call, so poison detection for genuinely bad chunks is unchanged.

    ``busy=True`` (tts_gateway#45 — the synth GPU stayed saturated) additionally
    clears ``started_at`` so busy time never counts toward the lifetime bound, and
    stamps ``busy_since`` to arm the absolute busy backstop instead.
    """
    stmt = (
        "SELECT release_tts_claim_busy(:j, :w) AS released"
        if busy
        else "SELECT release_tts_claim(:j, :w) AS released"
    )
    result = await session.execute(
        text(stmt),
        {"j": job_id, "w": worker_id},
    )
    row = result.first()
    await session.commit()
    return bool(row and (row.released or 0) > 0)


_RENEW_SQL = (
    "UPDATE tts_jobs SET lease_expires_at = now() + make_interval(secs => :lease) "
    "WHERE job_id = :j AND status = 'claimed'"
)
_RENEW_OWNED_SQL = _RENEW_SQL + " AND claimed_by = :w"


async def renew_lease(
    session: AsyncSession, *, job_id: str, lease_seconds: int, worker_id: str | None = None
) -> bool:
    """Heartbeat a claimed job's lease; False if it is no longer ``claimed``.

    With ``worker_id`` the renewal is guarded on ``claimed_by``: worker A, whose
    lease expired and whose job worker B re-claimed, must not keep renewing B's
    lease (and must learn — via False → 409 — that it lost the claim). Without it,
    today's unguarded behaviour, for workers that predate the field.
    """
    if worker_id is not None:
        result = await session.execute(
            text(_RENEW_OWNED_SQL), {"j": job_id, "lease": lease_seconds, "w": worker_id}
        )
    else:
        result = await session.execute(text(_RENEW_SQL), {"j": job_id, "lease": lease_seconds})
    await session.commit()
    return (cast("CursorResult[Any]", result).rowcount or 0) > 0


# ⚠️ ROLLOUT COMPATIBILITY. ``worker_id`` is OPTIONAL on complete()/fail() ONLY because
# the remote kokoro workers are manually-deployed launchd units on the Macs — CI cannot
# redeploy them. Making it required would instantly stop every not-yet-updated worker
# from reporting results: a worse outage than the race it closes. When it is None we keep
# today's exact (unguarded) behaviour; when supplied we add the ``claimed_by`` guard.
# FOLLOW-UP: make it REQUIRED once every Mac worker is on the new build.
#
# The optional `AND claimed_by = :w` suffix is appended as a FIXED literal — never a
# caller-supplied string — and every value stays a bound parameter.
_COMPLETE_SQL = (
    "UPDATE tts_jobs SET status = 'done', result_key = :k, duration_seconds = :d, "
    "completed_at = now(), lease_expires_at = NULL "
    "WHERE job_id = :j AND status = 'claimed'"
)

# ``AND status = 'claimed'`` is LOAD-BEARING, exactly as it is on _COMPLETE_SQL: a
# failure report may only act on a job that is currently in flight. Without it this
# UPDATE matched on ``job_id`` alone and happily rewrote a job that was already
# ``done``.
#
# That was not theoretical. ``worker/worker.py::_process`` wraps synthesis AND the
# ``POST /result`` AND its ``raise_for_status()`` in ONE ``try``, and the ``except``
# reports ``/fail``. So when the result upload SUCCEEDS server-side but the worker
# never sees the 204 — a read timeout on a multi-MB clip over the Tailnet, a
# connection reset, the gateway pod rolling right after COMMIT — the worker fails a
# job the gateway already finished. ``_COMPLETE_SQL`` does not clear ``claimed_by``,
# so even the ``worker_id`` guard matched: same job, same worker.
#
# The outcome was a finished chunk dragged back to ``queued`` with ``completed_at``
# cleared (its stored clip re-synthesized from scratch — paid GPU time for audio that
# already exists), or, once ``attempts`` had reached the cap, flipped straight to
# ``failed`` while its perfectly good ``result_key`` sat in MinIO. The orchestrator's
# audiobook narrate stage raises ``StagePreconditionError`` on ANY failed job, so one
# lost 204 aborted a whole book.
#
# A miss now returns ``unknown`` (a truthful 200), which is precisely what the route
# already documents for the other no-op case — no retry bait, no resurrection.
_FAIL_SQL = (
    "UPDATE tts_jobs SET "
    "  status = CASE WHEN attempts >= :max OR CAST(:terminal AS boolean) "
    "    THEN 'failed' ELSE 'queued' END, "
    "  error_message = :e, "
    "  claimed_by = NULL, lease_expires_at = NULL, "
    "  completed_at = CASE WHEN attempts >= :max OR CAST(:terminal AS boolean) "
    "    THEN now() ELSE NULL END "
    "WHERE job_id = :j AND status = 'claimed'"
)

_OWNER_GUARD = " AND claimed_by = :w"


async def complete(
    session: AsyncSession,
    *,
    job_id: str,
    result_key: str,
    duration_seconds: float | None,
    worker_id: str | None = None,
) -> bool:
    """Mark a claimed job ``done`` with its stored clip key + duration.

    ``duration_seconds=None`` is not "no duration yet" — it is the DELIVERED value for
    a clip whose length nobody measured (``tts_backend`` emits ``duration_sec=0.0``
    when it has no sample rate). It is stored as SQL NULL so that the number the
    orchestrator's narration timeline consumes can never be a fabricated zero; the
    absence is republished by ``get_batch`` as ``duration_unmeasured``.

    Guarded on ``claimed_by`` when ``worker_id`` is supplied: worker A can hang past its
    lease, have the job reclaimed by worker B, and only THEN come back with a result —
    without the guard that stale write would finish B's ACTIVE claim out from under it
    (and publish A's audio for a synthesis B is still running). 0 rows → benign no-op.
    """
    owner_guard = _OWNER_GUARD if worker_id is not None else ""
    params: dict[str, Any] = {"j": job_id, "k": result_key, "d": duration_seconds}
    if worker_id is not None:
        params["w"] = worker_id
    stmt = _COMPLETE_SQL + owner_guard
    result = await session.execute(text(stmt), params)
    await session.commit()
    return (cast("CursorResult[Any]", result).rowcount or 0) > 0


async def fail(
    session: AsyncSession,
    *,
    job_id: str,
    error: str,
    max_attempts: int,
    worker_id: str | None = None,
    terminal: bool = False,
) -> str:
    """Report a synth failure. Requeue if attempts remain, else mark failed.

    ``terminal=True`` marks it failed NOW regardless of attempts left: a failure no
    retry can fix (e.g. the synth target's 402 for a byo_required tenant with no
    configured provider). Re-claiming it would only repeat the same 402.

    Returns the resulting status (``queued`` or ``failed``), or ``unknown`` when the
    UPDATE matched nothing — which, with ``worker_id`` supplied, is exactly the stale
    worker case: A's late failure must not requeue (or poison) the claim B now holds.
    """
    owner_guard = _OWNER_GUARD if worker_id is not None else ""
    params: dict[str, Any] = {
        "j": job_id,
        "e": error[:2000],
        "max": max_attempts,
        "terminal": terminal,
    }
    if worker_id is not None:
        params["w"] = worker_id
    stmt = _FAIL_SQL + owner_guard + " RETURNING status"
    result = await session.execute(text(stmt), params)
    row = result.first()
    await session.commit()
    return row.status if row else "unknown"
