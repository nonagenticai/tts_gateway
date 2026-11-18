"""A dedupe hit must not publish a reference to a clip that is GONE from storage.

The dedupe key IS the object name (``{tenant}/{idempotency_key}.wav``), so the
``tts_jobs`` row and the audio are two facts that can disagree: a lifecycle expiry,
a bucket sweep or a half-failed upload deletes the object and leaves the ``done``
row. Production symptom (E2E run 33144609129, ``audiobook`` scenario): submit
answered ``done`` in 23s, the poll presigned a URL for an object that no longer
existed, and the downstream video_compile died on ``404 Not Found`` — the whole run
lost in 52s.

Every test here asserts OBSERVABLE state — the status the caller receives, the row
the queue holds, and above all whether ``claim_next`` now hands the chunk to a
worker (i.e. synthesis was genuinely re-queued) — never a mock call count.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text as sql

from src.services import job_manager as jm

# A real uuid: ``tts_jobs.tenant_id`` is a native ``uuid`` column since migration
# 0003, so a slug tenant no longer round-trips at all.
_TENANT = "11111111-1111-1111-1111-111111111111"


def _chunk(ref: str, text: str) -> dict:
    return {"ref": ref, "text": text, "voice": "af_heart", "speed": 1.0}


async def _row(session_factory, job_id: str) -> dict:
    async with session_factory() as s:
        r = (
            await s.execute(
                sql(
                    "SELECT status, result_key, duration_seconds, attempts, started_at, "
                    "completed_at FROM tts_jobs WHERE job_id = :j"
                ),
                {"j": job_id},
            )
        ).first()
    assert r is not None
    return {
        "status": r.status,
        "result_key": r.result_key,
        "duration": r.duration_seconds,
        "attempts": r.attempts,
        "started_at": r.started_at,
        "completed_at": r.completed_at,
    }


async def _submit_and_finish(session_factory, *, batch_id: str = "run1") -> tuple[str, str]:
    """Enqueue one chunk, synthesize it, and return ``(job_id, result_key)``."""
    async with session_factory() as s:
        first = await jm.submit_batch(
            s,
            tenant_id=_TENANT,
            batch_id=batch_id,
            chunks=[_chunk("r1", "Chapter one.")],
        )
    job_id = first[0]["job_id"]
    # ``job_id`` is ``str | None`` in the JobRow contract (submit's read-back has a
    # defensive fallback); narrow it once here so the queue calls below stay typed.
    assert job_id is not None
    key = f"{_TENANT}/{first[0]['idempotency_key']}.wav"
    async with session_factory() as s:
        claimed = await jm.claim_next(
            s, worker_id="w", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
        assert claimed is not None and claimed["job_id"] == job_id
        await jm.complete(s, job_id=job_id, result_key=key, duration_seconds=4.0)
    return job_id, key


def _probe(verdict: bool | None):
    async def probe(key: str) -> bool | None:
        return verdict

    return probe


async def _resubmit(session_factory, clip_exists, *, batch_id: str = "run2") -> list[jm.JobRow]:
    async with session_factory() as s:
        jobs = await jm.submit_batch(
            s,
            tenant_id=_TENANT,
            batch_id=batch_id,
            chunks=[_chunk("r1", "Chapter one.")],
        )
        return await jm.requeue_jobs_with_missing_clips(
            s, tenant_id=_TENANT, jobs=jobs, clip_exists=clip_exists
        )


# ── Direction 1: the clip is GONE → re-queue, never return the dead ref ──────────


async def test_dedupe_hit_on_a_vanished_clip_requeues_instead_of_returning_it(
    session_factory,
) -> None:
    job_id, key = await _submit_and_finish(session_factory)
    assert (await _row(session_factory, job_id))["result_key"] == key

    jobs = await _resubmit(session_factory, _probe(False))

    # What the CALLER receives: not a done reference.
    assert jobs[0]["job_id"] == job_id  # same row — no duplicate job was created
    assert jobs[0]["status"] == "queued"

    row = await _row(session_factory, job_id)
    assert row["status"] == "queued"
    assert row["result_key"] is None  # the dead pointer is gone, not merely shadowed
    assert row["duration"] is None
    assert row["completed_at"] is None

    # ⚠️ The row MUST come back with a clean sheet. It carries a spent attempt and a
    # ``started_at`` from its original synthesis; ``claim_next_tts_job`` fails a job
    # outright past ``max_job_lifetime_seconds`` (measured from started_at) or past
    # ``max_attempts``. Without the reset the very next claim would poison it into
    # ``status='failed'`` — and the orchestrator raises StagePreconditionError on ANY
    # failed job, converting a recoverable 404 into a hard whole-book failure.
    assert row["attempts"] == 0
    assert row["started_at"] is None


async def test_a_requeued_job_is_actually_synthesized_again(session_factory) -> None:
    """The load-bearing observable: a worker can claim it, and the clip comes back.

    ``status='queued'`` in the DB is only a claim ABOUT re-queueing. This drives the
    real queue to prove it, under PRODUCTION queue settings and a realistically AGED
    row: the clip was synthesized hours ago and had spent its attempts. Those two
    columns are exactly what ``claim_next_tts_job`` fails a job on, so without the
    reset in the re-queue this claim returns ``None`` and the row lands in
    ``status='failed'`` — a hard whole-book abort instead of a re-render.
    """
    job_id, key = await _submit_and_finish(session_factory)
    async with session_factory() as s:
        await s.execute(
            sql(
                "UPDATE tts_jobs SET started_at = now() - interval '5 hours', attempts = 4 "
                "WHERE job_id = :j"
            ),
            {"j": job_id},
        )
        await s.commit()

    await _resubmit(session_factory, _probe(False))

    async with session_factory() as s:
        claimed = await jm.claim_next(
            s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=3600
        )
    assert claimed is not None, "a re-queued job must be claimable by a worker"
    assert claimed["job_id"] == job_id
    assert claimed["text"] == "Chapter one."  # the synthesis inputs survived the re-queue

    async with session_factory() as s:
        await jm.complete(s, job_id=job_id, result_key=key, duration_seconds=4.0)
    async with session_factory() as s:
        rows = await jm.get_batch(s, tenant_id=_TENANT, batch_id="run2")
    assert [(r["status"], r["result_key"]) for r in rows] == [("done", key)]


# ── Direction 2: the clip is THERE → dedupe must still work ──────────────────────


async def test_dedupe_still_short_circuits_when_the_clip_is_present(session_factory) -> None:
    """The anti-regression half. A fix that re-queues unconditionally would pass the
    tests above while destroying the whole point of the gateway: never re-synthesize
    audio that already exists."""
    job_id, key = await _submit_and_finish(session_factory)

    jobs = await _resubmit(session_factory, _probe(True))

    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["status"] == "done"

    row = await _row(session_factory, job_id)
    assert row["status"] == "done"
    assert row["result_key"] == key
    assert float(row["duration"]) == 4.0

    async with session_factory() as s:
        rows = await jm.get_batch(s, tenant_id=_TENANT, batch_id="run2")
    assert [(r["status"], r["result_key"]) for r in rows] == [("done", key)]

    # Nothing was handed back to the workers: no GPU time is spent re-rendering.
    async with session_factory() as s:
        assert (
            await jm.claim_next(
                s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
            is None
        )


async def test_an_indeterminate_probe_keeps_the_dedupe_hit(session_factory) -> None:
    """``None`` means "MinIO did not tell us" (down / unreachable / permissions).

    Treating it as absent would let a read-side blip re-queue an entire cached book —
    hours of paid synthesis caused by a storage hiccup. Only a definitive ``False``
    may act.
    """
    job_id, key = await _submit_and_finish(session_factory)

    jobs = await _resubmit(session_factory, _probe(None))

    assert jobs[0]["status"] == "done"
    assert (await _row(session_factory, job_id))["result_key"] == key


async def test_a_first_time_submit_probes_nothing(session_factory) -> None:
    """Cost floor: a book that has never been synthesized pays zero storage round
    trips. The probe raises, so any call at all fails the test."""

    async def exploding(key: str) -> bool | None:
        raise AssertionError(f"probed {key} on a submit with no dedupe hits")

    async with session_factory() as s:
        jobs = await jm.submit_batch(
            s,
            tenant_id=_TENANT,
            batch_id="fresh",
            chunks=[_chunk("r1", "One."), _chunk("r2", "Two.")],
        )
        jobs = await jm.requeue_jobs_with_missing_clips(
            s, tenant_id=_TENANT, jobs=jobs, clip_exists=exploding
        )
    assert {j["status"] for j in jobs} == {"queued"}


# ── Races ───────────────────────────────────────────────────────────────────────


async def test_two_concurrent_submits_requeue_the_job_exactly_once(session_factory) -> None:
    """Both callers see the clip missing; the row must be re-opened once, and the
    chunk must become claimable by exactly ONE worker (no double synthesis, no lost
    update, no second job row for the same dedupe key)."""
    job_id, _ = await _submit_and_finish(session_factory)

    a, b = await asyncio.gather(
        _resubmit(session_factory, _probe(False), batch_id="runA"),
        _resubmit(session_factory, _probe(False), batch_id="runB"),
    )
    # Neither caller is told "done" — the loser of the CAS re-reads the truth.
    assert a[0]["status"] == "queued"
    assert b[0]["status"] == "queued"
    assert a[0]["job_id"] == b[0]["job_id"] == job_id

    async with session_factory() as s:
        n = (
            await s.execute(
                sql(
                    "SELECT count(*) AS n FROM tts_jobs WHERE tenant_id = '11111111-1111-1111-1111-111111111111'"
                )
            )
        ).scalar_one()
    assert n == 1  # one dedupe key, one row, still

    async with session_factory() as s:
        first = await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
        second = await jm.claim_next(
            s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
    assert first is not None and first["job_id"] == job_id
    assert second is None, "a doubly-requeued job would be synthesized twice"


async def test_a_worker_recompleting_during_the_probe_wins(session_factory) -> None:
    """The probe's verdict is a snapshot. If a worker re-completes the job with FRESH
    bytes between the probe and the re-queue, the compare-and-swap on ``completed_at``
    must lose so we do not throw away audio that now exists."""
    job_id, key = await _submit_and_finish(session_factory)

    async def probe_then_recomplete(_key: str) -> bool | None:
        # Simulate the real window: the object was absent when we looked, and a
        # worker finished a re-synthesis a moment later.
        async with session_factory() as s:
            await s.execute(
                sql("UPDATE tts_jobs SET status = 'claimed', claimed_by = 'w9' WHERE job_id = :j"),
                {"j": job_id},
            )
            await s.commit()
            await jm.complete(
                s, job_id=job_id, result_key=key, duration_seconds=9.5, worker_id="w9"
            )
        return False

    jobs = await _resubmit(session_factory, probe_then_recomplete)

    row = await _row(session_factory, job_id)
    assert row["status"] == "done", "the re-queue must not clobber a fresh completion"
    assert row["result_key"] == key
    assert float(row["duration"]) == 9.5
    assert jobs[0]["status"] == "done"


# ── The ROUTE, not just the function ────────────────────────────────────────────


async def test_the_submit_route_itself_refuses_to_publish_a_dead_reference(
    session_factory, monkeypatch
) -> None:
    """Wiring gate. Every test above calls ``requeue_jobs_with_missing_clips``
    directly, so all of them stay green if the route never calls it — the fix would
    ship INERT. This drives ``POST /api/tts/batches`` end to end with a real DB and a
    storage stub that reports the clip absent, and asserts on the HTTP body."""
    import httpx
    from fastapi import FastAPI
    from sqlalchemy import event

    from src.api.routes import batches
    from src.database.session import bind_tenant_every_transaction, get_tenant_session

    job_id, _ = await _submit_and_finish(session_factory)

    probed: list[str] = []
    bound: list[str | None] = []

    class _Storage:
        async def exists(self, key: str) -> bool | None:
            probed.append(key)
            return False

    monkeypatch.setattr(batches, "get_storage", lambda: _Storage())

    app = FastAPI()
    app.include_router(batches.router)

    @app.middleware("http")
    async def _tenant_state(request, call_next):
        request.state.tenant_id = _TENANT
        return await call_next(request)

    async def _session():
        # Bind ``app.current_tenant`` exactly as the real dependency does. Yielding a
        # bare session here would exercise neither the GUC nor RLS, and it is precisely
        # that gap that let a ``tenant_id`` bound as ``text`` pass this test against a
        # native ``uuid`` column. Binding through the production helper also pins the
        # other half of that contract: ``set_config`` is ``(text, text, bool)``, so a
        # ``uuid.UUID`` reaching it raises ``UndefinedFunction`` and every transaction
        # on the session dies. (RLS POLICIES themselves live in the migrations, not in
        # ``create_all``, so this binds the GUC without enforcing policies here — the
        # policy behaviour is covered by ``test_tenant_session_scope``.)
        async with session_factory() as s:
            bind_tenant_every_transaction(s, _TENANT)

            # Registered AFTER the bind, so it runs after it: records what
            # ``app.current_tenant`` actually holds inside every transaction the route
            # opens. Asserting on this — rather than on the fact that we called the
            # binder — is what makes "the route ran tenant-bound" an observable value.
            @event.listens_for(s.sync_session, "after_begin")
            def _observe(_sess, _txn, connection) -> None:
                bound.append(
                    connection.exec_driver_sql(
                        "SELECT current_setting('app.current_tenant', true)"
                    ).scalar()
                )

            yield s

    app.dependency_overrides[get_tenant_session] = _session

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/tts/batches",
            json={
                "batch_id": "route-run",
                "chunks": [{"ref": "r1", "text": "Chapter one.", "voice": "af_heart"}],
            },
        )
    assert resp.status_code == 202
    body = resp.json()["jobs"][0]
    assert body["job_id"] == job_id
    assert body["status"] == "queued", "the route published a reference to a missing clip"
    assert probed, "the route never asked storage whether the clip still exists"
    assert bound, "the route opened no transaction — the session was never used"
    assert set(bound) == {_TENANT}, (
        f"the route ran on an UNBOUND session (app.current_tenant={bound!r}); it must "
        "carry the caller's tenant, or neither the GUC nor RLS is exercised here"
    )
    assert (await _row(session_factory, job_id))["result_key"] is None


# ── The tenant bind: parsed in Python, never handed to Postgres as text ──────────


async def test_a_non_uuid_tenant_fails_loudly_without_poisoning_the_transaction(
    session_factory,
) -> None:
    """The re-queue SELECT binds the tenant through ``_tenant_uuid``, not as a bare str.

    ``tts_jobs.tenant_id`` is a native ``uuid`` column since migration 0003. This
    pins the property that actually differs between the two binds, because the
    obvious one does NOT: psycopg3 sends a bare ``str`` with an *unspecified* type
    OID, so Postgres infers ``uuid`` from context and ``tenant_id = :t`` quietly
    works. (Measured here, 2026-08-31 — so the ``operator does not exist: uuid =
    text`` failure that the convention is written against is not what a bare bind
    produces on this driver.) What still differs, and is what this asserts:

    * ``_tenant_uuid`` rejects a non-uuid tenant in PYTHON, as a ``ValueError``,
      before a statement is sent. A caller arriving with a legacy slug is a
      licence/middleware defect and must be loud — silently matching no rows would
      present it as an empty queue and re-synthesize an entire already-paid book.
    * Because nothing was sent, the caller's transaction is UNHARMED. A bare bind
      instead reaches Postgres as ``invalid input syntax for type uuid``, which
      aborts the transaction: every subsequent statement on that session raises
      ``InFailedSqlTransaction``, so one bad tenant takes down the whole request —
      including the submit work that had already succeeded on that session.

    The second assertion is the load-bearing half: without it this test passes on
    either bind, since both raise *something*.
    """
    job_id, _ = await _submit_and_finish(session_factory)

    async with session_factory() as s:
        # A live transaction with real work in it, exactly as the route has by the
        # time the re-queue runs.
        n_before = (await s.execute(sql("SELECT count(*) FROM tts_jobs"))).scalar_one()
        assert n_before == 1

        with pytest.raises(ValueError):
            await jm.requeue_jobs_with_missing_clips(
                s,
                tenant_id="n8n-dev",  # the real legacy slug this fleet carried
                jobs=[jm.JobRow(job_id=job_id, ref="r1", idempotency_key="k", status="done")],
                clip_exists=_probe(False),
            )

        # The session is still usable: nothing was ever sent, so nothing was aborted.
        n_after = (await s.execute(sql("SELECT count(*) FROM tts_jobs"))).scalar_one()
        assert n_after == n_before, (
            "the tenant reached Postgres and aborted the transaction — it must be "
            "parsed in Python before any statement is sent"
        )
