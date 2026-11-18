"""tts_jobs claim-queue: idempotent submit, atomic claim, lease reclaim, poison."""

from __future__ import annotations

import asyncio

from sqlalchemy import text as sql

from src.services import job_manager as jm


def _chunk(ref: str, text: str, voice: str = "af_heart", speed: float = 1.0) -> dict:
    return {"ref": ref, "text": text, "voice": voice, "speed": speed}


async def _row(session_factory, job_id: str) -> dict:
    """Raw queue-row state — attempts/claimed_by are not exposed by get_batch."""
    async with session_factory() as s:
        r = (
            await s.execute(
                sql(
                    "SELECT status, attempts, requeue_count, claimed_by, error_message, "
                    "started_at, completed_at FROM tts_jobs WHERE job_id = :j"
                ),
                {"j": job_id},
            )
        ).first()
    assert r is not None
    return {
        "status": r.status,
        "attempts": r.attempts,
        "requeue_count": r.requeue_count,
        "claimed_by": r.claimed_by,
        "error": r.error_message,
        "started_at": r.started_at,
        "completed_at": r.completed_at,
    }


async def _backdate_start(session_factory, job_id: str, hours: int) -> None:
    async with session_factory() as s:
        await s.execute(
            sql(
                f"UPDATE tts_jobs SET started_at = now() - interval '{hours} hours' "  # noqa: S608
                "WHERE job_id = :j"
            ),
            {"j": job_id},
        )
        await s.commit()


def test_idempotency_key_golden_pins_cache_key_scheme() -> None:
    """Freeze the full dedupe-key scheme (format + PREPROC_VERSION) to a literal.

    A tts_backend pre-processing change (e.g. the misaki G2P activation) MUST bump
    PREPROC_VERSION IN LOCKSTEP with w2v_orchestrator or ON CONFLICT DO NOTHING replays
    stale espeak audio. This golden hash breaks the moment the version or raw-string
    format changes, so a bump is always a deliberate, reviewed edit — never silent drift.
    """
    assert jm.PREPROC_VERSION == "2"
    assert (
        jm.idempotency_key(
            "e148e8af-0000-0000-0000-000000000000", "af_heart", 1.0, "The quick brown fox."
        )
        == "60069fac2a18a2eee05b3ca7440a503f3161ffaf17fa90b0cfe1f21962c678a3"  # pragma: allowlist secret
    )


async def test_submit_is_idempotent(session_factory) -> None:
    async with session_factory() as s:
        first = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", "Hello."), _chunk("r2", "Bye.")],
        )
    # re-submit the SAME book (a resumed narrate) → same job_ids, no new rows
    async with session_factory() as s:
        again = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", "Hello."), _chunk("r2", "Bye.")],
        )
    assert {r["job_id"] for r in first} == {r["job_id"] for r in again}

    async with session_factory() as s:
        rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
        )
    assert len(rows) == 2  # deduped — not 4


async def test_identical_text_dedupes_to_one_job(session_factory) -> None:
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", "Yes."), _chunk("r2", "Yes.")],
        )
    # two refs, identical (voice,speed,text) → one underlying job
    assert out[0]["job_id"] == out[1]["job_id"]
    async with session_factory() as s:
        assert (
            len(
                await jm.get_batch(
                    s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
                )
            )
            == 1
        )


async def test_reattempt_new_batch_reuses_done_clip_without_stealing_the_old_one(
    session_factory,
) -> None:
    """Both halves of dedupe-across-batches, in one test.

    REPLACES ``test_reattempt_new_batch_reuses_done_clip``, which asserted
    ``old_rows == []`` — i.e. it PINNED the defect: the ``ON CONFLICT DO UPDATE SET
    batch_id`` re-tag physically MOVED the shared job row out of the first batch. The
    re-attempt half of that test was right and is kept; the "and the old batch is now
    empty" half was the bug, and is inverted here.
    """
    async with session_factory() as s:
        first = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="run1",
            chunks=[_chunk("r1", "Chapter one.")],
        )
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
        )
        assert job is not None
        await jm.complete(s, job_id=job["job_id"], result_key="t1/clip.wav", duration_seconds=4.0)

    # re-attempt: identical chunk under a fresh batch_id
    async with session_factory() as s:
        again = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="run2",
            chunks=[_chunk("r1", "Chapter one.")],
        )
    assert again[0]["job_id"] == first[0]["job_id"]  # same underlying job, no re-render
    assert again[0]["status"] == "done"

    async with session_factory() as s:
        new_rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="run2"
        )
        old_rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="run1"
        )
    assert len(new_rows) == 1 and new_rows[0]["status"] == "done"
    assert new_rows[0]["result_key"] == "t1/clip.wav"  # reused clip is pollable
    # ...AND run1 still sees its own work. Membership is many-to-many; the second
    # submit ADDS a batch, it does not re-point the row.
    assert len(old_rows) == 1
    assert old_rows[0]["job_id"] == first[0]["job_id"]
    assert old_rows[0]["status"] == "done" and old_rows[0]["result_key"] == "t1/clip.wav"

    # Dedupe still holds: ONE job row, so exactly one paid synthesis.
    async with session_factory() as s:
        n = (await s.execute(sql("SELECT count(*) AS n FROM tts_jobs"))).first()
    assert n is not None and n.n == 1


async def test_a_concurrent_submit_does_not_steal_an_in_flight_batchs_rows(
    session_factory,
) -> None:
    """THE defect. Run A is mid-poll when run B submits an OVERLAPPING book.

    The two batches must genuinely overlap on a dedupe key or the assertion cannot
    fail, so B re-submits A's "Shared line." verbatim (same voice + speed + text ⇒
    same ``idempotency_key``) alongside a chunk of its own.

    Before the fix, B's ``ON CONFLICT DO UPDATE SET batch_id`` re-tagged the shared row
    onto ``runB``; ``get_batch(runA)`` then returned ONE row instead of two, the
    orchestrator's narrate loop never saw its chunk reach a terminal status, and it
    raised ``StagePreconditionError`` for a chunk that had in fact been synthesized.
    """
    tenant = "11111111-1111-1111-1111-111111111111"
    async with session_factory() as s:
        run_a = await jm.submit_batch(
            s,
            tenant_id=tenant,
            batch_id="runA",
            chunks=[_chunk("a1", "Shared line."), _chunk("a2", "Only in A.")],
        )
    # A's chunk is synthesized while A is still polling.
    for _ in range(2):
        async with session_factory() as s:
            job = await jm.claim_next(
                s, worker_id="w", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
            )
            assert job is not None
            await jm.complete(
                s, job_id=job["job_id"], result_key=f"t/{job['job_id']}.wav", duration_seconds=1.0
            )

    # Run B submits an OVERLAPPING book while A has not finished polling.
    async with session_factory() as s:
        run_b = await jm.submit_batch(
            s,
            tenant_id=tenant,
            batch_id="runB",
            chunks=[_chunk("b1", "Shared line."), _chunk("b2", "Only in B.")],
        )

    shared = {r["job_id"] for r in run_a} & {r["job_id"] for r in run_b}
    assert len(shared) == 1, "the two batches must actually share a job for this to test anything"

    async with session_factory() as s:
        a_rows = await jm.get_batch(s, tenant_id=tenant, batch_id="runA")
        b_rows = await jm.get_batch(s, tenant_id=tenant, batch_id="runB")

    # A still owns BOTH of its jobs — including the one B now also references.
    assert {r["job_id"] for r in a_rows} == {r["job_id"] for r in run_a}
    assert len(a_rows) == 2
    assert all(r["status"] == "done" for r in a_rows)
    # B sees the shared, already-done clip too — no re-synthesis, and no theft.
    assert {r["job_id"] for r in b_rows} == {r["job_id"] for r in run_b}
    b_shared = next(r for r in b_rows if r["job_id"] in shared)
    assert b_shared["status"] == "done" and b_shared["result_key"]

    # Dedupe HOLDS: 3 distinct job rows for 4 submitted chunks — "Shared line." was
    # synthesized exactly once and is NOT paid for twice.
    async with session_factory() as s:
        rows = list(await s.execute(sql("SELECT idempotency_key FROM tts_jobs")))
    assert len(rows) == 3
    assert len({r.idempotency_key for r in rows}) == 3


async def test_claim_marks_claimed_and_returns_inputs(session_factory) -> None:
    async with session_factory() as s:
        await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", "Read me.")],
        )
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="mini2", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
        )
    assert job is not None
    assert job["text"] == "Read me." and job["voice"] == "af_heart" and job["speed"] == 1.0
    async with session_factory() as s:
        rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
        )
    assert rows[0]["status"] == "claimed"


async def test_concurrent_claims_are_disjoint(session_factory) -> None:
    async with session_factory() as s:
        await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk(f"r{i}", f"Line {i}.") for i in range(6)],
        )

    async def claim_one() -> str | None:
        async with session_factory() as s:
            job = await jm.claim_next(
                s, worker_id="w", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
            )
            return job["job_id"] if job else None

    claimed = await asyncio.gather(*(claim_one() for _ in range(6)))
    got = [j for j in claimed if j]
    assert len(got) == len(set(got)) == 6  # SKIP LOCKED → no double-claim


async def test_expired_lease_is_reclaimed(session_factory) -> None:
    async with session_factory() as s:
        await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", "One.")],
        )
    # claim with a lease of 0s → immediately expired
    async with session_factory() as s:
        first = await jm.claim_next(
            s, worker_id="dead", lease_seconds=0, max_attempts=5, max_lifetime_seconds=0
        )
    assert first is not None
    async with session_factory() as s:
        second = await jm.claim_next(
            s, worker_id="alive", lease_seconds=60, max_attempts=5, max_lifetime_seconds=0
        )
    assert second is not None and second["job_id"] == first["job_id"]  # same job reclaimed


async def test_poison_job_fails_past_max_attempts(session_factory) -> None:
    async with session_factory() as s:
        await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", "Boom.")],
        )
    # claim + fail-requeue up to the cap, then the claim past the cap fails it
    for _ in range(3):
        async with session_factory() as s:
            job = await jm.claim_next(
                s, worker_id="w", lease_seconds=0, max_attempts=3, max_lifetime_seconds=0
            )
            if job:
                await jm.fail(s, job_id=job["job_id"], error="kaboom", max_attempts=3)
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
        )
        rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
        )
    assert rows[0]["status"] == "failed"


# ── Re-opening a terminally-failed job ──────────────────────────────────────────
#
# The test above stops exactly where the production defect began: it proves a row
# reaches ``failed`` and never RE-SUBMITS it. Because submit dedupes with
# ``ON CONFLICT DO NOTHING``, a failed row wins that conflict — the insert is a
# no-op, the read-back replays the stored ``error_message``, nothing is enqueued and
# no worker is ever called. That made the failure permanent and CONTENT-scoped:
# four orchestrator runs across 23 minutes all resolved to one ``job_id`` and
# replayed one stale 503, and only a manual UPDATE recovered it.
_T = "11111111-1111-1111-1111-111111111111"


async def _submit_and_fail(session_factory, *, text: str = "Boom.", batch_id: str = "b1") -> str:
    """Enqueue one chunk and drive it to TERMINAL ``failed`` in a single attempt."""
    async with session_factory() as s:
        jobs = await jm.submit_batch(
            s, tenant_id=_T, batch_id=batch_id, chunks=[_chunk("r1", text)]
        )
    job_id = jobs[0]["job_id"]
    assert job_id is not None
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w", lease_seconds=60, max_attempts=1, max_lifetime_seconds=0
        )
        assert job is not None
        # attempts is now 1 and max_attempts is 1, so _FAIL_SQL lands on 'failed'.
        await jm.fail(s, job_id=job_id, error="upstream 503 from kokoro", max_attempts=1)
    assert (await _row(session_factory, job_id))["status"] == "failed"
    return job_id


async def _resubmit(session_factory, *, text: str, batch_id: str, cap: int = 3) -> list[jm.JobRow]:
    async with session_factory() as s:
        jobs = await jm.submit_batch(
            s, tenant_id=_T, batch_id=batch_id, chunks=[_chunk("r1", text)]
        )
        return await jm.reopen_failed_jobs(s, tenant_id=_T, jobs=jobs, max_requeues=cap)


async def test_resubmitting_a_failed_job_reopens_it(session_factory) -> None:
    """The whole defect, end to end: a re-submit must SYNTHESIZE, not replay an error.

    Asserts observable state only — the status the caller is handed, the row the
    queue holds, the error the poll reports, and that a worker can actually claim the
    chunk again (which is what "re-queued" has to mean).
    """
    job_id = await _submit_and_fail(session_factory, text="Boom.")

    jobs = await _resubmit(session_factory, text="Boom.", batch_id="b2")

    # The caller is told queued, not failed, and gets the SAME row (dedupe intact).
    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["status"] == "queued"

    row = await _row(session_factory, job_id)
    assert row["status"] == "queued"
    assert row["attempts"] == 0, "a re-open is a NEW synthesis; a spent budget re-fails it"
    assert row["requeue_count"] == 1
    assert row["started_at"] is None and row["completed_at"] is None
    assert row["claimed_by"] is None

    # The stale error is GONE from the poll surface — the orchestrator raises
    # StagePreconditionError on any failed job and on the error it reports.
    async with session_factory() as s:
        polled = await jm.get_batch(s, tenant_id=_T, batch_id="b2")
    assert polled[0]["status"] == "queued"
    assert polled[0]["error"] is None

    # And it is genuinely claimable again — the production symptom was that NO
    # worker was ever called.
    async with session_factory() as s:
        claimed = await jm.claim_next(
            s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
    assert claimed is not None and claimed["job_id"] == job_id


async def test_reopening_stops_at_the_requeue_cap(session_factory) -> None:
    """The treadmill guard. ``attempts`` is RESET by the re-open, so it cannot bound
    how often the re-open happens; only the monotonic ``requeue_count`` can. Past the
    cap the row must behave exactly as it does today — stay ``failed`` and replay its
    stored error — or an orchestrator retrying a book every 20 minutes re-burns
    ``max_attempts`` GPU claims forever on text kokoro will never accept."""
    job_id = await _submit_and_fail(session_factory, text="Poison.")

    for expected in (1, 2):
        jobs = await _resubmit(session_factory, text="Poison.", batch_id=f"c{expected}", cap=2)
        assert jobs[0]["status"] == "queued"
        assert (await _row(session_factory, job_id))["requeue_count"] == expected
        # Fail it again, as a genuinely poison chunk does on every re-synthesis.
        async with session_factory() as s:
            await jm.claim_next(
                s, worker_id="w", lease_seconds=60, max_attempts=1, max_lifetime_seconds=0
            )
            await jm.fail(s, job_id=job_id, error="kokoro rejected the text", max_attempts=1)

    # Cap spent: the third re-submit gets today's behaviour back, deliberately.
    jobs = await _resubmit(session_factory, text="Poison.", batch_id="c3", cap=2)
    assert jobs[0]["status"] == "failed"
    row = await _row(session_factory, job_id)
    assert row["status"] == "failed"
    assert row["requeue_count"] == 2, "the counter is monotonic and was NOT incremented"
    assert row["error"] == "kokoro rejected the text"

    async with session_factory() as s:
        polled = await jm.get_batch(s, tenant_id=_T, batch_id="c3")
    assert polled[0]["error"] == "kokoro rejected the text"

    # Nothing is claimable: converged to permanently-failed, no GPU is burned.
    async with session_factory() as s:
        assert (
            await jm.claim_next(
                s, worker_id="w9", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
            is None
        )


async def test_two_concurrent_submits_reopen_a_failed_job_exactly_once(session_factory) -> None:
    """Both callers see ``failed`` and both issue the CAS. Postgres serializes them on
    the row lock; the loser matches 0 rows and must RE-READ rather than assume — a
    0-row outcome is also what a spent cap looks like. Exactly one re-open, one
    increment, one row, one claimable chunk."""
    job_id = await _submit_and_fail(session_factory, text="Race.")

    a, b = await asyncio.gather(
        _resubmit(session_factory, text="Race.", batch_id="runA"),
        _resubmit(session_factory, text="Race.", batch_id="runB"),
    )
    # Neither caller is told "failed" — the loser re-read the truth.
    assert a[0]["status"] == "queued"
    assert b[0]["status"] == "queued"
    assert a[0]["job_id"] == b[0]["job_id"] == job_id

    row = await _row(session_factory, job_id)
    assert row["requeue_count"] == 1, "a doubly-counted re-open spends the cap twice as fast"

    async with session_factory() as s:
        n = (
            await s.execute(
                sql("SELECT count(*) AS n FROM tts_jobs WHERE tenant_id = :t"), {"t": _T}
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
    assert second is None, "a doubly-reopened job would be synthesized twice"


async def test_a_done_job_is_deduped_and_never_reopened(session_factory) -> None:
    """Negative control. Re-opening is scoped to ``failed`` by the SQL guard; a
    ``done`` row must keep returning its cached clip. If this ever goes red the fix
    has started re-synthesizing every already-paid-for chunk in every book."""
    async with session_factory() as s:
        jobs = await jm.submit_batch(
            s, tenant_id=_T, batch_id="d1", chunks=[_chunk("r1", "Cached.")]
        )
    job_id = jobs[0]["job_id"]
    assert job_id is not None
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
        await jm.complete(s, job_id=job_id, result_key="t/abc.wav", duration_seconds=3.5)

    again = await _resubmit(session_factory, text="Cached.", batch_id="d2")

    assert again[0]["job_id"] == job_id
    assert again[0]["status"] == "done"
    row = await _row(session_factory, job_id)
    assert row["status"] == "done"
    assert row["requeue_count"] == 0
    async with session_factory() as s:
        assert (
            await jm.claim_next(
                s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
            is None
        ), "a done job must never become claimable again"


async def test_the_submit_route_itself_reopens_a_failed_dedupe_hit(
    session_factory, monkeypatch
) -> None:
    """Wiring gate. Every re-open test above calls ``jm.reopen_failed_jobs`` DIRECTLY,
    so all of them stay green if the submit route never calls it — the fix would ship
    INERT and nothing would notice it being deleted. (Measured: removing the call at
    ``src/api/routes/batches.py`` left the whole suite green.) This drives
    ``POST /api/tts/batches`` end to end against a real Postgres and asserts on the
    HTTP body plus the row the queue holds, so the ROUTE is what is under test.

    Same harness as ``tests/test_dedupe_clip_presence.py``'s sibling wiring gate; the
    storage probe is stubbed because a re-opened row is ``queued`` and must never be
    probed at all.
    """
    import httpx
    from fastapi import FastAPI

    from src.api.routes import batches
    from src.database.session import bind_tenant_every_transaction, get_tenant_session

    job_id = await _submit_and_fail(session_factory, text="Boom.")

    probed: list[str] = []

    class _Storage:
        async def exists(self, key: str) -> bool | None:
            probed.append(key)
            return True

    monkeypatch.setattr(batches, "get_storage", lambda: _Storage())

    app = FastAPI()
    app.include_router(batches.router)

    @app.middleware("http")
    async def _tenant_state(request, call_next):
        request.state.tenant_id = _T
        return await call_next(request)

    async def _session():
        # Bind ``app.current_tenant`` exactly as the production dependency does: the
        # re-open SELECT is tenant-scoped, so a bare session would exercise neither
        # the GUC nor RLS.
        async with session_factory() as s:
            bind_tenant_every_transaction(s, _T)
            yield s

    app.dependency_overrides[get_tenant_session] = _session

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/tts/batches",
            json={
                "batch_id": "route-reopen",
                "chunks": [{"ref": "r1", "text": "Boom.", "voice": "af_heart", "speed": 1.0}],
            },
        )
        assert resp.status_code == 202
        body = resp.json()["jobs"][0]
        assert body["job_id"] == job_id, "dedupe must still collapse onto the same row"
        assert body["status"] == "queued", (
            "the route published the dedupe hit's stale FAILED status — "
            "reopen_failed_jobs is not wired into submit_batch"
        )

        # The stale error is gone from the poll surface too, read through the ROUTE:
        # the orchestrator raises StagePreconditionError on any error it reports.
        polled = await client.get("/api/tts/batches/route-reopen")
    assert polled.status_code == 200
    polled_job = polled.json()["jobs"][0]
    assert polled_job["status"] == "queued"
    assert polled_job["error"] is None, "the route replayed the stale error_message"

    row = await _row(session_factory, job_id)
    assert row["status"] == "queued"
    assert row["attempts"] == 0, "a re-open is a NEW synthesis; a spent budget re-fails it"
    assert row["requeue_count"] == 1
    assert row["error"] is None
    assert not probed, "a re-opened row is queued and has no clip to probe"

    # And a worker can genuinely claim it again — the production symptom was that
    # no worker was ever called for this content again.
    async with session_factory() as s:
        claimed = await jm.claim_next(
            s, worker_id="w9", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
    assert claimed is not None and claimed["job_id"] == job_id


async def test_complete_records_result(session_factory) -> None:
    async with session_factory() as s:
        await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", "Done.")],
        )
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
        )
        assert job is not None
        ok = await jm.complete(
            s, job_id=job["job_id"], result_key="t1/abc.wav", duration_seconds=3.5
        )
    assert ok
    async with session_factory() as s:
        rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
        )
    assert rows[0]["status"] == "done"
    assert rows[0]["result_key"] == "t1/abc.wav" and rows[0]["duration_seconds"] == 3.5


async def _submit_one(session_factory, txt: str = "One.") -> str:
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[_chunk("r1", txt)],
        )
    job_id = out[0]["job_id"]
    # ``job_id`` is ``str | None`` in the JobRow contract (submit's read-back has a
    # defensive fallback); narrow it once here so the queue calls below stay typed.
    assert job_id is not None
    return job_id


async def test_release_refunds_the_attempt(session_factory) -> None:
    """A surrendered claim gives the attempt back — the core of the poison fix."""
    job_id = await _submit_one(session_factory)
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
    assert job is not None
    assert (await _row(session_factory, job_id))["attempts"] == 1  # the claim charged one

    async with session_factory() as s:
        assert await jm.release_claim(s, job_id=job_id, worker_id="w1") is True

    row = await _row(session_factory, job_id)
    assert row["status"] == "queued"  # immediately re-claimable, no lease wait
    assert row["claimed_by"] is None
    assert row["attempts"] == 0  # REFUNDED — the job was never given a verdict


async def test_repeated_rollouts_never_poison_a_healthy_job(session_factory) -> None:
    """THE PROD BUG, proven dead.

    tts-gateway-worker is a k8s RollingUpdate deployment and the kokoro workers are
    Macs that reboot. Before the refund, every interrupted claim burned an attempt,
    so max_attempts=4 meant FOUR rollouts permanently poisoned a perfectly healthy
    chunk. Six graceful claim/release cycles — well past the cap — must leave the
    job untouched and still synthesizable.
    """
    job_id = await _submit_one(session_factory, "Healthy chunk.")

    for _ in range(6):
        async with session_factory() as s:
            job = await jm.claim_next(
                s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
        assert job is not None, "a released job must always be re-claimable"
        async with session_factory() as s:
            assert await jm.release_claim(s, job_id=job_id, worker_id="w1") is True
        assert (await _row(session_factory, job_id))["status"] != "failed"

    # The 7th claim (a worker that actually survives) still gets the job.
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
    assert job is not None
    row = await _row(session_factory, job_id)
    assert row["status"] == "claimed"
    assert row["attempts"] == 1  # every rollout refunded; only this live claim counts


async def test_crashing_job_fails_on_max_lifetime(session_factory) -> None:
    """The refund removes `attempts` as a bound, so wall-clock life is the backstop."""
    job_id = await _submit_one(session_factory, "Never finishes.")
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w1", lease_seconds=0, max_attempts=4, max_lifetime_seconds=0
        )
    await _backdate_start(session_factory, job_id, hours=2)

    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=3600
        )
    assert job is None  # aged out instead of being handed to another worker

    row = await _row(session_factory, job_id)
    assert row["status"] == "failed"
    assert "max job lifetime" in (row["error"] or "")


async def test_lifetime_zero_disables_the_bound(session_factory) -> None:
    job_id = await _submit_one(session_factory, "Old but fine.")
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w1", lease_seconds=0, max_attempts=4, max_lifetime_seconds=0
        )
    await _backdate_start(session_factory, job_id, hours=2)

    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w2", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
    assert job is not None  # 0 = no bound, however old
    assert (await _row(session_factory, job_id))["status"] == "claimed"


async def test_lifetime_never_ages_out_a_backlogged_job(session_factory) -> None:
    """A never-claimed row has consumed nothing — ageing it out would mass-fail a backlog."""
    job_id = await _submit_one(session_factory, "Waiting in line.")
    assert (await _row(session_factory, job_id))["started_at"] is None

    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=1
        )
    assert job is not None  # exempt: the lifetime clock starts at the FIRST claim
    assert (await _row(session_factory, job_id))["status"] == "claimed"


async def test_release_after_complete_is_a_noop(session_factory) -> None:
    """A shutdown racing a just-finished job must not resurrect it."""
    job_id = await _submit_one(session_factory, "Finished.")
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
        await jm.complete(s, job_id=job_id, result_key="t1/x.wav", duration_seconds=1.0)

    async with session_factory() as s:
        assert await jm.release_claim(s, job_id=job_id, worker_id="w1") is False

    row = await _row(session_factory, job_id)
    assert row["status"] == "done"  # not dragged back to queued
    assert row["attempts"] == 1  # unchanged


async def test_release_by_a_different_worker_is_a_noop(session_factory) -> None:
    """After a lease expiry hands the job to w2, w1's late shutdown must not steal it."""
    job_id = await _submit_one(session_factory, "Owned by w1.")
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )

    async with session_factory() as s:
        assert await jm.release_claim(s, job_id=job_id, worker_id="w2") is False

    row = await _row(session_factory, job_id)
    assert row["status"] == "claimed" and row["claimed_by"] == "w1"
    assert row["attempts"] == 1


async def test_double_release_does_not_double_refund(session_factory) -> None:
    """Both the signal handler and atexit can fire; attempts must never go negative."""
    job_id = await _submit_one(session_factory, "Released twice.")
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )

    async with session_factory() as s:
        assert await jm.release_claim(s, job_id=job_id, worker_id="w1") is True
    async with session_factory() as s:
        assert await jm.release_claim(s, job_id=job_id, worker_id="w1") is False

    assert (await _row(session_factory, job_id))["attempts"] == 0  # floors at 0, never -1


async def test_three_arg_shim_keeps_an_old_pod_claiming_during_a_rollout(session_factory) -> None:
    """BOTH signatures must work — this is the rollout-safety pin. Do not delete.

    tts-gateway is a 2-replica RollingUpdate whose migration runs in an initContainer,
    so for the length of every deploy the OLD image (which calls the 3-arg
    claim_next_tts_job) is live against the ALREADY-migrated schema. If a future
    "cleanup" drops the 3-arg shim before the fleet is fully on the new image, every
    claim 500s for the duration of the rollout. This test fails first instead.

    The shim must also behave as lifetime-DISABLED (it delegates with 0), i.e. exactly
    the semantics the old image was written against — while the 4-arg form on the very
    same row enforces the bound.
    """
    job_id = await _submit_one(session_factory, "Old pod's job.")

    # Age the job well past any bound, via a first claim that leaves an expired lease.
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="w1", lease_seconds=0, max_attempts=4, max_lifetime_seconds=0
        )
    await _backdate_start(session_factory, job_id, hours=2)

    # An OLD pod calls the 3-arg signature directly — it must still claim, and must NOT
    # be subject to a lifetime bound it never knew about.
    async with session_factory() as s:
        row = (
            await s.execute(
                sql("SELECT * FROM claim_next_tts_job(:w, :lease, :max)"),
                {"w": "old-pod", "lease": 0, "max": 4},
            )
        ).first()
        await s.commit()
    assert row is not None, "the 3-arg shim was dropped — this breaks every rollout"
    assert row.job_id == job_id
    assert (await _row(session_factory, job_id))["status"] == "claimed"

    # A NEW pod, same 2h-old row, 4-arg form → the bound IS enforced.
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="new-pod", lease_seconds=60, max_attempts=4, max_lifetime_seconds=3600
        )
    assert job is None
    assert (await _row(session_factory, job_id))["status"] == "failed"


async def _claim_expiring(session_factory, worker_id: str) -> dict | None:
    """Claim with an already-expired lease, so the next claim reclaims it."""
    async with session_factory() as s:
        return await jm.claim_next(
            s, worker_id=worker_id, lease_seconds=0, max_attempts=4, max_lifetime_seconds=0
        )


async def test_complete_by_a_stale_worker_cannot_finish_another_workers_claim(
    session_factory,
) -> None:
    """A hung worker's late result must not publish audio for a claim it has LOST.

    A claims → A's synthesis hangs past the lease → B reclaims → A finally returns and
    POSTs /result. Keyed on job_id alone, A's write lands on B's ACTIVE claim.
    """
    job_id = await _submit_one(session_factory, "Slow synth.")
    assert await _claim_expiring(session_factory, "A") is not None
    async with session_factory() as s:  # B reclaims the expired lease
        assert (
            await jm.claim_next(
                s, worker_id="B", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
            is not None
        )
    assert (await _row(session_factory, job_id))["claimed_by"] == "B"

    async with session_factory() as s:  # A's stale completion
        ok = await jm.complete(
            s, job_id=job_id, result_key="t1/a.wav", duration_seconds=9.9, worker_id="A"
        )
    assert ok is False  # benign no-op, not a completion

    row = await _row(session_factory, job_id)
    assert row["status"] == "claimed" and row["claimed_by"] == "B"  # B still owns it

    async with session_factory() as s:  # and B can still finish its own job
        assert (
            await jm.complete(
                s, job_id=job_id, result_key="t1/b.wav", duration_seconds=4.0, worker_id="B"
            )
            is True
        )
    row = await _row(session_factory, job_id)
    assert row["status"] == "done"
    async with session_factory() as s:
        batch = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
        )
    assert batch[0]["result_key"] == "t1/b.wav"  # B's audio, never A's


async def test_keyinfo_exposes_the_current_owner(session_factory) -> None:
    """``get_job_keyinfo`` must report ``claimed_by``, not just the key parts.

    The result route builds the clip key from this row and then writes to the
    object store. That write is only safe if the caller still owns the claim, and
    the route cannot know that unless this query returns the owner — the DB-level
    ``claimed_by`` guard on ``complete()`` fires far too late, after the bytes have
    already replaced the owner's clip under the deterministic key.
    """
    job_id = await _submit_one(session_factory, "Who owns me.")
    async with session_factory() as s:
        info = await jm.get_job_keyinfo(s, job_id=job_id)
    assert info is not None
    assert info["claimed_by"] is None, "an unclaimed job has no owner"

    assert await _claim_expiring(session_factory, "A") is not None
    async with session_factory() as s:
        info = await jm.get_job_keyinfo(s, job_id=job_id)
    assert info is not None
    assert info["claimed_by"] == "A"

    async with session_factory() as s:  # B reclaims the expired lease
        assert (
            await jm.claim_next(
                s, worker_id="B", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
            is not None
        )
    async with session_factory() as s:
        info = await jm.get_job_keyinfo(s, job_id=job_id)
    assert info is not None
    assert info["claimed_by"] == "B", (
        "the row must name the LIVE owner — this is what lets the route refuse a "
        "stale worker's write before it reaches MinIO"
    )
    # And the key parts the route still needs are unchanged.
    assert info["tenant_id"] == "11111111-1111-1111-1111-111111111111" and info["idempotency_key"]


async def test_fail_by_a_stale_worker_cannot_requeue_another_workers_claim(
    session_factory,
) -> None:
    """A stale failure must not requeue — or at the cap, POISON — B's active claim."""
    job_id = await _submit_one(session_factory, "Slow synth.")
    assert await _claim_expiring(session_factory, "A") is not None
    async with session_factory() as s:
        assert (
            await jm.claim_next(
                s, worker_id="B", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
            is not None
        )

    async with session_factory() as s:  # A's stale failure arrives
        status = await jm.fail(s, job_id=job_id, error="A timed out", max_attempts=4, worker_id="A")
    assert status == "unknown"  # matched nothing — same shape, no 4xx to retry on

    row = await _row(session_factory, job_id)
    assert row["status"] == "claimed" and row["claimed_by"] == "B"  # B works on, undisturbed
    assert row["error"] is None  # A's error never written


async def test_legacy_worker_without_worker_id_behaves_exactly_as_today(session_factory) -> None:
    """The compat path: a not-yet-redeployed Mac worker sends no worker_id.

    It MUST keep working unguarded — CI cannot redeploy those launchd units, so a
    required worker_id would instantly stop every old worker reporting results.
    """
    job_id = await _submit_one(session_factory, "Legacy path.")
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="old-mac", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )

    async with session_factory() as s:  # no worker_id → unguarded, as before
        assert (
            await jm.complete(s, job_id=job_id, result_key="t1/legacy.wav", duration_seconds=2.0)
            is True
        )
    assert (await _row(session_factory, job_id))["status"] == "done"

    # and the legacy fail path still requeues an under-cap job
    job2 = await _submit_one(session_factory, "Legacy fail.")
    async with session_factory() as s:
        await jm.claim_next(
            s, worker_id="old-mac", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
    async with session_factory() as s:
        assert await jm.fail(s, job_id=job2, error="boom", max_attempts=4) == "queued"
    assert (await _row(session_factory, job2))["status"] == "queued"


async def test_resolver_set_matches_the_migrated_schema(session_factory) -> None:
    """The create_all path and the alembic path must converge on the SAME functions.

    Migrations now inline frozen SQL snapshots (a migration is a historical record and must
    never import live code — importing it makes an old revision replay today's code on a
    fresh DB). The price of freezing is drift: someone edits queue_resolvers.py and forgets
    the migration. This pins the create_all end-state — which is what conftest builds from
    the LIVE CLAIM_FN_DDL — to exactly the set `0002` creates, so drift fails here.
    """
    async with session_factory() as s:
        rows = (
            await s.execute(
                sql(
                    "SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS args, "
                    "       l.lanname "
                    "FROM pg_proc p "
                    "JOIN pg_namespace n ON n.oid = p.pronamespace "
                    "JOIN pg_language l ON l.oid = p.prolang "
                    "WHERE n.nspname = 'public' ORDER BY p.proname, args"
                )
            )
        ).all()

    assert [(r.proname, r.args, r.lanname) for r in rows] == [
        # the 3-arg rollout shim — SQL, delegates to the 4-arg with lifetime 0
        (
            "claim_next_tts_job",
            "p_worker_id text, p_lease_seconds integer, p_max_attempts integer",
            "sql",
        ),
        # the pre-#45 4-arg signature — now a SQL shim (busy backstop disabled)
        (
            "claim_next_tts_job",
            "p_worker_id text, p_lease_seconds integer, p_max_attempts integer, "
            "p_max_lifetime_seconds integer",
            "sql",
        ),
        # the real claim fn (0006: + absolute busy backstop)
        (
            "claim_next_tts_job",
            "p_worker_id text, p_lease_seconds integer, p_max_attempts integer, "
            "p_max_lifetime_seconds integer, p_max_busy_age_seconds integer",
            "plpgsql",
        ),
        # the refund
        ("release_tts_claim", "p_job_id text, p_worker_id text", "plpgsql"),
        # the GPU-busy refund (0006): also clears started_at, stamps busy_since
        ("release_tts_claim_busy", "p_job_id text, p_worker_id text", "plpgsql"),
    ]


async def test_fairness_interleaves_tenants(session_factory) -> None:
    # tenant A submits a big book first; tenant B a single job after.
    async with session_factory() as s:
        await jm.submit_batch(
            s,
            tenant_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            batch_id="a",
            chunks=[_chunk(f"a{i}", f"A{i}.") for i in range(5)],
        )
        await jm.submit_batch(
            s,
            tenant_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            batch_id="b",
            chunks=[_chunk("b0", "B0.")],
        )
    # first claim holds an A lease; the SECOND claim should prefer B (fewest leases)
    async with session_factory() as s:
        first = await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
        )
    async with session_factory() as s:
        second = await jm.claim_next(
            s, worker_id="w2", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
        )
    assert first is not None and second is not None
    assert {first["text"][0], second["text"][0]} == {"A", "B"}  # not both A
