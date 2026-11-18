"""A failure report must never act on a job the gateway already finished.

``worker/worker.py::_process`` wraps synthesis, the ``POST /result`` and its
``raise_for_status()`` in ONE ``try``, and its ``except`` posts ``/fail``. So a
result upload that lands server-side but whose 204 the worker never sees — a read
timeout on a multi-MB clip, a reset connection, the gateway pod rolling right
after COMMIT — makes the worker report a failure for a job that is already
``done``. ``_COMPLETE_SQL`` leaves ``claimed_by`` set, so even the ``worker_id``
guard matches: same job, same worker.

Two distinct harms, one per test below: the finished chunk is dragged back to
``queued`` (its stored clip re-synthesized from scratch — paid GPU time for audio
that already exists), or, once ``attempts`` has reached the cap, flipped straight
to ``failed`` while its good ``result_key`` sits in MinIO. The orchestrator's
audiobook narrate stage raises ``StagePreconditionError`` on ANY failed job, so
one lost 204 aborts a whole book.
"""

from __future__ import annotations

from sqlalchemy import text as sql

from src.services import job_manager as jm


async def _row(session_factory, job_id: str) -> dict:
    async with session_factory() as s:
        r = (
            await s.execute(
                sql(
                    "SELECT status, attempts, claimed_by, result_key, completed_at "
                    "FROM tts_jobs WHERE job_id = :j"
                ),
                {"j": job_id},
            )
        ).first()
    assert r is not None
    return {
        "status": r.status,
        "attempts": r.attempts,
        "claimed_by": r.claimed_by,
        "result_key": r.result_key,
        "completed_at": r.completed_at,
    }


async def _claim_and_complete(session_factory, *, text: str, attempts_before: int = 0) -> str:
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[{"ref": "r1", "text": text, "voice": "af_heart", "speed": 1.0}],
        )
    job_id = out[0]["job_id"]
    # ``job_id`` is ``str | None`` in the JobRow contract (submit's read-back has a
    # defensive fallback); narrow it once here so the queue calls below stay typed.
    assert job_id is not None
    if attempts_before:
        async with session_factory() as s:
            await s.execute(
                sql("UPDATE tts_jobs SET attempts = :a WHERE job_id = :j"),
                {"a": attempts_before, "j": job_id},
            )
            await s.commit()
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=99, max_lifetime_seconds=0
        )
        assert job is not None and job["job_id"] == job_id
        assert await jm.complete(
            s, job_id=job_id, result_key="t1/clip.wav", duration_seconds=9.0, worker_id="w1"
        )
    return job_id


async def test_late_fail_cannot_requeue_a_completed_job(session_factory) -> None:
    job_id = await _claim_and_complete(session_factory, text="Chapter one, narrated.")

    # PRECONDITION: the job really is finished, and `claimed_by` is still this worker
    # — which is exactly why the owner guard does not save us here.
    done = await _row(session_factory, job_id)
    assert done["status"] == "done", done
    assert done["claimed_by"] == "w1", done
    assert done["result_key"] == "t1/clip.wav", done

    # The worker's `except` branch: it never saw the 204, so it reports a failure.
    async with session_factory() as s:
        status = await jm.fail(
            s,
            job_id=job_id,
            error="httpx.ReadTimeout on POST /result",
            max_attempts=4,
            worker_id="w1",
        )

    assert status == "unknown", (
        f"fail() reported {status!r} for an already-done job — it matched and rewrote it"
    )
    after = await _row(session_factory, job_id)
    assert after["status"] == "done", (
        "a completed job was dragged back to the queue by a late failure report; "
        f"row is now {after!r} — the stored clip will be re-synthesized"
    )
    assert after["completed_at"] is not None
    assert after["result_key"] == "t1/clip.wav"

    # And the batch poll the orchestrator reads must still show the finished chunk.
    async with session_factory() as s:
        rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
        )
    assert [r["status"] for r in rows] == ["done"], rows


async def test_late_fail_at_the_attempt_cap_cannot_destroy_a_completed_job(
    session_factory,
) -> None:
    """The same lost 204, but on a chunk that has already used its attempts.

    Here the CASE arm resolves to ``'failed'``, so the job does not merely get
    re-rendered — it lands in a terminal state the orchestrator treats as fatal for
    the entire book, with valid audio already in storage.
    """
    job_id = await _claim_and_complete(
        session_factory, text="Chapter two, narrated.", attempts_before=3
    )

    done = await _row(session_factory, job_id)
    assert done["status"] == "done", done
    # PRECONDITION: attempts is at/over the cap the worker will report against, so a
    # matching UPDATE would take the 'failed' arm — this is the destructive case.
    assert done["attempts"] >= 4, done

    async with session_factory() as s:
        status = await jm.fail(
            s,
            job_id=job_id,
            error="httpx.ReadTimeout on POST /result",
            max_attempts=4,
            worker_id="w1",
        )

    assert status == "unknown", status
    after = await _row(session_factory, job_id)
    assert after["status"] == "done", (
        "a completed chunk with stored audio was marked FAILED by a late failure "
        f"report; row is now {after!r} — the orchestrator aborts the whole book"
    )


async def test_a_legacy_worker_without_worker_id_also_cannot_undo_a_completion(
    session_factory,
) -> None:
    """The Mac workers still POST ``/fail`` without ``worker_id`` mid-rollout.

    For them there is no owner guard at all, so the status guard is the ONLY thing
    standing between a stray failure report and a finished job.
    """
    job_id = await _claim_and_complete(session_factory, text="Chapter three, narrated.")
    assert (await _row(session_factory, job_id))["status"] == "done"

    async with session_factory() as s:
        status = await jm.fail(s, job_id=job_id, error="boom", max_attempts=4)

    assert status == "unknown", status
    assert (await _row(session_factory, job_id))["status"] == "done"


async def test_a_genuine_failure_on_a_claimed_job_still_requeues(session_factory) -> None:
    """Guard rail: the status guard must not blunt real failure reporting."""
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b1",
            chunks=[{"ref": "r1", "text": "Boom.", "voice": "af_heart", "speed": 1.0}],
        )
    job_id = out[0]["job_id"]
    # ``job_id`` is ``str | None`` in the JobRow contract (submit's read-back has a
    # defensive fallback); narrow it once here so the queue calls below stay typed.
    assert job_id is not None
    async with session_factory() as s:
        assert (
            await jm.claim_next(
                s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
            )
            is not None
        )
    # PRECONDITION: the job is in flight, so this failure is a real verdict.
    assert (await _row(session_factory, job_id))["status"] == "claimed"

    async with session_factory() as s:
        assert await jm.fail(s, job_id=job_id, error="kokoro blew up", max_attempts=4) == "queued"
    assert (await _row(session_factory, job_id))["status"] == "queued"


async def _claimed_job(session_factory, *, text: str) -> str:
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id="11111111-1111-1111-1111-111111111111",
            batch_id="b-terminal",
            chunks=[{"ref": "r1", "text": text, "voice": "af_heart", "speed": 1.0}],
        )
    job_id = out[0]["job_id"]
    assert job_id is not None
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w1", lease_seconds=60, max_attempts=4, max_lifetime_seconds=0
        )
        assert job is not None and job["job_id"] == job_id
    return job_id


async def test_terminal_fail_marks_failed_with_attempts_left(session_factory) -> None:
    """#45: a 402 customer-config refusal must not be re-claimed 3 more times."""
    job_id = await _claimed_job(session_factory, text="Terminal: a byo_required tenant.")
    assert (await _row(session_factory, job_id))["attempts"] == 1  # 3 lives left

    async with session_factory() as s:
        status = await jm.fail(
            s, job_id=job_id, error="402", max_attempts=4, worker_id="w1", terminal=True
        )

    assert status == "failed"
    row = await _row(session_factory, job_id)
    assert row["status"] == "failed" and row["completed_at"] is not None


async def test_fail_route_forwards_terminal(session_factory) -> None:
    """The /fail route must pass ``terminal`` through, not drop it on the floor."""
    from src.api.routes.workers import report_failure

    job_id = await _claimed_job(session_factory, text="Terminal via the route handler.")
    async with session_factory() as s:
        out = await report_failure(job_id, error="402", worker_id="w1", terminal=True, session=s)
    assert out == {"status": "failed"}
    assert (await _row(session_factory, job_id))["status"] == "failed"
