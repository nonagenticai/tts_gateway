"""GPU-busy releases must never end in a terminal lifetime failure (tts_gateway#45).

``release_tts_claim`` refunds the attempt but keeps ``started_at``; the claim
function's lifetime arm then failed the job once ``started_at`` was older than
``max_job_lifetime_seconds``. A worker releasing on every sustained GPU-busy spell
therefore hit "exceeded max job lifetime" after ~4 cycles. The busy release clears
``started_at`` and arms an absolute backstop (``busy_since`` + ``max_busy_age``)
that fails with the stable, re-drivable reason ``gpu_busy_exhausted``.

Time is simulated by back-dating timestamps in the row, not by sleeping.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import text as sql

from src.api.routes import workers as routes
from src.services import job_manager as jm

LIFE = 3600
BUSY_AGE = 86400


async def _submit(session_factory, text: str) -> str:
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id="22222222-2222-2222-2222-222222222222",
            batch_id="busy",
            chunks=[{"ref": "r1", "text": text, "voice": "af_heart", "speed": 1.0}],
        )
    job_id = out[0]["job_id"]
    assert job_id is not None
    return job_id


async def _claim(session_factory, worker: str = "w1") -> dict | None:
    async with session_factory() as s:
        return await jm.claim_next(
            s,
            worker_id=worker,
            lease_seconds=60,
            max_attempts=4,
            max_lifetime_seconds=LIFE,
            max_busy_age_seconds=BUSY_AGE,
        )


async def _row(session_factory, job_id: str):
    async with session_factory() as s:
        r = (
            await s.execute(
                sql(
                    "SELECT status, attempts, started_at, busy_since, error_message, "
                    "claimed_by FROM tts_jobs WHERE job_id = :j"
                ),
                {"j": job_id},
            )
        ).first()
    assert r is not None
    return r


async def _age_started(session_factory, job_id: str, seconds: int) -> None:
    """Simulate `seconds` of wall clock passing while the claim is held (busy wait)."""
    async with session_factory() as s:
        await s.execute(
            sql(
                "UPDATE tts_jobs SET started_at = started_at - make_interval(secs => :n) "
                "WHERE job_id = :j"
            ),
            {"n": seconds, "j": job_id},
        )
        await s.commit()


async def _release(session_factory, job_id: str, *, busy: bool, worker: str = "w1") -> bool:
    async with session_factory() as s:
        return await jm.release_claim(s, job_id=job_id, worker_id=worker, busy=busy)


async def test_six_busy_cycles_past_the_lifetime_still_requeue(session_factory) -> None:
    job_id = await _submit(session_factory, "Busy six times over an hour and a half.")

    for cycle in range(6):  # 6 x 1000 s = 6000 s of busy, well past LIFE=3600
        job = await _claim(session_factory)
        assert job is not None and job["job_id"] == job_id, f"cycle {cycle}: not re-claimed"
        await _age_started(session_factory, job_id, 1000)
        assert await _release(session_factory, job_id, busy=True)
        row = await _row(session_factory, job_id)
        assert row.status == "queued", f"cycle {cycle}: {row.error_message}"
        assert row.started_at is None, "busy time must not count toward lifetime"
        assert row.attempts == 0, "the busy release refunds the attempt"
        assert row.busy_since is not None

    assert (await _claim(session_factory)) is not None  # still alive and claimable


async def test_control_a_plain_release_does_hit_the_lifetime(session_factory) -> None:
    """Control: the same churn through the SIGTERM release DOES age out — the defect."""
    job_id = await _submit(session_factory, "Plain releases keep the lifetime clock.")
    for _ in range(4):
        assert (await _claim(session_factory)) is not None
        await _age_started(session_factory, job_id, 1000)
        assert await _release(session_factory, job_id, busy=False)
    assert (await _claim(session_factory)) is None
    row = await _row(session_factory, job_id)
    assert row.status == "failed"
    assert row.error_message.startswith("exceeded max job lifetime")


async def test_absolute_busy_cap_fails_with_gpu_busy_exhausted(session_factory) -> None:
    job_id = await _submit(session_factory, "Busy for more than a day.")
    assert (await _claim(session_factory)) is not None
    assert await _release(session_factory, job_id, busy=True)
    async with session_factory() as s:
        await s.execute(
            sql(
                "UPDATE tts_jobs SET busy_since = now() - make_interval(secs => :n) "
                "WHERE job_id = :j"
            ),
            {"n": BUSY_AGE + 60, "j": job_id},
        )
        await s.commit()

    assert (await _claim(session_factory)) is None
    row = await _row(session_factory, job_id)
    assert row.status == "failed"
    assert row.error_message.startswith("gpu_busy_exhausted"), row.error_message
    assert row.claimed_by is None


async def test_busy_age_below_the_cap_still_claims(session_factory) -> None:
    job_id = await _submit(session_factory, "Busy for most of a day.")
    assert (await _claim(session_factory)) is not None
    assert await _release(session_factory, job_id, busy=True)
    async with session_factory() as s:
        await s.execute(
            sql(
                "UPDATE tts_jobs SET busy_since = now() - make_interval(secs => :n) "
                "WHERE job_id = :j"
            ),
            {"n": BUSY_AGE - 600, "j": job_id},
        )
        await s.commit()
    assert (await _claim(session_factory)) is not None


async def test_plain_release_keeps_started_at_and_does_not_arm_busy(session_factory) -> None:
    job_id = await _submit(session_factory, "SIGTERM handoff.")
    assert (await _claim(session_factory)) is not None
    assert await _release(session_factory, job_id, busy=False)
    row = await _row(session_factory, job_id)
    assert row.started_at is not None
    assert row.busy_since is None


async def test_release_route_maps_reason_gpu_busy(session_factory) -> None:
    job_id = await _submit(session_factory, "Route-level busy release.")
    assert (await _claim(session_factory)) is not None
    async with session_factory() as s:
        out = await routes.release_claim(job_id, worker_id="w1", reason="gpu_busy", session=s)
    assert out == {"released": True}
    row = await _row(session_factory, job_id)
    assert row.started_at is None and row.busy_since is not None


async def test_reopen_clears_busy_since_so_exhausted_is_redrivable(session_factory) -> None:
    job_id = await _submit(session_factory, "Re-driven after gpu_busy_exhausted.")
    assert (await _claim(session_factory)) is not None
    assert await _release(session_factory, job_id, busy=True)
    async with session_factory() as s:
        await s.execute(
            sql("UPDATE tts_jobs SET busy_since = now() - interval '2 days' WHERE job_id = :j"),
            {"j": job_id},
        )
        await s.commit()
    assert (await _claim(session_factory)) is None
    assert (await _row(session_factory, job_id)).status == "failed"

    # A later submit of the same content re-opens the failed row (the submit route
    # calls reopen_failed_jobs on its dedupe hits).
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id="22222222-2222-2222-2222-222222222222",
            batch_id="busy",
            chunks=[
                {
                    "ref": "r1",
                    "text": "Re-driven after gpu_busy_exhausted.",
                    "voice": "af_heart",
                    "speed": 1.0,
                }
            ],
        )
        assert out[0]["status"] == "failed"
        await jm.reopen_failed_jobs(
            s, tenant_id="22222222-2222-2222-2222-222222222222", jobs=out, max_requeues=3
        )
        await s.commit()
    row = await _row(session_factory, job_id)
    assert row.status == "queued" and row.busy_since is None
    assert (await _claim(session_factory)) is not None


# ---------------------------------------------------------------- terminal guard


async def test_terminal_fail_without_worker_id_is_rejected(session_factory) -> None:
    job_id = await _submit(session_factory, "Unguarded terminal fail.")
    assert (await _claim(session_factory)) is not None
    async with session_factory() as s:
        with pytest.raises(HTTPException) as exc:
            await routes.report_failure(
                job_id, error="402", worker_id=None, terminal=True, session=s
            )
    assert exc.value.status_code == 422
    assert (await _row(session_factory, job_id)).status == "claimed", "job left untouched"


# ------------------------------------------------------------- heartbeat owner


async def test_heartbeat_from_another_worker_is_refused(session_factory) -> None:
    job_id = await _submit(session_factory, "Owner-guarded heartbeat.")
    assert (await _claim(session_factory, worker="owner")) is not None
    async with session_factory() as s:
        with pytest.raises(HTTPException) as exc:
            await routes.heartbeat(job_id, worker_id="intruder", session=s)
    assert exc.value.status_code == 409
    async with session_factory() as s:
        await routes.heartbeat(job_id, worker_id="owner", session=s)  # no raise
    async with session_factory() as s:
        # legacy worker without worker_id keeps the unguarded behaviour
        await routes.heartbeat(job_id, worker_id=None, session=s)


async def test_claim_route_reports_the_lease(session_factory, monkeypatch) -> None:
    await _submit(session_factory, "Lease travels with the claim.")
    from src.api.schemas import ClaimRequest

    async with session_factory() as s:
        resp = await routes.claim(ClaimRequest(worker_id="w1"), session=s)
    import json

    payload = json.loads(bytes(resp.body))
    assert payload["lease_seconds"] == routes.get_settings().queue.lease_seconds


async def test_claim_route_applies_the_busy_backstop(session_factory) -> None:
    """The /claim route must pass max_busy_age_seconds, or the backstop is dead code."""
    from src.api.schemas import ClaimRequest

    job_id = await _submit(session_factory, "Route-level busy backstop.")
    assert (await _claim(session_factory)) is not None
    assert await _release(session_factory, job_id, busy=True)
    age = routes.get_settings().queue.max_busy_age_seconds
    assert age > 0
    async with session_factory() as s:
        await s.execute(
            sql(
                "UPDATE tts_jobs SET busy_since = now() - make_interval(secs => :n) "
                "WHERE job_id = :j"
            ),
            {"n": age + 60, "j": job_id},
        )
        await s.commit()

    async with session_factory() as s:
        resp = await routes.claim(ClaimRequest(worker_id="w1"), session=s)
    assert resp.status_code == 204
    assert (await _row(session_factory, job_id)).error_message.startswith("gpu_busy_exhausted")
