"""A worker's self-reported ``duration_seconds`` must not be able to KILL the job.

``duration_seconds`` lands in ``TtsJob.duration_seconds`` — ``Numeric(9, 3)``, ceiling
999999.999. Unbounded at the route, a worker that reports 1e9 (or ``inf``/``nan``, which
``float()`` parses out of a form field without complaint) reaches the UPDATE in
``job_manager.complete`` and raises a psycopg ``DataError`` → HTTP 500. ``worker.py``
calls ``raise_for_status()``, converts that into POST ``/fail``, and the queue requeues
the job — so the NEXT claim charges another attempt. Repeat to ``max_attempts`` and a
job whose AUDIO WAS FINE lands in ``status='failed'``, which makes the orchestrator's
audiobook narrate stage raise ``StagePreconditionError`` and abort the whole run.

So the assertions here are not about the status code. They are about the job ROW: a
rejected report must leave ``attempts`` and ``status`` exactly as it found them. A
422-only test would still pass against the 500 regression's downstream cost.

The bound is one-sided ON PURPOSE. There is NO lower bound: ``tts_backend`` emits
``duration_sec=0.0`` on a live path with real audio, so a non-positive duration is a
normal production event that is logged and accepted (``tests/test_result_upload_guards``
pins the log). ``test_non_positive_duration_is_still_accepted`` pins the absence of a
lower bound so a later "symmetric bounds" tidy-up cannot quietly introduce the very
failure mode the upper bound exists to prevent. The clip is ACCEPTED; the unusable
number is normalised to NULL rather than stored (see
``tests/test_duration_unmeasured_is_not_zero.py``).

Unlike ``test_result_upload_guards`` this module runs against the REAL Postgres queue
(``session_factory`` from conftest) — ``attempts`` lives nowhere else. Only storage is
stubbed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text as sql

from src.api.routes import workers
from src.api.schemas import MAX_DURATION_SECONDS
from src.database.session import get_raw_session
from src.services import job_manager as jm

# tts_jobs.tenant_id is a real uuid column (migration 0003), so a tenant is a uuid.
TENANT = "11111111-1111-1111-1111-111111111111"

MAX_BYTES = 4096
AUDIO = b"RIFF-real-audio"
WORKER_ID = "mini2"

# The Numeric(9,3) ceiling itself — must be ACCEPTED, it round-trips exactly.
COLUMN_MAX = 999999.999
# Under 1e6, yet fatal: Postgres rounds to the declared scale BEFORE checking
# precision, so this becomes 1000000.000 → 10 digits → `numeric field overflow`.
# This is the value that proves the bound is anchored on the column ceiling and
# not on a lazy `lt=1e6`.
ROUNDS_PAST_SCALE = 999999.9995


class _RecordingStorage:
    """Stands in for ClipStorage; records what actually reached the object store."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, int]] = []

    def key_for(self, tenant_id: str, idempotency_key: str) -> str:
        return f"{tenant_id}/{idempotency_key}.wav"

    async def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        self.puts.append((key, len(data)))
        return key


@pytest.fixture
def storage() -> _RecordingStorage:
    return _RecordingStorage()


@pytest_asyncio.fixture
async def api(monkeypatch, session_factory, storage) -> AsyncIterator[httpx.AsyncClient]:
    """The workers router over the REAL queue DB; only MinIO is stubbed."""
    monkeypatch.setattr(workers, "get_storage", lambda: storage)
    monkeypatch.setattr(
        workers, "get_settings", lambda: SimpleNamespace(max_result_bytes=MAX_BYTES)
    )

    app = FastAPI()
    app.include_router(workers.router)

    async def _session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_raw_session] = _session
    # Worker-identity authz is covered in tests/test_worker_authz.py; bypass it here so
    # these cases reach the duration bound instead of 403-ing at the router dependency.
    app.dependency_overrides[workers.authorize_worker] = lambda: None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tts.test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def claimed_job(session_factory) -> str:
    """A real row taken through submit → claim, so ``attempts`` is already 1.

    ``claim_next_tts_job`` charges an attempt on EVERY claim, so 1 is the correct
    baseline and any increment past it is the burn this whole module guards.
    """
    async with session_factory() as session:
        submitted = await jm.submit_batch(
            session,
            tenant_id=TENANT,
            batch_id="b1",
            chunks=[{"ref": "r1", "text": "Hello.", "voice": "af_heart", "speed": 1.0}],
        )
    async with session_factory() as session:
        claimed = await jm.claim_next(
            session,
            worker_id=WORKER_ID,
            lease_seconds=180,
            max_attempts=4,
            max_lifetime_seconds=3600,
        )
    assert claimed is not None, "fixture setup: the submitted job must be claimable"
    assert claimed["job_id"] == submitted[0]["job_id"]
    return str(submitted[0]["job_id"])


async def _row(session_factory, job_id: str) -> dict:
    """Raw queue-row state — attempts/status are not exposed by any API response."""
    async with session_factory() as session:
        r = (
            await session.execute(
                sql("SELECT status, attempts, duration_seconds FROM tts_jobs WHERE job_id = :j"),
                {"j": job_id},
            )
        ).first()
    assert r is not None
    return {
        "status": r.status,
        "attempts": r.attempts,
        "duration": float(r.duration_seconds) if r.duration_seconds is not None else None,
    }


async def _report(api: httpx.AsyncClient, job_id: str, duration: str) -> httpx.Response:
    return await api.post(
        f"/api/tts/jobs/{job_id}/result",
        files={"audio": ("clip.wav", AUDIO, "audio/wav")},
        data={"duration_seconds": duration, "worker_id": WORKER_ID},
    )


# ── Rejected: unstorable values never reach the UPDATE ───────────────────────


@pytest.mark.parametrize(
    "duration",
    ["1e9", "1000000.0", "999999.9995", "inf", "-inf", "nan"],
    ids=["1e9", "one-million", "rounds-past-scale", "inf", "-inf", "nan"],
)
async def test_unstorable_duration_is_422_and_does_not_burn_an_attempt(
    api, session_factory, storage, claimed_job, duration: str
) -> None:
    """THE BUG: these values used to 500, and the 500 cost the job a life.

    422 is only half the requirement. The half that matters is that the job row is
    untouched — same ``attempts``, still ``claimed`` — so the worker's retry (or the
    lease expiry) finds a job with all of its lives intact instead of one that has
    been walked toward ``status='failed'`` by a number.
    """
    before = await _row(session_factory, claimed_job)
    assert before == {"status": "claimed", "attempts": 1, "duration": None}

    r = await _report(api, claimed_job, duration)

    assert r.status_code == 422, (
        f"duration_seconds={duration!r} is unstorable in Numeric(9,3); it must be a "
        f"deterministic client error, not a DataError-driven 500 the worker reports as /fail"
    )
    after = await _row(session_factory, claimed_job)
    assert after["attempts"] == before["attempts"], (
        f"rejecting duration_seconds={duration!r} burned an attempt "
        f"({before['attempts']} → {after['attempts']}) — repeat to max_attempts and a job "
        f"with perfectly good audio is poisoned by a bad number"
    )
    assert after["status"] == "claimed", "a rejected report must not requeue or fail the job"
    assert after["duration"] is None
    assert storage.puts == [], "nothing may reach MinIO on a rejected report"


# ── Accepted: everything the column can actually hold ────────────────────────


async def test_column_ceiling_is_accepted_and_stored_exactly(
    api, session_factory, storage, claimed_job
) -> None:
    """Positive control: the bound is the ceiling ITSELF, inclusive — not one below it.

    999999.999 round-trips through Numeric(9,3) losslessly. Rejecting it would mean the
    bound was picked by feel rather than from the column.
    """
    r = await _report(api, claimed_job, repr(COLUMN_MAX))

    assert r.status_code == 204
    row = await _row(session_factory, claimed_job)
    assert row == {"status": "done", "attempts": 1, "duration": COLUMN_MAX}
    assert len(storage.puts) == 1
    key, size = storage.puts[0]
    assert key.startswith(f"{TENANT}/") and size == len(AUDIO)


async def test_ordinary_duration_is_stored_unchanged(
    api, session_factory, storage, claimed_job
) -> None:
    """The everyday case must pass through byte-for-byte — no clamping, no coercion."""
    r = await _report(api, claimed_job, "12.345")

    assert r.status_code == 204
    row = await _row(session_factory, claimed_job)
    assert row["status"] == "done"
    assert row["duration"] == 12.345, "a legal duration must be stored verbatim"
    assert row["attempts"] == 1
    assert len(storage.puts) == 1


@pytest.mark.parametrize("duration", ["0.0", "-1.5"], ids=["zero", "negative"])
async def test_non_positive_duration_is_still_accepted(
    api, session_factory, claimed_job, duration: str
) -> None:
    """PINS THE ABSENCE OF A LOWER BOUND. Deliberate, not an omission.

    ``tts_backend`` emits ``duration_sec = ... if sr else 0.0``, so real audio arrives
    with a zero duration on a live path. A ``gt=0`` bound would 422 it →
    ``raise_for_status()`` → POST ``/fail`` → requeue → burn every attempt → the
    orchestrator aborts the entire narrate run over one good chunk. Adding the upper
    bound must not tempt anyone into "making it symmetric".

    What the value is STORED as is a different question, answered in
    ``tests/test_duration_unmeasured_is_not_zero.py``: a non-positive report is a
    measurement gap, so it lands as NULL rather than as the fabricated number. This
    test's subject is the acceptance, and it asserts the NULL only to stay honest
    about where the value went.
    """
    r = await _report(api, claimed_job, duration)

    assert r.status_code == 204, (
        "a non-positive duration is a NORMAL tts_backend event; it is logged and "
        "accepted, and adding an upper bound must not have introduced a lower one"
    )
    row = await _row(session_factory, claimed_job)
    assert row["status"] == "done"
    assert row["attempts"] == 1, "accepting the clip must not burn an attempt"
    assert row["duration"] is None, (
        "a non-positive report is a measurement gap, not a length — storing it "
        "verbatim republishes a fabricated zero to the narration timeline"
    )


# ── The constant is derived from the column, not typed by hand ───────────────


def test_bound_matches_the_column_ceiling() -> None:
    """Structural: if the migration ever widens the column, this is what breaks first."""
    assert MAX_DURATION_SECONDS == COLUMN_MAX
    # And the discriminating property: the fatal value is UNDER a million, so any
    # bound expressed as `lt=1e6` would admit it.
    assert ROUNDS_PAST_SCALE < 1_000_000.0
    assert ROUNDS_PAST_SCALE > MAX_DURATION_SECONDS
