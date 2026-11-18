"""API-boundary behaviour of POST /api/tts/jobs/{job_id}/result.

* **Bounded body read.** ``await audio.read(max+1)`` keeps an oversized clip out of
  the object store and out of a whole-body ``bytes`` object in the handler. On its
  own it is not an OOM defence — Starlette's multipart parser has already spooled the
  file part into an unbounded ``SpooledTemporaryFile`` (max_size 1 MiB) before the
  endpoint runs. The transport-layer gate that fixes that is
  ``BodySizeLimitMiddleware``, covered by ``tests/test_body_limit_middleware.py``;
  this module mounts the router WITHOUT it, so what it pins is the route's own
  per-file bound.
* **Self-reported duration is accepted, not policed.** ``tts_backend`` emits
  ``duration_sec=0.0`` on a live path with perfectly good audio, so a zero duration
  is a normal event. Rejecting it would make ``worker.py`` POST ``/fail``, requeue a
  deterministically-failing job to ``max_attempts``, and abort the orchestrator's
  whole narrate stage. The route logs it, completes the job, and hands
  ``duration_seconds=None`` (not ``0.0``) to ``complete()`` so the gap stays
  expressible downstream — see ``tests/test_duration_unmeasured_is_not_zero.py``.

No DB and no MinIO: the session dependency, the job_manager calls and the storage
facade are all stubbed. What is under test is the HTTP contract of the route.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import workers
from src.database.session import get_raw_session

if TYPE_CHECKING:
    # starlette's TestClient returns httpx2 responses when httpx2 is installed.
    import httpx2

MAX_BYTES = 1024  # tiny cap so a test payload can exceed it without allocating anything real


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
def storage(monkeypatch) -> _RecordingStorage:
    stub = _RecordingStorage()
    monkeypatch.setattr(workers, "get_storage", lambda: stub)
    return stub


@pytest.fixture
def completions(monkeypatch) -> list[dict]:
    """Capture jm.complete kwargs — a bad duration must never reach the DB."""
    recorded: list[dict] = []

    async def _keyinfo(_session, *, job_id: str) -> dict[str, str]:
        # Mirrors the real SELECT, ``claimed_by`` included: the route needs the
        # current owner BEFORE it writes, not just afterwards. "mini2" is the owner
        # every test here posts as unless it is deliberately testing a stale worker.
        return {
            "tenant_id": "11111111-1111-1111-1111-111111111111",
            "idempotency_key": f"idem-{job_id}",
            "status": "claimed",
            "claimed_by": "mini2",
        }

    async def _complete(_session, **kwargs) -> bool:
        recorded.append(kwargs)
        return True

    monkeypatch.setattr(workers.jm, "get_job_keyinfo", _keyinfo)
    monkeypatch.setattr(workers.jm, "complete", _complete)
    return recorded


@pytest.fixture
def client(monkeypatch, storage, completions) -> Iterator[TestClient]:
    """The workers router alone — no auth middleware, no lifespan, no DB."""
    monkeypatch.setattr(
        workers,
        "get_settings",
        lambda: SimpleNamespace(max_result_bytes=MAX_BYTES),
    )
    app = FastAPI()
    app.include_router(workers.router)
    app.dependency_overrides[get_raw_session] = lambda: None
    # The router's worker-identity guard is out of scope here (it has its own module,
    # tests/test_worker_authz.py); override it so these cases exercise the upload
    # contract rather than 403-ing before the handler runs.
    app.dependency_overrides[workers.authorize_worker] = lambda: None
    with TestClient(app) as c:
        yield c


def _post(client: TestClient, audio: bytes, duration: str = "3.5") -> httpx2.Response:
    return client.post(
        "/api/tts/jobs/job-1/result",
        files={"audio": ("clip.wav", audio, "audio/wav")},
        data={"duration_seconds": duration, "worker_id": "mini2"},
    )


# ── Bounded read ─────────────────────────────────────────────────────────────


def test_oversized_clip_is_rejected_with_413(client, storage, completions) -> None:
    """A payload past the cap must 413 — and must never be stored or completed.

    What this check buys: no oversized object lands in MinIO and the handler never
    allocates the whole body as ``bytes``. It does not stop the upload from being
    spooled — that is ``BodySizeLimitMiddleware``'s job, and this app is mounted
    without it on purpose so the route's own bound is what gets asserted.
    """
    r = _post(client, b"x" * (MAX_BYTES + 512))

    assert r.status_code == 413, "an oversized clip must be refused, not buffered whole"
    assert storage.puts == []  # nothing reached MinIO
    assert completions == []  # and the job was not finished


def test_empty_clip_is_rejected_with_400(client, storage) -> None:
    """A zero-byte upload is a failed synth, not a finished job."""
    r = _post(client, b"")

    assert r.status_code == 400
    assert storage.puts == []


def test_clip_at_the_cap_is_accepted(client, storage) -> None:
    """The bound is inclusive — exactly max_result_bytes must still succeed."""
    r = _post(client, b"y" * MAX_BYTES)

    assert r.status_code == 204
    assert storage.puts == [("11111111-1111-1111-1111-111111111111/idem-job-1.wav", MAX_BYTES)]


# ── Self-reported duration ───────────────────────────────────────────────────


def test_zero_duration_is_accepted_and_warned(client, storage, completions, caplog) -> None:
    """A 0.0 duration must COMPLETE the job, and must be visible in the log.

    ``tts_backend`` emits ``duration_sec = ... if sr else 0.0`` on a live path, so a
    real clip legitimately arrives with duration 0.0. Refusing it would 422 →
    ``raise_for_status()`` in worker.py → POST ``/fail`` → requeue a deterministically
    failing job until ``max_attempts`` → ``status='failed'`` → the orchestrator's
    audiobook narrate stage raises ``StagePreconditionError`` and aborts the ENTIRE
    run over one chunk. Accepting keeps the audio.

    What must NOT be kept is the number: ``0.0`` is a sentinel for "no sample rate",
    and passing it through to ``complete()`` writes a fabricated measurement the
    narration timeline cannot distinguish from a real one. ``complete()`` receives
    ``None``.
    """
    with caplog.at_level(logging.WARNING, logger=workers.logger.name):
        r = _post(client, b"riff-real-audio", duration="0.0")

    assert r.status_code == 204, "a zero duration must not fail the job"
    assert storage.puts == [
        ("11111111-1111-1111-1111-111111111111/idem-job-1.wav", len(b"riff-real-audio"))
    ]
    assert completions[0]["duration_seconds"] is None, (
        "the route forwarded the sentinel 0.0 as if it were a measurement; the gap "
        "must reach the DB as NULL so the batch poll can report it"
    )

    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings, "a non-positive duration must be logged, not silently stored"
    assert "job-1" in warnings[0].getMessage()
    assert "duration_sec" in warnings[0].getMessage()


def test_positive_duration_is_not_warned(client, completions, caplog) -> None:
    """The warning is for the anomaly only — a normal duration must stay quiet."""
    with caplog.at_level(logging.WARNING, logger=workers.logger.name):
        r = _post(client, b"riff-real-audio", duration="2.0")

    assert r.status_code == 204
    assert [rec for rec in caplog.records if rec.levelno == logging.WARNING] == []


def test_positive_duration_still_completes(client, completions) -> None:
    r = _post(client, b"riff-real-audio", duration="4.25")

    assert r.status_code == 204
    assert completions[0]["duration_seconds"] == 4.25


# ── Ownership is checked BEFORE the object store is touched ──────────────────
#
# The clip key is ``{tenant}/{idempotency_key}.wav`` — DETERMINISTIC, not
# content-addressed. Two workers holding the same job at different times write the
# SAME key with DIFFERENT bytes (the deployed pair run different synth backends:
# in-cluster modality-gateway vs. off-cluster kokoro on mini1, and even one backend
# is not bit-reproducible). So a stale worker's ``put`` clobbers the rightful
# owner's audio, its duration diverges from the ``duration_seconds`` the owner
# reported, and the orchestrator's narration timeline drifts against the picture
# for a job that never failed.


def _post_as(client: TestClient, worker_id: str | None, audio: bytes = b"stale-audio"):
    data: dict[str, str] = {"duration_seconds": "9.5"}
    if worker_id is not None:
        data["worker_id"] = worker_id
    return client.post(
        "/api/tts/jobs/job-1/result",
        files={"audio": ("clip.wav", audio, "audio/wav")},
        data=data,
    )


def test_stale_worker_never_writes_over_the_owners_clip(client, storage, completions) -> None:
    """THE DEFECT: an unowned writer mutated the shared key before being refused.

    ``complete()`` is guarded on ``claimed_by`` and correctly no-ops for a worker
    that lost its claim — but that guard runs AFTER ``storage.put``. The stale
    worker was refused the DB row and granted the bytes, which is the wrong half.
    """
    r = _post_as(client, "mini1-stale")  # owner is "mini2" (see the keyinfo stub)

    assert storage.puts == [], (
        "a worker that no longer owns the claim wrote its audio to the owner's key — "
        "the rightful owner's clip is clobbered and its reported duration now lies"
    )
    assert completions == [], "and it must not attempt to finish another worker's claim"
    assert r.status_code == 204, "still a benign no-op, not retry bait for a lost job"


def test_the_owner_still_writes_normally(client, storage, completions) -> None:
    """The control: the guard must not break the only path that matters."""
    r = _post_as(client, "mini2")

    assert r.status_code == 204
    assert storage.puts == [
        ("11111111-1111-1111-1111-111111111111/idem-job-1.wav", len(b"stale-audio"))
    ]
    assert completions and completions[0]["worker_id"] == "mini2"


def test_legacy_worker_without_worker_id_is_unchanged(client, storage, completions) -> None:
    """Rollout compatibility: the manually-deployed Mac workers may still omit
    ``worker_id``. There is no identity to compare, so behaviour must be exactly
    today's — write and complete unguarded. Making this a 403/409 would stop every
    not-yet-redeployed launchd worker from reporting results.
    """
    r = _post_as(client, None)

    assert r.status_code == 204
    assert storage.puts == [
        ("11111111-1111-1111-1111-111111111111/idem-job-1.wav", len(b"stale-audio"))
    ]
    assert completions and completions[0]["worker_id"] is None
