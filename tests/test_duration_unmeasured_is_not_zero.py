"""A duration nobody measured must not be published as ``0.0`` seconds.

``tts_backend`` emits ``duration_sec = frames / sr if sr else 0.0`` — so on a live
path a clip with perfectly good audio arrives with NO usable length. The worker used
to coerce that to a float (``float(d.get("duration_sec") or 0.0)``), the gateway
logged a warning and completed the job anyway, and ``get_batch`` published
``duration_seconds: 0.0`` on a ``done`` job. The orchestrator's narrate stage builds
the narration timeline out of exactly that number: it allocates ZERO seconds for a
chunk that holds real speech, the narration desynchronises against the picture for
every later chunk, and the batch and the run both report success. Nothing is
measurable after the fact — a fabricated 0.0 is byte-identical to a real one.

The audio is REAL, so the job must still complete: dropping it (422/failed) is
strictly worse — the orchestrator's narrate stage raises ``StagePreconditionError``
on any failed job and kills the whole book for a chunk that synthesized fine.

So the gap is made EXPRESSIBLE instead: ``duration_seconds`` stays NULL (the column
is already nullable) and ``JobStatus.duration_unmeasured`` carries the fact to the
consumer. NULL is not a value the timeline builder can silently add up, and the flag
says WHY it is absent — "audio of unknown length", never "zero seconds of audio".

Three layers are pinned, because a fix at only one of them does not reach the
consumer:

* the worker reports the absence instead of inventing a number;
* the gateway persists NULL rather than the reported non-positive value;
* ``GET /api/tts/batches/{id}`` — the only surface the orchestrator ever reads —
  exposes ``duration_unmeasured`` alongside a null ``duration_seconds``.
"""

from __future__ import annotations

import base64
import importlib
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text as sql

from src.api.routes import batches, workers
from src.database.session import get_raw_session, get_tenant_session
from src.services import job_manager as jm

TENANT = "22222222-2222-2222-2222-222222222222"
WORKER_ID = "mini2"
AUDIO = b"RIFF-real-audio"
MAX_BYTES = 4096


# ── layer 1: the worker must not invent a number ──────────────────────────────


@pytest.fixture
def worker(monkeypatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", "gw-token")  # pragma: allowlist secret
    monkeypatch.setenv("TTS_SYNTH_TOKEN", "synth-token")  # pragma: allowlist secret
    monkeypatch.delenv("TTS_SYNTH_ALLOW_SHARED_TOKEN", raising=False)
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", WORKER_ID)
    sys.modules.pop("worker.worker", None)
    mod = importlib.import_module("worker.worker")
    yield importlib.reload(mod)
    sys.modules.pop("worker.worker", None)


class _Response:
    status_code = 200

    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _process_one(worker: ModuleType, client: object, job: dict) -> None:
    """Run ``_process`` with ONE fake as both clients, so every call lands in one log.

    Production splits them (gateway client signs with a cell token, synth client
    does not); the URL each call hits still tells the two apart.
    """
    worker._process(client, client, job)


class _FakeClient:
    """Answers /synthesize with a caller-chosen payload; records every POST."""

    def __init__(self, calls: list[dict], synth_payload: dict) -> None:
        self._calls = calls
        self._synth = synth_payload

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def post(self, url: str, **kwargs: Any) -> _Response:
        self._calls.append({"url": url, **kwargs})
        if "/synthesize" in url:
            return _Response(self._synth)
        return _Response()


def _audio_payload(**extra: Any) -> dict:
    return {"audio_base64": base64.b64encode(b"RIFFfake").decode(), **extra}


@pytest.mark.parametrize(
    ("payload", "case"),
    [
        (_audio_payload(), "duration_sec absent"),
        (_audio_payload(duration_sec=0.0), "duration_sec 0.0"),
        (_audio_payload(duration_sec=None), "duration_sec null"),
    ],
    ids=["absent", "zero", "null"],
)
def test_synthesize_reports_an_unmeasured_duration_as_none(
    worker: ModuleType, payload: dict, case: str
) -> None:
    """``float(... or 0.0)`` fabricates a measurement the backend never made."""
    client = _FakeClient([], payload)
    _wav, duration = worker._synthesize(client, "Hello.", None, 1.0, None)
    assert duration is None, (
        f"{case}: the worker invented a duration the synth backend never reported; "
        "0.0 is indistinguishable from a real measurement downstream"
    )


def test_synthesize_passes_a_real_duration_through(worker: ModuleType) -> None:
    """The anomaly path must not swallow a genuine measurement."""
    _wav, duration = worker._synthesize(
        _FakeClient([], _audio_payload(duration_sec=2.5)), "Hello.", None, 1.0, None
    )
    assert duration == 2.5


def test_process_omits_the_duration_field_when_it_is_unmeasured(worker: ModuleType) -> None:
    """Sending ``"0.0"`` on the wire re-fabricates the value one layer later."""
    calls: list[dict] = []
    _process_one(
        worker, _FakeClient(calls, _audio_payload()), {"job_id": "j1", "text": "Hi.", "speed": 1.0}
    )
    result = next(c for c in calls if c["url"].endswith("/result"))
    assert "duration_seconds" not in result["data"], (
        "the worker forwarded a fabricated duration to the gateway: "
        f"{result['data'].get('duration_seconds')!r}"
    )
    assert result["data"]["worker_id"] == WORKER_ID
    assert result["files"], "the clip itself must still be uploaded — the audio is real"


# ── layer 1b: the two guards a gutting left green ─────────────────────────────
#
# ``_measured_duration`` has three guards. An adversarial verifier deleted the
# ``isinstance`` type guard and the ``math.isfinite`` check, ran the whole suite, and
# got 216 passed — only the absent/0.0/null cases above were pinned. The cases below
# exist so that gutting either guard goes RED.
#
# Which case kills which mutation is NOT symmetric, and the asymmetry is the point:
#
# * ``float`` parses ``nan``/``inf`` out of a JSON body without complaint
#   (``json.loads`` decodes the bare literals ``NaN``/``Infinity``), and every
#   comparison against ``nan`` is False — so ``nan`` and ``+inf`` sail straight
#   through the ``<= 0.0`` test. ONLY ``isfinite`` stops them. ``-inf`` is caught by
#   ``<= 0.0`` as well; it is here for completeness, not as a mutation probe.
# * ``float("abc")``/``float({})`` raise, and the ``except`` already returns None — so
#   a bad string or a dict does NOT depend on the type guard. What does is ``True``:
#   ``bool`` is an ``int`` subclass, so without the guard a JSON ``true`` becomes a
#   1.0-SECOND MEASUREMENT. ``False`` is caught by ``<= 0.0``. ``True`` is the only
#   input that distinguishes the type guard from its neighbours, which is exactly why
#   deleting the guard changed nothing before.
#
# The cost of a leak is not a wrong number, it is a DEAD JOB: the gateway's
# ``duration_seconds`` Form field is bound ``allow_inf_nan=False`` (422 on non-finite —
# pinned in ``tests/test_result_duration_bound.py``), the worker's
# ``raise_for_status`` turns that into POST ``/fail``, and the job burns attempts until
# it lands in ``status='failed'``. That fails the orchestrator's whole narrate stage for
# a clip whose audio was fine — the exact outcome this change exists to avoid.


@pytest.mark.parametrize(
    ("reported", "case"),
    [
        (float("nan"), "nan sails through `<= 0.0` — every comparison against it is False"),
        (float("inf"), "+inf is > 0.0, so only isfinite rejects it"),
        (float("-inf"), "-inf"),
        ("nan", "the string 'nan' — float() parses it just as happily"),
        ("Infinity", "the string 'Infinity'"),
    ],
    ids=["nan", "inf", "neg-inf", "str-nan", "str-infinity"],
)
def test_a_non_finite_duration_is_unmeasured(
    worker: ModuleType, reported: object, case: str
) -> None:
    """A non-finite reading is a broken measurement, not a length."""
    _wav, duration = worker._synthesize(
        _FakeClient([], _audio_payload(duration_sec=reported)), "Hello.", None, 1.0, None
    )
    assert duration is None, (
        f"{case}: the worker accepted {duration!r} as a measurement. Forwarded, it 422s at "
        "the gateway's allow_inf_nan=False bound and turns a clip with real audio into a "
        "failed job"
    )


@pytest.mark.parametrize(
    ("reported", "case"),
    [
        (True, "JSON `true` — bool is an int subclass, so float(True) is a 1.0s 'measurement'"),
        (False, "JSON `false`"),
        ("not-a-number", "a non-numeric string"),
        ("", "an empty string"),
        ({}, "an object"),
        ({"seconds": 2.5}, "an object that merely looks like a duration"),
        ([2.5], "an array"),
    ],
    ids=["true", "false", "str", "empty-str", "dict", "dict-shaped", "list"],
)
def test_a_non_numeric_duration_is_unmeasured(
    worker: ModuleType, reported: object, case: str
) -> None:
    """Anything that is not a number is an absence, never a coercion."""
    _wav, duration = worker._synthesize(
        _FakeClient([], _audio_payload(duration_sec=reported)), "Hello.", None, 1.0, None
    )
    assert duration is None, (
        f"{case}: the worker coerced it into a duration of {duration!r} seconds"
    )


def test_a_numeric_string_duration_is_still_honoured(worker: ModuleType) -> None:
    """Positive control: the type guard admits str on purpose, so it must not over-reject."""
    _wav, duration = worker._synthesize(
        _FakeClient([], _audio_payload(duration_sec="2.5")), "Hello.", None, 1.0, None
    )
    assert duration == 2.5, "rejecting every string would break a backend that quotes its numbers"


@pytest.mark.parametrize("reported", [float("nan"), float("inf"), True], ids=["nan", "inf", "true"])
def test_process_never_puts_a_bogus_duration_on_the_wire(
    worker: ModuleType, reported: object
) -> None:
    """The value-level guard is only worth what reaches the gateway."""
    calls: list[dict] = []
    _process_one(
        worker,
        _FakeClient(calls, _audio_payload(duration_sec=reported)),
        {"job_id": "j1", "text": "Hi.", "speed": 1.0},
    )
    result = next(c for c in calls if c["url"].endswith("/result"))
    assert "duration_seconds" not in result["data"], (
        f"the worker forwarded duration_seconds={result['data'].get('duration_seconds')!r} "
        f"for a reported {reported!r}"
    )
    assert result["files"], "the clip itself must still be uploaded — the audio is real"


def test_process_sends_a_real_duration(worker: ModuleType) -> None:
    calls: list[dict] = []
    _process_one(
        worker,
        _FakeClient(calls, _audio_payload(duration_sec=3.25)),
        {"job_id": "j1", "text": "Hi.", "speed": 1.0},
    )
    result = next(c for c in calls if c["url"].endswith("/result"))
    assert result["data"]["duration_seconds"] == "3.25"


# ── layer 2: the gateway must persist the absence, not a zero ─────────────────


class _RecordingStorage:
    def __init__(self) -> None:
        self.puts: list[tuple[str, int]] = []

    def key_for(self, tenant_id: str, idempotency_key: str) -> str:
        return f"{tenant_id}/{idempotency_key}.wav"

    async def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        self.puts.append((key, len(data)))
        return key

    async def presign(self, key: str) -> str | None:
        return f"http://minio.test/{key}?sig=x" if key else None


@pytest.fixture
def storage() -> _RecordingStorage:
    return _RecordingStorage()


@pytest_asyncio.fixture
async def api(monkeypatch, session_factory, storage) -> AsyncIterator[httpx.AsyncClient]:
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
    app.dependency_overrides[workers.authorize_worker] = lambda: None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tts.test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def claimed_job(session_factory) -> str:
    async with session_factory() as session:
        submitted = await jm.submit_batch(
            session,
            tenant_id=TENANT,
            batch_id="b1",
            chunks=[{"ref": "r1", "text": "Chapter one.", "voice": "af_heart", "speed": 1.0}],
        )
    async with session_factory() as session:
        claimed = await jm.claim_next(
            session,
            worker_id=WORKER_ID,
            lease_seconds=180,
            max_attempts=4,
            max_lifetime_seconds=3600,
        )
    assert claimed is not None
    return str(submitted[0]["job_id"])


async def _row(session_factory, job_id: str) -> dict:
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


async def _report(api: httpx.AsyncClient, job_id: str, duration: str | None) -> httpx.Response:
    data: dict[str, str] = {"worker_id": WORKER_ID}
    if duration is not None:
        data["duration_seconds"] = duration
    return await api.post(
        f"/api/tts/jobs/{job_id}/result",
        files={"audio": ("clip.wav", AUDIO, "audio/wav")},
        data=data,
    )


@pytest.mark.parametrize("duration", [None, "0.0", "-1.5"], ids=["omitted", "zero", "negative"])
async def test_an_unmeasured_duration_is_stored_as_null_not_zero(
    api, session_factory, storage, claimed_job, duration, caplog
) -> None:
    """The row must say "unknown", and the clip must still be kept."""
    with caplog.at_level(logging.WARNING, logger="src.api.routes.workers"):
        r = await _report(api, claimed_job, duration)

    assert r.status_code == 204, "real audio must never be dropped over a missing measurement"
    assert storage.puts, "the clip must still reach storage"

    row = await _row(session_factory, claimed_job)
    assert row["status"] == "done"
    assert row["attempts"] == 1, "an unmeasured duration must not burn an attempt"
    assert row["duration"] is None, (
        f"an unmeasured duration was persisted as {row['duration']!r} — a fabricated "
        "measurement the timeline builder cannot tell from a real zero"
    )
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records), (
        "a measurement gap must be LOUD, not silent"
    )


async def test_a_real_duration_is_still_persisted(api, session_factory, claimed_job) -> None:
    r = await _report(api, claimed_job, "12.345")
    assert r.status_code == 204
    row = await _row(session_factory, claimed_job)
    assert row["duration"] == 12.345


# ── layer 3: the consumer surface — this is what the orchestrator reads ───────


async def _poll(session_factory, storage: Any, batch_id: str) -> list[dict[str, Any]]:
    app = FastAPI()
    app.include_router(batches.router)

    @app.middleware("http")
    async def _tenant_state(request, call_next):
        request.state.tenant_id = TENANT
        return await call_next(request)

    async def _session():
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_tenant_session] = _session
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        with patch.object(batches, "get_storage", lambda: storage):
            resp = await client.get(f"/api/tts/batches/{batch_id}")
    assert resp.status_code == 200
    return list(resp.json()["jobs"])


async def test_the_batch_poll_says_unmeasured_rather_than_zero(
    api, session_factory, storage, claimed_job
) -> None:
    """A fix that only logs better never reaches the timeline builder."""
    assert (await _report(api, claimed_job, None)).status_code == 204

    jobs = await _poll(session_factory, storage, "b1")
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "done", "the audio is real; the job is done"
    assert job["result_url"], "the clip must still be fetchable"
    assert job["duration_seconds"] is None, (
        f"the consumer received duration_seconds={job['duration_seconds']!r}; "
        "0.0 allocates no time on the narration timeline for real speech"
    )
    assert job["duration_unmeasured"] is True, (
        "the consumer cannot distinguish 'audio of unknown length' from "
        "'zero seconds of audio' — the gap must survive to the response"
    )


async def test_a_measured_job_is_not_flagged_unmeasured(
    api, session_factory, storage, claimed_job
) -> None:
    """The flag is for the anomaly only — a normal job must stay quiet."""
    assert (await _report(api, claimed_job, "7.5")).status_code == 204

    jobs = await _poll(session_factory, storage, "b1")
    assert jobs[0]["duration_seconds"] == 7.5
    assert jobs[0]["duration_unmeasured"] is False


async def test_a_pending_job_is_not_flagged_unmeasured(session_factory) -> None:
    """A queued job has no duration YET — that is not a measurement gap."""
    async with session_factory() as s:
        await jm.submit_batch(
            s,
            tenant_id=TENANT,
            batch_id="b2",
            chunks=[{"ref": "r1", "text": "Later.", "voice": "af_heart", "speed": 1.0}],
        )
    jobs = await _poll(session_factory, _RecordingStorage(), "b2")
    assert jobs[0]["status"] == "queued"
    assert jobs[0]["duration_seconds"] is None
    assert jobs[0]["duration_unmeasured"] is False, (
        "flagging an unfinished job would make the signal meaningless"
    )
