"""GPU admission control on the worker side (tts_gateway#45).

Upstream GPU services answer ``503`` + ``Retry-After`` + ``{"error":"gpu_busy",...}``
when their lane is full; a gateway in between may answer ``429``. Before this change
``_synthesize`` called ``raise_for_status()`` and ``_process`` POSTed ``/fail`` — every
busy signal burned one of the job's ``max_attempts`` lives, and four busy spells
poisoned a healthy chunk.

The contract pinned here:
  * busy → wait ``Retry-After`` (header, then body ``retry_after``, default 5, clamped
    1..120) and retry IN PLACE, with the claim heartbeated meanwhile;
  * busy budget exhausted → ``/release`` (refunds the attempt), never ``/fail``;
  * any other 4xx/5xx → ``/fail``, exactly as before;
  * a refused heartbeat (409) during the wait → abandon: neither fail nor release.

No network: real ``httpx.Client`` objects over ``httpx.MockTransport``, so status
codes, headers and ``raise_for_status`` are httpx's own, not a hand-rolled fake.
"""

from __future__ import annotations

import base64
import importlib
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import ModuleType

import httpx
import pytest

JOB = {"job_id": "job-busy-0001", "text": "Hello there.", "speed": 1.0, "provider": "kokoro"}
_OK_SYNTH = {"audio_base64": base64.b64encode(b"RIFFfake").decode(), "duration_sec": 1.5}
_BUSY_BODY = {"error": "gpu_busy", "lane": "tts", "retry_after": 1}


@pytest.fixture
def worker(monkeypatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", "gw-tok")  # pragma: allowlist secret
    monkeypatch.setenv("TTS_SYNTH_TOKEN", "synth-tok")  # pragma: allowlist secret
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", "busy-worker")
    monkeypatch.delenv("TTS_LEASE_SECONDS", raising=False)
    monkeypatch.delenv("TTS_BUSY_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("TTS_BUSY_MAX_RETRIES", raising=False)
    sys.modules.pop("worker.worker", None)
    mod = importlib.import_module("worker.worker")
    yield importlib.reload(mod)
    mod._CURRENT.clear()
    sys.modules.pop("worker.worker", None)


class _Gateway:
    """Records every gateway call (with a monotonic timestamp)."""

    def __init__(self, heartbeat_status: int = 204) -> None:
        self.calls: list[tuple[float, str, bytes]] = []
        self._lock = threading.Lock()
        self.heartbeat_status = heartbeat_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.calls.append((time.monotonic(), request.url.path, request.read()))
        path = request.url.path
        if path.endswith("/heartbeat"):
            return httpx.Response(self.heartbeat_status)
        if path.endswith("/release"):
            return httpx.Response(200, json={"released": True})
        if path.endswith("/fail"):
            return httpx.Response(200, json={"status": "queued"})
        return httpx.Response(204)

    def paths(self, suffix: str) -> list[tuple[float, str, bytes]]:
        with self._lock:
            return [c for c in self.calls if c[1].endswith(suffix)]


class _Synth:
    """Answers from a script of responses; the last one repeats forever."""

    def __init__(self, script: list[Callable[[], httpx.Response]]) -> None:
        self.script = script
        self.times: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tts/synthesize"
        self.times.append(time.monotonic())
        idx = min(len(self.times) - 1, len(self.script) - 1)
        return self.script[idx]()


def _busy(status: int = 503, header: str | None = "1", body: dict | None = None) -> Callable:
    def make() -> httpx.Response:
        headers = {"Retry-After": header} if header is not None else {}
        return httpx.Response(
            status, headers=headers, json=body if body is not None else _BUSY_BODY
        )

    return make


def _ok() -> httpx.Response:
    return httpx.Response(200, json=_OK_SYNTH)


def _run(worker: ModuleType, gw: _Gateway, synth: _Synth) -> float:
    with (
        httpx.Client(transport=httpx.MockTransport(gw.handler)) as g,
        httpx.Client(transport=httpx.MockTransport(synth.handler)) as s,
    ):
        return worker._process(g, s, dict(JOB))


# --------------------------------------------------------------------------- retry


def test_two_busy_503s_then_200_completes_without_fail_or_release(worker) -> None:
    gw, synth = _Gateway(), _Synth([_busy(), _busy(), _ok])

    cooldown = _run(worker, gw, synth)

    assert len(synth.times) == 3, "retried in place: 2 busy answers + 1 success"
    assert len(gw.paths("/result")) == 1, "the job completed"
    assert gw.paths("/fail") == [], "a busy GPU must never burn an attempt via /fail"
    assert gw.paths("/release") == [], "the job succeeded — nothing to hand back"
    assert cooldown == 0.0
    # Retry-After: 1 was honoured between attempts (not a hot loop).
    gaps = [b - a for a, b in zip(synth.times, synth.times[1:], strict=False)]
    assert all(g >= 0.9 for g in gaps), gaps


def test_429_without_header_uses_body_retry_after(worker) -> None:
    gw = _Gateway()
    synth = _Synth([_busy(429, header=None, body={"error": "gpu_busy", "retry_after": 1}), _ok])

    _run(worker, gw, synth)

    assert len(synth.times) == 2
    assert len(gw.paths("/result")) == 1
    assert gw.paths("/fail") == []
    assert synth.times[1] - synth.times[0] >= 0.9


# ----------------------------------------------------------------------- exhausted


def test_busy_retry_count_exhausted_releases_instead_of_failing(worker, monkeypatch) -> None:
    monkeypatch.setattr(worker, "BUSY_MAX_RETRIES", 1)
    gw, synth = _Gateway(), _Synth([_busy(header="1")])

    cooldown = _run(worker, gw, synth)

    assert len(synth.times) == 2, "one initial attempt + exactly BUSY_MAX_RETRIES retries"
    releases = gw.paths("/release")
    assert len(releases) == 1, "exhausted busy budget → /release (attempt refunded)"
    assert b"worker_id=busy-worker" in releases[0][2], "release must name the claim holder"
    assert gw.paths("/fail") == [], "exhausted busy must NOT /fail"
    assert gw.paths("/result") == []
    assert cooldown == 1.0, "claim loop backs off by the last Retry-After before re-claiming"
    assert worker._CURRENT.get("job_id") is None


def test_wait_that_would_cross_the_deadline_is_not_started(worker, monkeypatch) -> None:
    monkeypatch.setattr(worker, "BUSY_DEADLINE_SECONDS", 0.5)
    gw, synth = _Gateway(), _Synth([_busy(header="1")])

    started = time.monotonic()
    _run(worker, gw, synth)

    assert len(synth.times) == 1, "a 1s wait cannot fit a 0.5s budget — release at once"
    assert time.monotonic() - started < 0.9
    assert len(gw.paths("/release")) == 1
    assert gw.paths("/fail") == []


# ------------------------------------------------------------ non-busy unchanged


@pytest.mark.parametrize("status", [400, 403, 500, 502])
def test_other_errors_still_fail_the_job_without_retry(worker, status) -> None:
    gw = _Gateway()
    synth = _Synth([lambda: httpx.Response(status, headers={"Retry-After": "1"}, json={})])

    cooldown = _run(worker, gw, synth)

    assert len(synth.times) == 1, f"HTTP {status} is not a busy signal — no retry"
    assert len(gw.paths("/fail")) == 1
    assert b"terminal" not in gw.paths("/fail")[0][2], "an ordinary failure stays retryable"
    assert gw.paths("/release") == []
    assert cooldown == 0.0


# --------------------------------------------------------------------- heartbeat


def test_heartbeats_keep_flowing_during_the_busy_sleep(worker, monkeypatch) -> None:
    monkeypatch.setattr(worker, "HEARTBEAT_SECONDS", 0.1)
    gw, synth = _Gateway(), _Synth([_busy(header="1"), _ok])

    _run(worker, gw, synth)

    first, second = synth.times
    during = [c for c in gw.paths("/heartbeat") if first < c[0] < second]
    assert len(during) >= 5, f"lease must be renewed while sleeping; got {len(during)} beats"


def test_heartbeat_cadence_is_at_most_a_third_of_the_lease(monkeypatch) -> None:
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", "gw-tok")  # pragma: allowlist secret
    monkeypatch.setenv("TTS_SYNTH_TOKEN", "synth-tok")  # pragma: allowlist secret
    monkeypatch.setenv("TTS_LEASE_SECONDS", "30")
    sys.modules.pop("worker.worker", None)
    try:
        mod = importlib.import_module("worker.worker")
        assert mod.HEARTBEAT_SECONDS == 10.0
        monkeypatch.delenv("TTS_LEASE_SECONDS")
        mod = importlib.reload(mod)
        # Default lease 180 s → 60 s cadence; and longer than the max busy sleep is
        # irrelevant because the heartbeat runs on its own thread.
        assert mod.HEARTBEAT_SECONDS == 60.0
        assert mod.HEARTBEAT_SECONDS <= mod.LEASE_SECONDS / 3
    finally:
        sys.modules.pop("worker.worker", None)


def test_refused_heartbeat_during_busy_wait_abandons_without_fail_or_release(
    worker, monkeypatch
) -> None:
    monkeypatch.setattr(worker, "HEARTBEAT_SECONDS", 0.05)
    gw = _Gateway(heartbeat_status=409)
    synth = _Synth([_busy(header="60"), _ok])

    started = time.monotonic()
    _run(worker, gw, synth)

    assert time.monotonic() - started < 5, "a lost lease must cut the 60s wait short"
    assert len(synth.times) == 1
    assert gw.paths("/fail") == [], "not our claim any more — must not fail it"
    assert gw.paths("/release") == [], "not our claim any more — must not release it"
    assert gw.paths("/result") == []


# ------------------------------------------------------------- Retry-After parse


def _resp(headers: dict | None = None, body: object = None) -> httpx.Response:
    if body is None:
        return httpx.Response(503, headers=headers or {})
    return httpx.Response(503, headers=headers or {}, json=body)


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        ({"Retry-After": "7"}, {"retry_after": 30}, 7.0),  # header wins
        ({}, {"retry_after": 30}, 30.0),  # body fallback
        ({}, None, 5.0),  # default
        ({"Retry-After": "garbage"}, {"retry_after": 9}, 9.0),  # bad header → body
        ({"Retry-After": "0"}, None, 1.0),  # clamp low
        ({"Retry-After": "9999"}, None, 120.0),  # clamp high
        ({}, {"retry_after": True}, 5.0),  # bool is not a number
        ({}, {"retry_after": -4}, 1.0),
    ],
)
def test_retry_after_resolution(worker, headers, body, expected) -> None:
    assert worker._retry_after_seconds(_resp(headers, body)) == expected


def test_retry_after_http_date(worker) -> None:
    when = datetime.now(UTC) + timedelta(seconds=40)
    value = worker._retry_after_seconds(_resp({"Retry-After": format_datetime(when, usegmt=True)}))
    assert 35 <= value <= 41


# --------------------------------------------------------------- 402 = terminal


def test_402_is_a_terminal_customer_config_failure(worker) -> None:
    """modality_gateway 402s a byo_required tenant with no provider (incl. kokoro, #50)."""
    gw = _Gateway()
    detail = "tenant must configure its own tts provider before use"
    synth = _Synth([lambda: httpx.Response(402, json={"detail": detail})])

    cooldown = _run(worker, gw, synth)

    assert len(synth.times) == 1, "402 is not retryable — no in-place retry"
    fails = gw.paths("/fail")
    assert len(fails) == 1
    form = dict(httpx.QueryParams(fails[0][2].decode()))
    assert form["terminal"] == "true", "a 402 must fail TERMINALLY, not requeue"
    assert form["worker_id"] == "busy-worker"
    assert "402" in form["error"] and "customer configuration" in form["error"]
    assert detail in form["error"], "the upstream's own reason must reach the job"
    assert gw.paths("/release") == []
    assert cooldown == 0.0


# ------------------------------------------- gateway contract (audit follow-ups)


def test_busy_release_names_reason_gpu_busy(worker, monkeypatch) -> None:
    """Without reason=gpu_busy the gateway keeps the lifetime clock running."""
    monkeypatch.setattr(worker, "BUSY_MAX_RETRIES", 0)
    gw, synth = _Gateway(), _Synth([_busy(header="1")])

    _run(worker, gw, synth)

    form = dict(httpx.QueryParams(gw.paths("/release")[0][2].decode()))
    assert form == {"worker_id": "busy-worker", "reason": "gpu_busy"}


def test_heartbeat_carries_worker_id(worker, monkeypatch) -> None:
    monkeypatch.setattr(worker, "HEARTBEAT_SECONDS", 0.05)
    gw, synth = _Gateway(), _Synth([_busy(header="1"), _ok])

    _run(worker, gw, synth)

    beats = gw.paths("/heartbeat")
    assert beats
    assert all(
        dict(httpx.QueryParams(b[2].decode())) == {"worker_id": "busy-worker"} for b in beats
    )


def test_heartbeat_interval_follows_the_claimed_lease(worker) -> None:
    assert worker._heartbeat_interval(30) == 10.0
    assert worker._heartbeat_interval(600) == worker.HEARTBEAT_SECONDS  # capped at 60
    assert worker._heartbeat_interval(None) == worker.HEARTBEAT_SECONDS  # older gateway
    assert worker._heartbeat_interval(0) == worker.HEARTBEAT_SECONDS
    assert worker._heartbeat_interval(True) == worker.HEARTBEAT_SECONDS


def test_process_uses_the_lease_from_the_claim(worker) -> None:
    """A 0.3 s lease → beats every 0.1 s, though HEARTBEAT_SECONDS is still 60."""
    gw, synth = _Gateway(), _Synth([_busy(header="1"), _ok])
    with (
        httpx.Client(transport=httpx.MockTransport(gw.handler)) as g,
        httpx.Client(transport=httpx.MockTransport(synth.handler)) as s,
    ):
        worker._process(g, s, {**JOB, "lease_seconds": 0.3})

    first, second = synth.times
    during = [c for c in gw.paths("/heartbeat") if first < c[0] < second]
    assert len(during) >= 5, len(during)
