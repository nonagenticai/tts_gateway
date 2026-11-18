"""The worker's SIGTERM path — it must hand the claim back, and never die noisily.

No DB and no network: httpx is monkeypatched. What matters here is the shape of the
shutdown contract, because it is the half of the poison fix that lives on the worker:
a k8s RollingUpdate sends SIGTERM, and if the worker does not release its claim the
gateway charges the interrupted job an attempt it never earned.
"""

from __future__ import annotations

import importlib
import signal
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest


@pytest.fixture
def worker(monkeypatch) -> Iterator[ModuleType]:
    """Import worker.worker with its required env present (it reads env at import)."""
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", "tok-123")  # pragma: allowlist secret
    # REQUIRED since the synth fallback was removed — import raises without it.
    monkeypatch.setenv("TTS_SYNTH_TOKEN", "synth-tok-456")  # pragma: allowlist secret
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_WORKER_ID", "mini2-worker")
    sys.modules.pop("worker.worker", None)
    mod = importlib.import_module("worker.worker")
    yield importlib.reload(mod)
    sys.modules.pop("worker.worker", None)


class _FakeResponse:
    status_code = 200


class _FakeClient:
    """Records the single release POST; usable as a context manager like httpx.Client."""

    def __init__(self, calls: list[dict], exc: Exception | None = None, **kwargs) -> None:
        self._calls = calls
        self._exc = exc
        self.init_kwargs = kwargs

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def post(self, url: str, **kwargs) -> _FakeResponse:
        self._calls.append({"url": url, **kwargs})
        if self._exc is not None:
            raise self._exc
        return _FakeResponse()


def test_sigterm_releases_the_claim_with_this_workers_id(worker, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(worker.httpx, "Client", lambda **kw: _FakeClient(calls, **kw))
    worker._CURRENT["job_id"] = "job-abc-123"

    with pytest.raises(SystemExit) as exit_info:
        worker._on_shutdown_signal(signal.SIGTERM, None)

    assert exit_info.value.code == 0
    assert len(calls) == 1, "exactly one release, no retries — a shutdown must not stall"
    assert calls[0]["url"] == "http://gw.test/api/tts/jobs/job-abc-123/release"
    # The gateway only refunds a claim the requesting worker actually holds.
    assert calls[0]["data"] == {"worker_id": "mini2-worker"}
    assert worker._CURRENT.get("job_id") is None  # cleared → atexit won't double-release


def test_release_failure_still_exits_and_does_not_re_raise(worker, monkeypatch) -> None:
    """A gateway that is unreachable at shutdown must not turn SIGTERM into a crash.

    The lease expiry is the backstop: the job is reclaimed a little later instead of
    immediately. Slower, but correct — and never a stack trace on the way out.
    """
    calls: list[dict] = []
    boom = RuntimeError("connection refused")
    monkeypatch.setattr(worker.httpx, "Client", lambda **kw: _FakeClient(calls, exc=boom, **kw))
    worker._CURRENT["job_id"] = "job-doomed"

    with pytest.raises(SystemExit) as exit_info:  # SystemExit, NOT RuntimeError
        worker._on_shutdown_signal(signal.SIGTERM, None)

    assert exit_info.value.code == 0
    assert len(calls) == 1


def test_release_is_a_noop_when_no_job_is_held(worker, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(worker.httpx, "Client", lambda **kw: _FakeClient(calls, **kw))
    worker._CURRENT.pop("job_id", None)

    worker._release_current()  # idle worker between polls

    assert calls == []  # nothing to hand back → no call at all
