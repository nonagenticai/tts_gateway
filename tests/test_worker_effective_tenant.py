"""The synth hop must be attributed to the JOB's tenant, not the worker's.

The Bearer token on the ``TTS_SYNTH_URL`` hop authenticates the WORKER — it
introspects to a service-account tenant. tts_backend scopes the request, and the
cost event it emits at the end of synthesis, to whatever tenant it resolves. With
no ``X-Effective-Tenant-Id`` on the request, that is the service account: every
customer narration was billed to the fleet SA tenant, and the gateway emits no
cost event of its own, so no row anywhere corrected it.

These tests pin the three behaviours that matter:

* a job with a tenant stamps it, as a UUID, under w2v_common's header constant;
* a job with NO tenant is sent exactly as before — absent, not empty. This path
  caused a narration outage; degrading attribution is acceptable, degrading
  synthesis is not;
* a non-UUID tenant is dropped rather than forwarded. w2v_common REJECTS a
  delegating subject that asserts a non-UUID effective tenant (401 INVALID_TENANT,
  deliberately with no fallback), so forwarding garbage would kill the job.

No network: httpx is monkeypatched, as in test_worker_headers.
"""

from __future__ import annotations

import base64
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest
from w2v_common.auth.outbound import EFFECTIVE_TENANT_HEADER

GATEWAY_TOKEN = "gw-secret-token"  # noqa: S105 # pragma: allowlist secret
SYNTH_TOKEN = "synth-only-token"  # noqa: S105 # pragma: allowlist secret
JOB_TENANT = "3f1c2d4e-5a6b-4c7d-8e9f-0a1b2c3d4e5f"


@pytest.fixture
def worker_module(monkeypatch) -> Iterator[ModuleType]:
    """The worker as an operator who has set NO flags gets it: stamp OFF."""
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("TTS_SYNTH_TOKEN", SYNTH_TOKEN)
    monkeypatch.delenv("TTS_SYNTH_ALLOW_SHARED_TOKEN", raising=False)
    monkeypatch.delenv("SYNTH_EFFECTIVE_TENANT_ENABLED", raising=False)
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", "mini1-worker")
    sys.modules.pop("worker.worker", None)
    mod = importlib.import_module("worker.worker")
    yield importlib.reload(mod)
    sys.modules.pop("worker.worker", None)


@pytest.fixture
def worker(worker_module, monkeypatch) -> ModuleType:
    """…with the stamp switched ON, as a deliberate per-deployment opt-in would."""
    monkeypatch.setattr(worker_module, "SYNTH_EFFECTIVE_TENANT_ENABLED", True)
    return worker_module


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
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def post(self, url: str, **kwargs) -> _Response:
        self._calls.append({"url": url, **kwargs})
        if "/synthesize" in url:
            return _Response(
                {"audio_base64": base64.b64encode(b"RIFFfake").decode(), "duration_sec": 2.5}
            )
        return _Response()


def _synth_headers_for(worker: ModuleType, job: dict) -> dict:
    calls: list[dict] = []
    _process_one(worker, _FakeClient(calls), job)
    synth = [c for c in calls if "/synthesize" in c["url"]]
    assert len(synth) == 1, f"expected exactly one synth call, got {len(synth)}"
    return synth[0]["headers"]


def test_synth_hop_carries_the_jobs_tenant(worker) -> None:
    """THE FIX: the customer's tenant reaches tts_backend, so the cost event lands there."""
    headers = _synth_headers_for(
        worker,
        {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0, "tenant_id": JOB_TENANT},
    )

    assert headers[EFFECTIVE_TENANT_HEADER] == JOB_TENANT, (
        "the synth hop did not assert the job's tenant — tts_backend will scope the "
        "request, and bill the synthesis, to the worker's service-account tenant"
    )
    # Attribution must not have cost the credential split (test_worker_headers).
    assert headers["X-Tenant-Id"] == f"Bearer {SYNTH_TOKEN}"
    assert GATEWAY_TOKEN not in str(headers)


@pytest.mark.parametrize("job_extra", [{}, {"tenant_id": None}, {"tenant_id": ""}])
def test_unknown_tenant_sends_the_request_unchanged(worker, job_extra) -> None:
    """FAIL SAFE: absent, not empty. Narration must survive a claim with no tenant."""
    headers = _synth_headers_for(
        worker, {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0, **job_extra}
    )

    assert EFFECTIVE_TENANT_HEADER not in headers, (
        "an unknown tenant must be an ABSENT header, never a blank one — a strict "
        "receiver would refuse the blank and take narration down"
    )
    assert headers["X-Tenant-Id"] == f"Bearer {SYNTH_TOKEN}"


def test_non_uuid_tenant_is_dropped_not_forwarded(worker) -> None:
    """w2v_common 401s a delegator that asserts a non-UUID tenant — it does not fall back."""
    headers = _synth_headers_for(
        worker,
        {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0, "tenant_id": "acme-corp"},
    )

    assert EFFECTIVE_TENANT_HEADER not in headers


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param(lambda w: w._synth_headers(JOB_TENANT), id="tenant-path"),
        pytest.param(lambda w: w._synth_headers(None), id="fail-safe-absent"),
        pytest.param(lambda w: w._synth_headers("acme-corp"), id="fail-safe-non-uuid"),
    ],
)
def test_the_shared_constant_is_never_mutated(worker, returned) -> None:
    """``_SYNTH_H`` is module-level and shared; a stamp onto it would leak across jobs.

    Every path must return a COPY, not the constant itself — including the fail-safe
    paths, which used to `return _SYNTH_H`. Aliasing is not a bug today only because
    no caller mutates the result; the next one that does would corrupt the credential
    header for every subsequent job in the process.
    """
    before = dict(worker._SYNTH_H)

    headers = returned(worker)
    headers["X-Injected-By-A-Future-Caller"] = "boom"

    assert headers is not worker._SYNTH_H, "a fail-safe path returned the shared constant itself"
    assert before == worker._SYNTH_H, "mutating the returned dict corrupted the module constant"
    assert EFFECTIVE_TENANT_HEADER not in worker._SYNTH_H


# ── The stamp ships DARK ─────────────────────────────────────────────────────


def test_the_stamp_is_off_unless_explicitly_enabled(worker_module) -> None:
    """DEFAULT OFF. On the in-cluster hop the receiver ACCEPTS the delegation and binds
    license_id=None, so an unprovisioned customer tenant 4xxes and the job poisons.
    The mechanism deploys dark and is enabled per hop after the receiver is checked."""
    assert worker_module.SYNTH_EFFECTIVE_TENANT_ENABLED is False

    headers = _synth_headers_for(
        worker_module,
        {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0, "tenant_id": JOB_TENANT},
    )

    assert EFFECTIVE_TENANT_HEADER not in headers, (
        "the synth hop re-scoped live traffic with the flag unset — this must ship dark"
    )
    assert headers["X-Tenant-Id"] == f"Bearer {SYNTH_TOKEN}"


@pytest.mark.parametrize("raw", ["", "  ", "0", "false", "no", "off", "False", "maybe"])
def test_non_affirmative_flag_values_stay_off(monkeypatch, worker_module, raw: str) -> None:
    """An operator writing ``=false`` must not be read as consent by a truthiness test."""
    monkeypatch.setenv("SYNTH_EFFECTIVE_TENANT_ENABLED", raw)
    assert worker_module._env_flag("SYNTH_EFFECTIVE_TENANT_ENABLED") is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " True "])
def test_affirmative_flag_values_turn_it_on(monkeypatch, worker_module, raw: str) -> None:
    monkeypatch.setenv("SYNTH_EFFECTIVE_TENANT_ENABLED", raw)
    assert worker_module._env_flag("SYNTH_EFFECTIVE_TENANT_ENABLED") is True


# ── The ENV VAR, not the module attribute ────────────────────────────────────
#
# Every test above enables the stamp with `monkeypatch.setattr(worker_module,
# "SYNTH_EFFECTIVE_TENANT_ENABLED", True)` — it patches the module ATTRIBUTE and
# never sets the environment variable an operator actually sets. That leaves a
# hole big enough to drive the feature through:
#
#   * the off-tests pass with a typo'd key (a typo'd key is also unset);
#   * the on-tests pass with a typo'd key (the attribute is patched directly);
#   * the `_env_flag` tests pass with a typo'd key (the test SUPPLIES the key).
#
# So a single-character slip in the key string at the `SYNTH_EFFECTIVE_TENANT_ENABLED
# = _env_flag(...)` line leaves the whole suite green and the flag permanently dead
# in the cluster — the inert-wiring shape this fleet keeps shipping. This test is the
# only one that binds the literal key an operator types to the header on the wire.


@pytest.fixture
def worker_enabled_via_env(monkeypatch) -> Iterator[ModuleType]:
    """The worker as the CLUSTER builds it: env var set, module imported fresh.

    No `setattr` on the module anywhere in this fixture. The flag must come from the
    environment through the module-level `_env_flag(...)` call, exactly as it does in
    a running pod.
    """
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("TTS_SYNTH_TOKEN", SYNTH_TOKEN)
    monkeypatch.delenv("TTS_SYNTH_ALLOW_SHARED_TOKEN", raising=False)
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", "mini1-worker")
    # THE POINT OF THIS FIXTURE: the real variable name, spelled out here so that a
    # typo at the worker.py call site makes this diverge and the assertions below fail.
    monkeypatch.setenv("SYNTH_EFFECTIVE_TENANT_ENABLED", "true")
    sys.modules.pop("worker.worker", None)
    mod = importlib.import_module("worker.worker")
    yield importlib.reload(mod)
    sys.modules.pop("worker.worker", None)


def test_the_env_var_alone_turns_the_flag_on(worker_enabled_via_env) -> None:
    """Setting only the environment variable must flip the module-level flag."""
    assert worker_enabled_via_env.SYNTH_EFFECTIVE_TENANT_ENABLED is True, (
        "SYNTH_EFFECTIVE_TENANT_ENABLED=true in the environment did not enable the "
        "flag — the key passed to _env_flag() in worker.py does not match the key an "
        "operator sets, so the feature is dead in the cluster and every other test "
        "still passes because they patch the module attribute instead"
    )


def test_the_env_var_alone_puts_the_tenant_on_the_wire(worker_enabled_via_env) -> None:
    """END TO END through the real switch: env var in, header out.

    This is the assertion that a typo'd key at the `_env_flag(...)` call site cannot
    survive. It exercises the full production path — environment variable, module
    import, `_env_flag`, the module global, `_synth_headers`, the outbound request —
    with nothing monkeypatched except the environment and the HTTP client.
    """
    headers = _synth_headers_for(
        worker_enabled_via_env,
        {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0, "tenant_id": JOB_TENANT},
    )

    assert headers[EFFECTIVE_TENANT_HEADER] == JOB_TENANT, (
        "the stamp did not reach the wire when enabled the way the cluster enables "
        "it. The mechanism is inert: an operator setting the documented variable gets "
        "nothing, and no other test in this file can see it"
    )
    assert headers["X-Tenant-Id"] == f"Bearer {SYNTH_TOKEN}"
    assert GATEWAY_TOKEN not in str(headers)
