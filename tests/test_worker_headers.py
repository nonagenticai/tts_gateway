"""The worker must not hand its GATEWAY credential to the synth endpoint.

``TTS_SYNTH_URL`` is operator-configurable and points at a box the gateway does not
own. A single header dict shared across both destinations means every synth call
ships the service-account license token — the credential that can claim, complete
and fail jobs for EVERY tenant — to whatever host that env var names. A typo, a
stale ConfigMap or a compromised kokoro box is then a full gateway-credential
disclosure, and nothing in the gateway would ever notice.

So the two destinations get two distinct header sets: the gateway calls carry the
gateway token, the synth call carries only its own (``TTS_SYNTH_TOKEN``).

The split used to be INERT, because ``TTS_SYNTH_TOKEN`` fell back to the gateway
token when unset — which is exactly how production ran. The SILENT fallback is gone:
absent a scoped token the worker refuses to start, unless the operator opts in with
``TTS_SYNTH_ALLOW_SHARED_TOKEN``, which warns at every boot. Sharing cannot simply be
banned — one worker's synth target 402s a synth-scoped tenant (see below).

No network: httpx is monkeypatched, as in test_worker_shutdown.
"""

from __future__ import annotations

import base64
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

GATEWAY_TOKEN = "gw-secret-token"  # noqa: S105 # pragma: allowlist secret
SYNTH_TOKEN = "synth-only-token"  # noqa: S105 # pragma: allowlist secret


@pytest.fixture
def worker(monkeypatch) -> Iterator[ModuleType]:
    """Import worker.worker with distinct gateway/synth credentials in env."""
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("TTS_SYNTH_TOKEN", SYNTH_TOKEN)
    monkeypatch.delenv("TTS_SYNTH_ALLOW_SHARED_TOKEN", raising=False)
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", "mini2-worker")
    sys.modules.pop("worker.worker", None)
    mod = importlib.import_module("worker.worker")
    yield importlib.reload(mod)
    sys.modules.pop("worker.worker", None)


@pytest.fixture
def worker_with_duplicate_synth_token(monkeypatch) -> Iterator[ModuleType]:
    """TTS_SYNTH_TOKEN set, but to the SAME value as the gateway token.

    This is the likely result of a rushed remediation: GitOps'
    `tts-gateway-secrets` holds exactly one token (DOWNSTREAM_LICENSE_TOKEN, which
    TTS_GATEWAY_TOKEN already reads), so "wire TTS_SYNTH_TOKEN in GitOps" resolves
    to the same value. Byte-identical means the synth host receives exactly what it
    received before — the configuration changed, the exposure did not.
    """
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("TTS_SYNTH_TOKEN", GATEWAY_TOKEN)
    monkeypatch.delenv("TTS_SYNTH_ALLOW_SHARED_TOKEN", raising=False)
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", "mini2-worker")
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
    """Records every POST; answers the synth URL with a valid kokoro payload."""

    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def post(self, url: str, **kwargs) -> _Response:
        self._calls.append({"url": url, **kwargs})
        if "/synthesize" in url:
            return _Response(
                {"audio_base64": base64.b64encode(b"RIFFfake").decode(), "duration_sec": 2.5}
            )
        return _Response()


def _run_one_job(worker: ModuleType) -> list[dict]:
    calls: list[dict] = []
    _process_one(worker, _FakeClient(calls), {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0})
    return calls


def _by_url(calls: list[dict], needle: str) -> dict:
    matches = [c for c in calls if needle in c["url"]]
    assert len(matches) == 1, f"expected exactly one {needle} call, got {len(matches)}"
    return matches[0]


def test_synth_call_does_not_carry_the_gateway_token(worker) -> None:
    """THE LEAK: a shared ``_H`` sends the gateway credential to a third party."""
    calls = _run_one_job(worker)

    synth_headers = _by_url(calls, "/synthesize")["headers"]
    leaked = [v for v in synth_headers.values() if GATEWAY_TOKEN in str(v)]
    assert leaked == [], (
        "the gateway service-account license token was sent to TTS_SYNTH_URL — "
        "a misconfigured or compromised synth host now holds fleet-wide queue credentials"
    )
    # It still authenticates, just with its own credential (tts_backend enforces auth).
    assert synth_headers["X-Tenant-Id"] == f"Bearer {SYNTH_TOKEN}"
    assert synth_headers["Accept"] == "application/json"


def test_gateway_calls_still_carry_the_gateway_token(worker) -> None:
    """The split must not silently de-authenticate the gateway side."""
    calls = _run_one_job(worker)

    result_headers = _by_url(calls, "/result")["headers"]
    # Authorization: Bearer — the deprecated X-Tenant-Id: Bearer must be gone.
    assert result_headers["Authorization"] == f"Bearer {GATEWAY_TOKEN}"
    assert "X-Tenant-Id" not in result_headers
    assert SYNTH_TOKEN not in str(result_headers)


def test_failure_report_uses_the_gateway_headers(worker) -> None:
    """The /fail path is a gateway call too — same credential, not the synth one."""
    calls: list[dict] = []

    class _FailingSynth(_FakeClient):
        def post(self, url: str, **kwargs) -> _Response:
            if "/synthesize" in url:
                self._calls.append({"url": url, **kwargs})
                raise RuntimeError("kokoro down")
            return super().post(url, **kwargs)

    _process_one(worker, _FailingSynth(calls), {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0})

    fail_headers = _by_url(calls, "/fail")["headers"]
    assert fail_headers["Authorization"] == f"Bearer {GATEWAY_TOKEN}"
    assert "X-Tenant-Id" not in fail_headers


def test_synth_headers_and_gateway_headers_are_not_the_same_object(worker) -> None:
    """Structural guard: one shared dict is how the leak happened in the first place.

    A future gateway-only header (``X-License-Id`` / ``X-Subject-Id`` under the S2S
    forwarding convention) added to a shared dict would leak all over again.
    """
    assert worker._GW_H is not worker._SYNTH_H
    assert GATEWAY_TOKEN not in str(worker._SYNTH_H)


def test_explicit_synth_token_does_not_warn(worker, capsys) -> None:
    """A correctly split configuration must stay silent — no warning fatigue."""
    assert worker.SYNTH_TOKEN_IS_EXPLICIT_DUPLICATE is False

    worker._warn_if_sharing_gateway_token()

    assert capsys.readouterr().out == ""


def test_synth_token_identical_to_gateway_token_still_warns(
    worker_with_duplicate_synth_token, capsys
) -> None:
    """Setting the variable is not the fix; setting it to a DIFFERENT token is.

    Warning only on the unset case would let the no-op read as remediation: the
    operator sets TTS_SYNTH_TOKEN from the one token the secret has, the warning
    disappears, and the synth host still holds the fleet credential. Silence would
    then be a lie, which is worse than the original warning.
    """
    w = worker_with_duplicate_synth_token

    assert w.SYNTH_TOKEN_IS_EXPLICIT_DUPLICATE is True, "it IS set — just uselessly"

    w._warn_if_sharing_gateway_token()

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "no-op" in out, "the operator must be told this configuration changed nothing"
    assert GATEWAY_TOKEN not in out, "the warning must not print the credential itself"


def test_duplicate_flag_is_false_only_for_a_genuinely_distinct_token(worker) -> None:
    """The control: a real split is the one configuration that stays quiet."""
    assert worker.SYNTH_TOKEN_IS_EXPLICIT_DUPLICATE is False
    assert worker.SYNTH_TOKEN != worker.TOKEN


# ── Silent reuse of the fleet token is unreachable: three states, default closed ──
#
# Sharing cannot be banned outright. Verified live 2026-08-03: the mini2 worker's
# synth target (in-cluster modality-gateway) enforces BYO posture on the CALLER's
# tenant and answers a synth-scoped token with HTTP 402 "tenant must configure its
# own tts provider before use", while the gateway token gets a 200 with real audio.
# The mini1 worker (off-cluster kokoro) accepts either. So the guard is: scoped token
# wins; absent, sharing requires an explicit opt-in; absent and un-opted-in, refuse.


def _import_worker(
    monkeypatch, *, synth_token: str | None = None, allow_shared: str | None = None
) -> ModuleType:
    """Import worker.worker with a precisely controlled credential env."""
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("TTS_GATEWAY_URL", "http://gw.test")
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", "mini2-worker")
    for name, value in (
        ("TTS_SYNTH_TOKEN", synth_token),
        ("TTS_SYNTH_ALLOW_SHARED_TOKEN", allow_shared),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    sys.modules.pop("worker.worker", None)
    return importlib.import_module("worker.worker")


def test_absent_synth_token_never_ships_the_gateway_credential(monkeypatch) -> None:
    """THE DEFECT: unset TTS_SYNTH_TOKEN silently reused the FLEET-WIDE token.

    ``TTS_GATEWAY_TOKEN`` is wired from ``DOWNSTREAM_LICENSE_TOKEN`` — the credential
    every w2v service authenticates cross-service calls with. ``TTS_SYNTH_URL`` points
    at an operator-configurable box the gateway does not own. A warning is not a
    control: the worker logged one and then made the call anyway, so in production the
    synth host received the fleet credential on every synthesize.

    Two acceptable outcomes with no opt-in present, and only two: the worker refuses to
    start, or it starts and the gateway token appears in no synth-bound header.
    Degrading to "warn and send it" is neither.
    """
    try:
        mod = _import_worker(monkeypatch)
    except RuntimeError as exc:
        # Fail-closed: refused to start. The message must name the missing knob and
        # must not print the credential it declined to reuse.
        assert "TTS_SYNTH_TOKEN" in str(exc)
        assert GATEWAY_TOKEN not in str(exc)
        return
    finally:
        sys.modules.pop("worker.worker", None)

    synth_headers = _by_url(_run_one_job(mod), "/synthesize")["headers"]
    leaked = [k for k, v in synth_headers.items() if GATEWAY_TOKEN in str(v)]
    assert leaked == [], (
        f"the worker started without TTS_SYNTH_TOKEN and sent the fleet-wide "
        f"DOWNSTREAM_LICENSE_TOKEN to TTS_SYNTH_URL in header(s) {leaked} — sharing the "
        f"fleet credential must be an explicit opt-in, never a silent fallback"
    )


@pytest.mark.parametrize("synth_token", [None, ""], ids=["unset", "empty"])
@pytest.mark.parametrize(
    "allow_shared",
    [None, "", "false", "0", "no", "off"],
    ids=["absent", "blank", "false", "zero", "no", "off"],
)
def test_worker_refuses_to_start_without_a_token_or_an_optin(
    monkeypatch, synth_token, allow_shared
) -> None:
    """The DEFAULT is closed, and a negative opt-in stays closed.

    Empty string counts as an absent token: a Secret key present but blank is the same
    operational state as a missing one, and ``"" or TOKEN`` was the exact expression
    that produced the leak. And the opt-in must be AFFIRMATIVE — a bare truthiness test
    would read ``TTS_SYNTH_ALLOW_SHARED_TOKEN=false`` as consent, so an operator
    *disabling* the share would silently re-enable it.
    """
    try:
        with pytest.raises(RuntimeError) as exc:
            _import_worker(monkeypatch, synth_token=synth_token, allow_shared=allow_shared)
    finally:
        sys.modules.pop("worker.worker", None)

    msg = str(exc.value)
    assert "TTS_SYNTH_TOKEN" in msg, "the message must name the knob the operator must set"
    assert "TTS_SYNTH_ALLOW_SHARED_TOKEN" in msg, "and the escape hatch, since one exists"
    assert GATEWAY_TOKEN not in msg, "a startup error must not print the credential"


@pytest.mark.parametrize("allow_shared", ["1", "true", "TRUE", "yes", "on", " true "])
def test_explicit_optin_shares_the_gateway_token_and_says_so(
    monkeypatch, allow_shared, capsys
) -> None:
    """The mini2 shape: its synth target 402s a scoped tenant, so sharing is the only
    thing that works. It must still start — and it must never be quiet about it.
    """
    mod = _import_worker(monkeypatch, allow_shared=allow_shared)
    try:
        assert mod.SYNTH_TOKEN == mod.TOKEN, "the opt-in is what keeps this worker alive"
        assert mod.SYNTH_TOKEN_SHARING_IS_OPTED_IN is True
        assert mod.SYNTH_TOKEN_IS_SHARED is True

        mod._warn_if_sharing_gateway_token()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "TTS_SYNTH_ALLOW_SHARED_TOKEN" in out, "name the knob that caused this"
        assert "ACCEPTED" in out, "an accepted exposure must not read as a fixed one"
        assert GATEWAY_TOKEN not in out, "the warning must not print the credential itself"
    finally:
        sys.modules.pop("worker.worker", None)


def test_a_scoped_token_wins_over_the_optin(monkeypatch, capsys) -> None:
    """Opt-in is a fallback, not an override — it must not downgrade a good config.

    A worker carrying both (the likely transitional state while a scoped license is
    rolled out) must use the scoped token, and must go quiet.
    """
    mod = _import_worker(monkeypatch, synth_token=SYNTH_TOKEN, allow_shared="true")
    try:
        assert mod.SYNTH_TOKEN == SYNTH_TOKEN
        assert mod.SYNTH_TOKEN != mod.TOKEN
        assert mod.SYNTH_TOKEN_IS_SHARED is False

        mod._warn_if_sharing_gateway_token()
        assert capsys.readouterr().out == "", (
            "a genuinely separated credential must not warn, even with the opt-in set"
        )

        synth_headers = _by_url(_run_one_job(mod), "/synthesize")["headers"]
        assert GATEWAY_TOKEN not in str(synth_headers)
    finally:
        sys.modules.pop("worker.worker", None)


def test_the_synth_token_is_never_the_gateway_token_by_default(monkeypatch) -> None:
    """Structural: the old fallback was a single ``or`` and would be trivially
    reintroduced by a well-meaning "keep the old boxes running" patch. If SYNTH_TOKEN
    can equal TOKEN without the operator having said so, the leak is back.
    """
    mod = _import_worker(monkeypatch, synth_token=SYNTH_TOKEN)
    try:
        assert mod.SYNTH_TOKEN == SYNTH_TOKEN
        assert mod.SYNTH_TOKEN != mod.TOKEN
        assert mod.SYNTH_TOKEN_SHARING_IS_OPTED_IN is False
        assert not hasattr(mod, "SYNTH_TOKEN_IS_GATEWAY_FALLBACK"), (
            "the silent fallback flag is gone because the silent fallback is gone"
        )
    finally:
        sys.modules.pop("worker.worker", None)


# ── The claim's ``provider`` must actually reach the synth call ──────────────
#
# The gateway already SELECTs `provider` for every claim (job_manager.claim_next)
# and ships it in ClaimResponse (src/api/schemas.py). The worker dropped it on the
# floor and sent only {text, voice, speed}, so the synth body's `provider` stayed
# null all the way down.
#
# That null is load-bearing at the in-cluster hop: modality_gateway's
# `/api/tts/synthesize` skips the ENTIRE BYO resolve when `provider` is one of its
# local providers ("kokoro"/"kitten") and proxies straight to the platform GPU
# backend. With provider null the resolve runs, and a synth-SCOPED tenant hits
# `ModalityBYORequiredError` → HTTP 402 — which is the sole reason the worker has to
# be booted with TTS_SYNTH_ALLOW_SHARED_TOKEN and hand the fleet credential over.
# Sending the provider the job already carries makes that 402 branch unreachable and
# is the prerequisite for retiring the shared-token opt-in.
#
# Today's behaviour is unchanged either way: every tts_jobs row is born
# provider='kokoro' (DB server_default), tts_backend's SynthesizeRequest accepts the
# same literal, and TTS_DEFAULT_PROVIDER is 'kokoro' on both deployed backends — so
# the value sent equals the value that was being defaulted to.


def _job(**over) -> dict:
    return {"job_id": "job-xyz", "text": "Hi.", "speed": 1.0, **over}


def test_synth_body_carries_the_jobs_provider(worker) -> None:
    """THE GAP: the claim's provider never reached the synth request body."""
    calls: list[dict] = []
    _process_one(worker, _FakeClient(calls), _job(provider="kokoro"))

    body = _by_url(calls, "/synthesize")["json"]
    assert body.get("provider") == "kokoro", (
        "the worker dropped the claim's provider, so modality_gateway ran its BYO "
        "resolve and a synth-scoped tenant got HTTP 402 — the shared gateway token "
        "cannot be retired while this field is null"
    )
    # The rest of the :9202 contract is untouched.
    assert body["text"] == "Hi." and body["speed"] == 1.0


def test_a_non_default_provider_is_forwarded_verbatim(worker) -> None:
    """The field is WIRED, not hardcoded — a row set to 'kitten' must say kitten."""
    calls: list[dict] = []
    _process_one(worker, _FakeClient(calls), _job(provider="kitten"))

    assert _by_url(calls, "/synthesize")["json"]["provider"] == "kitten"


def test_a_missing_provider_is_omitted_not_sent_as_null(worker) -> None:
    """A claim without the field (an older gateway mid-rollout) must leave the key
    OUT, exactly as ``voice`` does — an explicit null is not the same as absent to a
    pydantic model that keys behaviour off it, and the backend's own default
    (TTS_DEFAULT_PROVIDER) is the right answer when we have nothing to say.
    """
    calls: list[dict] = []
    _process_one(worker, _FakeClient(calls), _job())

    assert "provider" not in _by_url(calls, "/synthesize")["json"]
