"""The worker can run on its service cell token alone — no shared licence.

``TTS_GATEWAY_TOKEN`` is wired from the fleet-wide ``DOWNSTREAM_LICENSE_TOKEN``. Live,
both workers already sign every gateway call with a ``scope=service`` cell token
(gateway: ``w2v_cell_auth_total{lane="dual",scope="service"}``), and the gateway admits
that token on its own (``test_worker_cell_auth``: dual-without-licence and enforce).
What kept the shared token mandatory was the worker itself:
``os.environ["TTS_GATEWAY_TOKEN"]`` — a KeyError at import, i.e. a CrashLoop.

Pinned here, against the real w2v_common signer (no auth mocks):

* cell key only            → boots; gateway calls carry ONLY the cell token;
* neither                  → refuses to boot (loud), never an anonymous claim loop;
* key + licence (dual)     → unchanged: both credentials on every gateway call;
* cell-only + a gateway URL the signer will not mint for → refuses to boot;
* cell-only + TTS_SYNTH_ALLOW_SHARED_TOKEN → refuses to boot (nothing to share).
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jwt.algorithms import OKPAlgorithm
from w2v_common.auth.cell_jwt import CELL_TOKEN_HEADER, CellConfigError, reset_cell_state
from w2v_common.auth.identity_context import clear_identity_context

WORKER = "tts-gateway-worker"
IN_CELL_GATEWAY = "http://tts-gateway.w2v:6320"
GATEWAY_TOKEN = "gw-licence-token"  # noqa: S105 # pragma: allowlist secret
SYNTH_TOKEN = "synth-only-token"  # noqa: S105 # pragma: allowlist secret

_ENV = (
    "CELL_AUTH_MODE",
    "CELL_SERVICE_NAME",
    "CELL_DEPLOYMENT_ID",
    "CELL_JWKS_PATH",
    "CELL_JWT_SIGNING_KEY",
    "CELL_JWT_KID",
    "CELL_ALLOWED_CALLERS",
    "W2V_LOCAL",
    "W2V_TTS_GATEWAY_URL",
    "TTS_GATEWAY_TOKEN",
    "TTS_SYNTH_TOKEN",
    "TTS_SYNTH_ALLOW_SHARED_TOKEN",
    "TTS_GATEWAY_URL",
)


@pytest.fixture(scope="module")
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: Ed25519PrivateKey) -> Iterator[None]:
    """A clean env with NO gateway credential; each test adds exactly what it needs."""
    entry = json.loads(OKPAlgorithm.to_jwk(key.public_key()))
    entry.update({"kid": f"{WORKER}-kid", "iss": WORKER})
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [entry]}))
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", "strato-w2v")
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))
    monkeypatch.setenv("TTS_GATEWAY_URL", IN_CELL_GATEWAY)
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_SYNTH_TOKEN", SYNTH_TOKEN)
    monkeypatch.setenv("TTS_WORKER_ID", "cell-only-worker")
    reset_cell_state()
    clear_identity_context()
    sys.modules.pop("worker.worker", None)
    yield
    sys.modules.pop("worker.worker", None)
    reset_cell_state()
    clear_identity_context()


def _with_cell_key(monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey) -> None:
    monkeypatch.setenv("CELL_SERVICE_NAME", WORKER)
    monkeypatch.setenv("CELL_JWT_KID", f"{WORKER}-kid")
    monkeypatch.setenv(
        "CELL_JWT_SIGNING_KEY",
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
    )


def _import_worker() -> ModuleType:
    sys.modules.pop("worker.worker", None)
    return importlib.import_module("worker.worker")


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/synthesize"):
            return httpx.Response(
                200,
                json={"audio_base64": base64.b64encode(b"RIFFfake").decode(), "duration_sec": 2.0},
            )
        return httpx.Response(204)


def _run_one_job(worker: ModuleType) -> tuple[list[httpx.Request], list[httpx.Request]]:
    """A claim poll plus one full job, through the worker's own clients and headers."""
    gw_rec, synth_rec = _Recorder(), _Recorder()
    with (
        worker._gateway_client(transport=httpx.MockTransport(gw_rec.handler)) as gw,
        worker._synth_client(transport=httpx.MockTransport(synth_rec.handler)) as synth,
    ):
        gw.post(f"{worker.GATEWAY}/api/tts/claim", headers=worker._GW_H, json={"worker_id": "w"})
        worker._process(gw, synth, {"job_id": "job-1", "text": "Hi.", "speed": 1.0})
    return gw_rec.requests, synth_rec.requests


# ── cell key only ─────────────────────────────────────────────────────────────


def test_cell_key_only_boots_and_sends_only_the_cell_token(
    monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey
) -> None:
    _with_cell_key(monkeypatch, key)

    worker = _import_worker()

    assert worker.TOKEN is None
    assert worker.CELL_SIGNER_CONFIGURED is True
    assert worker._GW_H == {}
    gateway_calls, synth_calls = _run_one_job(worker)
    paths = [r.url.path for r in gateway_calls]
    assert "/api/tts/claim" in paths
    assert "/api/tts/jobs/job-1/result" in paths
    for request in gateway_calls:
        claims = jwt.decode(request.headers[CELL_TOKEN_HEADER], options={"verify_signature": False})
        assert (claims["aud"], claims["iss"], claims["scope"]) == ("tts-gateway", WORKER, "service")
        assert "Authorization" not in request.headers, request.url.path
        assert "X-Tenant-Id" not in request.headers, request.url.path
    # The synth hop is untouched: its own token, never a cell token.
    (synth_request,) = synth_calls
    assert synth_request.headers["X-Tenant-Id"] == f"Bearer {SYNTH_TOKEN}"
    assert CELL_TOKEN_HEADER not in synth_request.headers


def test_cell_key_only_with_a_blank_token_is_still_cell_only(
    monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey
) -> None:
    """``TTS_GATEWAY_TOKEN: ""`` (an emptied Secret key) must not become ``Bearer ``."""
    _with_cell_key(monkeypatch, key)
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", "   ")

    worker = _import_worker()

    assert worker.TOKEN is None
    assert worker._GW_H == {}


# ── neither ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("token", [None, "", "  "], ids=["unset", "empty", "blank"])
def test_neither_cell_key_nor_token_refuses_to_boot(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    if token is not None:
        monkeypatch.setenv("TTS_GATEWAY_TOKEN", token)

    with pytest.raises(RuntimeError, match="no gateway credential"):
        _import_worker()


def test_half_configured_signer_without_token_refuses_to_boot(
    monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey
) -> None:
    """A key without its kid is a config error, not 'no signer' — fail loud on it."""
    _with_cell_key(monkeypatch, key)
    monkeypatch.delenv("CELL_JWT_KID")

    with pytest.raises(CellConfigError):
        _import_worker()


# ── dual (key + licence) is unchanged ─────────────────────────────────────────


def test_cell_key_and_token_send_both_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey
) -> None:
    _with_cell_key(monkeypatch, key)
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)

    worker = _import_worker()

    assert {"Authorization": f"Bearer {GATEWAY_TOKEN}"} == worker._GW_H
    gateway_calls, synth_calls = _run_one_job(worker)
    claim = next(r for r in gateway_calls if r.url.path == "/api/tts/claim")
    assert claim.headers["Authorization"] == f"Bearer {GATEWAY_TOKEN}"
    assert (
        jwt.decode(claim.headers[CELL_TOKEN_HEADER], options={"verify_signature": False})["scope"]
        == "service"
    )
    assert GATEWAY_TOKEN not in str(synth_calls[0].headers)


def test_token_only_legacy_lane_still_boots(monkeypatch: pytest.MonkeyPatch) -> None:
    """No signer, licence set: today's pre-secret rollout shape keeps working."""
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)

    worker = _import_worker()

    assert worker.CELL_SIGNER_CONFIGURED is False
    assert {"Authorization": f"Bearer {GATEWAY_TOKEN}"} == worker._GW_H


# ── cell-only guards ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    ["https://tts-gateway.test", "http://tts-gateway:6320"],
    ids=["internal-alias", "short-name"],
)
def test_cell_only_with_a_gateway_url_the_signer_will_not_mint_for_refuses_to_boot(
    monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey, url: str
) -> None:
    """Cell-only has no fallback lane: an unsigned alias would 401 forever."""
    _with_cell_key(monkeypatch, key)
    monkeypatch.setenv("TTS_GATEWAY_URL", url)

    with pytest.raises(RuntimeError, match="not the gateway's in-cell endpoint"):
        _import_worker()


def test_cell_only_cannot_share_a_token_it_does_not_have(
    monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey
) -> None:
    """The mini2 worker's synth hop still shares the licence — dropping it must be loud."""
    _with_cell_key(monkeypatch, key)
    monkeypatch.delenv("TTS_SYNTH_TOKEN")
    monkeypatch.setenv("TTS_SYNTH_ALLOW_SHARED_TOKEN", "true")

    with pytest.raises(RuntimeError, match="no token to share"):
        _import_worker()
