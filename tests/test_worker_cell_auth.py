"""P7: workers authenticate to the gateway with SERVICE-scope cell tokens.

Live before this change: both worker Deployments polled ``POST /api/tts/claim``
~37x/min with only the shared w2v service-account licence on the deprecated
``X-Tenant-Id: Bearer`` header — no tenant, so no cell token was minted, and the
gateway counted every poll as legacy (``w2v_cell_legacy_delegation_total``). Under
``CELL_AUTH_MODE=enforce`` that is a 401 on every worker route: a dead narration queue.

Pinned here, against the REAL w2v_common middleware, signer, verifier and keyring
(no auth mocks; only the queue/DB/storage behind the routes is stubbed):

* the worker's gateway client stamps a fresh ``scope=service`` token, ``aud``
  ``tts-gateway``, ``iss`` = its own cell identity — and the synth client never does;
* ``enforce``: a worker service token with NO licence reaches every worker route;
  any other cell caller is refused;
* ``dual``: the licence still decides, exactly as before.
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from jwt.algorithms import OKPAlgorithm
from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    mint_cell_token,
    mint_service_cell_token,
    reset_cell_state,
)
from w2v_common.auth.identity_context import clear_identity_context
from w2v_common.auth.middleware import AuthMiddleware
from w2v_common.discovery import service_url

from src.api.routes import workers
from src.database.session import get_raw_session

DEPLOYMENT = "strato-w2v"
GATEWAY_SERVICE = "tts-gateway"
WORKER = "tts-gateway-worker"
WORKER_MINI1 = "tts-gateway-worker-mini1"
ORCH = "w2v-orchestrator"
# In the keyring AND in CELL_ALLOWED_CALLERS, so the ONLY thing that can refuse it on
# a worker route is the route's own opt-in — the control under test.
INTRUDER = "analysis-service"

WORKER_SUBJECT = "11111111-1111-1111-1111-111111111111"
ATTACKER_SUBJECT = "99999999-9999-9999-9999-999999999999"
TENANT = "0198f1d2-0000-7000-8000-0000000c3111"
GATEWAY_TOKEN = "gw-licence-token"  # noqa: S105 # pragma: allowlist secret
IN_CELL_GATEWAY = "http://tts-gateway.w2v:6320"

_CELL_ENV = (
    "CELL_AUTH_MODE",
    "CELL_SERVICE_NAME",
    "CELL_DEPLOYMENT_ID",
    "CELL_JWKS_PATH",
    "CELL_JWT_SIGNING_KEY",
    "CELL_JWT_KID",
    "CELL_ALLOWED_CALLERS",
    "AUTH_DELEGATING_SUBJECTS",
    "W2V_LOCAL",
    "W2V_TTS_GATEWAY_URL",
)


@pytest.fixture(scope="module")
def keys() -> dict[str, Ed25519PrivateKey]:
    return {name: Ed25519PrivateKey.generate() for name in (WORKER, WORKER_MINI1, ORCH, INTRUDER)}


def _pem(key: Ed25519PrivateKey) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture(autouse=True)
def cell_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keys: dict[str, Ed25519PrivateKey]
) -> Iterator[None]:
    entries = []
    for name, key in keys.items():
        entry = json.loads(OKPAlgorithm.to_jwk(key.public_key()))
        entry.update({"kid": f"{name}-kid", "iss": name})
        entries.append(entry)
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": entries}))
    for name in _CELL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", DEPLOYMENT)
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))
    reset_cell_state()
    clear_identity_context()
    yield
    reset_cell_state()
    clear_identity_context()


def _sign_as(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey], name: str
) -> None:
    monkeypatch.setenv("CELL_SERVICE_NAME", name)
    monkeypatch.setenv("CELL_JWT_KID", f"{name}-kid")
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem(keys[name]))


def _receive_as_gateway(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """This process becomes tts-gateway with its LIVE allow-list (+ the intruder)."""
    for name in ("CELL_JWT_SIGNING_KEY", "CELL_JWT_KID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CELL_SERVICE_NAME", GATEWAY_SERVICE)
    monkeypatch.setenv("CELL_AUTH_MODE", mode)
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", f"{ORCH},{WORKER},{WORKER_MINI1},{INTRUDER}")


def _decode(token: str) -> dict[str, Any]:
    return jwt.decode(token, options={"verify_signature": False})


# ── Worker side ──────────────────────────────────────────────────────────────


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("TTS_GATEWAY_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("TTS_SYNTH_TOKEN", "synth-only-token")  # pragma: allowlist secret
    monkeypatch.delenv("TTS_SYNTH_ALLOW_SHARED_TOKEN", raising=False)
    monkeypatch.setenv("TTS_GATEWAY_URL", IN_CELL_GATEWAY)
    monkeypatch.setenv("TTS_SYNTH_URL", "http://synth.test")
    monkeypatch.setenv("TTS_WORKER_ID", "incluster-mini2")
    sys.modules.pop("worker.worker", None)
    mod = importlib.import_module("worker.worker")
    yield importlib.reload(mod)
    sys.modules.pop("worker.worker", None)


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
        if request.url.path.endswith("/claim"):
            return httpx.Response(204)
        return httpx.Response(204)


def test_default_gateway_url_is_the_in_cell_endpoint(worker: ModuleType) -> None:
    """The hook mints only for the discovery endpoint — keep the two in lockstep."""
    assert service_url("tts_gateway") == IN_CELL_GATEWAY == worker.GATEWAY


@pytest.mark.parametrize("identity", [WORKER, WORKER_MINI1])
def test_claim_poll_carries_a_service_token_for_the_gateway(
    worker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
    identity: str,
) -> None:
    _sign_as(monkeypatch, keys, identity)
    rec = _Recorder()
    with worker._gateway_client(transport=httpx.MockTransport(rec.handler)) as gw:
        gw.post(f"{worker.GATEWAY}/api/tts/claim", headers=worker._GW_H, json={"worker_id": "w"})

    (request,) = rec.requests
    claims = _decode(request.headers[CELL_TOKEN_HEADER])
    assert claims["aud"] == GATEWAY_SERVICE
    assert claims["iss"] == identity
    assert claims["scope"] == "service"
    assert claims["tenant"] is None
    # Legacy lane moved off the deprecated header, same licence.
    assert request.headers["Authorization"] == f"Bearer {GATEWAY_TOKEN}"
    assert "X-Tenant-Id" not in request.headers


def test_every_poll_gets_a_fresh_token(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]
) -> None:
    """A replayed jti is refused by the gateway: the token must not be cached."""
    _sign_as(monkeypatch, keys, WORKER)
    rec = _Recorder()
    with worker._gateway_client(transport=httpx.MockTransport(rec.handler)) as gw:
        for _ in range(2):
            gw.post(f"{worker.GATEWAY}/api/tts/claim", headers=worker._GW_H, json={})
    jtis = {_decode(r.headers[CELL_TOKEN_HEADER])["jti"] for r in rec.requests}
    assert len(jtis) == 2


def test_job_calls_are_signed_and_the_synth_hop_never_is(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]
) -> None:
    """result (and the gateway half of a job) is signed; the synth host gets nothing."""
    _sign_as(monkeypatch, keys, WORKER)
    gw_rec, synth_rec = _Recorder(), _Recorder()
    with (
        worker._gateway_client(transport=httpx.MockTransport(gw_rec.handler)) as gw,
        worker._synth_client(transport=httpx.MockTransport(synth_rec.handler)) as synth,
    ):
        worker._process(gw, synth, {"job_id": "job-1", "text": "Hi.", "speed": 1.0})

    (synth_request,) = synth_rec.requests
    assert CELL_TOKEN_HEADER not in synth_request.headers
    assert GATEWAY_TOKEN not in str(synth_request.headers)
    result = [r for r in gw_rec.requests if r.url.path.endswith("/result")]
    assert len(result) == 1
    assert _decode(result[0].headers[CELL_TOKEN_HEADER])["scope"] == "service"


def test_failure_report_is_signed(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]
) -> None:
    _sign_as(monkeypatch, keys, WORKER)
    gw_rec = _Recorder()

    def _synth_down(_request: httpx.Request) -> httpx.Response:
        # 500, not 503: since #45 a 503/429 is a GPU-BUSY signal that is retried in
        # place and then RELEASED, never /fail'ed. A genuine failure is what reaches
        # /fail, which is the signed call this test is about.
        return httpx.Response(500)

    with (
        worker._gateway_client(transport=httpx.MockTransport(gw_rec.handler)) as gw,
        worker._synth_client(transport=httpx.MockTransport(_synth_down)) as synth,
    ):
        worker._process(gw, synth, {"job_id": "job-1", "text": "Hi.", "speed": 1.0})

    fail = [r for r in gw_rec.requests if r.url.path.endswith("/fail")]
    assert len(fail) == 1
    assert _decode(fail[0].headers[CELL_TOKEN_HEADER])["iss"] == WORKER


def test_gpu_busy_release_is_signed(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]
) -> None:
    """The #45 busy-exhausted /release rides the GATEWAY client, so it is signed too."""
    _sign_as(monkeypatch, keys, WORKER)
    monkeypatch.setattr(worker, "BUSY_MAX_RETRIES", 0)
    gw_rec = _Recorder()

    def _synth_busy(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"Retry-After": "1"})

    with (
        worker._gateway_client(transport=httpx.MockTransport(gw_rec.handler)) as gw,
        worker._synth_client(transport=httpx.MockTransport(_synth_busy)) as synth,
    ):
        worker._process(gw, synth, {"job_id": "job-1", "text": "Hi.", "speed": 1.0})

    release = [r for r in gw_rec.requests if r.url.path.endswith("/release")]
    assert len(release) == 1
    assert not [r for r in gw_rec.requests if r.url.path.endswith("/fail")]
    assert _decode(release[0].headers[CELL_TOKEN_HEADER])["iss"] == WORKER


def test_shutdown_release_is_signed(
    worker: ModuleType, monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]
) -> None:
    _sign_as(monkeypatch, keys, WORKER)
    rec = _Recorder()
    real_client = httpx.Client

    def _client(*, transport: object = None, **kwargs: Any) -> httpx.Client:
        del transport  # the worker passes None; route the release to the recorder instead
        return real_client(transport=httpx.MockTransport(rec.handler), **kwargs)

    monkeypatch.setattr(worker.httpx, "Client", _client)
    worker._CURRENT["job_id"] = "job-1"
    worker._release_current()

    (request,) = rec.requests
    assert request.url.path == "/api/tts/jobs/job-1/release"
    assert _decode(request.headers[CELL_TOKEN_HEADER])["aud"] == GATEWAY_SERVICE


@pytest.mark.parametrize(
    "url",
    ["https://tts-gateway.test/api/tts/claim", "http://tts-gateway:6320/api/tts/claim"],
    ids=["internal-alias", "short-name"],
)
def test_non_discovery_gateway_url_gets_no_token(
    worker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
    url: str,
) -> None:
    """Why TTS_GATEWAY_URL must stay the in-cluster Service URL: an alias is unsigned."""
    _sign_as(monkeypatch, keys, WORKER)
    rec = _Recorder()
    with worker._gateway_client(transport=httpx.MockTransport(rec.handler)) as gw:
        gw.post(url, headers={**worker._GW_H, CELL_TOKEN_HEADER: "stale"}, json={})
    assert CELL_TOKEN_HEADER not in rec.requests[0].headers


def test_unconfigured_signer_sends_no_token(worker: ModuleType) -> None:
    """No key (local dev, pre-secret rollout) degrades to the licence lane alone."""
    rec = _Recorder()
    with worker._gateway_client(transport=httpx.MockTransport(rec.handler)) as gw:
        gw.post(f"{worker.GATEWAY}/api/tts/claim", headers=worker._GW_H, json={})
    assert CELL_TOKEN_HEADER not in rec.requests[0].headers


# ── Gateway side: the real AuthMiddleware in front of the worker router ──────


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def _claim_next(_session: object, **_kw: object) -> dict[str, object]:
        calls.append("claim_next")
        return {
            "job_id": "job-1",
            "voice": "af_heart",
            "speed": 1.0,
            "provider": "kokoro",
            "text": "hello",
            "tenant_id": TENANT,
        }

    async def _renew_lease(_session: object, **_kw: object) -> bool:
        calls.append("renew_lease")
        return True

    async def _keyinfo(_session: object, **_kw: object) -> dict[str, str]:
        return {
            "tenant_id": TENANT,
            "idempotency_key": "idem-1",
            "status": "claimed",
            "claimed_by": "w",
        }

    async def _complete(_session: object, **_kw: object) -> bool:
        calls.append("complete")
        return True

    async def _release_claim(_session: object, **_kw: object) -> bool:
        calls.append("release_claim")
        return True

    async def _fail(_session: object, **_kw: object) -> str:
        calls.append("fail")
        return "requeued"

    class _Storage:
        def key_for(self, tenant_id: str, idempotency_key: str) -> str:
            return f"{tenant_id}/{idempotency_key}.wav"

        async def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
            return key

    monkeypatch.setattr(workers.jm, "claim_next", _claim_next)
    monkeypatch.setattr(workers.jm, "renew_lease", _renew_lease)
    monkeypatch.setattr(workers.jm, "get_job_keyinfo", _keyinfo)
    monkeypatch.setattr(workers.jm, "complete", _complete)
    monkeypatch.setattr(workers.jm, "release_claim", _release_claim)
    monkeypatch.setattr(workers.jm, "fail", _fail)
    monkeypatch.setattr(workers, "get_storage", lambda: _Storage())
    monkeypatch.setattr(
        workers,
        "get_settings",
        lambda: SimpleNamespace(
            max_result_bytes=1024,
            queue=SimpleNamespace(
                lease_seconds=180,
                max_attempts=4,
                max_job_lifetime_seconds=3600,
                max_busy_age_seconds=86400,
            ),
            worker_subject_id_set=lambda: frozenset({WORKER_SUBJECT}),
        ),
    )
    return calls


@pytest_asyncio.fixture
async def gateway(reached: list[str]) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(workers.router)
    app.dependency_overrides[get_raw_session] = lambda: None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=IN_CELL_GATEWAY
    ) as client:
        yield client


Call = Callable[[httpx.AsyncClient, dict[str, str]], Any]


async def _claim(c: httpx.AsyncClient, h: dict[str, str]) -> httpx.Response:
    return await c.post("/api/tts/claim", json={"worker_id": "w"}, headers=h)


async def _heartbeat(c: httpx.AsyncClient, h: dict[str, str]) -> httpx.Response:
    return await c.post("/api/tts/jobs/job-1/heartbeat", headers=h)


async def _result(c: httpx.AsyncClient, h: dict[str, str]) -> httpx.Response:
    return await c.post(
        "/api/tts/jobs/job-1/result",
        files={"audio": ("clip.wav", b"RIFFfake", "audio/wav")},
        data={"duration_seconds": "2.5", "worker_id": "w"},
        headers=h,
    )


async def _release(c: httpx.AsyncClient, h: dict[str, str]) -> httpx.Response:
    return await c.post("/api/tts/jobs/job-1/release", data={"worker_id": "w"}, headers=h)


async def _fail(c: httpx.AsyncClient, h: dict[str, str]) -> httpx.Response:
    return await c.post("/api/tts/jobs/job-1/fail", data={"error": "x"}, headers=h)


ROUTES = [
    pytest.param(_claim, 200, "claim_next", id="claim"),
    pytest.param(_heartbeat, 204, "renew_lease", id="heartbeat"),
    pytest.param(_result, 204, "complete", id="result"),
    pytest.param(_release, 200, "release_claim", id="release"),
    pytest.param(_fail, 200, "fail", id="fail"),
]


def _service_token(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey], issuer: str
) -> str:
    _sign_as(monkeypatch, keys, issuer)
    return mint_service_cell_token(GATEWAY_SERVICE)


def _introspect(subject: str) -> Any:
    return patch(
        "w2v_common.auth.middleware._introspect_license",
        new_callable=AsyncMock,
        return_value={"active": True, "tenant_id": TENANT, "subject_id": subject},
    )


@pytest.mark.parametrize(("call", "expected_ok", "marker"), ROUTES)
@pytest.mark.parametrize("identity", [WORKER, WORKER_MINI1])
async def test_enforce_admits_a_worker_service_token_with_no_licence(
    gateway: httpx.AsyncClient,
    reached: list[str],
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
    identity: str,
    call: Call,
    expected_ok: int,
    marker: str,
) -> None:
    token = _service_token(monkeypatch, keys, identity)
    _receive_as_gateway(monkeypatch, "enforce")

    with _introspect(ATTACKER_SUBJECT) as introspect:
        response = await call(gateway, {CELL_TOKEN_HEADER: token})

    assert response.status_code == expected_ok, response.text
    assert marker in reached
    introspect.assert_not_awaited()  # no licence was involved at all


@pytest.mark.parametrize(("call", "expected_ok", "marker"), ROUTES)
async def test_enforce_refuses_a_service_token_from_another_cell_caller(
    gateway: httpx.AsyncClient,
    reached: list[str],
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
    call: Call,
    expected_ok: int,
    marker: str,
) -> None:
    """Keyring + CELL_ALLOWED_CALLERS both pass it: only the route's opt-in refuses."""
    token = _service_token(monkeypatch, keys, INTRUDER)
    _receive_as_gateway(monkeypatch, "enforce")

    response = await call(gateway, {CELL_TOKEN_HEADER: token})

    assert response.status_code in {401, 403}, response.text
    assert reached == []


@pytest.mark.parametrize(("call", "expected_ok", "marker"), ROUTES)
async def test_enforce_refuses_a_tenant_scope_token_even_from_the_orchestrator(
    gateway: httpx.AsyncClient,
    reached: list[str],
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
    call: Call,
    expected_ok: int,
    marker: str,
) -> None:
    """A valid in-cell TENANT identity is still not the worker."""
    _sign_as(monkeypatch, keys, ORCH)
    token = mint_cell_token(GATEWAY_SERVICE, tenant=TENANT)
    _receive_as_gateway(monkeypatch, "enforce")

    response = await call(gateway, {CELL_TOKEN_HEADER: token})

    assert response.status_code in {401, 403}, response.text
    assert reached == []


async def test_enforce_refuses_the_worker_licence_alone(
    gateway: httpx.AsyncClient, reached: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this migration exists to avoid: today's poll, under enforce.

    Live, the worker licence's subject is a DELEGATING service account
    (AUTH_DELEGATING_SUBJECTS), so enforce refuses it on the licence lane — every
    worker route would 401 if the worker did not also sign.
    """
    _receive_as_gateway(monkeypatch, "enforce")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", WORKER_SUBJECT)
    with _introspect(WORKER_SUBJECT):
        response = await _claim(gateway, {"Authorization": f"Bearer {GATEWAY_TOKEN}"})
    assert response.status_code == 401, response.text
    assert reached == []


@pytest.mark.parametrize(("subject", "expected"), [(WORKER_SUBJECT, 200), (ATTACKER_SUBJECT, 403)])
async def test_dual_with_a_licence_the_licence_still_decides(
    gateway: httpx.AsyncClient,
    reached: list[str],
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
    subject: str,
    expected: int,
) -> None:
    """Dual + licence: the worker token is only recorded; WORKER_SUBJECT_IDS decides."""
    token = _service_token(monkeypatch, keys, WORKER)
    _receive_as_gateway(monkeypatch, "dual")

    with _introspect(subject):
        response = await _claim(
            gateway, {CELL_TOKEN_HEADER: token, "Authorization": f"Bearer {GATEWAY_TOKEN}"}
        )

    assert response.status_code == expected, response.text
    assert ("claim_next" in reached) is (expected == 200)


async def test_dual_without_a_licence_admits_the_worker_token(
    gateway: httpx.AsyncClient,
    reached: list[str],
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
) -> None:
    token = _service_token(monkeypatch, keys, WORKER)
    _receive_as_gateway(monkeypatch, "dual")

    response = await _claim(gateway, {CELL_TOKEN_HEADER: token})

    assert response.status_code == 200, response.text
    assert "claim_next" in reached


def test_worker_callers_match_the_deployments() -> None:
    assert frozenset({WORKER, WORKER_MINI1}) == workers.WORKER_CELL_CALLERS


@pytest.mark.parametrize("mode", ["enforce", "dual"])
@pytest.mark.parametrize(("call", "expected_ok", "marker"), ROUTES)
async def test_a_tenant_cell_token_cannot_borrow_the_worker_subject_by_header(
    gateway: httpx.AsyncClient,
    reached: list[str],
    monkeypatch: pytest.MonkeyPatch,
    keys: dict[str, Ed25519PrivateKey],
    mode: str,
    call: Call,
    expected_ok: int,
    marker: str,
) -> None:
    """The cell TENANT lane binds the subject from the caller's ``X-Subject-Id`` header.

    That header is a forwarded run attribution, not an authenticated identity, so the
    legacy ``WORKER_SUBJECT_IDS`` fallback must not read it: otherwise any allowed
    cell caller (the orchestrator) reaches the cross-tenant queue by naming the
    worker's subject — with no licence at all (enforce, or dual without one).
    """
    _sign_as(monkeypatch, keys, ORCH)
    token = mint_cell_token(GATEWAY_SERVICE, tenant=TENANT)
    _receive_as_gateway(monkeypatch, mode)

    response = await call(gateway, {CELL_TOKEN_HEADER: token, "X-Subject-Id": WORKER_SUBJECT})

    assert response.status_code == 403, response.text
    assert reached == []
