"""The entitlement gate on /api/tts/batches actually refuses an unlicensed tenant.

Until w2v_common 0.75.0 this service authenticated the licence token for tenant
IDENTITY and stopped there: ``AuthMiddleware`` answers "who are you", never "may
you". A tenant whose licence was ``expired`` or ``revoked`` still presented a
syntactically valid token, so both ``/api/tts/batches`` routes served them.
``LICENSE_GATE`` is mounted router-wide on ``src/api/routes/batches.py`` to
close that.

Every test here drives a REAL HTTP request through the REAL ``batches.router``
(and the REAL ``health.router``), because the interesting question is not
whether ``require_license`` works — w2v_common owns that — but whether THIS
service's routes actually carry it. This FastAPI build includes sub-routers
lazily (``_IncludedRouter``), so a parent router's ``dependencies=`` are applied
when a request is MATCHED, not baked into route objects at import time: static
introspection of ``route.dependant`` reports sub-router routes as ungated and
would be a *false negative*. A request is the only honest probe.

The paired 200s are what stop this file from passing vacuously: if the gate
were mounted wrongly (or the fixture were broken) the "entitled" and
"flag off" cases would not be 200s.
"""

from __future__ import annotations

from typing import Any, Self

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.routes import batches, health
from src.database.session import get_tenant_session

TENANT_1 = "11111111-1111-1111-1111-111111111111"
SUBJECT = "user-under-test"

#: The cheapest genuinely tenant-scoped route on the gated router: it needs no
#: request body, its only route-level dependency is ``get_tenant_session``
#: (overridable), and its only real work is ``jm.get_batch`` (patched). So a
#: non-200 here can only be the gate — which is the whole point.
GATED_PATH = "/api/tts/batches/batch-under-test"

#: One of the three probes this service exposes. Must answer 200 for a revoked
#: tenant: a 402/503 on a liveness/readiness/startup probe CrashLoops the pod,
#: and an unlicensed tenant cannot be billed by killing our own deployment.
HEALTH_PATH = "/health"
READYZ_PATH = "/readyz"
LIVEZ_PATH = "/livez"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT_1, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TENANT_1,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TENANT_1,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stand-in tenant session. The gate must refuse before this is ever used."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class _OkDbSession:
    """Stand-in for readyz's own DB probe, so this file needs no live Postgres."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, *_: object, **__: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _readyz_db_answers(monkeypatch):
    """/readyz runs a real ``SELECT 1``; stub it so this file needs no live
    Postgres and stays focused on the gate, not on database reachability."""
    monkeypatch.setattr(health, "session_factory", lambda: _OkDbSession)


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def _no_batch_db(monkeypatch):
    """Make the gated handler's own work free, so only the gate can fail it."""

    async def _empty(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr(batches.jm, "get_batch", _empty)


@pytest.fixture
def app() -> FastAPI:
    """The REAL routers, with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = TENANT_1
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(health.router)
    application.include_router(batches.router)

    async def _fake_tenant_session():
        async with _FakeSession() as session:
            yield session

    application.dependency_overrides[get_tenant_session] = _fake_tenant_session
    return application


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(app, monkeypatch, inventory, code):
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


@pytest.mark.asyncio
async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_armed_gate_serves_an_entitled_tenant(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["jobs"] == []


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated — all THREE of this service's unusual probe paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [HEALTH_PATH, READYZ_PATH, LIVEZ_PATH])
async def test_probe_is_served_for_a_revoked_tenant(app, monkeypatch, path):
    """A gated probe path CrashLoops the pod. All three probe paths this service
    exposes — /health (liveness), /readyz (readiness), /livez (startup) — must
    stay 200 for a revoked tenant."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, path)

    assert response.status_code == 200, f"{path}: {response.text}"


@pytest.mark.asyncio
async def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths."""
    from w2v_common.auth import is_license_exempt_path

    for path in (HEALTH_PATH, READYZ_PATH, LIVEZ_PATH, "/healthz", "/ready"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)
