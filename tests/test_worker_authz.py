"""Only the worker service account may reach the worker surface.

THE HOLE this closes: ``/api`` is guarded by w2v_common's license middleware, which
checks exactly two things — the token introspects ``active``, and it carries a
non-empty ``tenant_id``. Every customer license satisfies both. It checks no role, no
scope and no service-account identity, and ``workers.py`` used to add nothing on top.

So any tenant holding any valid license token could POST ``/api/tts/claim``, which
pops from the GLOBAL queue via a SECURITY DEFINER function that binds no tenant GUC,
and receive another tenant's **full narration text** together with its ``job_id`` — a
cross-tenant read. That ``job_id`` then addressed ``/result`` (inject arbitrary audio
into the victim's job), ``/heartbeat``, ``/release`` and ``/fail``. Claiming also
de-queues the job from the real worker fleet, so the read doubles as a DoS.

Tenant-scoping the by-id routes would NOT have fixed this. ``claim`` takes no job id
and is cross-tenant by construction; there is no tenant to scope it to. The only
control that expresses the requirement is an IDENTITY check: is the caller the worker
service account? Hence ``WORKER_SUBJECT_IDS`` + ``require_worker_subject``.

Every route is asserted SEPARATELY and each in its own parametrised case. A single
test asserting "claim 403s and heartbeat 403s" can pass while protecting neither if
the shared fixture is what fails, and would pass unchanged if a future edit moved the
guard from the router down onto one route.

No DB, no MinIO: the session dependency, job_manager and storage are stubbed, so a
non-403 response proves the request reached the handler.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.routing import APIRoute
from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

from src.api.routes import workers
from src.config import Settings
from src.database.session import get_raw_session

WORKER_SUBJECT = "11111111-1111-1111-1111-111111111111"
OTHER_WORKER_SUBJECT = "22222222-2222-2222-2222-222222222222"
ATTACKER_SUBJECT = "99999999-9999-9999-9999-999999999999"
VICTIM_TENANT = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

# The queue row an unguarded claim would have handed to whoever asked.
VICTIM_JOB_TEXT = "chapter one of the victim tenant's unpublished manuscript"


class _Storage:
    def key_for(self, tenant_id: str, idempotency_key: str) -> str:
        return f"{tenant_id}/{idempotency_key}.wav"

    async def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        return key


@pytest.fixture
def reached(monkeypatch) -> list[str]:
    """Record every job_manager call, so 'the handler ran' is observable.

    A 403 that still executed the handler would be a leak dressed as a denial —
    ``claim_next`` de-queues before anything looks at the response status.
    """
    calls: list[str] = []

    async def _claim_next(_session, **_kw) -> dict[str, object]:
        calls.append("claim_next")
        return {
            "job_id": "job-1",
            "voice": "af_heart",
            "speed": 1.0,
            "provider": "kokoro",
            "text": VICTIM_JOB_TEXT,
        }

    async def _renew_lease(_session, **_kw) -> bool:
        calls.append("renew_lease")
        return True

    async def _keyinfo(_session, **_kw) -> dict[str, str]:
        calls.append("get_job_keyinfo")
        # Mirrors the real SELECT. ``claimed_by`` matches the ``worker_id`` every
        # request below sends, so these cases exercise the AUTHZ guard rather than
        # short-circuiting on the result route's stale-writer check (that one has
        # its own module, tests/test_result_upload_guards.py).
        return {
            "tenant_id": VICTIM_TENANT,
            "idempotency_key": "idem-1",
            "status": "claimed",
            "claimed_by": "mini2",
        }

    async def _complete(_session, **_kw) -> bool:
        calls.append("complete")
        return True

    async def _release_claim(_session, **_kw) -> bool:
        calls.append("release_claim")
        return True

    async def _fail(_session, **_kw) -> str:
        calls.append("fail")
        return "requeued"

    monkeypatch.setattr(workers.jm, "claim_next", _claim_next)
    monkeypatch.setattr(workers.jm, "renew_lease", _renew_lease)
    monkeypatch.setattr(workers.jm, "get_job_keyinfo", _keyinfo)
    monkeypatch.setattr(workers.jm, "complete", _complete)
    monkeypatch.setattr(workers.jm, "release_claim", _release_claim)
    monkeypatch.setattr(workers.jm, "fail", _fail)
    monkeypatch.setattr(workers, "get_storage", lambda: _Storage())
    return calls


@pytest.fixture
def allowlist(monkeypatch) -> Callable[[str], None]:
    """Set WORKER_SUBJECT_IDS for the router's dependency (settings are lru_cached)."""

    def _set(value: str) -> None:
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
                worker_subject_id_set=lambda: frozenset(
                    s.strip() for s in value.split(",") if s.strip()
                ),
            ),
        )

    _set(f"{WORKER_SUBJECT},{OTHER_WORKER_SUBJECT}")
    return _set


@pytest_asyncio.fixture
async def api(reached, allowlist) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(workers.router)
    app.dependency_overrides[get_raw_session] = lambda: None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tts.test"
    ) as client:
        yield client
    clear_identity_context()


def as_subject(subject: str) -> None:
    """Stand in for AuthMiddleware: an ACTIVE license for a real tenant.

    Deliberately a fully valid identity — the point is that a valid license is
    precisely what the old code accepted, so the guard cannot be satisfied by
    authentication alone.
    """
    set_identity_context(
        tenant_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        license_id="lic-1",
        subject_id=subject,
    )


# Every worker route, each callable independently. `expected_ok` is the status the
# route returns once the guard passes — asserting it proves the allowed case reaches
# the handler rather than merely failing differently.
async def _claim(api: httpx.AsyncClient) -> httpx.Response:
    return await api.post("/api/tts/claim", json={"worker_id": "mini2"})


async def _heartbeat(api: httpx.AsyncClient) -> httpx.Response:
    return await api.post("/api/tts/jobs/job-1/heartbeat")


async def _result(api: httpx.AsyncClient) -> httpx.Response:
    return await api.post(
        "/api/tts/jobs/job-1/result",
        files={"audio": ("clip.wav", b"RIFFfake", "audio/wav")},
        data={"duration_seconds": "2.5", "worker_id": "mini2"},
    )


async def _release(api: httpx.AsyncClient) -> httpx.Response:
    return await api.post("/api/tts/jobs/job-1/release", data={"worker_id": "mini2"})


async def _fail(api: httpx.AsyncClient) -> httpx.Response:
    return await api.post("/api/tts/jobs/job-1/fail", data={"error": "kokoro down"})


ROUTES = [
    pytest.param(_claim, 200, "claim_next", id="claim"),
    pytest.param(_heartbeat, 204, "renew_lease", id="heartbeat"),
    pytest.param(_result, 204, "complete", id="result"),
    pytest.param(_release, 200, "release_claim", id="release"),
    pytest.param(_fail, 200, "fail", id="fail"),
]


# ── A merely-authenticated tenant is refused, route by route ─────────────────


@pytest.mark.parametrize(("call", "expected_ok", "handler_marker"), ROUTES)
async def test_non_allowlisted_subject_is_refused(
    api, reached, call, expected_ok, handler_marker
) -> None:
    """A valid license whose subject is not the worker's must not get in."""
    as_subject(ATTACKER_SUBJECT)

    response = await call(api)

    assert response.status_code == 403, (
        f"an authenticated non-worker tenant reached this route with {response.status_code}; "
        "an active license alone must not authorize the cross-tenant worker surface"
    )
    assert reached == [], (
        f"the guard returned 403 but the handler still ran ({reached}) — for /claim that "
        "means the job was de-queued and its text read before the denial"
    )


async def test_claim_does_not_leak_another_tenants_text(api, reached) -> None:
    """The concrete harm, pinned separately: no narration text in a refused response."""
    as_subject(ATTACKER_SUBJECT)

    response = await _claim(api)

    assert response.status_code == 403
    assert VICTIM_JOB_TEXT not in response.text
    assert "job-1" not in response.text, "a job_id is the key to result/heartbeat/fail"
    assert reached == []


# ── The worker itself still works ────────────────────────────────────────────


@pytest.mark.parametrize(("call", "expected_ok", "handler_marker"), ROUTES)
async def test_allowlisted_subject_is_admitted(
    api, reached, call, expected_ok, handler_marker
) -> None:
    """The guard must not break the fleet: the worker's subject passes through."""
    as_subject(WORKER_SUBJECT)

    response = await call(api)

    assert response.status_code == expected_ok, response.text
    assert handler_marker in reached, "the allowed call must actually reach the handler"


@pytest.mark.parametrize(("call", "expected_ok", "handler_marker"), ROUTES)
async def test_second_allowlisted_subject_is_admitted(
    api, reached, call, expected_ok, handler_marker
) -> None:
    """Multiple ids are honoured — a credential rotation must not need a queue outage."""
    as_subject(OTHER_WORKER_SUBJECT)

    response = await call(api)

    assert response.status_code == expected_ok, response.text
    assert handler_marker in reached


# ── Fail-closed: unconfigured means DENY ─────────────────────────────────────


@pytest.mark.parametrize(("call", "expected_ok", "handler_marker"), ROUTES)
@pytest.mark.parametrize(
    "value", ["", "   ", ",", " , "], ids=["empty", "blank", "comma", "commas"]
)
async def test_empty_allowlist_denies_every_route(
    api, reached, allowlist, value, call, expected_ok, handler_marker
) -> None:
    """An unset (or whitespace-only) WORKER_SUBJECT_IDS must deny, never allow-all.

    "Empty means unconfigured, so let everyone in" is how a guard becomes a no-op the
    day it ships: the gateway deploys before the ConfigMap lands, the fleet works, and
    nobody notices the hole is still open. The parse must also treat separators-only
    input as empty rather than as a subject named "".
    """
    allowlist(value)
    as_subject(WORKER_SUBJECT)

    response = await call(api)

    assert response.status_code == 403, (
        "an empty allowlist admitted a caller — unconfigured must fail CLOSED"
    )
    assert reached == []


async def test_missing_identity_is_refused(api, reached) -> None:
    """No introspected subject at all (empty ContextVar) must not match the allowlist.

    ``set_identity_context`` coerces a missing subject to ``""``, so a naive membership
    test against a set that somehow contained an empty string would admit everyone.
    """
    clear_identity_context()

    response = await _claim(api)

    assert response.status_code == 403
    assert reached == []


# ── Structural: the guard is on the ROUTER, not sprinkled per route ──────────


def test_guard_is_attached_to_the_router_itself() -> None:
    """So a route added later cannot be born unguarded.

    Per-route dependencies are opt-in, and the next worker endpoint would be written
    by copying a neighbour — which is exactly how the original omission happened.
    """
    router_deps = [d.dependency for d in workers.router.dependencies]
    assert workers.authorize_worker in router_deps

    # A route's resolved `dependant.dependencies` holds `Dependant` objects, whose
    # callable is `.call` — not the `.dependency` attribute that `params.Depends`
    # (the router-level list above) exposes. Mixing the two silently AttributeErrors.
    #
    # `routes` is typed as list[BaseRoute], and only APIRoute carries `dependant`,
    # so narrow before touching it rather than silencing the type checker: an
    # entry that is NOT an APIRoute is exactly the case we want to notice, since
    # it would be a worker endpoint the guard could not have been applied to.
    api_routes = [route for route in workers.router.routes if isinstance(route, APIRoute)]
    assert len(api_routes) == len(workers.router.routes), (
        "every route on the worker router must be an APIRoute — anything else "
        "cannot carry the dependency and would be silently unguarded"
    )
    guarded = [
        route
        for route in api_routes
        if workers.authorize_worker in [d.call for d in route.dependant.dependencies]
    ]
    assert len(guarded) == len(api_routes) >= 5, "every worker route must inherit the guard"


class TestRealSettingsParseIsFailClosed:
    """Exercise the REAL ``Settings`` parser, not a test double.

    The rest of this module monkeypatches ``get_settings`` with a stand-in that
    reimplements the same parse, so those cases prove the fixture correct and say
    nothing about production. That gap is not theoretical: deleting the
    ``if s.strip()`` filter in ``Settings.worker_subject_id_set`` leaves every
    other test in this file green while making the DEFAULT empty config parse to
    ``frozenset({""})`` — which then matches a subjectless caller and reopens the
    cross-tenant surface completely. These tests are what make that mutation fail.
    """

    @pytest.mark.parametrize("raw", ["", "   ", ",", " , ", ",,,", "\t\n"])
    def test_blank_shaped_values_all_parse_to_deny_all(self, raw: str) -> None:
        parsed = Settings(worker_subject_ids=raw).worker_subject_id_set()
        assert parsed == frozenset(), f"{raw!r} must parse to deny-all, got {parsed!r}"
        # The specific catastrophic shape: never admit the empty subject.
        assert "" not in parsed

    def test_default_is_deny_all(self) -> None:
        assert Settings().worker_subject_id_set() == frozenset()

    def test_values_are_stripped_and_blanks_dropped(self) -> None:
        parsed = Settings(worker_subject_ids="  a , ,b,  ").worker_subject_id_set()
        assert parsed == frozenset({"a", "b"})
        assert "" not in parsed
