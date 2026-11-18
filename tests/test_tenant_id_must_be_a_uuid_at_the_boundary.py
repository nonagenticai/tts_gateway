"""A non-uuid tenant is a 400, never a 500 and never a 404.

``tts_jobs.tenant_id`` became a real ``uuid`` column in migration
``0003_tenant_id_uuid``. That flip has a regression attached to it that is easy to
ship and hard to see: a request carrying a legacy tenant slug (``n8n-dev``) or no
tenant at all used to be inert — the value was text, it matched nothing, the caller
got an empty batch. Against a ``uuid`` column the same request reaches Postgres as::

    invalid input syntax for type uuid: "n8n-dev"

and the caller gets a **500**. Malformed client input reported as a server fault
pages the wrong on-call and buries a licence/middleware defect under an "outage".

So ``get_tenant_session`` — the single dependency both tenant-scoped routes hang
off — parses the tenant with ``w2v_common.tenant.parse_tenant_id`` and turns
``TenantError`` into a 400.

Three things are asserted, and the last two are what make the first non-vacuous:

* a slug tenant is **400** — specifically NOT 500 (the regression) and NOT 404
  ("unusable tenant" and "no such batch" are different answers);
* a missing tenant is **400** for the same reason — it would otherwise reach
  ``uuid.UUID("")`` and 500 identically;
* a real uuid tenant still reaches the handler, so the gate is proven to be a gate
  and not a blanket refusal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request

from src.api.routes import batches
from src.database import session as session_mod

VALID_TENANT = "11111111-1111-1111-1111-111111111111"
# The real legacy slug this fleet actually carried, not an invented string.
LEGACY_SLUG_TENANT = "n8n-dev"
# Unique to this file: the suite shares one database, so scoping the control read to
# a batch nothing else writes keeps it deterministic under any test order. Asserting
# "the whole table is empty" here would pass or fail depending on who ran first.
CONTROL_BATCH = "tenant-gate-control-batch"


@pytest_asyncio.fixture
async def api(engine, monkeypatch) -> AsyncIterator[httpx.AsyncClient]:
    """The batches router behind a stand-in for AuthMiddleware.

    The middleware only copies the header onto ``request.state`` — exactly what
    ``w2v_common``'s AuthMiddleware does with the tenant it resolves — so the gate
    under test is the real one in ``get_tenant_session``, not a test-local check.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    monkeypatch.setattr(
        session_mod, "_sessionmaker", async_sessionmaker(engine, expire_on_commit=False)
    )

    app = FastAPI()

    @app.middleware("http")
    async def _bind_tenant(request: Request, call_next):  # type: ignore[no-untyped-def]
        header = request.headers.get("X-Tenant-Id")
        if header is not None:
            request.state.tenant_id = header
        return await call_next(request)

    app.include_router(batches.router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        yield client


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"X-Tenant-Id": LEGACY_SLUG_TENANT}, id="legacy-slug"),
        pytest.param({"X-Tenant-Id": ""}, id="empty"),
        pytest.param({}, id="absent"),
    ],
)
async def test_an_unusable_tenant_is_rejected_as_client_error(api, headers) -> None:
    r = await api.get(f"/api/tts/batches/{CONTROL_BATCH}", headers=headers)
    assert r.status_code == 400, (
        f"an unusable tenant returned {r.status_code}. 500 means the uuid cast reached "
        "Postgres and a client mistake is being reported as a server fault; 404 means it "
        "was conflated with 'no such batch'."
    )


async def test_a_real_uuid_tenant_still_reaches_the_handler(api) -> None:
    """The control: without this the test above passes on a router that 400s everything."""
    r = await api.get(f"/api/tts/batches/{CONTROL_BATCH}", headers={"X-Tenant-Id": VALID_TENANT})
    assert r.status_code == 200, r.text
    assert r.json()["jobs"] == []
