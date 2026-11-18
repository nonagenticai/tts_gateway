"""``app.current_tenant`` must be TRANSACTION-scoped and re-bound per transaction.

These pods talk to Postgres through PgBouncer in transaction pooling mode, and
this deployment's ``server_reset_query`` scrubs only ``row_security``. A
session-scoped ``set_config(..., false)`` that gets COMMITTED therefore sticks to
the pooled server connection and is inherited by whoever lands on that backend
next — one tenant's connection carrying another tenant's identity.

The assertions below are on observable values: the SQL actually sent to the
driver, and what Postgres reports for the GUC on a REUSED pooled connection.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast

import pytest
import pytest_asyncio
from fastapi import HTTPException, Request
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from src.database import session as session_mod

_TENANT = "11111111-1111-1111-1111-111111111111"
_READ_GUC = text("SELECT current_setting('app.current_tenant', true)")


@pytest_asyncio.fixture
async def pinned_engine(engine: AsyncEngine) -> Any:
    """A one-connection pool, so "the next request" provably reuses the backend.

    Depends on the session-scoped ``engine`` fixture purely to guarantee the
    schema exists before we connect.
    """
    _ = engine
    eng = create_async_engine(os.environ["DATABASE_URL"], pool_size=1, max_overflow=0)
    yield eng
    await eng.dispose()


async def _read_guc(session: Any) -> str | None:
    return cast("str | None", (await session.execute(_READ_GUC)).scalar())


async def test_tenant_guc_is_transaction_scoped_and_survives_a_commit(
    pinned_engine: AsyncEngine, monkeypatch: Any
) -> None:
    executed: list[str] = []

    @event.listens_for(pinned_engine.sync_engine, "before_cursor_execute")
    def _record(
        _conn: Any, _cursor: Any, statement: str, *_rest: Any
    ) -> None:  # pragma: no cover - trivial
        executed.append(statement)

    monkeypatch.setattr(
        session_mod,
        "_sessionmaker",
        async_sessionmaker(pinned_engine, expire_on_commit=False),
    )

    request = cast("Request", SimpleNamespace(state=SimpleNamespace(tenant_id=_TENANT)))
    agen = cast("AsyncGenerator[Any]", session_mod.get_tenant_session(request))
    session = await agen.__anext__()

    # First transaction: the caller's tenant is bound.
    assert await _read_guc(session) == _TENANT

    # ``job_manager.submit_batch`` commits while the request is still running.
    # Transaction scope kills the GUC here, so the per-transaction re-bind is
    # what keeps the NEXT statement tenant-correct instead of silently
    # fail-closing under RLS enforce.
    await session.commit()
    assert await _read_guc(session) == _TENANT

    await session.commit()
    await agen.aclose()

    # THE ISOLATION ASSERTION, first because it is the actual defect: a later,
    # tenant-less request reusing the same pooled connection must not inherit
    # the previous tenant's identity.
    raw_gen = cast("AsyncGenerator[Any]", session_mod.get_raw_session())
    raw = await raw_gen.__anext__()
    leaked = await _read_guc(raw)
    await raw_gen.aclose()
    assert leaked in (None, ""), f"tenant GUC leaked across pooled connections: {leaked!r}"

    # The bind must be transaction-local: exactly the flag that keeps it off the
    # pooled backend.
    assert "SELECT set_config('app.current_tenant', %s, true)" in executed
    binds = [s for s in executed if "set_config('app.current_tenant'" in s]
    assert len(binds) == 2, f"expected one bind per transaction, got {binds}"


async def test_no_tenant_on_request_is_refused_before_a_session_exists(
    pinned_engine: AsyncEngine, monkeypatch: Any
) -> None:
    """A tenant-less request is a 400 — it never gets an unbound session at all.

    REWRITTEN with migration ``0003_tenant_id_uuid``. This test used to assert that
    a tenant-less request simply "binds nothing" and proceeded with an unbound
    session. That was defensible while ``tenant_id`` was ``varchar``: the request
    went on to match a text value, found nothing, and returned an empty batch.

    Against a ``uuid`` column it is not. ``batches._tenant`` turns a missing tenant
    into ``""``, which reaches Postgres as ``invalid input syntax for type uuid``
    and the caller gets a **500** — a client mistake reported as a server fault,
    several layers away from the middleware that actually failed to resolve a
    tenant. So the unbound session is no longer a benign state to hand out, and
    ``get_tenant_session`` refuses at the gate instead.

    The original isolation property this file exists for is unaffected and still
    proven by the test above: nothing is bound here, so nothing can leak onto the
    pooled backend — now because no session is ever produced.
    """
    monkeypatch.setattr(
        session_mod,
        "_sessionmaker",
        async_sessionmaker(pinned_engine, expire_on_commit=False),
    )
    request = cast("Request", SimpleNamespace(state=SimpleNamespace(tenant_id=None)))
    agen = cast("AsyncGenerator[Any]", session_mod.get_tenant_session(request))
    with pytest.raises(HTTPException) as excinfo:
        await agen.__anext__()
    assert excinfo.value.status_code == 400, (
        "a missing tenant must be a client error, not a 500 from the uuid cast "
        "and not a silently unbound session"
    )
    await agen.aclose()
