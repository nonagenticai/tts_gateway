"""Async engine + session dependencies.

Two flavours: a tenant-scoped session for the submit/poll API (binds
``app.current_tenant`` so RLS enforce scopes the tenant's own jobs), and a raw
session for the worker claim path (cross-tenant — the SECURITY DEFINER claim fn
binds no tenant, the worker binds the RETURNED tenant per job).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from sqlalchemy import Connection, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, SessionTransaction
from w2v_common.tenant import TenantError, parse_tenant_id

from src.config import get_settings

_settings = get_settings()
_engine = create_async_engine(_settings.effective_db_url(), pool_pre_ping=True)
_sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)


def session_factory() -> async_sessionmaker[AsyncSession]:
    return _sessionmaker


async def get_raw_session() -> AsyncIterator[AsyncSession]:
    """Session with no tenant bound (worker claim/heartbeat/result path)."""
    async with _sessionmaker() as session:
        yield session


def bind_tenant_every_transaction(session: AsyncSession, tenant_id: str) -> None:
    """Re-apply ``app.current_tenant`` TRANSACTION-scoped at every BEGIN on ``session``.

    Two things are load-bearing here, and each one alone is wrong:

    1. ``is_local=true``. These pods reach Postgres through PgBouncer in
       *transaction* pooling mode. A session-scoped ``set_config(..., false)``
       that is COMMITTED sticks to the pooled server connection after the client
       transaction ends, and this deployment's ``server_reset_query`` scrubs only
       ``row_security`` — it never clears ``app.current_tenant``. The next
       request to land on that backend would inherit the previous tenant's
       identity: a cross-tenant isolation failure, not a hygiene nit. (Same class
       as the stray ``default_transaction_read_only=on`` that once poisoned one
       of four backends here.) Transaction scope dies with the transaction, so
       nothing can be inherited.

    2. The ``after_begin`` listener. Transaction scope means the GUC dies at
       COMMIT — and ``job_manager.submit_batch`` *does* commit while the request
       is still running. A one-line flag flip would therefore leave any DB work
       that follows a commit in a fresh, UNBOUND transaction, where RLS enforce
       fail-closes SILENTLY (0 rows / ``InsufficientPrivilege``) rather than
       erroring loudly. Re-binding on every BEGIN makes the session
       tenant-correct for its whole lifetime without ever leaving state on the
       connection. Modelled on ``analysis_service``'s
       ``src/shared/database/rls_events.py``, but registered on THIS session
       instance rather than the ``Session`` class, so it is scoped to one request
       and needs no global install/uninstall bookkeeping.

    ``tenant_id`` must already be ``str``: ``set_config`` is
    ``(text, text, bool)`` and psycopg3 adapts a bare ``uuid.UUID`` to the
    postgres ``uuid`` type, which raises ``UndefinedFunction``.
    """

    @event.listens_for(session.sync_session, "after_begin")
    def _rebind(
        _session: Session, _transaction: SessionTransaction, connection: Connection
    ) -> None:
        connection.exec_driver_sql(
            "SELECT set_config('app.current_tenant', %s, true)", (tenant_id,)
        )


async def get_tenant_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Session with ``app.current_tenant`` bound to the caller (submit/poll).

    This is the ONE choke point where a caller-supplied tenant enters the
    tenant-scoped surface: both ``/api/tts/batches`` routes depend on it, and
    ``batches._tenant`` only re-reads the value this dependency has already
    vetted. Validating here (rather than at each call site) is what keeps the two
    from drifting.

    **Why validation became mandatory with 0003.** ``tts_jobs.tenant_id`` is now a
    real ``uuid`` column. A request carrying a legacy slug (``n8n-dev``) or no
    tenant at all used to be harmless — it wrote/matched a text value and simply
    returned nothing. Against a ``uuid`` column the same request reaches Postgres
    as ``invalid input syntax for type uuid`` and the caller gets a **500**:
    malformed *input* reported as a *server* fault, which pages the wrong person
    and hides a licence/middleware defect behind an "outage". So the tenant is
    parsed here and a bad one is a 400 — the repo's existing status for bad input
    (see ``routes/workers.py``). Never a 404: "your tenant id is unusable" and
    "that batch does not exist" are different answers and must not be conflated.

    The parsed value is deliberately thrown away and the ORIGINAL string is bound:
    see ``bind_tenant_every_transaction`` — ``set_config`` is ``(text, text, bool)``
    and a ``uuid.UUID`` there raises ``UndefinedFunction``, killing every
    transaction on the session. Parsing is a GATE here, not a conversion.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    try:
        parse_tenant_id(tenant_id, source="request tenant identity")
    except TenantError as exc:
        # The message never echoes the value: an X-Tenant-Id field can carry a
        # `Bearer <license token>` and w2v_common keeps it out of the text for
        # exactly that reason. Chained so the real cause stays in the traceback.
        raise HTTPException(400, "request has no usable tenant identity") from exc

    async with _sessionmaker() as session:
        bind_tenant_every_transaction(session, str(tenant_id))
        yield session
