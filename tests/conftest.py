"""Queue-core test fixtures: a real Postgres 18 with the claim fn installed."""

from __future__ import annotations

import os

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.database.models import Base
from src.database.queue_resolvers import register_claim_fn

# Install the SECURITY DEFINER claim fn on create_all (prod == test).
register_claim_fn(Base.metadata)


def _db_url() -> str:
    # Fleet convention: the CI archetype exports DATABASE_URL for the per-run CI
    # database. Fall back to TEST_DATABASE_URL / a local default for dev.
    url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("TEST_DATABASE_URL")
        or "postgresql+psycopg://postgres:postgres@localhost:5434/tts_gateway_test"  # pragma: allowlist secret
    )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    if "sslmode=" not in url:
        url = f"{url}{'&' if '?' in url else '?'}sslmode=disable"
    return url


DB_URL = _db_url()

# ``src.database.session`` builds its engine from ``get_settings()`` AT IMPORT, and
# ``Settings.database_url`` is required — so importing any route module without
# DATABASE_URL present blows up at collection. Publish the resolved test URL so
# route-level tests can import the app surface. ``setdefault``: CI already exports
# its own per-run DATABASE_URL and must win.
os.environ.setdefault("DATABASE_URL", DB_URL)


@pytest_asyncio.fixture(scope="session")
async def engine():
    eng = create_async_engine(DB_URL)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    async with engine.begin() as conn:
        # Both tables, and CASCADE: ``tts_batch_jobs`` has an FK to ``tts_jobs``,
        # and Postgres refuses to TRUNCATE a referenced table otherwise.
        await conn.execute(text("TRUNCATE tts_jobs, tts_batch_jobs RESTART IDENTITY CASCADE"))
    yield async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s
