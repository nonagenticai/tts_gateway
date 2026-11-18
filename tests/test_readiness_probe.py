"""Readiness must answer 503 when Postgres is down -- never 500.

``/readyz`` used to run its ``SELECT 1`` completely unguarded, so a dead
Postgres escaped as an unhandled exception: a 500, or an ASGI-level error with
no body at all. It is also the exact path the gitops readinessProbe targets, so
it is the one handler in this service that must not misreport.

Both tests below fail if the guard is removed: the unreachable-database test
sees the raw exception surface as a 500 instead of a 503, and the healthy test
would be unaffected -- which is why the pair is the proof, not either alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.api.routes import health


class _RaisingSession:
    """Stand-in session whose ``execute`` fails the way a dead server does."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, *_: object, **__: object) -> None:
        raise OperationalError("SELECT 1", {}, ConnectionRefusedError("connection refused"))


class _OkSession:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, *_: object, **__: object) -> None:
        return None


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Mount only the health router, so no lifespan or real engine is needed."""
    app = FastAPI()
    app.include_router(health.router)
    with TestClient(app) as c:
        yield c


def test_readyz_returns_503_when_database_unreachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "session_factory", lambda: _RaisingSession)

    response = client.get("/readyz")

    assert response.status_code == 503, (
        f"a dead database must read as not-ready, got {response.status_code}"
    )
    assert response.status_code != 500


def test_readyz_returns_200_when_database_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "session_factory", lambda: _OkSession)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_liveness_stays_unconditional(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness must NOT depend on the database, or a DB blip restarts the pod."""
    monkeypatch.setattr(health, "session_factory", lambda: _RaisingSession)

    for path in ("/health", "/livez"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.json()["status"] == "ok"
