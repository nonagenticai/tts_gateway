"""A batch whose artifact cannot be presigned is NOT ``done``.

``ClipStorage.presign`` used to collapse "MinIO unreachable" into the same ``None``
it returns for "there is legitimately no URL here", and the poll route reported the
job ``done`` regardless. The orchestrator's narrate stage treats ``done`` as
terminal, so a read-side outage was published as a SUCCESSFUL batch carrying no
fetchable audio — fail-open, silently.

Two directions are pinned here, because collapsing either into the other is a bug:

* transient-unreachable  → the job is reported ``storage_unavailable`` (not terminal
  for the orchestrator, so it re-polls and self-heals), and the failure is LOUD;
* legitimately-artifact-less → unchanged behaviour, no error, no status rewrite.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from src.api.routes import batches
from src.config import StorageSettings
from src.database.session import get_tenant_session
from src.services import job_manager as jm
from src.services.storage import ClipStorage, ClipStorageUnavailableError

_TENANT = "11111111-1111-1111-1111-111111111111"


def _minio_settings(**overrides: Any) -> StorageSettings:
    base = StorageSettings(
        backend="minio",
        minio_endpoint="minio.test:9000",
        minio_access_key="key",  # pragma: allowlist secret
        minio_secret_key="secret",  # noqa: S106 # pragma: allowlist secret
        minio_bucket="tts-gateway",
        minio_secure=False,
    )
    return base.model_copy(update=overrides) if overrides else base


def _patch_settings(settings: StorageSettings) -> Any:
    return patch(
        "src.services.storage.get_settings",
        return_value=SimpleNamespace(storage=settings),
    )


def _fake_minio() -> Any:
    client = SimpleNamespace()
    client.bucket_exists = AsyncMock(return_value=True)
    client.make_bucket = AsyncMock()
    client.presigned_get_object = AsyncMock(
        return_value="http://minio.test:9000/tts-gateway/k?sig=x"
    )
    return client


# ── the storage layer: the two verdicts must not be the same value ──────────────


async def test_presign_raises_and_logs_when_minio_is_unreachable_at_init(caplog) -> None:
    client = SimpleNamespace()
    client.bucket_exists = AsyncMock(side_effect=ConnectionError("minio down"))
    client.make_bucket = AsyncMock()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
        caplog.at_level(logging.ERROR, logger="src.services.storage"),
    ):
        storage = ClipStorage()
        with pytest.raises(ClipStorageUnavailableError):
            await storage.presign("tenant-a/idem-1.wav")
    assert any(r.levelno >= logging.ERROR for r in caplog.records), (
        "an unreachable backend must be LOUD, not a silent None"
    )


async def test_presign_raises_when_the_minio_call_itself_fails(caplog) -> None:
    """The facade swallows the transport error into ``None``; we must not accept it."""
    client = _fake_minio()
    client.presigned_get_object = AsyncMock(side_effect=ConnectionError("minio down"))
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
        caplog.at_level(logging.ERROR, logger="src.services.storage"),
    ):
        storage = ClipStorage()
        await storage.initialize()
        with pytest.raises(ClipStorageUnavailableError):
            await storage.presign("tenant-a/idem-1.wav")
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


async def test_presign_of_an_empty_key_is_still_a_quiet_none() -> None:
    """A job that legitimately has NO artifact must not become a failure."""
    with _patch_settings(_minio_settings()):
        assert await ClipStorage().presign("") is None


async def test_presign_on_a_local_backend_is_a_quiet_none(tmp_path) -> None:
    """``local`` does not presign at all — a configuration, not an outage."""
    settings = StorageSettings(backend="local", local_path=str(tmp_path))
    with _patch_settings(settings):
        storage = ClipStorage()
        await storage.initialize()
        assert await storage.presign("tenant-a/idem-1.wav") is None


# ── the route: what the orchestrator actually reads ─────────────────────────────


async def _poll(session_factory, storage: Any, batch_id: str) -> list[dict[str, Any]]:
    app = FastAPI()
    app.include_router(batches.router)

    @app.middleware("http")
    async def _tenant_state(request, call_next):
        request.state.tenant_id = _TENANT
        return await call_next(request)

    async def _session():
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_tenant_session] = _session
    transport = httpx.ASGITransport(app=app)
    with patch.object(batches, "get_storage", lambda: storage):
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.get(f"/api/tts/batches/{batch_id}")
    assert resp.status_code == 200
    return list(resp.json()["jobs"])


async def _one_done_job(session_factory, *, result_key: str | None) -> str:
    async with session_factory() as s:
        out = await jm.submit_batch(
            s,
            tenant_id=_TENANT,
            batch_id="b1",
            chunks=[{"ref": "r1", "text": "Chapter one.", "voice": "af_heart", "speed": 1.0}],
        )
    async with session_factory() as s:
        job = await jm.claim_next(
            s, worker_id="w", lease_seconds=60, max_attempts=3, max_lifetime_seconds=0
        )
        assert job is not None
        await jm.complete(
            s, job_id=job["job_id"], result_key=result_key or "", duration_seconds=2.0
        )
    return str(out[0]["job_id"])


async def test_a_presign_failure_is_not_reported_as_done(session_factory) -> None:
    """The live shape: the row IS done, the bucket cannot answer, the batch is not."""
    await _one_done_job(session_factory, result_key="t/clip.wav")

    class _Down:
        async def presign(self, key: str) -> str | None:
            raise ClipStorageUnavailableError(f"storage unreachable while presigning {key!r}")

    jobs = await _poll(session_factory, _Down(), "b1")
    assert len(jobs) == 1
    assert jobs[0]["status"] != "done", (
        "a batch with no fetchable audio was published as done — the orchestrator "
        "stops polling on done and the run dies for work that succeeded"
    )
    assert jobs[0]["status"] == batches.STORAGE_UNAVAILABLE_STATUS
    assert jobs[0]["status"] not in ("done", "failed"), (
        "the status must be OUTSIDE the orchestrator's terminal set so the poll re-tries"
    )
    assert jobs[0]["result_url"] is None
    assert jobs[0]["error"]


async def test_a_job_with_no_artifact_at_all_still_reports_done(session_factory) -> None:
    """The control. An empty ``result_key`` never reaches presign and never was an
    error; turning THAT into a failure would be the opposite over-correction."""
    await _one_done_job(session_factory, result_key=None)

    class _Exploding:
        async def presign(self, key: str) -> str | None:  # pragma: no cover -- must not run
            raise AssertionError("presign must not be called for an artifact-less job")

    jobs = await _poll(session_factory, _Exploding(), "b1")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "done"
    assert jobs[0]["result_url"] is None
    assert jobs[0]["error"] is None


async def test_a_healthy_presign_is_unchanged(session_factory) -> None:
    await _one_done_job(session_factory, result_key="t/clip.wav")

    class _Up:
        async def presign(self, key: str) -> str | None:
            return f"http://minio/{key}?sig=x"

    jobs = await _poll(session_factory, _Up(), "b1")
    assert jobs[0]["status"] == "done"
    assert jobs[0]["result_url"] == "http://minio/t/clip.wav?sig=x"
