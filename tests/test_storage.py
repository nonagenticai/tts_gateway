"""ClipStorage delegates to the shared w2v_common StorageService facade.

No live MinIO: the miniopy-async ``Minio`` client is mocked, so these assert the
wiring (backend resolution, key delegation, presign host-rewrite) rather than
object round-trips.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from src.config import StorageSettings
from src.services.storage import ClipStorage, ClipStorageUnavailableError


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


def _fake_minio(presigned_url: str = "http://minio.test:9000/tts-gateway/k?sig=x") -> Any:
    client = SimpleNamespace()
    client.bucket_exists = AsyncMock(return_value=True)
    client.make_bucket = AsyncMock()
    client.put_object = AsyncMock()
    client.presigned_get_object = AsyncMock(return_value=presigned_url)
    return client


def _patch_settings(settings: StorageSettings) -> Any:
    return patch(
        "src.services.storage.get_settings",
        return_value=SimpleNamespace(storage=settings),
    )


async def test_initialize_minio_backend_does_not_raise() -> None:
    """A minio-backed StorageSettings resolves and initialize() is clean."""
    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()  # must not raise
    client.bucket_exists.assert_awaited_once_with("tts-gateway")


async def test_put_delegates_to_facade() -> None:
    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        ref = await storage.put("tenant-a/idem-1.wav", b"riff", content_type="audio/wav")
    assert ref == "tenant-a/idem-1.wav"
    client.put_object.assert_awaited_once()
    args, kwargs = client.put_object.call_args
    assert args[0] == "tts-gateway"
    assert args[1] == "tenant-a/idem-1.wav"
    assert kwargs["content_type"] == "audio/wav"


async def test_presign_rewrites_internal_host_to_external() -> None:
    """When minio_external_endpoint is set the facade rewrites the presigned host."""
    client = _fake_minio()
    settings = _minio_settings(minio_external_endpoint="minio-ext.example.com:9000")
    with (
        _patch_settings(settings),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        url = await storage.presign("tenant-a/idem-1.wav")
    assert url is not None
    assert "minio-ext.example.com:9000" in url
    assert "minio.test:9000" not in url


async def test_presign_no_rewrite_without_external_endpoint() -> None:
    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        url = await storage.presign("tenant-a/idem-1.wav")
    assert url == "http://minio.test:9000/tts-gateway/k?sig=x"


async def test_presign_empty_key_returns_none() -> None:
    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        assert await storage.presign("") is None


async def test_put_lazily_initializes_when_startup_init_never_ran() -> None:
    """A pod that booted with a swallowed/absent startup init (``_facade`` None) but
    with MinIO now reachable must self-heal on first ``put()`` — not 500 forever."""
    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        assert storage._facade is None  # simulate startup init that never ran
        ref = await storage.put("tenant-a/idem-1.wav", b"riff", content_type="audio/wav")
    assert ref == "tenant-a/idem-1.wav"
    client.put_object.assert_awaited_once()


async def test_presign_lazily_initializes_when_facade_none() -> None:
    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        assert storage._facade is None
        url = await storage.presign("tenant-a/idem-1.wav")
    assert url == "http://minio.test:9000/tts-gateway/k?sig=x"


async def test_put_still_raises_when_minio_genuinely_down() -> None:
    """Lazy init must NOT swallow a real persistent failure: a down MinIO still
    errors (and does not hang) rather than silently succeeding."""
    client = SimpleNamespace()
    client.bucket_exists = AsyncMock(side_effect=ConnectionError("minio down"))
    client.make_bucket = AsyncMock()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        with pytest.raises(Exception):  # noqa: B017 -- any error, must not hang/succeed
            await storage.put("k", b"x")


async def test_lazy_init_happens_once_under_concurrent_puts() -> None:
    """Thundering-herd guard: N concurrent first-puts must initialize exactly once."""
    import asyncio

    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client) as minio_ctor,
    ):
        storage = ClipStorage()
        await asyncio.gather(*(storage.put(f"k{i}", b"x") for i in range(8)))
    # One StorageService facade → one Minio client constructed despite 8 racing puts.
    assert minio_ctor.call_count == 1


async def test_close_drops_facade() -> None:
    client = _fake_minio()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        await storage.close()
    assert storage._facade is None
    # And with the facade dropped AND settings un-patched, a presign can no longer be
    # built — which is now a LOUD failure rather than a silent None (see
    # ``test_presign_failure_is_not_done``: the silent None was reported as ``done``).
    with pytest.raises(ClipStorageUnavailableError):
        await storage.presign("k")


# ── exists(): the tri-state probe the dedupe path acts on ───────────────────────
#
# `submit_batch` re-queues a dedupe hit ONLY on a definitive `False`. These pin the
# three verdicts end-to-end through the real w2v_common facade, driving the object
# absent and then restoring it — the same two directions the queue tests assert.


def _s3_error(code: str) -> Exception:
    from miniopy_async.error import S3Error

    return S3Error(code, code, "res", "req", "host", response=cast(Any, SimpleNamespace()))


async def test_exists_is_true_when_the_clip_is_there() -> None:
    client = _fake_minio()
    client.stat_object = AsyncMock(return_value=SimpleNamespace(size=1024))
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        assert await storage.exists("tenant-a/idem-1.wav") is True
    client.stat_object.assert_awaited_once_with("tts-gateway", "tenant-a/idem-1.wav")


async def test_exists_is_false_when_the_clip_has_been_swept() -> None:
    """The production case: the ``tts_jobs`` row survived, the object did not."""
    client = _fake_minio()
    client.stat_object = AsyncMock(side_effect=_s3_error("NoSuchKey"))
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        assert await storage.exists("tenant-a/idem-1.wav") is False


async def test_exists_is_none_when_minio_cannot_answer() -> None:
    """Down / unreachable / permission-denied is DON'T KNOW, never "gone".

    Returning False here would let one MinIO blip re-queue every cached clip in a
    book — hours of paid GPU synthesis caused by a read-side outage.
    """
    client = _fake_minio()
    client.stat_object = AsyncMock(side_effect=ConnectionError("minio down"))
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        assert await storage.exists("tenant-a/idem-1.wav") is None

    # An S3Error that is not NoSuchKey (e.g. AccessDenied) is equally indeterminate.
    client.stat_object = AsyncMock(side_effect=_s3_error("AccessDenied"))
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        await storage.initialize()
        assert await storage.exists("tenant-a/idem-1.wav") is None


async def test_exists_is_none_when_storage_is_unreachable_at_init() -> None:
    """A facade that cannot even be built must not read as "the clip is gone"."""
    client = SimpleNamespace()
    client.bucket_exists = AsyncMock(side_effect=ConnectionError("minio down"))
    client.make_bucket = AsyncMock()
    with (
        _patch_settings(_minio_settings()),
        patch("miniopy_async.Minio", return_value=client),
    ):
        storage = ClipStorage()
        assert await storage.exists("tenant-a/idem-1.wav") is None


async def test_exists_empty_key_returns_none() -> None:
    with _patch_settings(_minio_settings()):
        assert await ClipStorage().exists("") is None
