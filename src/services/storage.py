"""MinIO clip storage — the gateway owns the bucket; workers upload via the API.

Deterministic key ``{tenant_id}/{idempotency_key}.wav`` (tenant in the key so two
tenants with identical text never share a clip). ``result_url`` is presigned at
READ time (in the batch poll) so it never expires before the orchestrator
assembles — mirrors the audiobook stitch ``refresh_shot_asset_urls`` concern.

Object I/O is delegated to the shared :class:`w2v_common.storage.StorageService`
facade (built once from ``settings.storage``); this class is a thin domain
wrapper that owns only the ``{tenant}/{key}.wav`` key convention. Both the
``local`` and ``minio`` backends stay live via ``settings.storage.backend`` —
tts_gateway is a real MinIO user, so we do NOT force a single backend. The
internal→external host rewrite on presigned URLs is handled by the facade when
``settings.storage.minio_external_endpoint`` is set.
"""

from __future__ import annotations

import asyncio
import logging

from w2v_common.storage import StorageService

from src.config import get_settings

logger = logging.getLogger(__name__)


class ClipStorageUnavailableError(RuntimeError):
    """Storage could not ANSWER — distinct from "this job has no artifact".

    ``presign`` used to collapse both into ``None``, and the poll route reported the
    job ``done`` with ``result_url=None`` regardless. The orchestrator treats ``done``
    as terminal, so a MinIO outage was published as a successful batch carrying no
    fetchable audio: the exact fail-open shape that has cost this fleet a run at a
    time. A raise forces the caller to decide, and the caller marks the job
    non-terminal so the poll re-tries when storage comes back.
    """


class ClipStorage:
    def __init__(self) -> None:
        self._facade: StorageService | None = None
        # Serializes lazy (re-)initialization so a burst of first-use callers after
        # a swallowed startup init (or a MinIO blip during rollout) rebuilds the
        # facade exactly once, not once per concurrent request.
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        facade = StorageService(get_settings().storage)
        await facade.initialize()
        self._facade = facade

    async def close(self) -> None:
        # The facade holds no close(); the underlying miniopy session belongs to
        # a process-lifetime singleton and is released at interpreter exit. Drop
        # the reference so a re-initialize rebuilds cleanly.
        self._facade = None

    async def _ensure_facade(self) -> StorageService:
        """Return a live facade, lazily (re-)initializing if startup init never ran
        or was swallowed (``_facade is None``). Makes the module docstring's "bucket
        is re-ensured on first use" TRUE without requiring a successful startup init.

        A genuinely-down MinIO still raises here (initialize() propagates the backend
        error) — a real persistent failure is NOT swallowed; only a transient blip
        self-heals on the next call. When init already succeeded this is a no-op.
        """
        facade = self._facade
        if facade is not None:
            return facade
        async with self._init_lock:
            if self._facade is None:
                await self.initialize()
            facade = self._facade
        if facade is None:  # pragma: no cover -- initialize() sets it or raises
            raise RuntimeError("ClipStorage: storage facade unavailable after initialize()")
        return facade

    def key_for(self, tenant_id: str, idempotency_key: str) -> str:
        return f"{tenant_id}/{idempotency_key}.wav"

    async def put(self, key: str, data: bytes, content_type: str = "audio/wav") -> str:
        facade = await self._ensure_facade()
        return await facade.put_bytes(key, data, content_type=content_type)

    async def exists(self, key: str) -> bool | None:
        """TRI-STATE presence probe for one clip key — never a plain bool.

        ``True`` present, ``False`` DEFINITIVELY absent (bucket reachable, key
        gone), ``None`` don't-know (MinIO down/unreachable, credentials, bucket
        missing, lazy init failed). The facade's ``object_exists_key`` already
        collapses every transport/permission error to ``None``; this wrapper only
        adds the same treatment for a failed lazy init.

        The distinction is load-bearing for the caller: only ``False`` may be acted
        on. Treating ``None`` as absent would let a MinIO blip re-queue an entire
        already-synthesized book — turning a read-side outage into hours of paid
        GPU re-synthesis.
        """
        if not key:
            return None
        try:
            facade = await self._ensure_facade()
        except Exception:  # noqa: BLE001 -- an unreachable backend is "don't know", not "gone"
            return None
        return await facade.object_exists_key(key)

    async def presign(self, key: str) -> str | None:
        """Presigned GET URL for ``key``, or ``None`` when there is legitimately none.

        ``None`` means ONLY the two benign cases:

        * an empty ``key`` — the job has no artifact to point at;
        * a non-``minio`` backend (``local``), where presigning is not a concept and
          the facade has always returned ``None`` by design.

        Everything else — MinIO unreachable, credentials rejected, a failed lazy
        init — raises :class:`ClipStorageUnavailableError` and is logged at ERROR.
        The distinction cannot be made downstream, because
        ``StorageService.presign_get`` collapses "wrong backend" and "transport
        error" into the same ``None``; it has to be made here, where the configured
        backend is known.

        This is the same tri-state discipline ``exists()`` already applies, and for
        the same reason: only the definitive verdict may be acted on.
        """
        if not key:
            return None
        try:
            facade = await self._ensure_facade()
        except Exception as exc:
            logger.error(
                "presign: storage backend is UNREACHABLE (key=%s) — refusing to report "
                "the job done with no fetchable audio",
                key,
                exc_info=exc,
            )
            raise ClipStorageUnavailableError(
                f"storage unreachable while presigning {key!r}"
            ) from exc

        if get_settings().storage.backend != "minio":
            # Not an outage: this deployment does not presign at all.
            return None

        url = await facade.presign_get(key)
        if url is None:
            # minio backend + no URL == the facade swallowed a transport/permission
            # error (it logs at WARNING and returns None). Presigning does NOT probe
            # the object, so "the key does not exist" is NOT a path to None here.
            logger.error(
                "presign: minio backend returned no URL for key=%s — treating as a "
                "storage failure, not as a job without audio",
                key,
            )
            raise ClipStorageUnavailableError(f"storage could not presign {key!r}")
        return url


_storage: ClipStorage | None = None


def get_storage() -> ClipStorage:
    global _storage
    if _storage is None:
        _storage = ClipStorage()
    return _storage
