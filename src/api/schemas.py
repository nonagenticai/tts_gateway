"""Request/response models for the gateway API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Largest value that round-trips into ``TtsJob.duration_seconds`` — ``Numeric(9, 3)``,
# i.e. 9 significant digits with 3 after the point, so 999999.999 and not one unit of
# the third decimal more. The bound is the column ceiling ITSELF (``le``, inclusive)
# rather than "just under a million": Postgres ROUNDS to the declared scale before it
# checks precision, so 999999.9995 — which is under 1e6 — rounds to 1000000.000, needs
# 10 digits, and raises ``numeric field overflow``. Anchoring on the exact ceiling
# rejects that value; a ``lt=1e6`` bound would have let it through to the DataError.
MAX_DURATION_SECONDS = 999999.999

# Column widths in ``TtsJob``: ``voice`` and ``batch_id`` are both ``String(64)``.
# Bounding them HERE is the same argument the ``speed`` bound below makes, for the
# same statement: ``submit_batch`` inserts every chunk of a book in ONE ``pg_insert``,
# so a single over-width value raises ``StringDataRightTruncation`` and takes the
# WHOLE batch down with a 500 — every other chunk in the book, all of them fine, is
# lost with it. Unbounded, these two fields were the only client-controlled strings
# reaching a length-limited column with no guard in front of them. A 422 naming the
# offending field is the correct answer; a 500 on the book is not.
MAX_VOICE_LEN = 64
MAX_BATCH_ID_LEN = 64


class ChunkIn(BaseModel):
    ref: str = Field(..., description="Caller's opaque id (e.g. orchestrator chunk id)")
    text: str
    voice: str | None = Field(default=None, max_length=MAX_VOICE_LEN)
    # Bounded to kokoro's accepted range (kokoro-onnx asserts 0.5 <= speed <= 2.0;
    # see tts_backend/src/services/ssml_processor.py + tts_backend models.py). This
    # is the batch's ONLY guard: submit_batch inserts every chunk in ONE pg_insert
    # into a Numeric(4,3) column, so a single out-of-range speed (12.0/inf) would
    # overflow the whole statement → 500 → the entire book fails to enqueue; and a
    # nan/0.0 would store fine only to poison the job at kokoro. A Field numeric
    # bound also rejects inf/nan. 0.0 is intentionally NOT a "means default" alias:
    # it is rejected (422) rather than silently coerced to 1.0 downstream.
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class SubmitBatchRequest(BaseModel):
    batch_id: str = Field(
        ..., max_length=MAX_BATCH_ID_LEN, description="One book's narrate submission id"
    )
    chunks: list[ChunkIn]


class JobRef(BaseModel):
    ref: str | None
    job_id: str | None
    idempotency_key: str
    status: str


class SubmitBatchResponse(BaseModel):
    batch_id: str
    jobs: list[JobRef]


class JobStatus(BaseModel):
    """One job as the orchestrator reads it while polling a batch.

    ``duration_seconds`` is ``None`` in TWO different situations, and a consumer that
    cannot tell them apart mis-builds the narration timeline:

    * the job has not finished yet — there is no duration to have;
    * the job IS ``done`` and its clip is real, but the synth backend never measured
      a length (``tts_backend`` emits ``duration_sec = frames / sr if sr else 0.0``).

    ``duration_unmeasured`` names the second case. It exists because the second case
    used to be published as the number ``0.0``, which is indistinguishable from a
    genuine measurement: the timeline allocated no time at all for a chunk full of
    speech, narration drifted against picture for everything after it, and the batch
    and the run both reported success. A consumer must treat ``duration_unmeasured``
    as "audio of unknown length" — measure the clip, or degrade loudly — and never as
    "zero seconds of audio".
    """

    job_id: str
    idempotency_key: str
    status: str
    result_url: str | None = None
    duration_seconds: float | None = None
    # True only for a ``done`` job whose stored duration is NULL. Defaults to False so
    # an unrelated construction site cannot accidentally assert a gap.
    duration_unmeasured: bool = False
    error: str | None = None


class BatchStatusResponse(BaseModel):
    batch_id: str
    jobs: list[JobStatus]


class ClaimRequest(BaseModel):
    worker_id: str


class ClaimedJob(BaseModel):
    job_id: str
    voice: str | None
    speed: float
    provider: str
    text: str
    # The OWNING tenant of the job, so the worker can attribute the synth hop.
    #
    # Why it has to be here: the worker's only knowledge of a job is this payload.
    # Without the tenant it called the synth backend with nothing but its own
    # service-account licence, so tts_backend scoped the request — and the cost
    # event it emits — to the SERVICE ACCOUNT's tenant. A customer narration billed
    # to the fleet SA tenant, and no other service emitted a correcting row.
    # ``claim_next`` already SELECTs ``tenant_id``; this payload simply dropped it.
    #
    # Optional, and the worker must tolerate its absence: an older gateway paired
    # with a newer worker (or the reverse) must keep synthesizing, unattributed,
    # rather than fail. Narration outages are more expensive than a mis-billed row.
    tenant_id: str | None = None
    # The gateway's lease length, so the worker derives its heartbeat cadence from
    # the real value rather than a hand-copied constant. Optional for the same
    # mixed-version reason as tenant_id.
    lease_seconds: int | None = None


class ResultRequest(BaseModel):
    result_key: str = Field(..., description="MinIO key the worker stored the clip at")
    # Same column, same ceiling as the multipart /result route (see MAX_DURATION_SECONDS).
    duration_seconds: float = Field(..., le=MAX_DURATION_SECONDS, allow_inf_nan=False)


class FailRequest(BaseModel):
    error: str
