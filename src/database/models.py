"""Durable TTS job-queue schema.

One ``tts_jobs`` row per synthesis request (a chunk/span of a book). A UNIQUE
``(tenant_id, idempotency_key)`` where ``idempotency_key = sha256(tenant|voice|
speed|text)`` makes submission idempotent — re-submitting the same book (a
resumed orchestrator narrate) collapses onto the existing rows, many already
``done``, so no synthesis is repeated. Lease columns mirror ``html_render_service``'s
proven claim-queue so a crashed worker's job is reclaimed on lease expiry.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TtsJob(Base):
    __tablename__ = "tts_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # External opaque handle (uuid7 hex) the submitter/worker address the job by.
    job_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    # A real ``uuid`` COLUMN in Postgres (fleet ruling; see migration
    # 0003_tenant_id_uuid) whose Python-side value stays a ``str``.
    #
    # ⚠️ ``as_uuid=False`` is LOAD-BEARING, not style. With ``as_uuid=True`` SQLAlchemy
    # hands back ``uuid.UUID`` objects, and ``uuid.UUID(t) == t`` is False for every
    # string ``t``. Every ``tenant_id == <str>`` comparison in the service — ownership
    # guards, scope predicates, storage-key building, a pydantic ``tenant_id: str``
    # response field — would silently start failing, and NO type checker flags it:
    # ``UUID == str`` is legal Python that just returns False. ty and ruff stay green
    # while the service 404s owners on their own rows. (Found the hard way in
    # kag_orchestrator #148.) ``as_uuid=False`` gets the real uuid column — the entire
    # point of the ruling — with zero comparison-site blast radius.
    # ``tests/test_tenant_id_is_uuid_column.py`` pins both halves of this.
    tenant_id: Mapped[str] = mapped_column(PgUUID(as_uuid=False), index=True)
    # sha256(tenant|voice|speed|text) — dedupe key; identical requests collapse.
    idempotency_key: Mapped[str] = mapped_column(String(64))
    # The batch this job was FIRST submitted under — provenance only, WRITE-ONCE.
    #
    # ⚠️ This column is deliberately NOT the answer to "which batches contain this
    # job". Dedupe is global by (tenant_id, idempotency_key), so one row is legitimately
    # shared by every batch that submits the same chunk. It used to be RE-TAGGED onto
    # the newest batch on every dedupe hit, which silently STOLE the row out of an
    # in-flight batch: run A polls ``get_batch(A)``, run B submits an overlapping book,
    # A's rows move to B, and A polls an empty batch until its budget expires and the
    # orchestrator raises for work that actually succeeded. Batch membership now lives
    # in ``tts_batch_jobs`` (many-to-many), so both batches observe the same completed
    # work and neither can take it from the other. See ``job_manager.submit_batch``.
    batch_id: Mapped[str] = mapped_column(String(64), index=True)

    # --- synthesis inputs (the durable substitute for the sync request body) ---
    voice: Mapped[str | None] = mapped_column(String(64), nullable=True)
    speed: Mapped[float] = mapped_column(Numeric(4, 3), default=1.0, server_default="1.0")
    provider: Mapped[str] = mapped_column(String(32), default="kokoro", server_default="kokoro")
    text: Mapped[str] = mapped_column(Text)

    # --- lifecycle (mirror html_render's lease queue) ---
    status: Mapped[str] = mapped_column(
        String(20), default="queued", server_default="queued", index=True
    )
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # How many times this row has been re-opened out of a TERMINAL ``failed`` state
    # by ``job_manager.reopen_failed_jobs``. MONOTONIC — nothing resets it, ever.
    #
    # ⚠️ This cannot be ``attempts``. Re-opening deliberately sets ``attempts = 0``
    # (a fresh synthesis needs a full claim budget), so ``attempts`` can never bound
    # how many times the re-open itself happens. Without a separate monotonic counter
    # the fix for "a failed row blacklists its content forever" becomes an infinite
    # treadmill: an orchestrator re-submitting the same book every 20 minutes re-burns
    # ``max_attempts`` GPU claims each time on text kokoro will never accept. With it,
    # poison converges — after ``max_requeues`` re-opens the row stays ``failed`` and
    # the stored error is replayed, i.e. exactly today's behaviour, but only after a
    # bounded number of genuine retries.
    requeue_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Fairness / manual bump; higher runs sooner within the same tenant balance.
    priority: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # --- result ---
    result_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Numeric(9, 3), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # First GPU-busy release (tts_gateway#45); arms the absolute busy backstop in
    # claim_next_tts_job. Cleared by every re-open, so a re-driven job starts fresh.
    busy_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Dedupe: the same (tenant, voice, speed, text) is one row.
        Index("uq_tts_jobs_idem", "tenant_id", "idempotency_key", unique=True),
        # Cheap claim scan: only ever touches claimable rows even as done rows pile up.
        Index(
            "ix_tts_jobs_claimable",
            "priority",
            "created_at",
            postgresql_where=sql_text("status IN ('queued','claimed')"),
        ),
        Index("ix_tts_jobs_batch_lookup", "batch_id", "status"),
    )


class TtsBatchJob(Base):
    """Which batches contain which jobs — the membership relation ``batch_id`` cannot be.

    A job row is uniquely keyed by ``(tenant_id, idempotency_key)`` and is therefore
    SHARED by every batch that submits the same chunk (a resumed narrate, a re-run
    under a fresh ``run_id``, two concurrent runs of the same book). Membership is
    consequently many-to-many, and modelling it as a column on ``tts_jobs`` made
    submission of batch B destructive to the in-flight poll of batch A.

    Rows are append-only and never re-pointed: ``submit_batch`` inserts
    ``ON CONFLICT DO NOTHING``, so a re-submit is a no-op and no batch ever loses a
    member. The FK cascades so a job deletion cannot strand membership.
    """

    __tablename__ = "tts_batch_jobs"

    tenant_id: Mapped[str] = mapped_column(PgUUID(as_uuid=False), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tts_jobs.job_id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # The poll path: every job of one (tenant, batch). The PK's leading
        # (tenant_id, batch_id) already serves it, so no extra index is needed there.
        # This one serves the reverse lookup (every batch a job belongs to) and the
        # FK's cascade.
        Index("ix_tts_batch_jobs_job", "job_id"),
    )
