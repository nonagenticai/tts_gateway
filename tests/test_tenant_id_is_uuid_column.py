"""``tts_jobs.tenant_id`` is a uuid COLUMN whose Python value is still a ``str``.

Two halves, and BOTH must hold. Pinning only one of them is what makes this bug
class survive review.

**Half 1 — the column really is ``uuid``.** The fleet ruling is that ``tenant_id`` is
a real uuid, never text. ``String(255)`` would satisfy every test in this repo, so
the type is asserted against the live database, not against the model.

**Half 2 — the loaded value is still ``str``, i.e. ``as_uuid=False``.** This is the
half with no type-checker coverage at all. Flipping to ``as_uuid=True`` makes
SQLAlchemy return ``uuid.UUID`` objects, and::

    uuid.UUID("aaaaaaaa-...-aaaaaaaaaaaa") == "aaaaaaaa-...-aaaaaaaaaaaa"  # False

``UUID.__eq__`` returns ``NotImplemented`` for a non-UUID, so Python falls back to
identity and the comparison is simply False — legal, silent, and invisible to ``ty``
and ``ruff``. Every ``tenant_id == <str>`` site in a service degrades at once:
ownership guards 404 the owner on their own rows, scope predicates match nothing,
storage keys built by f-string change shape, and a pydantic ``tenant_id: str``
response field rejects the value outright. The suite stays green; production breaks.
(Observed in kag_orchestrator #148.)

So this file asserts the exact type, not "truthy" or "uuid-like": ``type(...) is str``
fails the moment someone flips ``as_uuid``, and the third test demonstrates *why* that
flip is not cosmetic.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy import text as sql

from src.database.models import TtsJob
from src.services import job_manager as jm

TENANT = "11111111-1111-1111-1111-111111111111"


def _chunk(ref: str, txt: str) -> dict[str, object]:
    return {"ref": ref, "text": txt, "voice": "af_heart", "speed": 1.0}


async def test_the_database_column_is_a_real_uuid(session_factory) -> None:
    """Half 1: the ruling is about the COLUMN, so read it off the database."""
    async with session_factory() as s:
        data_type = (
            await s.execute(
                sql(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = 'tts_jobs' AND column_name = 'tenant_id'"
                )
            )
        ).scalar_one()
    assert data_type == "uuid", (
        f"tts_jobs.tenant_id is {data_type!r}, not uuid — migration 0003 or the model "
        "regressed to a text column"
    )


async def test_the_orm_attribute_round_trips_as_str_not_uuid(session_factory) -> None:
    """Half 2: ``as_uuid=False`` — the value Python sees must still be a ``str``."""
    async with session_factory() as s:
        await jm.submit_batch(
            s, tenant_id=TENANT, batch_id="b1", chunks=[_chunk("r1", "Chapter one.")]
        )
    async with session_factory() as s:
        job = (await s.execute(select(TtsJob))).scalars().one()

    assert type(job.tenant_id) is str, (
        f"TtsJob.tenant_id loaded as {type(job.tenant_id).__name__}, not str — someone "
        "flipped PgUUID(as_uuid=False) to as_uuid=True. See the module docstring: every "
        "`tenant_id == <str>` site in the service silently becomes False and no type "
        "checker will tell you."
    )
    # And it is the value the caller handed in, unchanged — so an f-string storage key
    # and an == against the request's tenant both still behave.
    assert job.tenant_id == TENANT


def test_why_as_uuid_true_would_be_silent() -> None:
    """The failure mode itself, pinned so the docstring above cannot rot into folklore.

    No exception, no type error — just ``False``. That is the whole problem.
    """
    assert uuid.UUID(TENANT) != TENANT
    assert (uuid.UUID(TENANT) == TENANT) is False
    # ...while the str form compares exactly as every call site expects.
    assert str(uuid.UUID(TENANT)) == TENANT
