"""Client strings that reach a length-limited column must be bounded at the API.

``submit_batch`` inserts a whole book in ONE ``pg_insert``, so a single over-width
value raises ``StringDataRightTruncation`` and the ENTIRE batch 500s — every other
chunk, all of them fine, is lost with it. This is the identical argument
``test_schema_speed.py`` makes for ``speed``; ``voice`` and ``batch_id`` are the
other two client-controlled strings hitting a ``String(64)`` column, and they had
no guard in front of them.

The bounds are read off the MODEL, never hardcoded here, so widening a column can
never leave the API guard behind. The DB half of the proof is
``test_overlong_voice_kills_the_whole_batch_in_the_db``: it drives the real
Postgres and shows what one unvalidated value costs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy import String, inspect
from sqlalchemy.exc import DataError

from src.api.schemas import ChunkIn, SubmitBatchRequest
from src.database.models import TtsJob
from src.services import job_manager as jm


def _column_width(name: str) -> int:
    col_type = inspect(TtsJob).columns[name].type
    assert isinstance(col_type, String), f"{name} is not a bounded String column"
    width = col_type.length
    assert isinstance(width, int) and width > 0, f"{name} is not a bounded String column"
    return width


VOICE_WIDTH = _column_width("voice")
BATCH_ID_WIDTH = _column_width("batch_id")


def test_a_realistic_voice_is_still_accepted() -> None:
    assert ChunkIn(ref="r1", text="hello", voice="af_heart").voice == "af_heart"
    assert ChunkIn(ref="r1", text="hello").voice is None
    # Exactly at the column width must pass — the bound is inclusive, not off-by-one.
    assert ChunkIn(ref="r1", text="hello", voice="v" * VOICE_WIDTH).voice is not None


def test_overlong_voice_is_rejected_at_the_schema() -> None:
    # PRECONDITION: one char shorter is legal, so this pins the boundary, not the type.
    assert ChunkIn(ref="r1", text="hello", voice="v" * VOICE_WIDTH).voice is not None
    with pytest.raises(ValidationError):
        ChunkIn(ref="r1", text="hello", voice="v" * (VOICE_WIDTH + 1))


def test_a_realistic_batch_id_is_still_accepted() -> None:
    uuid_like = "019fd0e2-1a2b-7c3d-8e4f-5a6b7c8d9e0f"
    assert len(uuid_like) <= BATCH_ID_WIDTH
    assert SubmitBatchRequest(batch_id=uuid_like, chunks=[]).batch_id == uuid_like


def test_overlong_batch_id_is_rejected_at_the_schema() -> None:
    assert SubmitBatchRequest(batch_id="b" * BATCH_ID_WIDTH, chunks=[]).batch_id
    with pytest.raises(ValidationError):
        SubmitBatchRequest(batch_id="b" * (BATCH_ID_WIDTH + 1), chunks=[])


async def test_overlong_voice_kills_the_whole_batch_in_the_db(session_factory) -> None:
    """WHY the schema bound matters: one bad chunk destroys the entire submission."""
    good = {"ref": "r2", "text": "A perfectly fine chunk.", "voice": "af_heart", "speed": 1.0}
    bad = {
        "ref": "r1",
        "text": "Another fine chunk.",
        "voice": "v" * (VOICE_WIDTH + 1),
        "speed": 1.0,
    }

    # PRECONDITION: the good chunk alone enqueues without complaint.
    async with session_factory() as s:
        assert (
            len(
                await jm.submit_batch(
                    s,
                    tenant_id="11111111-1111-1111-1111-111111111111",
                    batch_id="ok",
                    chunks=[good],
                )
            )
            == 1
        )

    with pytest.raises(DataError):
        async with session_factory() as s:
            await jm.submit_batch(
                s,
                tenant_id="11111111-1111-1111-1111-111111111111",
                batch_id="b1",
                chunks=[bad, good],
            )

    async with session_factory() as s:
        rows = await jm.get_batch(
            s, tenant_id="11111111-1111-1111-1111-111111111111", batch_id="b1"
        )
    assert rows == [], "the good chunk was lost with the bad one — the whole book 500s"
