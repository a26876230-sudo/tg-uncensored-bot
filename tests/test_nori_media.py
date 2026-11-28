import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from nori_media import VISION_MODEL, VoiceSynthesizer, read_media


def test_voice_transcription_preserves_the_question_at_the_end():
    transcript = "Предыстория. " * 400 + "Какой сегодня день?"
    media = SimpleNamespace(file_size=4, duration=180)
    media.get_file = AsyncMock(return_value=SimpleNamespace(
        file_size=4, download_as_bytearray=AsyncMock(return_value=bytearray(b"OggS")),
    ))
    create = AsyncMock(return_value=SimpleNamespace(text=transcript))
    groq = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    _, content = asyncio.run(read_media(groq, SimpleNamespace(voice=media), VISION_MODEL))
    assert content == transcript


def test_voice_cache_checks_content_and_never_keeps_failed_download(tmp_path):
    expected = b"verified model bytes"
    checksum = hashlib.sha256(expected).hexdigest()
    calls = 0

    def server(request):
        nonlocal calls
        calls += 1
        data = b"!" * len(expected) if calls == 1 else expected
        return httpx.Response(200, content=data)

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as http:
            speech = VoiceSynthesizer(http, tmp_path)
            with pytest.raises(ValueError):
                await speech._ensure_asset("model", len(expected), checksum)
            assert list(tmp_path.iterdir()) == []

            await speech._ensure_asset("model", len(expected), checksum)
            assert (tmp_path / "model").read_bytes() == expected
            await speech._ensure_asset("model", len(expected), checksum)
            assert calls == 2

    asyncio.run(check())
