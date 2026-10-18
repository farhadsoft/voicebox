"""Tests for OpenRouter cloud STT (backend/services/openrouter_stt.py).

No network access: the module's ``_http_client`` factory is monkeypatched to
return clients backed by ``httpx.MockTransport``.

Usage:
    python -m pytest backend/tests/test_openrouter_stt.py -v
"""

import base64
import json
import logging
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

# Repo root on sys.path so ``backend`` imports as a package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services import openrouter_stt, transcribe

TEST_KEY = "sk-or-test-SECRET"
OK_BODY = {"text": " Hello world. ", "usage": {"seconds": 1.5, "cost": 0.0001}}


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv(openrouter_stt.API_KEY_ENV_VAR, TEST_KEY)


@pytest.fixture
def wav(tmp_path):
    """A real 1.5 s of 16 kHz mono silence on disk."""
    path = tmp_path / "clip.wav"
    sf.write(str(path), np.zeros(24000, dtype="float32"), 16000)
    return str(path)


@pytest.fixture
def http(monkeypatch):
    """Patch _http_client; returns (recorded requests, mutable overrides).

    With no override the handler answers a fresh 200 OK_BODY per request; set
    ``overrides["response"]`` to a Response or an exception to force a case.
    """
    requests: list[httpx.Request] = []
    overrides: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        override = overrides.get("response")
        if isinstance(override, Exception):
            raise override
        if override is not None:
            return override
        return httpx.Response(200, json=OK_BODY)

    monkeypatch.setattr(
        openrouter_stt,
        "_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return requests, overrides


async def test_success_sends_base64_wav_and_trims_text(key, wav, http):
    requests, _ = http

    text = await openrouter_stt.transcribe(wav, None, "openrouter-whisper-turbo")

    assert text == "Hello world."
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == openrouter_stt.TRANSCRIPTIONS_URL
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "openai/whisper-large-v3-turbo"
    assert body["input_audio"]["format"] == "wav"
    assert base64.b64decode(body["input_audio"]["data"]) == Path(wav).read_bytes()
    assert "language" not in body


async def test_language_and_large_v3_slug_pass_through(key, wav, http):
    requests, _ = http

    await openrouter_stt.transcribe(wav, "de", "openrouter-whisper-large-v3")

    body = json.loads(requests[0].content)
    assert body["language"] == "de"
    assert body["model"] == "openai/whisper-large-v3"


async def test_missing_key_raises_before_any_request(wav, http, monkeypatch):
    requests, _ = http
    monkeypatch.delenv(openrouter_stt.API_KEY_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match="VOICEBOX_OPENROUTER_API_KEY is not set"):
        await openrouter_stt.transcribe(wav, None, "openrouter-whisper-turbo")

    assert requests == []


ERROR_CASES = [
    ("401", httpx.Response(401, json={"error": {"code": 401, "message": "No auth"}}), "HTTP 401"),
    ("402", httpx.Response(402, json={"error": {"code": 402, "message": "No credits"}}), "insufficient credits"),
    (
        "429",
        httpx.Response(429, headers={"Retry-After": "7"}, json={"error": {"code": 429, "message": "Slow down"}}),
        "Wait 7 s",
    ),
    ("408", httpx.Response(408, json={"error": {"code": 408, "message": "Upstream timeout"}}), "timed out"),
    ("500", httpx.Response(500, json={"error": {"code": 500, "message": "boom"}}), "HTTP 500\\): boom"),
    ("read-timeout", httpx.ReadTimeout("t"), "timed out"),
    ("connect-error", httpx.ConnectError("c"), "Could not reach OpenRouter \\(ConnectError\\)"),
    ("empty-text", httpx.Response(200, json={"text": "  "}), "empty transcript"),
]


@pytest.mark.parametrize("label,response,match", ERROR_CASES, ids=[c[0] for c in ERROR_CASES])
async def test_error_classes_become_readable_runtime_errors(key, wav, http, label, response, match):
    _, overrides = http
    overrides["response"] = response

    with pytest.raises(RuntimeError, match=match):
        await openrouter_stt.transcribe(wav, None, "openrouter-whisper-turbo")


async def test_logs_external_send_and_never_leaks_the_key(key, wav, http, caplog):
    _, overrides = http
    caplog.set_level(logging.INFO)

    await openrouter_stt.transcribe(wav, None, "openrouter-whisper-turbo")

    expected = (
        "Sending clip.wav (1.5 s) to external service OpenRouter for transcription "
        "(model=openai/whisper-large-v3-turbo)"
    )
    hits = [r for r in caplog.records if expected in r.getMessage()]
    assert len(hits) == 1
    assert TEST_KEY not in caplog.text

    overrides["response"] = httpx.Response(401, json={"error": {"code": 401, "message": "No auth"}})
    with pytest.raises(RuntimeError) as excinfo:
        await openrouter_stt.transcribe(wav, None, "openrouter-whisper-turbo")

    assert TEST_KEY not in str(excinfo.value)
    assert TEST_KEY not in caplog.text


async def test_dispatcher_splits_cloud_and_local(key, wav, http, monkeypatch):
    requests, _ = http

    assert await transcribe.transcribe_audio(wav, None, "openrouter-whisper-turbo") == "Hello world."

    class _FakeLocal:
        async def transcribe(self, path, language, model_size):
            return f"local:{model_size}"

    monkeypatch.setattr(transcribe, "get_stt_backend", lambda: _FakeLocal())
    sent_before = len(requests)

    assert await transcribe.transcribe_audio(wav, None, "turbo") == "local:turbo"
    assert len(requests) == sent_before  # a local id never touches the network
