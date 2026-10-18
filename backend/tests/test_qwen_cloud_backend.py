"""Tests for the qwen_cloud backend. No network access; dashscope not required
(_synthesize is monkeypatched, and it is the only place dashscope is imported)."""

import asyncio
import io

import numpy as np
import pytest
import soundfile as sf

from backend.backends import qwen_cloud_backend
from backend.backends.qwen_cloud_backend import QWEN_CLOUD_DEFAULT_VOICE, QwenCloudBackend


@pytest.fixture
def mp3_bytes():
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(np.arange(22050) / 10)).astype("float32"), 22050, format="MP3")
    return buf.getvalue()


@pytest.fixture
def api_key_env(monkeypatch):
    monkeypatch.setenv("VOICEBOX_CLOUD_API_KEY", "sk-sp-test123")


class _RequestFailureError(Exception):
    """Mimics dashscope.common.error.RequestFailure (attributes: name, message)."""

    def __init__(self, name, message):
        super().__init__(message)
        self.name = name
        self.message = message


def _recording_fake(monkeypatch, result):
    """Patch _synthesize to record its call args and return result."""
    calls = []

    def fake(api_key, voice, text, seed):
        calls.append((api_key, voice, text, seed))
        return result

    monkeypatch.setattr(qwen_cloud_backend, "_synthesize", fake)
    return calls


def _raising_fake(monkeypatch, exc):
    calls = []

    def fake(api_key, voice, text, seed):
        calls.append((api_key, voice, text, seed))
        raise exc

    monkeypatch.setattr(qwen_cloud_backend, "_synthesize", fake)
    return calls


def test_preset_voice_id_and_seed_are_passed_through(api_key_env, mp3_bytes, monkeypatch):
    calls = _recording_fake(monkeypatch, mp3_bytes)
    backend = QwenCloudBackend()

    audio, sr = asyncio.run(
        backend.generate(
            "Hello from QwenCloud.",
            {"voice_type": "preset", "preset_engine": "qwen_cloud", "preset_voice_id": "my-cloned-voice-1"},
            seed=70000,
        )
    )

    api_key, voice, text, seed = calls[0]
    assert voice == "my-cloned-voice-1"
    assert api_key == "sk-sp-test123"
    assert text == "Hello from QwenCloud."
    assert seed == 70000 % 65536
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert sr == 22050
    assert len(audio) > 0


def test_missing_voice_id_falls_back_to_default_voice(api_key_env, mp3_bytes, monkeypatch):
    calls = _recording_fake(monkeypatch, mp3_bytes)
    backend = QwenCloudBackend()

    asyncio.run(backend.generate("Hi.", {}))

    assert calls[0][1] == QWEN_CLOUD_DEFAULT_VOICE


def test_missing_api_key_raises_before_any_call(monkeypatch, mp3_bytes):
    monkeypatch.delenv("VOICEBOX_CLOUD_API_KEY", raising=False)
    calls = _recording_fake(monkeypatch, mp3_bytes)
    backend = QwenCloudBackend()

    with pytest.raises(RuntimeError, match="VOICEBOX_CLOUD_API_KEY"):
        asyncio.run(backend.generate("Hi.", {}))

    assert calls == []


def test_timeout_is_mapped_to_readable_error(api_key_env, monkeypatch):
    _raising_fake(monkeypatch, TimeoutError("x"))
    backend = QwenCloudBackend()

    with pytest.raises(RuntimeError, match="did not respond in time"):
        asyncio.run(backend.generate("Hi.", {}))


@pytest.mark.parametrize(
    ("error_name", "fragment"),
    [
        ("Throttling.RateQuota", "rate limit"),
        ("InvalidApiKey", "rejected the API key"),
        ("InternalError", "synthesis failed (InternalError)"),
    ],
)
def test_request_failure_mapping_never_leaks_key(api_key_env, monkeypatch, error_name, fragment):
    _raising_fake(monkeypatch, _RequestFailureError(error_name, "server said no"))
    backend = QwenCloudBackend()

    with pytest.raises(RuntimeError) as err_info:
        asyncio.run(backend.generate("Hi.", {}))

    assert fragment in str(err_info.value)
    assert "sk-sp-test123" not in str(err_info.value)


def test_empty_response_raises(api_key_env, monkeypatch):
    _recording_fake(monkeypatch, None)
    backend = QwenCloudBackend()

    with pytest.raises(RuntimeError, match="empty response"):
        asyncio.run(backend.generate("Hi.", {}))


def test_undecodable_audio_raises(api_key_env, monkeypatch):
    _recording_fake(monkeypatch, b"garbage")
    backend = QwenCloudBackend()

    with pytest.raises(RuntimeError, match="could not be decoded"):
        asyncio.run(backend.generate("Hi.", {}))
