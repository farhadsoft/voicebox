"""OpenRouter cloud speech-to-text (Whisper via /api/v1/audio/transcriptions).

The API key is read only from VOICEBOX_OPENROUTER_API_KEY, per call. Never logged.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
import soundfile as sf

logger = logging.getLogger(__name__)

API_KEY_ENV_VAR = "VOICEBOX_OPENROUTER_API_KEY"
TRANSCRIPTIONS_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
# Voicebox model id -> OpenRouter model slug
OPENROUTER_STT_MODELS = {
    "openrouter-whisper-turbo": "openai/whisper-large-v3-turbo",
    "openrouter-whisper-large-v3": "openai/whisper-large-v3",
}
AUDIO_FORMATS = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".m4a": "m4a",
    ".ogg": "ogg",
    ".webm": "webm",
    ".aac": "aac",
}
REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)  # upload + OpenRouter's 60 s processing window
TIMEOUT_MESSAGE = "OpenRouter transcription timed out. Try a shorter clip or a local Whisper model."


def is_openrouter_model(model: Optional[str]) -> bool:
    return model in OPENROUTER_STT_MODELS


def _http_client() -> httpx.AsyncClient:
    """Client factory — tests monkeypatch this to inject a MockTransport."""
    return httpx.AsyncClient(timeout=REQUEST_TIMEOUT)


def _api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV_VAR} is not set. Export it in the shell that starts the "
            "Voicebox backend to use OpenRouter transcription."
        )
    return api_key


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status == 200:
        return
    try:
        message = str(response.json()["error"]["message"])[:300]
    except Exception:
        message = response.text[:300]
    if status == 401:
        raise RuntimeError(
            f"OpenRouter rejected {API_KEY_ENV_VAR} (HTTP 401). Check the key at https://openrouter.ai/settings/keys."
        )
    if status == 402:
        raise RuntimeError(
            "OpenRouter: insufficient credits (HTTP 402). Add credits at https://openrouter.ai/settings/credits and retry."
        )
    if status == 429:
        retry_after = response.headers.get("Retry-After")
        wait = f"Wait {retry_after} s" if retry_after else "Wait a moment"
        raise RuntimeError(f"OpenRouter rate limit reached (HTTP 429). {wait} and retry.")
    if status in (408, 504, 524):
        raise RuntimeError(TIMEOUT_MESSAGE)
    raise RuntimeError(f"OpenRouter transcription failed (HTTP {status}): {message}")


async def transcribe(audio_path: str, language: Optional[str], model: str) -> str:
    """Transcribe a local audio file with an OpenRouter Whisper model."""
    slug = OPENROUTER_STT_MODELS[model]
    api_key = _api_key()
    path = Path(audio_path)
    try:
        seconds = f"{sf.info(str(path)).duration:.1f}"
    except Exception:  # log-only; a header soundfile can't read must not fail transcription
        seconds = "?"
    payload: dict = {
        "model": slug,
        "input_audio": {
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "format": AUDIO_FORMATS.get(path.suffix.lower(), "wav"),
        },
    }
    if language:
        payload["language"] = language

    logger.info(
        "Sending %s (%s s) to external service OpenRouter for transcription (model=%s)",
        path.name,
        seconds,
        slug,
    )
    try:
        async with _http_client() as client:
            response = await client.post(
                TRANSCRIPTIONS_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.TimeoutException:
        raise RuntimeError(TIMEOUT_MESSAGE) from None
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Could not reach OpenRouter ({type(exc).__name__}). Check your network connection."
        ) from None
    _raise_for_status(response)
    try:
        text = str(response.json().get("text") or "").strip()
    except Exception:
        raise RuntimeError("OpenRouter returned an unreadable response (HTTP 200).") from None
    if not text:
        raise RuntimeError("OpenRouter returned an empty transcript. The audio may be silent or too short.")
    return text
