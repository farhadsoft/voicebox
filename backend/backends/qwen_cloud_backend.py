"""
QwenCloud Token Plan cloud TTS backend.

Synthesises speech through QwenCloud's Token Plan WebSocket endpoint
(model ``qwen-audio-3.0-tts-plus``) using the DashScope SDK.

Voices work per-profile: a preset profile with
``preset_engine="qwen_cloud"`` carries a ``preset_voice_id`` (a system
voice, or a voice the user cloned outside Voicebox). A cloned profile
carries a ``cloud_voice_id`` when QwenCloud voice cloning is enabled
(``VOICEBOX_QWEN_CLOUD_CLONING=1``): the enrollment — uploading the
sample to DashScope temporary storage and creating the voice with the
pay-as-you-go key ``VOICEBOX_QWEN_ENROLL_API_KEY`` — happens in
``backend/services/qwen_cloud_voices.py``; this backend only receives
the resulting id and never uploads audio itself. Any other profile
falls back to the preset system voice ``longanlingxin``.

The synthesis API key is read only from the environment variable
``VOICEBOX_CLOUD_API_KEY`` (a Token Plan key, ``sk-sp-...``). Requires
the ``dashscope`` package.
"""

import asyncio
import io
import logging
import os

import numpy as np
import soundfile as sf

from .base import combine_voice_prompts as _combine_voice_prompts

logger = logging.getLogger(__name__)

QWEN_CLOUD_WS_URL = "wss://token-plan.maas.qwencloudapi.com/api-ws/v1/inference"
QWEN_CLOUD_MODEL = "qwen-audio-3.0-tts-plus"
QWEN_CLOUD_DEFAULT_VOICE = "longanlingxin"
API_KEY_ENV_VAR = "VOICEBOX_CLOUD_API_KEY"
SYNTH_TIMEOUT_MS = 120_000


def _synthesize(api_key: str, voice: str, text: str, seed: int) -> bytes | None:
    """Run one blocking DashScope synthesis. Called from a worker thread."""
    import dashscope
    from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

    dashscope.api_key = api_key  # SDK has no per-instance key; process-global, only this engine uses dashscope
    synthesizer = SpeechSynthesizer(
        model=QWEN_CLOUD_MODEL,
        voice=voice,
        format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
        seed=seed,
        url=QWEN_CLOUD_WS_URL,
    )
    return synthesizer.call(text, timeout_millis=SYNTH_TIMEOUT_MS)


class QwenCloudBackend:
    """Cloud TTS backend — no local model; the remote endpoint owns everything."""

    def __init__(self):
        self.model_size = "default"

    def is_loaded(self) -> bool:
        """Always True — there is no local model to load."""
        return True

    async def load_model(self, model_size: str = "default") -> None:
        """No-op — synthesis happens remotely."""
        return

    def unload_model(self) -> None:
        """No-op — nothing to unload locally."""
        return

    def _get_model_path(self, model_size: str) -> str:
        """No HF repo; report the endpoint so UIs/logs show something meaningful."""
        return QWEN_CLOUD_WS_URL

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Always True — used by /generate/stream's ensure_model_cached_or_raise."""
        return True

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """Return an empty prompt — qwen_cloud voice prompts are not built here.

        Cloned profiles are enrolled by ``backend/services/qwen_cloud_voices.py``
        when ``VOICEBOX_QWEN_CLOUD_CLONING=1``; kept for protocol parity only.
        """
        return {}, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        """Combine samples for protocol parity; generate() ignores the result.

        profiles.py saves the returned audio at 24000 Hz, so resample to
        that rate even though this backend never reads it back.
        """
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=24000)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesise ``text`` via QwenCloud and return (float32 mono audio, sample_rate).

        ``language`` and ``instruct`` are ignored: the SDK documents
        ``language_hints`` as zh/en only, so language is left to the
        model's auto-detection, and the endpoint has no instruct support.
        The voice comes from ``voice_prompt["preset_voice_id"]`` when the
        profile is a qwen_cloud preset profile, otherwise the default
        system voice is used.
        """
        api_key = os.environ.get(API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{API_KEY_ENV_VAR} is not set. Export your QwenCloud Token Plan key "
                "(sk-sp-...) in the terminal that starts the backend to use the qwen_cloud engine."
            )

        voice = (
            voice_prompt.get("preset_voice_id")
            or voice_prompt.get("cloud_voice_id")
            or QWEN_CLOUD_DEFAULT_VOICE
        )
        logger.info(
            "Sending %d chars of text to external service QwenCloud (model=%s, voice=%s)",
            len(text),
            QWEN_CLOUD_MODEL,
            voice,
        )

        try:
            audio_bytes = await asyncio.to_thread(_synthesize, api_key, voice, text, (seed or 0) % 65536)
        except ImportError as exc:
            raise RuntimeError(
                "The qwen_cloud engine needs the 'dashscope' package: pip install -c ~/constraints.txt dashscope"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "QwenCloud did not respond in time (5s connect / 120s synthesis). If it fails "
                "immediately, the WebSocket handshake was probably rejected: check "
                f"{API_KEY_ENV_VAR} and the 'websocket error: Handshake status ...' line in the server log."
            ) from exc
        except Exception as exc:
            # Covers the SDK's RequestFailure (attributes: name, message),
            # which has no common base class worth importing.
            name = getattr(exc, "name", None) or exc.__class__.__name__
            message = getattr(exc, "message", None) or str(exc)
            logger.error("QwenCloud synthesis failed: %s", name)
            if name.startswith("Throttling"):
                raise RuntimeError(
                    f"QwenCloud rate limit or quota reached ({name}): {message}. Wait and retry."
                ) from exc
            if name.startswith(("InvalidApiKey", "AccessDenied")):
                raise RuntimeError(
                    f"QwenCloud rejected the API key ({name}). Check {API_KEY_ENV_VAR} is your Token Plan key."
                ) from exc
            if voice_prompt.get("cloud_voice_id"):
                raise RuntimeError(
                    f"QwenCloud synthesis with cloned voice {voice} failed ({name}): {message}. "
                    "If QwenCloud reports the voice as unknown or not allowed, Token Plan "
                    "synthesis may not accept voices created with the pay-as-you-go enrollment key."
                ) from exc
            raise RuntimeError(f"QwenCloud synthesis failed ({name}): {message}") from exc

        if not audio_bytes:
            raise RuntimeError("QwenCloud returned an empty response (no audio).")

        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"QwenCloud returned audio that could not be decoded as MP3: {exc}") from exc

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            raise RuntimeError("QwenCloud returned an empty response (no audio).")

        # Keep the native 22050 Hz rate — the pipeline accepts any sample rate.
        return audio.astype(np.float32, copy=False), int(sr)
