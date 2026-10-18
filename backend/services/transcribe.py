"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""

from typing import Optional

from ..backends import get_stt_backend, STTBackend
from . import openrouter_stt


def get_whisper_model() -> STTBackend:
    """
    Get STT backend instance (MLX or PyTorch based on platform).
    
    Returns:
        STT backend instance
    """
    return get_stt_backend()


def unload_whisper_model():
    """Unload Whisper model to free memory."""
    backend = get_stt_backend()
    backend.unload_model()


async def transcribe_audio(audio_path: str, language: Optional[str], model_size: str) -> str:
    """Route openrouter-* ids to OpenRouter, everything else to the local Whisper backend."""
    if openrouter_stt.is_openrouter_model(model_size):
        return await openrouter_stt.transcribe(audio_path, language, model_size)
    return await get_stt_backend().transcribe(audio_path, language, model_size)
