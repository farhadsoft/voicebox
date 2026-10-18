"""QwenCloud voice cloning: enroll a profile's sample as a QwenCloud voice.

Opt-in via ``VOICEBOX_QWEN_CLOUD_CLONING=1``. On first use with the
``qwen_cloud`` engine, the profile's longest sample is uploaded to
QwenCloud's own temporary storage (DashScope ``/uploads`` policy flow,
auto-deleted after 48 h) and enrolled with the ``voice-enrollment``
model for ``qwen-audio-3.0-tts-plus``. The returned voice id is stored
on the profile together with a hash of the sample audio + reference
text, so later generations reuse it with zero HTTP calls until the
sample changes.

Enrollment needs a pay-as-you-go API key (``sk-ws-...``) in
``VOICEBOX_QWEN_ENROLL_API_KEY``: Token Plan keys (``sk-sp-...``) are
rejected by the enrollment API. Synthesis itself keeps using the Token
Plan key in ``VOICEBOX_CLOUD_API_KEY`` (see qwen_cloud_backend).

Deliberately does NOT use the dashscope SDK: ``dashscope.api_key`` is
process-global and synthesis sets it to the Token Plan key, so
enrolling with another key through the SDK would race; the SDK's
``VoiceEnrollmentService`` also doesn't send the OSS-resolve header.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import httpx
import soundfile

from .. import config
from ..backends.qwen_cloud_backend import QWEN_CLOUD_MODEL
from ..database import ProfileSample
from ..utils.cache import get_cache_key

logger = logging.getLogger(__name__)

ENROLL_API_KEY_ENV_VAR = "VOICEBOX_QWEN_ENROLL_API_KEY"
QWEN_CLOUD_HTTP_BASE = "https://maas.qwencloudapi.com/api/v1"
UPLOADS_URL = f"{QWEN_CLOUD_HTTP_BASE}/uploads"
CUSTOMIZATION_URL = f"{QWEN_CLOUD_HTTP_BASE}/services/audio/tts/customization"
ENROLLMENT_MODEL = "voice-enrollment"
VOICE_PREFIX = "voicebox"  # ≤10 alnum; lets list_voice filter Voicebox voices
MIN_SAMPLE_SECONDS = 5.0  # API: ≥5 s continuous speech
STATUS_POLL_INTERVAL_S = 2.0
STATUS_POLL_TIMEOUT_S = 60.0
LANGUAGE_HINTS = {"zh", "en", "fr", "de", "ja", "ko", "ru", "pt", "th", "id", "vi", "it", "es", "ms", "fil", "ar"}

# ponytail: one global lock; enrollment is rare, per-profile locks if it ever matters
_enroll_lock = asyncio.Lock()


def _http_client(timeout: float = 60.0) -> httpx.AsyncClient:
    """Client factory — tests monkeypatch this to inject a MockTransport."""
    return httpx.AsyncClient(timeout=timeout)


def _enroll_key() -> str:
    """Read and validate the enrollment API key. Never logged."""
    api_key = os.environ.get(ENROLL_API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{ENROLL_API_KEY_ENV_VAR} is not set. QwenCloud voice cloning needs a "
            "pay-as-you-go API key (sk-ws-...) exported in the terminal that starts the backend."
        )
    if api_key.startswith("sk-sp-"):
        raise RuntimeError(
            f"{ENROLL_API_KEY_ENV_VAR} holds a Token Plan key (sk-sp-...). Token Plan keys "
            "cannot call voice enrollment; use a pay-as-you-go key (sk-ws-...)."
        )
    return api_key


def _raise_for_api(response: httpx.Response, action: str) -> None:
    """Turn a non-200 QwenCloud response into a readable RuntimeError."""
    if response.status_code == 200:
        return
    code = ""
    message = ""
    try:
        body = response.json()
        code = str(body.get("code") or "")
        message = str(body.get("message") or "")
    except Exception:
        pass
    if not message:
        message = response.text[:300]
    status = response.status_code
    if status == 401 or code.startswith("InvalidApiKey"):
        raise RuntimeError(
            f"QwenCloud rejected {ENROLL_API_KEY_ENV_VAR} ({code}) while {action}. "
            "Use a pay-as-you-go key (sk-ws-...)."
        )
    if status == 429 or code.startswith("Throttling"):
        raise RuntimeError(f"QwenCloud rate limit or quota reached ({code}) while {action}: {message}. Wait and retry.")
    if status == 404 and str(response.request.url).startswith(UPLOADS_URL):
        raise RuntimeError(
            f"QwenCloud has no temporary file upload endpoint at {UPLOADS_URL} (HTTP 404), "
            "so the voice sample cannot be enrolled. Voicebox does not fall back to a third-party host."
        )
    raise RuntimeError(f"QwenCloud request failed while {action} (HTTP {status}, {code}): {message}")


async def _upload_sample(client: httpx.AsyncClient, api_key: str, path: Path) -> str:
    """Upload the sample to QwenCloud temporary storage; return its oss:// URL."""
    r = await client.get(
        UPLOADS_URL,
        params={"action": "getPolicy", "model": ENROLLMENT_MODEL},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    _raise_for_api(r, "requesting a temporary upload slot")
    policy = r.json()["data"]

    key = f"{policy['upload_dir']}/{path.name}"
    logger.info(
        "Uploading voice sample %s to external service QwenCloud "
        "(DashScope temporary storage, auto-deleted after 48 h) for voice cloning",
        path,
    )
    upload = await client.post(
        policy["upload_host"],
        data={
            "OSSAccessKeyId": policy["oss_access_key_id"],
            "Signature": policy["signature"],
            "policy": policy["policy"],
            "key": key,
            "x-oss-object-acl": policy["x_oss_object_acl"],
            "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
            "success_action_status": "200",
        },
        files={"file": (path.name, path.read_bytes(), "audio/wav")},
    )
    if upload.status_code != 200:
        raise RuntimeError(
            f"Uploading the voice sample to QwenCloud temporary storage failed "
            f"(HTTP {upload.status_code}): {upload.text[:300]}"
        )
    return f"oss://{key}"


async def _customization(
    client: httpx.AsyncClient,
    api_key: str,
    input: dict,
    action: str,
    resolve_oss: bool = False,
) -> dict:
    """POST one voice-enrollment action; return its ``output`` dict."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if resolve_oss:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    r = await client.post(
        CUSTOMIZATION_URL,
        json={"model": ENROLLMENT_MODEL, "input": input},
        headers=headers,
    )
    _raise_for_api(r, action)
    return r.json().get("output") or {}


async def _delete_voice(client: httpx.AsyncClient, api_key: str, voice_id: str) -> None:
    """Delete a cloned voice; best effort, never raises."""
    try:
        await _customization(client, api_key, {"action": "delete_voice", "voice_id": voice_id}, f"deleting voice {voice_id}")
        logger.info("Deleted QwenCloud cloned voice %s", voice_id)
    except Exception as exc:
        logger.warning("Could not delete QwenCloud cloned voice %s: %s", voice_id, exc)


async def delete_remote_voice_best_effort(voice_id: str) -> None:
    """Delete a cloned voice on QwenCloud after its profile was deleted.

    Runs regardless of VOICEBOX_QWEN_CLOUD_CLONING — cleanup must still work
    after the user turns cloning off. Never raises.
    """
    try:
        api_key = _enroll_key()
    except RuntimeError as exc:
        logger.warning("Not deleting QwenCloud cloned voice %s: %s", voice_id, exc)
        return
    try:
        async with _http_client(timeout=10.0) as client:
            await _delete_voice(client, api_key, voice_id)
    except Exception as exc:
        logger.warning("Could not delete QwenCloud cloned voice %s: %s", voice_id, exc)


def _pick_sample(profile_id: str, samples) -> tuple[ProfileSample, Path, float]:
    """The longest single sample of the profile (deterministic tie-break by id).

    Longest *single* sample, not the combined file: the API takes one URL,
    uses at most 10 s (default) of prompt audio and wants continuous
    single-recording speech with pauses ≤2 s — concatenating separate
    recordings adds nothing and risks room/level jumps.
    """
    best = None
    for sample in sorted(samples, key=lambda s: s.id):
        path = config.resolve_storage_path(sample.audio_path)
        if path is None or not path.exists():
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        duration = soundfile.info(str(path)).duration
        if best is None or duration > best[2]:
            best = (sample, path, duration)
    return best


async def _wait_until_ready(client: httpx.AsyncClient, api_key: str, voice_id: str, profile, db) -> None:
    """Poll query_voice until the cloned voice is OK; readable errors otherwise."""
    deadline = time.monotonic() + STATUS_POLL_TIMEOUT_S
    while True:
        output = await _customization(
            client, api_key, {"action": "query_voice", "voice_id": voice_id}, "checking the cloned voice status"
        )
        status = output.get("status")
        if status == "OK":
            return
        if status == "UNDEPLOYED":
            await _delete_voice(client, api_key, voice_id)
            profile.qwen_cloud_voice_id = None
            profile.qwen_cloud_voice_hash = None
            db.commit()
            raise RuntimeError(
                "QwenCloud rejected the cloned voice in review (status UNDEPLOYED). "
                "Use a clean 10-20 s single-speaker sample without background noise."
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"QwenCloud is still processing cloned voice {voice_id} after 60 s. "
                "Retry the generation in a minute."
            )
        await asyncio.sleep(STATUS_POLL_INTERVAL_S)


async def voice_prompt_for_profile(profile, samples, db) -> dict:
    """Voice prompt for a cloned profile on qwen_cloud: {"voice_type": "cloned", "cloud_voice_id": ...}.

    Enrolls the sample once and caches the voice id on the profile keyed by
    a hash of sample audio + reference text; a cache hit makes zero HTTP calls.
    """
    sample, path, duration = _pick_sample(profile.id, samples)
    digest = get_cache_key(str(path), sample.reference_text)

    async with _enroll_lock:
        db.refresh(profile)
        if profile.qwen_cloud_voice_id and profile.qwen_cloud_voice_hash == digest:
            return {"voice_type": "cloned", "cloud_voice_id": profile.qwen_cloud_voice_id}

        api_key = _enroll_key()
        if duration < MIN_SAMPLE_SECONDS:
            raise RuntimeError(
                "QwenCloud voice cloning needs at least 5 s of continuous speech; the longest "
                f"sample of this profile is {duration:.1f} s. Add a longer sample (10-20 s is ideal)."
            )
        old_voice_id = profile.qwen_cloud_voice_id
        try:
            async with _http_client() as client:
                oss_url = await _upload_sample(client, api_key, path)
                input = {
                    "action": "create_voice",
                    "target_model": QWEN_CLOUD_MODEL,
                    "prefix": VOICE_PREFIX,
                    "url": oss_url,
                }
                if profile.language in LANGUAGE_HINTS:
                    input["language_hints"] = [profile.language]
                output = await _customization(client, api_key, input, "enrolling the voice sample", resolve_oss=True)
                voice_id = output.get("voice_id")
                if not voice_id:
                    raise RuntimeError("QwenCloud voice enrollment returned no voice_id.")
                # Store id+hash right away so a later timeout/retry never re-uploads.
                profile.qwen_cloud_voice_id = voice_id
                profile.qwen_cloud_voice_hash = digest
                db.commit()
                logger.info("Enrolled QwenCloud cloned voice %s for profile %s", voice_id, profile.id)
                if old_voice_id and old_voice_id != voice_id:
                    await _delete_voice(client, api_key, old_voice_id)
                await _wait_until_ready(client, api_key, voice_id, profile, db)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not reach QwenCloud for voice cloning: {exc}") from exc

    return {"voice_type": "cloned", "cloud_voice_id": voice_id}
