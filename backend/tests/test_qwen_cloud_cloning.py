"""Tests for QwenCloud voice cloning enrollment (backend/services/qwen_cloud_voices.py).

No network access: the module's ``_http_client`` factory is monkeypatched to
return clients backed by ``httpx.MockTransport``.

Usage:
    python -m pytest backend/tests/test_qwen_cloud_cloning.py -v
"""

import json
import logging
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Repo root on sys.path so ``backend`` imports as a package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import config
from backend.database import Base, ProfileSample, VoiceProfile
from backend.services import profiles as profiles_service
from backend.services import qwen_cloud_voices
from backend.utils.cache import get_cache_key

POLICY = {
    "policy": "pol",
    "signature": "sig",
    "upload_dir": "tmp/abc",
    "upload_host": "https://oss.example.com/bucket",
    "oss_access_key_id": "ossid",
    "x_oss_object_acl": "private",
    "x_oss_forbid_overwrite": "true",
}
NEW_VOICE_ID = "qwen-audio-3.0-tts-plus-voicebox-new1"


def _kind(request: httpx.Request) -> str:
    """Classify a recorded request: getPolicy / upload / <enrollment action>."""
    url = str(request.url)
    if url.startswith(qwen_cloud_voices.UPLOADS_URL):
        return "getPolicy"
    if url.startswith(POLICY["upload_host"]):
        return "upload"
    return json.loads(request.content)["input"]["action"]


def _handler(requests: list, overrides: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        kind = _kind(request)
        if kind in overrides:
            override = overrides[kind]
            if isinstance(override, Exception):
                raise override
            return override
        if kind == "getPolicy":
            return httpx.Response(200, json={"data": POLICY})
        if kind == "upload":
            return httpx.Response(200)
        if kind == "create_voice":
            return httpx.Response(200, json={"output": {"voice_id": NEW_VOICE_ID}})
        if kind == "query_voice":
            return httpx.Response(200, json={"output": {"status": "OK"}})
        if kind == "delete_voice":
            return httpx.Response(200, json={"output": {}})
        raise AssertionError(f"unexpected request kind {kind}")

    return handler


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A session backed by a temp SQLite file, with the data dir redirected."""
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield session
    session.close()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("VOICEBOX_QWEN_ENROLL_API_KEY", "sk-ws-test")
    monkeypatch.setenv("VOICEBOX_QWEN_CLOUD_CLONING", "1")
    monkeypatch.setattr(qwen_cloud_voices, "STATUS_POLL_INTERVAL_S", 0)


@pytest.fixture
def http(monkeypatch):
    """Patch _http_client; returns (recorded requests, mutable overrides)."""
    requests: list[httpx.Request] = []
    overrides: dict = {}
    monkeypatch.setattr(
        qwen_cloud_voices,
        "_http_client",
        lambda timeout=60.0: httpx.AsyncClient(transport=httpx.MockTransport(_handler(requests, overrides))),
    )
    return requests, overrides


def _make_profile_with_sample(db, tmp_path, pid="p1", sid="s1", seconds=12.0, text="Hello there."):
    """Cloned profile + one real 24 kHz mono PCM_16 WAV sample on disk."""
    profile = VoiceProfile(id=pid, name=f"Voice {pid}", language="en", voice_type="cloned")
    db.add(profile)
    audio_dir = tmp_path / "profiles" / pid
    audio_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(audio_dir / f"{sid}.wav", seconds)
    db.add(
        ProfileSample(
            id=sid,
            profile_id=pid,
            audio_path=f"profiles/{pid}/{sid}.wav",
            reference_text=text,
        )
    )
    db.commit()
    return profile


def _write_wav(path: Path, seconds: float, amplitude=0.3) -> None:
    sr = 24000
    data = (amplitude * np.sin(np.arange(int(sr * seconds)) / 10)).astype("float32")
    sf.write(str(path), data, sr, subtype="PCM_16")


def _samples(db, pid="p1"):
    return db.query(ProfileSample).filter_by(profile_id=pid).all()


@pytest.mark.asyncio
async def test_cache_miss_enrolls_and_caches(db, tmp_path, env, http, caplog):
    requests, _ = http
    profile = _make_profile_with_sample(db, tmp_path)

    with caplog.at_level(logging.INFO, logger="backend.services.qwen_cloud_voices"):
        result = await qwen_cloud_voices.voice_prompt_for_profile(profile, _samples(db), db)

    assert result == {"voice_type": "cloned", "cloud_voice_id": NEW_VOICE_ID}
    assert [_kind(r) for r in requests] == ["getPolicy", "upload", "create_voice", "query_voice"]

    get_policy = requests[0]
    assert get_policy.method == "GET"
    assert get_policy.url.params["action"] == "getPolicy"
    assert get_policy.url.params["model"] == "voice-enrollment"
    assert get_policy.headers["authorization"] == "Bearer sk-ws-test"

    upload = requests[1]
    assert upload.method == "POST"
    assert str(upload.url).startswith(POLICY["upload_host"])
    assert "authorization" not in upload.headers
    assert b"tmp/abc/s1.wav" in upload.content  # OSS key = upload_dir/filename
    assert b"success_action_status" in upload.content

    create = requests[2]
    assert create.headers["x-dashscope-ossresourceresolve"] == "enable"
    assert json.loads(create.content) == {
        "model": "voice-enrollment",
        "input": {
            "action": "create_voice",
            "target_model": "qwen-audio-3.0-tts-plus",
            "prefix": "voicebox",
            "url": "oss://tmp/abc/s1.wav",
            "language_hints": ["en"],
        },
    }
    assert json.loads(requests[3].content)["input"] == {"action": "query_voice", "voice_id": NEW_VOICE_ID}

    db.refresh(profile)
    assert profile.qwen_cloud_voice_id == NEW_VOICE_ID
    path = config.resolve_storage_path("profiles/p1/s1.wav")
    assert profile.qwen_cloud_voice_hash == get_cache_key(str(path), "Hello there.")

    assert "QwenCloud" in caplog.text
    assert "s1.wav" in caplog.text
    assert "sk-ws-test" not in caplog.text


@pytest.mark.asyncio
async def test_cache_hit_makes_no_requests(db, tmp_path, env, http):
    requests, _ = http
    profile = _make_profile_with_sample(db, tmp_path)
    samples = _samples(db)

    first = await qwen_cloud_voices.voice_prompt_for_profile(profile, samples, db)
    count_after_first = len(requests)
    second = await qwen_cloud_voices.voice_prompt_for_profile(profile, samples, db)

    assert second == first
    assert len(requests) == count_after_first


@pytest.mark.asyncio
async def test_changed_reference_text_reenrolls_and_deletes_old(db, tmp_path, env, http):
    requests, overrides = http
    profile = _make_profile_with_sample(db, tmp_path)
    samples = _samples(db)

    first = await qwen_cloud_voices.voice_prompt_for_profile(profile, samples, db)
    old_id = first["cloud_voice_id"]
    overrides["create_voice"] = httpx.Response(
        200, json={"output": {"voice_id": "qwen-audio-3.0-tts-plus-voicebox-new2"}}
    )

    samples[0].reference_text = "Different text."
    db.commit()
    second = await qwen_cloud_voices.voice_prompt_for_profile(profile, samples, db)

    assert second["cloud_voice_id"] == "qwen-audio-3.0-tts-plus-voicebox-new2"
    delete_inputs = [json.loads(r.content)["input"] for r in requests if _kind(r) == "delete_voice"]
    assert {"action": "delete_voice", "voice_id": old_id} in delete_inputs
    db.refresh(profile)
    assert profile.qwen_cloud_voice_id == "qwen-audio-3.0-tts-plus-voicebox-new2"
    path = config.resolve_storage_path(samples[0].audio_path)
    assert profile.qwen_cloud_voice_hash == get_cache_key(str(path), "Different text.")


@pytest.mark.asyncio
async def test_changed_audio_reenrolls(db, tmp_path, env, http):
    requests, _ = http
    profile = _make_profile_with_sample(db, tmp_path)
    samples = _samples(db)

    await qwen_cloud_voices.voice_prompt_for_profile(profile, samples, db)
    count_after_first = len(requests)

    _write_wav(tmp_path / "profiles" / "p1" / "s1.wav", 12.0, amplitude=0.1)  # same text, new audio
    await qwen_cloud_voices.voice_prompt_for_profile(profile, samples, db)

    kinds = [_kind(r) for r in requests[count_after_first:]]
    assert kinds == ["getPolicy", "upload", "create_voice", "query_voice"]
    path = config.resolve_storage_path(samples[0].audio_path)
    db.refresh(profile)
    assert profile.qwen_cloud_voice_hash == get_cache_key(str(path), "Hello there.")


@pytest.mark.parametrize(
    "override,fragments",
    [
        ({"getPolicy": httpx.Response(404, json={})}, ["no temporary file upload endpoint"]),
        (
            {"getPolicy": httpx.Response(401, json={"code": "InvalidApiKey.NotFound", "message": "nope"})},
            ["VOICEBOX_QWEN_ENROLL_API_KEY", "pay-as-you-go"],
        ),
        (
            {"create_voice": httpx.Response(429, json={"code": "Throttling.RateQuota", "message": "slow down"})},
            ["rate limit or quota"],
        ),
        (
            {"create_voice": httpx.Response(400, json={"code": "Audio.AudioShortError", "message": "bad audio"})},
            ["Audio.AudioShortError", "bad audio"],
        ),
        ({"upload": httpx.Response(403, text="denied")}, ["Uploading the voice sample"]),
        ({"getPolicy": httpx.ConnectError("boom")}, ["Could not reach QwenCloud"]),
    ],
)
@pytest.mark.asyncio
async def test_api_errors_are_readable_and_cache_nothing(db, tmp_path, env, http, override, fragments):
    _, overrides = http
    overrides.update(override)
    profile = _make_profile_with_sample(db, tmp_path)

    with pytest.raises(RuntimeError) as err_info:
        await qwen_cloud_voices.voice_prompt_for_profile(profile, _samples(db), db)

    for fragment in fragments:
        assert fragment in str(err_info.value)
    db.refresh(profile)
    assert profile.qwen_cloud_voice_id is None
    assert profile.qwen_cloud_voice_hash is None


@pytest.mark.parametrize(
    "key,fragment",
    [(None, "is not set"), ("sk-sp-x", "Token Plan key")],
)
@pytest.mark.asyncio
async def test_bad_enroll_key_fails_before_any_request(db, tmp_path, monkeypatch, http, key, fragment):
    requests, _ = http
    if key is None:
        monkeypatch.delenv("VOICEBOX_QWEN_ENROLL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("VOICEBOX_QWEN_ENROLL_API_KEY", key)
    profile = _make_profile_with_sample(db, tmp_path)

    with pytest.raises(RuntimeError) as err_info:
        await qwen_cloud_voices.voice_prompt_for_profile(profile, _samples(db), db)

    assert fragment in str(err_info.value)
    assert requests == []


@pytest.mark.asyncio
async def test_short_sample_rejected_before_upload(db, tmp_path, env, http):
    requests, _ = http
    profile = _make_profile_with_sample(db, tmp_path, seconds=3.0)

    with pytest.raises(RuntimeError) as err_info:
        await qwen_cloud_voices.voice_prompt_for_profile(profile, _samples(db), db)

    assert "at least 5 s" in str(err_info.value)
    assert requests == []


@pytest.mark.asyncio
async def test_undeployed_deletes_voice_and_resets_columns(db, tmp_path, env, http):
    requests, overrides = http
    overrides["query_voice"] = httpx.Response(200, json={"output": {"status": "UNDEPLOYED"}})
    profile = _make_profile_with_sample(db, tmp_path)

    with pytest.raises(RuntimeError) as err_info:
        await qwen_cloud_voices.voice_prompt_for_profile(profile, _samples(db), db)

    assert "rejected" in str(err_info.value)
    delete_inputs = [json.loads(r.content)["input"] for r in requests if _kind(r) == "delete_voice"]
    assert {"action": "delete_voice", "voice_id": NEW_VOICE_ID} in delete_inputs
    db.refresh(profile)
    assert profile.qwen_cloud_voice_id is None
    assert profile.qwen_cloud_voice_hash is None


def test_validate_profile_engine_gates_qwen_cloud_on_flag(monkeypatch):
    profile = VoiceProfile(id="g1", name="G", language="en", voice_type="cloned")

    monkeypatch.delenv("VOICEBOX_QWEN_CLOUD_CLONING", raising=False)
    with pytest.raises(ValueError) as err_info:
        profiles_service.validate_profile_engine(profile, "qwen_cloud")
    assert "VOICEBOX_QWEN_CLOUD_CLONING" in str(err_info.value)

    monkeypatch.setenv("VOICEBOX_QWEN_CLOUD_CLONING", "1")
    profiles_service.validate_profile_engine(profile, "qwen_cloud")  # must not raise


@pytest.mark.asyncio
async def test_delete_profile_deletes_remote_voice(db, tmp_path, env, http):
    requests, _ = http
    profile = _make_profile_with_sample(db, tmp_path)
    profile.qwen_cloud_voice_id = "v1"
    db.commit()

    assert await profiles_service.delete_profile("p1", db) is True

    delete_inputs = [json.loads(r.content)["input"] for r in requests if _kind(r) == "delete_voice"]
    assert {"action": "delete_voice", "voice_id": "v1"} in delete_inputs
    assert db.query(VoiceProfile).filter_by(id="p1").first() is None


@pytest.mark.asyncio
async def test_delete_profile_survives_remote_failure(db, tmp_path, env, http):
    _, overrides = http
    overrides["delete_voice"] = httpx.Response(500, json={"code": "InternalError", "message": "boom"})
    profile = _make_profile_with_sample(db, tmp_path)
    profile.qwen_cloud_voice_id = "v1"
    db.commit()

    assert await profiles_service.delete_profile("p1", db) is True
    assert db.query(VoiceProfile).filter_by(id="p1").first() is None
