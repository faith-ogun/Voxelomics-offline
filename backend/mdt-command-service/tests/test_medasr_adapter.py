import asyncio
import base64
import json

import pytest

import agents
from agents import MedASRAdapter


class _FakeResponse:
    def __init__(self, payload=None, text=None, json_exc: Exception | None = None):
        self.status_code = 200
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload if payload is not None else {})
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload


def test_vertex_mode_reads_local_audio_and_calls_raw_predict(monkeypatch, tmp_path):
    monkeypatch.setenv("MDT_MEDASR_MODE", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "voxelomics")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("MDT_MEDASR_ENDPOINT_ID", "123456789")
    monkeypatch.setenv("MDT_MEDASR_ALLOW_MOCK_FALLBACK", "false")

    audio_bytes = b"RIFFFAKEAUDIO"
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(audio_bytes)

    captured = {}

    class _FakeEndpoint:
        def __init__(self, endpoint_id):
            captured["endpoint_id"] = endpoint_id

        def raw_predict(self, body, headers, use_dedicated_endpoint=False, timeout=None):
            captured["body"] = body
            captured["headers"] = headers
            captured["use_dedicated_endpoint"] = use_dedicated_endpoint
            captured["timeout"] = timeout
            return _FakeResponse({"text": "live transcript", "wer": 0.08})

    class _FakeAIPlatform:
        Endpoint = _FakeEndpoint

        @staticmethod
        def init(project, location):
            captured["init"] = (project, location)

    monkeypatch.setattr(agents, "aiplatform", _FakeAIPlatform)

    adapter = MedASRAdapter()
    result = asyncio.run(adapter.transcribe(transcript_text="fallback", audio_uri=str(audio_file)))

    assert result.transcript == "live transcript"
    assert result.wer_estimate == 0.08
    assert captured["init"] == ("voxelomics", "us-central1")
    assert captured["endpoint_id"] == "123456789"
    assert captured["use_dedicated_endpoint"] is True
    payload = json.loads(captured["body"].decode("utf-8"))
    assert base64.b64decode(payload["file"]) == audio_bytes
    assert captured["headers"]["Content-Type"] == "application/json"


def test_vertex_mode_requires_audio_uri_when_fallback_disabled(monkeypatch):
    monkeypatch.setenv("MDT_MEDASR_MODE", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "voxelomics")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("MDT_MEDASR_ENDPOINT_ID", "123456789")
    monkeypatch.setenv("MDT_MEDASR_ALLOW_MOCK_FALLBACK", "false")

    adapter = MedASRAdapter()
    with pytest.raises(ValueError, match="audio_uri is required"):
        asyncio.run(adapter.transcribe(transcript_text="hello board", audio_uri=None))


def test_vertex_mode_can_fallback_to_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("MDT_MEDASR_MODE", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "voxelomics")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("MDT_MEDASR_ENDPOINT_ID", "123456789")
    monkeypatch.setenv("MDT_MEDASR_ALLOW_MOCK_FALLBACK", "true")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFFFAKEAUDIO")

    class _FailingEndpoint:
        def __init__(self, _endpoint_id):
            pass

        def raw_predict(self, *_args, **_kwargs):
            raise RuntimeError("forced live failure")

    class _FakeAIPlatform:
        Endpoint = _FailingEndpoint

        @staticmethod
        def init(project, location):  # noqa: ARG004
            return None

    monkeypatch.setattr(agents, "aiplatform", _FakeAIPlatform)

    adapter = MedASRAdapter()
    result = asyncio.run(
        adapter.transcribe(
            transcript_text="board requested urgent review",
            audio_uri=str(audio_file),
        )
    )
    assert "board requested urgent review" in result.transcript
    assert result.notes and "fallback" in result.notes.lower()


def test_vertex_mode_extracts_transcript_from_nested_predictions(monkeypatch, tmp_path):
    monkeypatch.setenv("MDT_MEDASR_MODE", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "voxelomics")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("MDT_MEDASR_ENDPOINT_ID", "123456789")
    monkeypatch.setenv("MDT_MEDASR_ALLOW_MOCK_FALLBACK", "false")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFFFAKEAUDIO")

    class _FakeEndpoint:
        def __init__(self, _endpoint_id):
            pass

        def raw_predict(self, *_args, **_kwargs):
            return _FakeResponse(
                {
                    "predictions": [
                        {
                            "response": {"transcript": "nested transcript"},
                            "wer_estimate": "0.07",
                        }
                    ]
                }
            )

    class _FakeAIPlatform:
        Endpoint = _FakeEndpoint

        @staticmethod
        def init(project, location):  # noqa: ARG004
            return None

    monkeypatch.setattr(agents, "aiplatform", _FakeAIPlatform)

    adapter = MedASRAdapter()
    result = asyncio.run(adapter.transcribe(transcript_text="fallback", audio_uri=str(audio_file)))
    assert result.transcript == "nested transcript"
    assert result.wer_estimate == 0.07


def test_vertex_mode_accepts_plain_text_response_body(monkeypatch, tmp_path):
    monkeypatch.setenv("MDT_MEDASR_MODE", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "voxelomics")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("MDT_MEDASR_ENDPOINT_ID", "123456789")
    monkeypatch.setenv("MDT_MEDASR_ALLOW_MOCK_FALLBACK", "false")

    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFFFAKEAUDIO")

    class _FakeEndpoint:
        def __init__(self, _endpoint_id):
            pass

        def raw_predict(self, *_args, **_kwargs):
            return _FakeResponse(
                payload=None,
                text="Chair: recommend neoadjuvant anti-HER2 regimen.",
                json_exc=ValueError("not json"),
            )

    class _FakeAIPlatform:
        Endpoint = _FakeEndpoint

        @staticmethod
        def init(project, location):  # noqa: ARG004
            return None

    monkeypatch.setattr(agents, "aiplatform", _FakeAIPlatform)

    adapter = MedASRAdapter()
    result = asyncio.run(adapter.transcribe(transcript_text="fallback", audio_uri=str(audio_file)))
    assert "anti-HER2 regimen" in result.transcript


def test_local_mode_can_fallback_to_transcript_text_without_audio(monkeypatch):
    monkeypatch.setenv("MDT_MEDASR_MODE", "local")
    monkeypatch.setenv("MDT_MEDASR_LOCAL_ALLOW_TEXT_FALLBACK", "true")
    monkeypatch.setenv("MDT_MEDASR_ALLOW_MOCK_FALLBACK", "false")

    adapter = MedASRAdapter()
    result = asyncio.run(
        adapter.transcribe(
            transcript_text="Board discussion transcript available in text only.",
            audio_uri=None,
        )
    )

    assert "text only" in result.transcript
    assert result.notes and "local medasr" in result.notes.lower()
