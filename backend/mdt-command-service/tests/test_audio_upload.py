from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_upload_mdt_audio_returns_gcs_uri(monkeypatch):
    def fake_upload(file_bytes: bytes, filename: str, content_type: str) -> str:
        assert file_bytes
        assert filename == "sample.wav"
        assert content_type == "audio/wav"
        return "file:///tmp/mdt-audio/fake.wav"

    monkeypatch.setattr(main, "_upload_audio", fake_upload)

    response = client.post(
        "/mdt/audio/upload",
        files={"file": ("sample.wav", b"abc123", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["gcs_uri"].startswith("file:///")
    assert body["bytes_uploaded"] == 6


def test_upload_mdt_audio_rejects_empty_file():
    response = client.post(
        "/mdt/audio/upload",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_mdt_audio_rejects_unsupported_format():
    response = client.post(
        "/mdt/audio/upload",
        files={"file": ("sample.webm", b"abc123", "audio/webm")},
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()
