from fastapi.testclient import TestClient

from main import app
from models import DiagnosticorePrediction


client = TestClient(app)


def test_mdt_case_end_to_end_flow():
    start_resp = client.post("/mdt/start", json={"case_id": "MDT-001"})
    assert start_resp.status_code == 200
    start_body = start_resp.json()
    assert start_body["success"] is True
    assert start_body["status"] == "created"

    analyze_resp = client.post("/mdt/MDT-001/analyze")
    assert analyze_resp.status_code == 200
    analyze_body = analyze_resp.json()
    assert analyze_body["success"] is True
    assert analyze_body["status"] == "pending_approval"
    assert analyze_body["consensus"] is not None
    assert analyze_body["hitl_gate"] is not None

    draft_resp = client.get("/mdt/MDT-001/draft")
    assert draft_resp.status_code == 200
    draft_body = draft_resp.json()
    assert draft_body["success"] is True
    assert draft_body["artifacts"]["stage_one"] is not None
    assert draft_body["artifacts"]["soap_note"] is not None
    assert draft_body["artifacts"]["clinical_reasoning"] is not None
    assert draft_body["artifacts"]["transcription"] is not None

    status_resp = client.get("/mdt/MDT-001/status")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["status"] == "pending_approval"
    assert status_body["requires_approval"] is True

    approve_resp = client.post(
        "/mdt/MDT-001/approve",
        json={
            "decision": "approve",
            "clinician_name": "Dr. Ogundimu",
            "notes": "Approved after board review.",
        },
    )
    assert approve_resp.status_code == 200
    approve_body = approve_resp.json()
    assert approve_body["status"] == "approved"

    final_status_resp = client.get("/mdt/MDT-001/status")
    assert final_status_resp.status_code == 200
    final_status_body = final_status_resp.json()
    assert final_status_body["status"] == "approved"


def test_start_with_overrides_affects_generated_artifacts():
    custom_radio = "OVERRIDE RADIOLOGY: focal liver lesion now suspicious."
    custom_path = "OVERRIDE PATHOLOGY: poorly differentiated carcinoma."
    custom_gen = "OVERRIDE GENOMICS: BRCA2 pathogenic variant suspected."
    custom_transcript = "OVERRIDE TRANSCRIPT: board requests urgent second opinion."
    custom_audio_uri = "gs://demo-bucket/mdt-session-002.wav"

    start_resp = client.post(
        "/mdt/start",
        json={
            "case_id": "MDT-002",
            "overrides": {
                "radiology_notes": custom_radio,
                "pathology_notes": custom_path,
                "genomics_notes": custom_gen,
                "transcript_notes": custom_transcript,
                "transcript_audio_uri": custom_audio_uri,
            },
        },
    )
    assert start_resp.status_code == 200
    assert "overrides" in start_resp.json()["message"].lower()

    analyze_resp = client.post("/mdt/MDT-002/analyze")
    assert analyze_resp.status_code == 200

    draft_resp = client.get("/mdt/MDT-002/draft")
    assert draft_resp.status_code == 200
    draft = draft_resp.json()

    findings = draft["artifacts"]["stage_one"]["radiology"]["findings"]
    path_dx = draft["artifacts"]["stage_one"]["pathology"]["diagnosis"]
    genomics_interp = draft["artifacts"]["stage_one"]["genomics"]["interpretation"]
    transcript = draft["artifacts"]["transcription"]["transcript"]
    audio_uri = draft["artifacts"]["transcription"]["notes"]

    assert "OVERRIDE RADIOLOGY" in findings
    assert "OVERRIDE PATHOLOGY" in path_dx
    assert "OVERRIDE GENOMICS" in genomics_interp
    assert "OVERRIDE TRANSCRIPT" in transcript
    assert "Using transcript text supplied by the client UI" in audio_uri


def test_diagnosticore_override_is_integrated_with_safety_guardrails():
    start_resp = client.post(
        "/mdt/start",
        json={
            "case_id": "MDT-003",
            "overrides": {
                "diagnosticore": {
                    "source_service": "diagnosticore-service",
                    "target": "tp53_mutation",
                    "case_submitter_id": "TCGA-FAKE-0001",
                    "tp53_probability": 0.83,
                    "threshold": 0.383,
                    "predicted_label": "tp53_mutated",
                    "uncertainty_flags": ["borderline tissue quality"],
                    "model_version": "cnn_tp53_200",
                    "data_version": "tcga_brca_primary_200",
                    "is_confirmed_genomic_test": False,
                    "model_card": {
                        "cohort": "TCGA-BRCA primary-slide cohort",
                        "intended_use": "Research-only TP53 risk support, not standalone diagnosis.",
                        "limitations": [
                            "Single-cohort retrospective evaluation",
                            "External validation pending",
                        ],
                    },
                    "locked_threshold_report": {
                        "artifact_type": "tp53_locked_threshold_report_v1",
                        "selected_threshold": 0.383,
                        "test_ece_10": 0.072,
                        "test_recall_ci_low": 0.55,
                    },
                    "evidence": [
                        {
                            "slide_file_id": "slide-1",
                            "tile_path": "/tmp/tile_1.png",
                            "score": 0.92,
                            "note": "high-risk morphology tile",
                        }
                    ],
                }
            },
        },
    )
    assert start_resp.status_code == 200

    analyze_resp = client.post("/mdt/MDT-003/analyze")
    assert analyze_resp.status_code == 200

    draft_resp = client.get("/mdt/MDT-003/draft")
    assert draft_resp.status_code == 200
    draft = draft_resp.json()

    genomics_interp = draft["artifacts"]["stage_one"]["genomics"]["interpretation"]
    assert "DiagnostiCore inferred TP53 signal" in genomics_interp
    assert "not a confirmed molecular assay" in genomics_interp

    red_flags = draft["artifacts"]["consensus"]["red_flags"]
    assert isinstance(red_flags, list)

    checklist = draft["artifacts"]["hitl_gate"]["approval_checklist"]
    assert any("confirmatory sequencing/assay plan" in c for c in checklist)
    assert any("model card intended use" in c for c in checklist)

    safety_flags = draft["artifacts"]["hitl_gate"]["safety_flags"]
    assert any("requires confirmatory molecular testing" in f for f in safety_flags)
    assert not any("model card missing" in f.lower() for f in safety_flags)
    assert not any("locked-threshold report missing" in f.lower() for f in safety_flags)

    clinical_reasoning = draft["artifacts"]["clinical_reasoning"]
    assert clinical_reasoning["generation_mode"] == "local_medgemma"
    assert any("confirmatory molecular assay" in a.lower() for a in clinical_reasoning["confirmatory_actions"])


def test_diagnosticore_autofetch_applies_when_payload_not_manually_provided(monkeypatch):
    from main import orchestrator_agent

    def fake_fetch(_case_key: str):
        return DiagnosticorePrediction(
            source_service="diagnosticore-service",
            target="tp53_mutation",
            case_submitter_id="TCGA-A1-A0SP",
            tp53_probability=0.77,
            threshold=0.383,
            predicted_label="tp53_mutated",
            uncertainty_flags=["calibration_ece_high=0.190"],
            model_version="cnn_tp53_200",
            data_version="tcga_brca_primary_200",
            is_confirmed_genomic_test=False,
            model_card={
                "cohort": "TCGA-BRCA primary-slide cohort",
                "intended_use": "Research-only TP53 risk support, not standalone diagnosis.",
                "limitations": ["External validation pending"],
            },
            locked_threshold_report={
                "artifact_type": "tp53_locked_threshold_report_v1",
                "selected_threshold": 0.383,
                "test_ece_10": 0.190,
                "test_recall_ci_low": 0.538,
            },
            locked_threshold_report_uri="/tmp/tp53_clinical_report.json",
        )

    monkeypatch.setattr(orchestrator_agent.diagnosticore_client, "fetch_prediction", fake_fetch)

    start_resp = client.post("/mdt/start", json={"case_id": "MDT-001"})
    assert start_resp.status_code == 200

    analyze_resp = client.post("/mdt/MDT-001/analyze")
    assert analyze_resp.status_code == 200

    draft_resp = client.get("/mdt/MDT-001/draft")
    assert draft_resp.status_code == 200
    draft = draft_resp.json()

    genomics_interp = draft["artifacts"]["stage_one"]["genomics"]["interpretation"]
    assert "DiagnostiCore inferred TP53 signal" in genomics_interp
    assert "not a confirmed molecular assay" in genomics_interp
