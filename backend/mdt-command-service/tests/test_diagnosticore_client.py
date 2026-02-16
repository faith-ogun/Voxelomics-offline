import json

from diagnosticore_client import DiagnosticoreClient


def test_file_mode_fetches_prediction(tmp_path, monkeypatch):
    csv_path = tmp_path / "case_predictions.csv"
    csv_path.write_text(
        (
            "case_submitter_id,split,true_label,pred_prob,pred_label,n_tiles\n"
            "TCGA-AAA-0001,test,1,0.81,1,500\n"
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "artifact_type": "tp53_locked_threshold_report_v1",
                "locked_threshold": {"value": 0.383},
                "model_card": {
                    "cohort": "TCGA-BRCA",
                    "intended_use": "Research only",
                    "limitations": ["External validation pending"],
                },
                "mdt_handoff_summary": {
                    "test_ece_10": 0.08,
                    "test_recall_ci_low": 0.54,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MDT_DIAGNOSTICORE_FETCH_MODE", "file")
    monkeypatch.setenv("MDT_DIAGNOSTICORE_ALLOW_FALLBACK", "true")
    monkeypatch.setenv("MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV", str(csv_path))
    monkeypatch.setenv("MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON", str(report_path))

    client = DiagnosticoreClient()
    pred = client.fetch_prediction("TCGA-AAA-0001")
    assert pred is not None
    assert pred.case_submitter_id == "TCGA-AAA-0001"
    assert abs(pred.tp53_probability - 0.81) < 1e-6
    assert abs((pred.threshold or 0.0) - 0.383) < 1e-6
    assert pred.model_card is not None
    assert pred.locked_threshold_report is not None


def test_file_mode_missing_artifacts_fallbacks_when_enabled(monkeypatch):
    monkeypatch.setenv("MDT_DIAGNOSTICORE_FETCH_MODE", "file")
    monkeypatch.setenv("MDT_DIAGNOSTICORE_ALLOW_FALLBACK", "true")
    monkeypatch.setenv("MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV", "/tmp/does-not-exist.csv")
    monkeypatch.setenv("MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON", "/tmp/does-not-exist.json")

    client = DiagnosticoreClient()
    pred = client.fetch_prediction("TCGA-AAA-0001")
    assert pred is None
