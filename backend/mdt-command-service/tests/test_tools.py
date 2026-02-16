import pytest

from models import EvidenceItem, RadiologySummary, validate_structured_output
from tools import (
    _LIVE_RETRIEVAL,
    extract_gene_symbols,
    get_case_input,
    parse_json_payload,
    reset_retrieval_cache_for_tests,
    search_literature_evidence,
)


def test_get_case_input_returns_expected_case():
    case = get_case_input("MDT-001")
    assert case.case_id == "MDT-001"
    assert case.patient_name == "Sarah Johnson"
    assert case.diagnosis == "Invasive Ductal Carcinoma"


def test_extract_gene_symbols_from_mutations():
    case = get_case_input("MDT-001")
    genes = extract_gene_symbols(case)
    assert "PIK3CA" in genes
    assert "TP53" in genes


def test_literature_returns_matches(monkeypatch):
    monkeypatch.setenv("MDT_RETRIEVAL_MODE", "local")
    case = get_case_input("MDT-001")
    genes = extract_gene_symbols(case)

    evidence = search_literature_evidence(case.diagnosis, genes, max_results=3)

    assert len(evidence) >= 1


def test_parse_json_payload_handles_fenced_markdown():
    raw = '```json\n{"ok": true, "value": 12}\n```'
    payload = parse_json_payload(raw)
    assert payload["ok"] is True
    assert payload["value"] == 12


def test_schema_validation_rejects_invalid_payload():
    invalid = {"disease_burden": "high", "action_items": []}  # missing findings
    with pytest.raises(Exception):
        validate_structured_output(RadiologySummary, invalid)


def test_local_mode_uses_local_evidence(monkeypatch):
    monkeypatch.setenv("MDT_RETRIEVAL_MODE", "local")
    case = get_case_input("MDT-001")
    genes = extract_gene_symbols(case)
    evidence = search_literature_evidence(case.diagnosis, genes, max_results=1)
    assert len(evidence) >= 1
    assert evidence[0].identifier and evidence[0].identifier.startswith("PMID:")


def test_pubmed_live_client_uses_cache(monkeypatch):
    reset_retrieval_cache_for_tests()
    calls = {"count": 0}

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(_self, url, params=None):  # noqa: ARG001
        calls["count"] += 1
        if "esearch.fcgi" in url:
            return DummyResponse({"esearchresult": {"idlist": ["123456"]}})
        return DummyResponse(
            {
                "result": {
                    "uids": ["123456"],
                    "123456": {
                        "title": "Cached Live Article",
                        "fulljournalname": "Journal of Oncology",
                        "pubdate": "2025 Jan",
                    },
                }
            }
        )

    monkeypatch.setattr("httpx.Client.get", fake_get)

    first = _LIVE_RETRIEVAL.fetch_pubmed_evidence(
        diagnosis="Invasive Ductal Carcinoma",
        genes=["PIK3CA"],
        max_results=1,
    )
    second = _LIVE_RETRIEVAL.fetch_pubmed_evidence(
        diagnosis="Invasive Ductal Carcinoma",
        genes=["PIK3CA"],
        max_results=1,
    )

    assert calls["count"] == 2  # esearch + esummary once total due cache
    assert first[0].identifier == "PMID:123456"
    assert second[0].title == "Cached Live Article"


def test_local_mode_returns_empty_when_cache_missing(monkeypatch):
    monkeypatch.setenv("MDT_RETRIEVAL_MODE", "local")
    reset_retrieval_cache_for_tests()

    case = get_case_input("MDT-001")
    genes = extract_gene_symbols(case)
    evidence = search_literature_evidence(case.diagnosis, genes, max_results=3)
    assert isinstance(evidence, list)


def test_non_local_mode_rejected(monkeypatch):
    monkeypatch.setenv("MDT_RETRIEVAL_MODE", "live")

    case = get_case_input("MDT-001")
    genes = extract_gene_symbols(case)
    with pytest.raises(ValueError, match="supports only MDT_RETRIEVAL_MODE=local"):
        search_literature_evidence(case.diagnosis, genes, max_results=1)
