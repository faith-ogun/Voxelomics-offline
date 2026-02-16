import asyncio

import pytest

from agents import ADK_AVAILABLE, MDTCommandOrchestrator, vertexai
from models import AgentRunStatus


def test_adk_stage_one_has_no_mock_fallback(monkeypatch):
    orchestrator = MDTCommandOrchestrator(execution_mode="mock")
    record = orchestrator.start_case("MDT-001")

    # Force ADK branch in stage-one without requiring ADK init for this unit test.
    orchestrator.execution_mode = "adk"
    orchestrator.adk_stage_one_max_retries = 2

    async def fail_stage_one_adk(_case_input):
        raise RuntimeError("synthetic adk failure")

    async def unexpected_mock_path(*_args, **_kwargs):
        raise AssertionError("Mock fallback path should not execute in adk mode.")

    monkeypatch.setattr(orchestrator, "_run_stage_one_adk", fail_stage_one_adk)
    monkeypatch.setattr(orchestrator, "_run_radiology_synthesizer", unexpected_mock_path)

    with pytest.raises(RuntimeError, match="mock fallback disabled"):
        asyncio.run(orchestrator._run_stage_one(record, record.input_payload))

    failed_parallel_attempts = [
        t
        for t in record.traces
        if t.agent == "ParallelFanOut" and t.status == AgentRunStatus.FAILED
    ]
    assert len(failed_parallel_attempts) == 2


@pytest.mark.skipif(
    not ADK_AVAILABLE or vertexai is None,
    reason="ADK or vertexai runtime unavailable in this environment.",
)
def test_adk_mode_requires_google_cloud_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        MDTCommandOrchestrator(execution_mode="adk")


def test_unsupported_execution_mode_raises():
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        MDTCommandOrchestrator(execution_mode="invalid-mode")
