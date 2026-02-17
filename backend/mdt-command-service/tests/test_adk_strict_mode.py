import asyncio

import pytest

import agents
from agents import MDTCommandOrchestrator


def test_adk_stage_one_has_no_mock_fallback(monkeypatch):
    orchestrator = MDTCommandOrchestrator(execution_mode="local")
    record = orchestrator.start_case("MDT-001")

    # Force ADK branch in stage-one without requiring ADK init for this unit test.
    orchestrator.execution_mode = "adk_local"

    async def fail_stage_one_adk(_case_input):
        raise RuntimeError("synthetic adk failure")

    async def unexpected_local_path(*_args, **_kwargs):
        raise AssertionError("Local MedGemma path should not execute in adk_local mode.")

    monkeypatch.setattr(orchestrator, "_run_stage_one_adk", fail_stage_one_adk)
    monkeypatch.setattr(orchestrator, "_run_stage_one_with_medgemma_routing", unexpected_local_path)

    with pytest.raises(RuntimeError, match="synthetic adk failure"):
        asyncio.run(orchestrator._run_stage_one(record, record.input_payload))

    assert all(
        "local MedGemma routing" not in (trace.notes or "")
        for trace in record.traces
    )


def test_adk_local_mode_requires_adk_dependencies(monkeypatch):
    monkeypatch.setattr(agents, "ADK_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="adk_local requested"):
        MDTCommandOrchestrator(execution_mode="adk_local")


def test_unsupported_execution_mode_raises():
    with pytest.raises(ValueError, match="supports MDT_EXECUTION_MODE"):
        MDTCommandOrchestrator(execution_mode="invalid-mode")
