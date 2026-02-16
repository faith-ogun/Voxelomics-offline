import asyncio

from agents import MDTCommandOrchestrator


def test_transcription_artifact_persists_before_pipeline_completion(monkeypatch):
    orchestrator = MDTCommandOrchestrator(execution_mode="mock")
    orchestrator.start_case("MDT-001")

    original_consensus = orchestrator._run_consensus_synthesizer

    async def slow_consensus(record, case_input, stage_one):
        await asyncio.sleep(0.25)
        return await original_consensus(record, case_input, stage_one)

    monkeypatch.setattr(orchestrator, "_run_consensus_synthesizer", slow_consensus)

    async def run_check():
        analyze_task = asyncio.create_task(orchestrator.analyze_case("MDT-001"))
        await asyncio.sleep(0.12)

        mid_run = orchestrator.get_case("MDT-001")
        assert mid_run.status.value == "analyzing"
        assert mid_run.artifacts.transcription is not None
        assert mid_run.artifacts.consensus is None

        await analyze_task

    asyncio.run(run_check())
