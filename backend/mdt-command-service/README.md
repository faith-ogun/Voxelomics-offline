# Voxelomics MDT Command Service (Offline)

MDT Command is the local backend orchestrator for Voxelomics Offline tumor board workflow.

It is intentionally offline-at-runtime.

## Runtime modes

- `MDT_EXECUTION_MODE=local` (default): in-process orchestrator.
- `MDT_EXECUTION_MODE=adk_local`: Google ADK orchestration with local model backend (LiteLLM + Ollama).

In both modes, this offline profile enforces:
- `MDT_MEDASR_MODE=local`
- `MDT_AUDIO_UPLOAD_BACKEND=local`
- `MDT_CASE_STORE_BACKEND=sqlite`
- `MDT_RETRIEVAL_MODE=local`
- `MDT_MODEL_ROUTER_MODE=medgemma_only`

## 10-agent architecture

1. `MDTOrchestrator`
2. `RadiologySynthesizer`
3. `PathologySynthesizer`
4. `GenomicsSynthesizer`
5. `LiteratureAgent`
6. `TranscriptionAgent`
7. `ConsensusSynthesizer`
8. `SOAPGenerator`
9. `HITLGatekeeper`
10. `ClinicalReasoner`

The pipeline runs:
- Stage 1: parallel fan-out (specialists + literature), with concurrent transcription.
- Stage 2: sequential synthesis (consensus, SOAP, clinical reasoning).
- Stage 3: mandatory HITL safety gate before approval.

## API endpoints

Core:
- `POST /mdt/start`
- `POST /mdt/{case_id}/analyze`
- `GET /mdt/{case_id}/draft`
- `POST /mdt/{case_id}/approve`
- `GET /mdt/{case_id}/status`
- `POST /mdt/audio/upload`

DiagnostiCore viewer support:
- `GET /mdt/{case_id}/diagnosticore/tile-preview`
- `GET /mdt/{case_id}/diagnosticore/deepzoom.dzi`
- `GET /mdt/{case_id}/diagnosticore/deepzoom_tiles/{level}/{tile_name}`

Evidence cache:
- `GET /mdt/evidence/status`
- `POST /mdt/evidence/sync`

Patient case history:
- `GET /mdt/patients/{patient_id}/cases`
- `GET /mdt/cases/history/{snapshot_id}`
- `DELETE /mdt/cases/history/{snapshot_id}`

Health:
- `GET /health`

## DiagnostiCore handoff

Two supported paths:

1. Auto fetch (recommended):
- On `start`, the orchestrator can load DiagnostiCore payload from local file mode or HTTP mode.
- Fallback behavior is configurable (`MDT_DIAGNOSTICORE_ALLOW_FALLBACK=true`).

2. Manual override:
- Pass `overrides.diagnosticore` in `POST /mdt/start`.

Safety behavior:
- AI-inferred genomic signals are explicitly marked as not confirmatory molecular assays.
- HITL checklist enforces confirmatory testing language.
- Missing model-card / threshold-report artifacts are surfaced as safety flags.

### Auto-fetch env example

```bash
MDT_DIAGNOSTICORE_FETCH_MODE=file
MDT_DIAGNOSTICORE_ALLOW_FALLBACK=true
MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV=../diagnosticore-service/output/pathfoundation_tp53_200/case_predictions_calibrated_platt.csv
MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON=../diagnosticore-service/output/pathfoundation_tp53_200/tp53_clinical_report_pathfoundation_platt.json
MDT_DIAGNOSTICORE_DEEPZOOM_DIR=../diagnosticore-service/output/deepzoom
```

## Model routing (local MedGemma)

Key settings:
- `MDT_MODEL_ROUTER_MODE=medgemma_only`
- `MDT_MEDGEMMA_LOCAL_MODEL_ID=../../models/medgemma-4b-it`
- `MDT_MEDGEMMA_LOCAL_FILES_ONLY=true`
- `MDT_MEDGEMMA_TEMPERATURE=0.0`
- `MDT_MEDGEMMA_TIMEOUT_SECONDS=35`
- `MDT_AGENT_CALL_TIMEOUT_SECONDS=50`
- `MDT_AGENT_CALL_TIMEOUT_CAP_SECONDS=55`
- `MDT_ANALYZE_TIMEOUT_SECONDS=420`

## Runbook

### 1) Install dependencies

```bash
cd backend/mdt-command-service
python3 -m pip install -r requirements.txt
```

### 2) Start in local mode

```bash
cp .env.example .env
python3 -m uvicorn main:app --host 127.0.0.1 --port 8084
```

### 3) Optional: start in adk_local mode

```bash
cp .env.adk-local.example .env
# ensure Ollama is running and model is pulled
# ollama pull qwen2.5:7b-instruct
python3 -m uvicorn main:app --host 127.0.0.1 --port 8084
```

### 4) Smoke test

```bash
curl -X POST http://127.0.0.1:8084/mdt/start -H "Content-Type: application/json" -d '{"case_id":"MDT-001"}'
curl -X POST http://127.0.0.1:8084/mdt/MDT-001/analyze
curl http://127.0.0.1:8084/mdt/MDT-001/draft
curl -X POST http://127.0.0.1:8084/mdt/MDT-001/approve -H "Content-Type: application/json" -d '{"decision":"approve","clinician_name":"Dr. Faith"}'
curl http://127.0.0.1:8084/mdt/MDT-001/status
```

### Transcript-only pipeline path (desktop-recommended)

```bash
curl -X POST http://127.0.0.1:8084/mdt/start \
  -H "Content-Type: application/json" \
  -d '{"case_id":"MDT-001","overrides":{"transcript_notes":"<transcript text here>"}}'
```

## Local data

- Mock cases: `mock_db/cases.json`
- Mock literature baseline: `mock_db/literature_evidence.json`
- Case persistence: local SQLite (configured in `.env`)

## Tests

Run:

```bash
pytest -q
```

Current test files in this branch:
- `tests/test_audio_upload.py`
- `tests/test_diagnosticore_client.py`
- `tests/test_integration.py`
- `tests/test_tools.py`
- `tests/test_adk_strict_mode.py`

If ADK dependencies are not installed, ADK-specific paths may be skipped by test guards.

## Competition alignment

- Effective use of HAI-DEF: MedASR + MedGemma + Path Foundation handoff.
- Agentic workflow: staged specialist orchestration + HITL gate.
- Product feasibility: local-first runtime with explicit safety and provenance fields.

## Scope and safety

- Research/demo software only.
- Not standalone diagnosis.
- All generated outputs require clinician review and confirmatory testing when AI-inferred genomic signals are present.
