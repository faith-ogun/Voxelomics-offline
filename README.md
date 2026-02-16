# Voxelomics Offline

Offline-first clinical workflow demo for the **Kaggle MedGemma Impact Challenge**.

Voxelomics focuses on MDT (multidisciplinary tumor board) preparation with local-first execution, clinician approval gates, and transparent pathology support.

![Vite](https://img.shields.io/badge/Vite-6.x-646CFF?logo=vite&logoColor=white)
![React](https://img.shields.io/badge/React-18.x-20232A?logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Electron](https://img.shields.io/badge/Electron-Desktop-47848F?logo=electron&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Local%20Storage-003B57?logo=sqlite&logoColor=white)
![MedASR](https://img.shields.io/badge/HAI--DEF-MedASR-1f6feb)
![MedGemma](https://img.shields.io/badge/HAI--DEF-MedGemma-1f6feb)
![Path%20Foundation](https://img.shields.io/badge/HAI--DEF-Path%20Foundation-1f6feb)

## HAI-DEF model usage

- **MedASR**: local speech-to-text for clinical dictation (frontend worker path).
- **MedGemma**: local multimodal synthesis and recommendation drafting in MDT workflow.
- **Path Foundation + TP53 head**: pathology embedding workflow + case-level risk support artifacts.

## What this repo demonstrates

- Offline-first MDT orchestration (`backend/mdt-command-service`).
- Whole-slide pathology viewer + DiagnostiCore integration in UI.
- HITL gate before recommendation lock.
- Local patient case snapshots for review without rerunning all stages.

## Repository structure

- `App.tsx`, `components/`: frontend app (Vite + React + TypeScript)
- `backend/mdt-command-service/`: MDT orchestration backend (FastAPI)
- `backend/diagnosticore-service/`: pathology data/training utilities and artifacts
- `public/workers/medasr-onnx.worker.js`: browser MedASR worker
- `electron/`: desktop wrapper and packaging config
- `scripts/setup_medasr_web_assets.sh`: copies ONNX Runtime Web assets into `public/vendor/`

## Prerequisites

- Node.js 20+ and npm
- Python 3.12+ (3.14 works for MDT backend)
- Optional: Ollama (only if using ADK local mode)

## Quick start (local web app)

### 1) Install frontend dependencies
```bash
npm install
```

### 2) Configure and run MDT backend
```bash
cd backend/mdt-command-service
python3 -m pip install -r requirements.txt
cp .env.example .env
python3 -m uvicorn main:app --host 127.0.0.1 --port 8084
```

### 3) Start frontend
```bash
# from repo root
npm run dev
```

Open `http://127.0.0.1:5173`.

## Frontend MedASR (ONNX worker) setup

If `public/models/medasr.onnx` and ONNX runtime assets are not present, run:

```bash
cd backend/mdt-command-service
python3 scripts/export_medasr_onnx.py \
  --model-dir ../../models/medasr \
  --output-dir ../../public/models
```

```bash
# from repo root
npm install onnxruntime-web
bash scripts/setup_medasr_web_assets.sh
```

Optional `.env.local` overrides:
```bash
VITE_MEDASR_FRONTEND_ONNX=true
VITE_MEDASR_ONNX_URL=./models/medasr.onnx
VITE_MEDASR_VOCAB_URL=./models/medasr_vocab.json
VITE_MEDASR_ORT_URL=./vendor/onnxruntime-web/ort.min.js
VITE_MEDASR_EXECUTION_PROVIDERS=webgpu,wasm
VITE_MEDASR_CHUNK_SECONDS=10
VITE_MEDASR_OVERLAP_SECONDS=1
VITE_MDT_ANALYZE_TIMEOUT_MS=900000
```

## Desktop app (Electron)

### Dev mode
```bash
npm run desktop:dev
```

### Build installers/artifacts
```bash
npm run desktop:dist
```

Targets are configured in `package.json` (`zip`/`nsis`/`AppImage` etc.).

## Judge demo path (suggested)

1. Open app and enter `MDT Board Prep`.
2. Use `Load Demo Dictation` or upload audio.
3. Run MDT pipeline.
4. Review:
   - Clinical Review Workspace (risks/actions/evidence/uncertainty)
   - DiagnostiCore tab (WSI + pipeline + TP53 panel)
   - HITL approval checklist
5. Open `Patient Cases` and load a saved snapshot.

## Runtime behavior notes

- Pipeline can run from transcript text (backend MedASR not required for this path).
- Timeout guards are enabled to avoid infinite hangs:
  - `MDT_MEDGEMMA_TIMEOUT_SECONDS`
  - `MDT_AGENT_CALL_TIMEOUT_SECONDS`
  - `MDT_AGENT_CALL_TIMEOUT_CAP_SECONDS`
  - `MDT_ANALYZE_TIMEOUT_SECONDS`
- DiagnostiCore handoff defaults to local file-mode artifacts.

## Reproducibility pointers

- Path Foundation benchmark comparison artifact:
  - `backend/diagnosticore-service/output/pathfoundation_tp53_200/comparison_vs_cnn.json`
- Path Foundation clinical report artifact:
  - `backend/diagnosticore-service/output/pathfoundation_tp53_200/tp53_clinical_report_pathfoundation_platt.json`
- MDT service docs:
  - `backend/mdt-command-service/README.md`

## Safety and scope

- This project is **research/demo software**.
- It does not provide standalone diagnosis.
- AI-inferred outputs require clinician review and confirmatory testing.

## License

See `LICENSE`.
