# Cleanup Audit for GitHub Submission (Voxelomics Offline)

Date: 2026-02-16
Goal: remove stale/confusing files before public submission.

## A) Safe to delete now (very low risk)

These are generated artifacts, caches, local runtime residue, or clearly unused in current app flow.

### Generated / cache / local runtime state
- `.pytest_cache/`
- `node_modules/`
- `dist/`
- `audio/`
- `local_data/`
- `evidence_cache/`
- `mdt_cases.sqlite3`
- `backend/mdt-command-service/.venv/`
- `backend/mdt-command-service/.pytest_cache/`
- `backend/mdt-command-service/__pycache__/`
- `backend/mdt-command-service/local_data/`
- `backend/diagnosticore-service/.venv-pf/`
- `backend/diagnosticore-service/__pycache__/`

### OS clutter
- all `.DS_Store`

### Dead frontend files (not used by App)
- `components/AuthFlow.tsx` (not routed in `App.tsx`, references missing `contexts/AuthContext`)
- `components/ui/radial-orbital-timeline.tsx` (not imported anywhere)
- `components/ui/badge.tsx` (only used by unused radial timeline)
- `components/ui/button.tsx` (only used by unused radial timeline)
- `components/ui/card.tsx` (only used by unused radial timeline)
- `utils/googleCalendar.ts` (no imports found)

## B) Strongly consider deleting from submission repo (if you want a clean judge story)

### Planning/internal docs (not needed for runtime)
- `plan/` (contains internal notes + old Oncydra references)

### Notebooks (optional research history, not needed for demo runtime)
- `notebooks/`

### A/B harness scripts (evaluation utilities, not used in serving path)
- `backend/mdt-command-service/build_blinded_ab_packet.py`
- `backend/mdt-command-service/evaluate_mdt_ab_rubric.py`
- `backend/mdt-command-service/run_mdt_ab_harness.py`
- `backend/mdt-command-service/summarize_blinded_ab_ratings.py`
- `backend/mdt-command-service/tests/test_ab_harness.py`

## C) Keep (important for demo/runtime)

- `App.tsx`, `components/Home.tsx`, `components/MDTCommand.tsx`, `components/Technology.tsx`, etc.
- `public/models/medasr.onnx`
- `public/models/medasr_vocab.json`
- `public/workers/medasr-onnx.worker.js`
- `public/vendor/onnxruntime-web/*`
- `public/vendor/openseadragon/openseadragon.min.js`
- `backend/mdt-command-service/main.py`
- `backend/mdt-command-service/agents.py`
- `backend/mdt-command-service/models.py`
- `backend/mdt-command-service/tools.py`
- `backend/mdt-command-service/diagnosticore_client.py`
- `backend/diagnosticore-service/output/pathfoundation_tp53_200/*` (if used for live local handoff)
- `backend/diagnosticore-service/output/deepzoom/*` (if keeping WSI DeepZoom demo)
- `backend/diagnosticore-service/gdc_wsi/*` (if needed by your local pathology flow)

## D) Optional: shrink large diagnosticore output footprint

Current largest folders:
- `backend/diagnosticore-service/output/pathfoundation_tp53_200` (~210M)
- `backend/diagnosticore-service/output/deepzoom` (~120M)
- `backend/diagnosticore-service/output/cnn_tp53_200` (~66M)
- many baseline seed folders (~16-20M each)

If you want a lighter repo, you can keep only artifacts required by `diagnosticore_client.py` defaults and remove older baseline seed outputs.

## E) Suggested cleanup commands (manual run)

```bash
# from repo root
rm -rf .pytest_cache node_modules dist audio local_data evidence_cache mdt_cases.sqlite3
rm -rf backend/mdt-command-service/.venv backend/mdt-command-service/.pytest_cache backend/mdt-command-service/__pycache__ backend/mdt-command-service/local_data
rm -rf backend/diagnosticore-service/.venv-pf backend/diagnosticore-service/__pycache__

# remove dead frontend files
rm -f components/AuthFlow.tsx
rm -f components/ui/radial-orbital-timeline.tsx components/ui/badge.tsx components/ui/button.tsx components/ui/card.tsx
rm -f utils/googleCalendar.ts

# optional
rm -rf plan notebooks

# remove mac metadata
find . -name '.DS_Store' -delete
```

## F) Pre-commit verification after cleanup

```bash
npm install
npm run build
python3 -m uvicorn backend/mdt-command-service/main:app --host 127.0.0.1 --port 8084
```

Then run one full demo case end-to-end.
