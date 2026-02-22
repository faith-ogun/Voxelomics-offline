# Judge Quick Start (Download -> Open -> Run Demo)

This guide is for judges reviewing **Voxelomics Offline** with minimal setup.

## 1. Download

1. Open the latest GitHub Release:
   - `https://github.com/faith-ogun/Voxelomics-offline/releases`
2. Download your platform asset:
   - macOS (Apple Silicon): `Voxelomics Offline-0.0.0-arm64-mac.zip` (full desktop demo build with bundled DiagnostiCore pathology assets)
   - Windows/Linux: use the matching installer/zip asset when provided.

## 2. Open

1. Unzip the package.
2. Launch `Voxelomics Offline`.
3. If macOS blocks first launch:
   - Right-click app -> `Open` or double left click the app.

## 3. Run Demo

1. In **MDT Board Prep**, click `Load Demo Dictation`.
2. Click `Run MDT Pipeline`.
3. Wait for status to reach `PENDING_APPROVAL`.
4. Note: first run can take longer while local model weights initialize.

## 4. Review Core Features

1. **Clinical Review Workspace**
   - Risk synthesis
   - Recommended actions
   - Evidence links
2. **DiagnostiCore**
   - Interactive WSI viewer (zoom, pan, full-page) using bundled DeepZoom pathology assets
   - Fusion inference pipeline
   - TP53 probability output + validation metrics
3. **HITL Safety Gate**
   - Clinician checklist and sign-off controls
4. **Patient Cases**
   - Snapshot history
   - Load prior case state without rerunning pipeline

## 5. Offline Expectation

- This build is designed for local/offline workflow demonstration.
- No hosted cloud app is required for judging the desktop experience.
- DiagnostiCore pathology artifacts are bundled in the full desktop release so judges can review the WSI/TP53 workflow locally.

## 6. Known Scope

- Research/demo software only.
- Not for standalone diagnosis.
- Outputs require clinician oversight and confirmatory testing.
