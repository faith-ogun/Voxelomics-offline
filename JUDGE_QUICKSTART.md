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

## 3. Choose a Review Path (Recommended)

Judges can review Voxelomics in either of two ways:

1. **Quick review (fastest, recommended):**
   - Open a saved case snapshot in **Patient Cases** to inspect outputs immediately without waiting for a full rerun.
2. **Full end-to-end run (transparent demo path):**
   - Run the full pipeline from demo dictation and observe staged execution.

Note: on some machines, a full local pipeline run may take approximately **3-5 minutes** (especially first run / cold start).

## 4. Run Demo (Full End-to-End Path)

1. In **MDT Board Prep**, click `Load Demo Dictation`.
2. Click `Run MDT Pipeline`.
3. Wait for status to reach `PENDING_APPROVAL`.
4. First run can take longer while local model weights initialize.
5. If you prefer a faster review, use **Patient Cases -> Snapshot history** and open a saved case state.

## 5. Review Core Features

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

## 6. Offline Expectation

- This build is designed for local/offline workflow demonstration.
- No hosted cloud app is required for judging the desktop experience.
- DiagnostiCore pathology artifacts are bundled in the full desktop release so judges can review the WSI/TP53 workflow locally.

## 7. Known Scope

- Research/demo software only.
- Not for standalone diagnosis.
- Outputs require clinician oversight and confirmatory testing.
