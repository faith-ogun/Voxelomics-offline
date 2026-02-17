# DiagnostiCore Service (Voxelomics Offline)

DiagnostiCore is the pathology support module used by Voxelomics Offline.

In this repository, DiagnostiCore provides:
- TP53 pathology-risk artifacts used by MDT handoff.
- Path Foundation experiment scripts (embedding extraction + TP53 head training/comparison).
- DeepZoom pyramid generation for local whole-slide viewing in the MDT UI.

This offline repository is intentionally scoped to runtime demo and reproducible evidence artifacts.

## What is included here

### Runtime utility
- `build_deepzoom_pyramid.py`

### Path Foundation experiment scripts
- `path_foundation_experiment/extract_pathfoundation_embeddings.py`
- `path_foundation_experiment/train_tp53_head_from_embeddings.py`
- `path_foundation_experiment/compare_vs_cnn_baseline.py`

### Tracked benchmark/report artifacts (for submission evidence)
- `output/pathfoundation_tp53_200/comparison_vs_cnn.json`
- `output/pathfoundation_tp53_200/tp53_clinical_report_pathfoundation_platt.json`
- `output/cnn_tp53_200/tp53_clinical_report_calibrated.json`

## What is intentionally not included here

Legacy data-intake and multi-stage cohort-building scripts from the online repository variant are not part of this offline branch. If you need full end-to-end dataset construction from raw GDC sources, use the online pipeline repository.

## Setup

```bash
cd backend/diagnosticore-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Path Foundation experiments:

```bash
pip install -r path_foundation_experiment/requirements.txt
```

Note:
- Path Foundation Keras loading depends on TensorFlow.
- Use Python 3.11/3.12 for the Path Foundation experiment environment.

## Path Foundation experiment runbook

This experiment expects an existing tile manifest with `output_png`, `split`, `label`, and case identifiers.

### 1) Extract embeddings

```bash
python3 path_foundation_experiment/extract_pathfoundation_embeddings.py \
  --tile-manifest output/tcga_brca_tp53_tiles_manifest_full_200.csv \
  --output-dir output/pathfoundation_tp53_200 \
  --model-id google/path-foundation \
  --batch-size 32
```

Optional flags:
- `--local-files-only`
- `--hf-token <TOKEN>`
- `--max-tiles <N>`
- `--split-filter train,val,test`

### 2) Train TP53 head from embeddings

```bash
python3 path_foundation_experiment/train_tp53_head_from_embeddings.py \
  --embeddings-npy output/pathfoundation_tp53_200/embeddings.npy \
  --embedding-rows-csv output/pathfoundation_tp53_200/embedding_rows.csv \
  --output-dir output/pathfoundation_tp53_200 \
  --decision-threshold 0.5
```

### 3) Compare against CNN baseline

```bash
python3 path_foundation_experiment/compare_vs_cnn_baseline.py \
  --cnn-case-predictions output/cnn_tp53_200/case_predictions.csv \
  --pathfoundation-case-predictions output/pathfoundation_tp53_200/case_predictions.csv \
  --output-json output/pathfoundation_tp53_200/comparison_vs_cnn.json
```

## DeepZoom generation for MDT UI

Generate a DeepZoom pyramid for a local WSI file:

```bash
python3 build_deepzoom_pyramid.py \
  --wsi-path data/gdc_wsi/<file_id>/<slide>.svs \
  --out-dir output/deepzoom \
  --slug <case_submitter_id>
```

This produces:
- `output/deepzoom/<slug>.dzi`
- `output/deepzoom/<slug>_files/`

## MDT integration pointers

`mdt-command-service` can auto-load DiagnostiCore artifacts from local files using environment variables:

```bash
MDT_DIAGNOSTICORE_FETCH_MODE=file
MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV=../diagnosticore-service/output/pathfoundation_tp53_200/case_predictions_calibrated_platt.csv
MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON=../diagnosticore-service/output/pathfoundation_tp53_200/tp53_clinical_report_pathfoundation_platt.json
MDT_DIAGNOSTICORE_DEEPZOOM_DIR=../diagnosticore-service/output/deepzoom
```

## Validation and scope notes

- All TP53 outputs are research-only decision support artifacts.
- DiagnostiCore TP53 predictions are AI-inferred morphology signals, not confirmed molecular assay results.
- Use the model card + locked-threshold report fields in the clinical report JSON when presenting limitations.
