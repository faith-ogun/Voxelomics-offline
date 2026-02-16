# DiagnostiCore Service (Module 1: Data Intake)

This module is the first concrete slice for Voxelomics:

- Cancer cohort: `TCGA-BRCA`
- First genomic target: `TP53`
- Current focus: build reliable input manifests (WSI + mutation files) before any modeling

## What this script does

`build_tcga_brca_manifests.py` queries the GDC API and produces:

1. `output/tcga_brca_wsi_files.csv`
2. `output/tcga_brca_masked_maf_files.csv`
3. `output/tcga_brca_wsi_manifest.tsv` (for `gdc-client download -m ...`)
4. `output/tcga_brca_maf_manifest.tsv` (for `gdc-client download -m ...`)
5. `output/tcga_brca_case_pairing.csv` (which cases have WSI and/or MAF)

No model training is done here. This is intake only.

## Setup

```bash
cd backend/diagnosticore-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python build_tcga_brca_manifests.py
```

Optional arguments:

```bash
python build_tcga_brca_manifests.py \
  --project-id TCGA-BRCA \
  --output-prefix tcga_brca \
  --output-dir output \
  --page-size 1000
```

## Next step after this module

Use the generated MAF manifest to download mutation files, then parse TP53 mutation status per case.

### Download MAF files (Module 2 input)

With `gdc-client` installed:

```bash
mkdir -p data/gdc_maf
gdc-client download \
  -m output/tcga_brca_maf_manifest.tsv \
  -d data/gdc_maf
```

If `gdc-client` is not installed, use the built-in downloader (open-access files):

```bash
python download_gdc_open_files.py \
  --manifest output/tcga_brca_maf_manifest.tsv \
  --download-dir data/gdc_maf
```

### Build TP53 labels (Module 2)

```bash
python build_tp53_case_labels.py \
  --maf-download-dir data/gdc_maf \
  --maf-file-metadata output/tcga_brca_masked_maf_files.csv \
  --case-pairing output/tcga_brca_case_pairing.csv \
  --output-csv output/tcga_brca_tp53_labels.csv
```

This writes:

- `output/tcga_brca_tp53_labels.csv`

Fields include:
- `case_submitter_id`
- `tp53_mutated`
- `tp53_non_silent_variant_count`
- `tp53_total_variant_count`
- `maf_files_expected`
- `maf_files_parsed`
- `missing_maf_file_ids`

## Module 3: Build WSI + TP53 Dataset Manifests

```bash
python build_wsi_tp53_dataset_manifest.py \
  --wsi-metadata output/tcga_brca_wsi_files.csv \
  --tp53-labels output/tcga_brca_tp53_labels.csv \
  --output-dir output
```

This writes:
- `output/tcga_brca_tp53_wsi_all_slides.csv` (all labeled slides)
- `output/tcga_brca_tp53_wsi_primary_slide.csv` (one representative slide per case)
- `output/tcga_brca_tp53_case_dataset.csv` (case-level labels)
- `output/tcga_brca_tp53_case_splits.csv` (deterministic train/val/test split by case)

## Module 4: Tiling Pipeline

### 4.1 Build labeled WSI download manifest

Primary slides (1 slide per case, recommended first):

```bash
python build_tp53_wsi_download_manifest.py \
  --slides-csv output/tcga_brca_tp53_wsi_primary_slide.csv \
  --case-splits output/tcga_brca_tp53_case_splits.csv \
  --out-manifest output/tcga_brca_tp53_wsi_primary_manifest.tsv \
  --out-metadata output/tcga_brca_tp53_wsi_primary_download_metadata.csv
```

Optional: train-only download manifest:

```bash
python build_tp53_wsi_download_manifest.py \
  --slides-csv output/tcga_brca_tp53_wsi_primary_slide.csv \
  --case-splits output/tcga_brca_tp53_case_splits.csv \
  --split train \
  --out-manifest output/tcga_brca_tp53_wsi_primary_train_manifest.tsv \
  --out-metadata output/tcga_brca_tp53_wsi_primary_train_download_metadata.csv
```

### 4.2 Download WSI files

```bash
mkdir -p data/gdc_wsi
gdc-client download -m output/tcga_brca_tp53_wsi_primary_manifest.tsv -d data/gdc_wsi
```

### 4.3 Build tiling jobs (checks local WSI presence)

```bash
python build_wsi_tiling_jobs.py \
  --slides-csv output/tcga_brca_tp53_wsi_primary_slide.csv \
  --case-splits output/tcga_brca_tp53_case_splits.csv \
  --wsi-download-dir data/gdc_wsi \
  --output-csv output/tcga_brca_tp53_tiling_jobs.csv
```

### 4.4 Run tiling (smoke test first)

Install system + Python dependencies for OpenSlide:

```bash
brew install openslide
pip install -r requirements.txt
```

Run a small smoke test:

```bash
python run_wsi_tiling.py \
  --jobs-csv output/tcga_brca_tp53_tiling_jobs.csv \
  --output-root data/tiles \
  --output-manifest output/tcga_brca_tp53_tiles_manifest.csv \
  --max-slides 2 \
  --max-tiles-per-slide 200
```

Then scale up by removing or increasing limits.

## Module 5: Baseline TP53 Training

Run a lightweight baseline on extracted tiles:

```bash
python train_tp53_baseline.py \
  --tile-manifest output/tcga_brca_tp53_tiles_manifest_full.csv \
  --output-dir output/baseline_tp53 \
  --max-train-tiles 50000 \
  --max-eval-tiles-per-split 15000
```

Outputs:
- `output/baseline_tp53/metrics.json`
- `output/baseline_tp53/tile_predictions.csv`
- `output/baseline_tp53/slide_predictions.csv`
- `output/baseline_tp53/case_predictions.csv`
- `output/baseline_tp53/model.joblib`

## Clinical Validity Artifact (CIs + Subgroups + Calibration + Model Card)

### 1) Optional calibration correction (fit on val only)
```bash
python calibrate_tp53_case_predictions.py \
  --input-csv output/cnn_tp53_200/case_predictions.csv \
  --output-csv output/cnn_tp53_200/case_predictions_calibrated.csv \
  --output-json output/cnn_tp53_200/case_calibration_report.json \
  --method platt \
  --fit-split val
```

### 2) Tune threshold on calibrated case predictions
```bash
python tune_case_threshold.py \
  --predictions-csv output/cnn_tp53_200/case_predictions_calibrated.csv \
  --objective max_f1 \
  --output-json output/cnn_tp53_200/threshold_tuning_calibrated_max_f1.json \
  --output-sweep-csv output/cnn_tp53_200/threshold_sweep_calibrated_max_f1.csv
```

### 3) Build locked-threshold clinical report from calibrated predictions

```bash
python build_tp53_clinical_report.py \
  --predictions-csv output/cnn_tp53_200/case_predictions_calibrated.csv \
  --threshold-json output/cnn_tp53_200/threshold_tuning_calibrated_max_f1.json \
  --output-json output/cnn_tp53_200/tp53_clinical_report_calibrated.json \
  --model-name "TP53 CNN (ResNet18 transfer baseline)" \
  --model-version "cnn_tp53_200" \
  --cohort "TCGA-BRCA primary-slide cohort (train100/val50/test50)" \
  --intended-use "Research-only TP53 risk support from WSI; not standalone diagnosis." \
  --known-limit "Single-cohort retrospective evaluation" \
  --known-limit "External validation pending"
```

Artifact includes:
- locked threshold metadata
- validation/test metrics
- test bootstrap 95% confidence intervals
- calibration (Brier + ECE)
- subgroup report on test split
- model-card fields suitable for MDT handoff

For MDT auto-handoff, point these env vars to:
- `MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV=../diagnosticore-service/output/cnn_tp53_200/case_predictions_calibrated_isotonic.csv`
- `MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON=../diagnosticore-service/output/cnn_tp53_200/tp53_clinical_report_calibrated.json`

### Optional: Build OpenSeadragon DeepZoom pyramid for MDT UI

Use a case WSI slide to generate a `.dzi` descriptor and tile pyramid:

```bash
python build_deepzoom_pyramid.py \
  --wsi-path data/gdc_wsi/<file_id>/<slide>.svs \
  --out-dir output/deepzoom \
  --slug <case_submitter_id>
```

For MDT case `MDT-001` (Sarah / `TCGA-A1-A0SP`), this should produce:
- `output/deepzoom/TCGA-A1-A0SP.dzi`
- `output/deepzoom/TCGA-A1-A0SP_files/`

## Module 6: External Validation (Independent Cohort)

Goal: keep the BRCA model locked, then evaluate on an independent cohort without retraining.

### 6.1 Build external cohort manifests

Replace placeholder values before running commands:
- `<EXTERNAL_PROJECT_ID>` example: `TCGA-LUAD`
- `<external_prefix>` example: `tcga_luad`

```bash
python build_tcga_brca_manifests.py \
  --project-id <EXTERNAL_PROJECT_ID> \
  --output-prefix <external_prefix> \
  --output-dir output
```

This writes:
- `output/<external_prefix>_wsi_files.csv`
- `output/<external_prefix>_masked_maf_files.csv`
- `output/<external_prefix>_wsi_manifest.tsv`
- `output/<external_prefix>_maf_manifest.tsv`
- `output/<external_prefix>_case_pairing.csv`

### 6.2 Build external TP53 labels

Download external MAF files first:

```bash
mkdir -p data/gdc_maf_external
gdc-client download -m output/<external_prefix>_maf_manifest.tsv -d data/gdc_maf_external
```

Then build labels:

```bash
python build_tp53_case_labels.py \
  --maf-download-dir data/gdc_maf_external \
  --maf-file-metadata output/<external_prefix>_masked_maf_files.csv \
  --case-pairing output/<external_prefix>_case_pairing.csv \
  --output-csv output/<external_prefix>_tp53_labels.csv
```

### 6.3 Build external primary-slide cohort (with leakage exclusion)

```bash
python build_external_tp53_dataset_manifest.py \
  --wsi-metadata output/<external_prefix>_wsi_files.csv \
  --tp53-labels output/<external_prefix>_tp53_labels.csv \
  --exclude-cases-csv output/tcga_brca_tp53_case_splits.csv \
  --split-name external \
  --source-cohort <EXTERNAL_PROJECT_ID> \
  --output-slides-csv output/<external_prefix>_tp53_external_primary_slide.csv \
  --output-case-splits-csv output/<external_prefix>_tp53_external_case_splits.csv
```

### 6.4 Build external tiling jobs and tile manifest

Download external WSI files first:

```bash
mkdir -p data/gdc_wsi_external
gdc-client download -m output/<external_prefix>_wsi_manifest.tsv -d data/gdc_wsi_external
```

```bash
python build_wsi_tiling_jobs.py \
  --slides-csv output/<external_prefix>_tp53_external_primary_slide.csv \
  --case-splits output/<external_prefix>_tp53_external_case_splits.csv \
  --wsi-download-dir data/gdc_wsi_external \
  --output-csv output/<external_prefix>_tp53_external_tiling_jobs.csv
```

```bash
python run_wsi_tiling.py \
  --jobs-csv output/<external_prefix>_tp53_external_tiling_jobs.csv \
  --output-root data/tiles \
  --output-manifest output/<external_prefix>_tp53_external_tiles_manifest.csv \
  --split external \
  --max-tiles-per-slide 500
```

### 6.5 Run locked-model inference (no retraining)

```bash
python predict_tp53_cnn_from_tiles.py \
  --tile-manifest output/<external_prefix>_tp53_external_tiles_manifest.csv \
  --model-weights output/cnn_tp53_200/model.pt \
  --output-dir output/cnn_tp53_external \
  --split-filter external \
  --decision-threshold 0.383
```

### 6.6 Generate external validation report

Use the same locked threshold policy from internal validation.

```bash
python evaluate_external_case_predictions.py \
  --predictions-csv output/cnn_tp53_external/case_predictions.csv \
  --split-name external \
  --threshold-json output/cnn_tp53_200/threshold_tuning_calibrated_max_f1.json \
  --reference-report-json output/cnn_tp53_200/tp53_clinical_report_calibrated.json \
  --output-json output/cnn_tp53_external/tp53_external_validation_report.json
```
