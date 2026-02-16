# Path Foundation TP53 Experiment (Isolated)

This folder is a fully separate experiment branch for comparing:

- Existing DiagnostiCore CNN baseline
- Google Path Foundation embeddings + lightweight TP53 head

No files in the existing CNN/MDT runtime are modified by this workflow.

## What this produces

Under your chosen `--output-dir` (example: `../output/pathfoundation_tp53_200`):

- `embeddings.npy`
- `embedding_rows.csv`
- `embedding_manifest.json`
- `tile_predictions.csv`
- `slide_predictions.csv`
- `case_predictions.csv`
- `metrics.json`
- `model.joblib`
- `comparison_vs_cnn.json` (optional benchmark report)

The prediction file schemas intentionally mirror the CNN output style for apples-to-apples analysis.

## 1) Install extra deps (if needed)

From `backend/diagnosticore-service`:

```bash
python3 -m pip install -r path_foundation_experiment/requirements.txt
```

Important:
- Path Foundation HF loading in this branch uses TensorFlow Keras.
- TensorFlow wheels typically target Python 3.11/3.12 (not 3.14). If your default Python is 3.14, create a separate 3.11/3.12 venv for this folder.
- Keep `huggingface_hub` below 1.0 for `from_pretrained_keras` support.

## 2) Extract Path Foundation embeddings

```bash
python3 path_foundation_experiment/extract_pathfoundation_embeddings.py \
  --tile-manifest output/tcga_brca_tp53_tiles_manifest_full_200.csv \
  --output-dir output/pathfoundation_tp53_200 \
  --model-id google/path-foundation \
  --batch-size 32
```

Notes:
- If you already have model files locally, you can pass a local path to `--model-id`.
- If your environment is offline, run once with internet to cache model files, then reuse cache.
- If access is gated, add `--hf-token <YOUR_TOKEN>`.

## 3) Train TP53 head and generate predictions

```bash
python3 path_foundation_experiment/train_tp53_head_from_embeddings.py \
  --embeddings-npy output/pathfoundation_tp53_200/embeddings.npy \
  --embedding-rows-csv output/pathfoundation_tp53_200/embedding_rows.csv \
  --output-dir output/pathfoundation_tp53_200 \
  --max-train-tiles 50000 \
  --max-eval-tiles-per-split 15000 \
  --decision-threshold 0.5
```

## 4) Compare against current CNN benchmark

```bash
python3 path_foundation_experiment/compare_vs_cnn_baseline.py \
  --cnn-case-predictions output/cnn_tp53_200/case_predictions.csv \
  --pathfoundation-case-predictions output/pathfoundation_tp53_200/case_predictions.csv \
  --output-json output/pathfoundation_tp53_200/comparison_vs_cnn.json
```

## Reusing your existing data

You do not need to redownload WSI data for this experiment if these already exist:

- Tile images referenced in your tile manifest (`output_png` column)
- TP53 labels/splits in the same manifest (`label`, `split`)

This branch directly consumes your existing tile manifest output.
