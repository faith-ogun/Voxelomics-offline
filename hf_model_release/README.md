---
license: cc-by-4.0
library_name: scikit-learn
tags:
- healthcare
- pathology
- tp53
- classification
- hai-def
- path-foundation
base_model:
- google/path-foundation
---

# Voxelomics Path Foundation TP53 Head (Derived Artifact)

This repository contains a derived TP53 classifier head (`model.joblib`) and evaluation artifacts used in the Voxelomics offline pathology workflow demo.

## Provenance / Traceability to HAI-DEF
- Base HAI-DEF model: `google/path-foundation`
- Base model source: https://huggingface.co/google/path-foundation
- Embeddings were extracted with the Voxelomics Path Foundation embedding pipeline.
- This repo does **not** redistribute Path Foundation base weights.

## What is included
- `model.joblib` (derived TP53 classifier head)
- evaluation JSON artifacts
- `PROVENANCE.json` (artifact lineage and code references)

## Intended use
Research/demo use only. Not for clinical deployment or diagnosis.

## Code
https://github.com/faith-ogun/Voxelomics-offline
