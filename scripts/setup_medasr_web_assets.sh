#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_SRC="${ROOT_DIR}/node_modules/onnxruntime-web/dist"
RUNTIME_DEST="${ROOT_DIR}/public/vendor/onnxruntime-web"

if [[ ! -d "${RUNTIME_SRC}" ]]; then
  cat <<'MSG'
onnxruntime-web runtime files were not found in node_modules.

Install first:
  npm install onnxruntime-web

Then rerun:
  bash scripts/setup_medasr_web_assets.sh
MSG
  exit 1
fi

mkdir -p "${RUNTIME_DEST}"
cp "${RUNTIME_SRC}/ort.min.js" "${RUNTIME_DEST}/"
cp "${RUNTIME_SRC}/ort.wasm.min.js" "${RUNTIME_DEST}/"
cp "${RUNTIME_SRC}/ort.wasm.js" "${RUNTIME_DEST}/" 2>/dev/null || true
cp "${RUNTIME_SRC}"/ort-wasm*.wasm "${RUNTIME_DEST}/" 2>/dev/null || true
cp "${RUNTIME_SRC}"/ort-wasm*.mjs "${RUNTIME_DEST}/" 2>/dev/null || true

echo "Copied onnxruntime-web assets to ${RUNTIME_DEST}"
