"""
MDT -> DiagnostiCore handoff client.

Supports runtime fetch modes:
- off: disabled
- file: read case probabilities + clinical report JSON from local paths
- http: fetch per-case payload from a Diagnosticore HTTP endpoint
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from models import DiagnosticoreModelCard, DiagnosticorePrediction

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


class DiagnosticoreClient:
    def __init__(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2]
        self.mode = (os.getenv("MDT_DIAGNOSTICORE_FETCH_MODE", "off") or "off").strip().lower()
        if self.mode not in {"off", "file", "http"}:
            logger.warning("Unsupported MDT_DIAGNOSTICORE_FETCH_MODE=%s; defaulting to off.", self.mode)
            self.mode = "off"

        self.timeout_seconds = max(1.0, _env_float("MDT_DIAGNOSTICORE_TIMEOUT_SECONDS", 4.0))
        self.allow_fallback = (
            (os.getenv("MDT_DIAGNOSTICORE_ALLOW_FALLBACK", "true") or "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        # File mode config
        self.case_predictions_csv = Path(
            os.getenv(
                "MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV",
                "../diagnosticore-service/output/pathfoundation_tp53_200/case_predictions_calibrated_platt.csv",
            )
        )
        self.clinical_report_json = Path(
            os.getenv(
                "MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON",
                "../diagnosticore-service/output/pathfoundation_tp53_200/tp53_clinical_report_pathfoundation_platt.json",
            )
        )
        self.case_key_column = (os.getenv("MDT_DIAGNOSTICORE_CASE_KEY_COLUMN", "case_submitter_id") or "").strip()
        self.wsi_metadata_csv = Path(
            os.getenv(
                "MDT_DIAGNOSTICORE_WSI_METADATA_CSV",
                "../diagnosticore-service/output/tcga_brca_tp53_wsi_primary_slide.csv",
            )
        )
        self.wsi_download_dir = Path(
            os.getenv(
                "MDT_DIAGNOSTICORE_WSI_DOWNLOAD_DIR",
                "../diagnosticore-service/data/gdc_wsi",
            )
        )
        self.deepzoom_dir = Path(
            os.getenv(
                "MDT_DIAGNOSTICORE_DEEPZOOM_DIR",
                "../diagnosticore-service/output/deepzoom",
            )
        )
        tile_manifest_csvs_env = (
            os.getenv(
                "MDT_DIAGNOSTICORE_TILE_MANIFEST_CSVS",
                "../diagnosticore-service/output/tcga_brca_tp53_tiles_manifest_external20.csv,"
                "../diagnosticore-service/output/tcga_brca_tp53_tiles_manifest_full_200.csv",
            )
            or ""
        )
        self.tile_manifest_csvs = [Path(p.strip()) for p in tile_manifest_csvs_env.split(",") if p.strip()]

        # HTTP mode config
        self.base_url = (os.getenv("MDT_DIAGNOSTICORE_BASE_URL", "") or "").strip()
        self.path_template = (
            os.getenv(
                "MDT_DIAGNOSTICORE_HTTP_PATH_TEMPLATE",
                "/diagnosticore/cases/{case_key}/prediction",
            )
            or ""
        ).strip()

        self._file_case_map: Optional[Dict[str, Dict[str, str]]] = None
        self._file_report: Optional[Dict] = None
        self._file_wsi_map: Optional[Dict[str, Dict[str, str]]] = None
        self._tile_preview_map: Optional[Dict[str, Dict[str, str]]] = None

    def fetch_prediction(self, case_key: str) -> Optional[DiagnosticorePrediction]:
        if not case_key or self.mode == "off":
            return None
        try:
            if self.mode == "file":
                return self._fetch_prediction_file(case_key)
            if self.mode == "http":
                return self._fetch_prediction_http(case_key)
            return None
        except Exception as exc:
            if self.allow_fallback:
                logger.warning("DiagnostiCore fetch failed for key=%s; fallback engaged: %s", case_key, exc)
                return None
            raise

    def _fetch_prediction_file(self, case_key: str) -> Optional[DiagnosticorePrediction]:
        case_map = self._load_case_map()
        row = case_map.get(case_key)
        if not row:
            return None

        report = self._load_report()
        threshold = self._extract_threshold(report)
        prob = float(row.get("pred_prob", "0.5"))
        predicted_label = "tp53_mutated" if prob >= threshold else "tp53_wildtype"
        source_split = (row.get("split") or "").strip().lower() or None
        n_tiles = _to_int(row.get("n_tiles"))
        raw_pred_prob = _to_float(row.get("raw_pred_prob"))

        model_card_dict = report.get("model_card", {}) if isinstance(report, dict) else {}
        model_card = DiagnosticoreModelCard.model_validate(
            {
                "cohort": model_card_dict.get("cohort"),
                "intended_use": model_card_dict.get("intended_use"),
                "limitations": model_card_dict.get("limitations", []),
            }
        )

        handoff_summary = report.get("mdt_handoff_summary", {}) if isinstance(report, dict) else {}
        ece = handoff_summary.get("test_ece_10")
        recall_ci_low = handoff_summary.get("test_recall_ci_low")
        uncertainty_flags = []
        try:
            if ece is not None and float(ece) > 0.10:
                uncertainty_flags.append(f"calibration_ece_high={float(ece):.3f}")
        except Exception:
            pass
        try:
            if recall_ci_low is not None and float(recall_ci_low) < 0.50:
                uncertainty_flags.append(f"recall_ci_low={float(recall_ci_low):.3f}")
        except Exception:
            pass

        locked_threshold_report = {
            "artifact_type": report.get("artifact_type", "tp53_locked_threshold_report_v1"),
            "selected_threshold": float(threshold),
            "test_ece_10": ece,
            "test_recall_ci_low": recall_ci_low,
        }
        validation_metrics: Dict[str, float] = {}
        overall_metrics = report.get("overall_metrics", {}) if isinstance(report, dict) else {}
        test_metrics = overall_metrics.get("test", {}) if isinstance(overall_metrics, dict) else {}
        test_threshold_metrics = (
            test_metrics.get("threshold_metrics", {}) if isinstance(test_metrics, dict) else {}
        )
        test_rank_metrics = test_metrics.get("rank_metrics", {}) if isinstance(test_metrics, dict) else {}
        test_calibration = test_metrics.get("calibration", {}) if isinstance(test_metrics, dict) else {}
        for key, value in {
            "accuracy": _to_float(test_threshold_metrics.get("accuracy")),
            "precision": _to_float(test_threshold_metrics.get("precision")),
            "recall": _to_float(test_threshold_metrics.get("recall")),
            "f1": _to_float(test_threshold_metrics.get("f1")),
            "roc_auc": _to_float(test_rank_metrics.get("roc_auc")),
            "average_precision": _to_float(test_rank_metrics.get("average_precision")),
            "brier_score": _to_float(test_calibration.get("brier_score")),
            "ece_10": _to_float(test_calibration.get("ece_10")),
        }.items():
            if value is not None:
                validation_metrics[key] = value
        if validation_metrics:
            locked_threshold_report["overall_metrics_test"] = validation_metrics

        wsi_meta = self._load_wsi_map().get(case_key, {})
        wsi_file_id = (wsi_meta.get("file_id") or "").strip() or None
        wsi_file_name = (wsi_meta.get("file_name") or "").strip() or None
        wsi_project_id = (wsi_meta.get("project_id") or "").strip() or None
        wsi_local_path = self._resolve_wsi_path(wsi_file_id, wsi_file_name)

        tile_preview = self._load_tile_preview_map().get(case_key, {})
        tile_preview_png = (tile_preview.get("output_png") or "").strip() or None
        tile_preview_x = _to_int(tile_preview.get("x"))
        tile_preview_y = _to_int(tile_preview.get("y"))
        deepzoom_dzi_path, deepzoom_tile_dir = self._resolve_deepzoom_paths(case_key)

        calibration_method = "unknown"
        model_version_text = str(model_card_dict.get("model_version") or row.get("model_version") or "")
        pred_csv_name = self.case_predictions_csv.name.lower()
        if "iso" in model_version_text.lower() or "isotonic" in pred_csv_name:
            calibration_method = "isotonic"
        elif "platt" in model_version_text.lower() or "platt" in pred_csv_name:
            calibration_method = "platt"

        cohort_text = str(model_card_dict.get("cohort") or "")
        cohort_relation = None
        if source_split == "external":
            cohort_relation = (
                "same_cohort_external_split"
                if "TCGA-BRCA" in cohort_text.upper()
                else "cross_cohort_external"
            )
        elif source_split in {"train", "val", "test"}:
            cohort_relation = f"internal_{source_split}_split"

        return DiagnosticorePrediction(
            source_service="diagnosticore-service",
            target="tp53_mutation",
            case_submitter_id=case_key,
            tp53_probability=prob,
            threshold=float(threshold),
            predicted_label=predicted_label,
            uncertainty_flags=uncertainty_flags,
            model_version=(model_card_dict.get("model_version") or row.get("model_version") or None),
            data_version=(row.get("data_version") or model_card_dict.get("cohort") or None),
            is_confirmed_genomic_test=False,
            model_card=model_card,
            locked_threshold_report=locked_threshold_report,
            locked_threshold_report_uri=str(self.clinical_report_json.resolve()),
            source_split=source_split,
            n_tiles=n_tiles,
            raw_pred_prob=raw_pred_prob,
            validation_metrics=validation_metrics,
            calibration_method=calibration_method,
            wsi_project_id=wsi_project_id,
            wsi_file_id=wsi_file_id,
            wsi_file_name=wsi_file_name,
            wsi_local_path=wsi_local_path,
            tile_preview_png=tile_preview_png,
            tile_preview_x=tile_preview_x,
            tile_preview_y=tile_preview_y,
            cohort_relation=cohort_relation,
            deepzoom_dzi_path=deepzoom_dzi_path,
            deepzoom_tile_dir=deepzoom_tile_dir,
        )

    def _fetch_prediction_http(self, case_key: str) -> Optional[DiagnosticorePrediction]:
        if not self.base_url:
            raise RuntimeError("MDT_DIAGNOSTICORE_BASE_URL is required for http fetch mode.")
        rel_path = self.path_template.format(case_key=case_key)
        url = urljoin(self.base_url.rstrip("/") + "/", rel_path.lstrip("/"))
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()

        if isinstance(payload, dict) and "diagnosticore" in payload and isinstance(payload["diagnosticore"], dict):
            payload = payload["diagnosticore"]
        if not isinstance(payload, dict):
            raise RuntimeError("Diagnosticore HTTP payload must be a JSON object.")
        return DiagnosticorePrediction.model_validate(payload)

    def _load_case_map(self) -> Dict[str, Dict[str, str]]:
        if self._file_case_map is not None:
            return self._file_case_map

        path = self._resolve_file_path(self.case_predictions_csv)
        if not path.exists():
            raise FileNotFoundError(f"DiagnostiCore case predictions CSV not found: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        case_map: Dict[str, Dict[str, str]] = {}
        for r in rows:
            key = r.get(self.case_key_column)
            if key:
                case_map[str(key)] = r
        self._file_case_map = case_map
        return self._file_case_map

    def _load_report(self) -> Dict:
        if self._file_report is not None:
            return self._file_report
        path = self._resolve_file_path(self.clinical_report_json)
        if not path.exists():
            raise FileNotFoundError(f"DiagnostiCore clinical report JSON not found: {path}")
        self._file_report = json.loads(path.read_text(encoding="utf-8"))
        return self._file_report

    def _load_wsi_map(self) -> Dict[str, Dict[str, str]]:
        if self._file_wsi_map is not None:
            return self._file_wsi_map
        path = self._resolve_file_path(self.wsi_metadata_csv)
        if not path.exists():
            self._file_wsi_map = {}
            return self._file_wsi_map
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self._file_wsi_map = {}
        for row in rows:
            key = (row.get("case_submitter_id") or "").strip()
            if key and key not in self._file_wsi_map:
                self._file_wsi_map[key] = row
        return self._file_wsi_map

    def _load_tile_preview_map(self) -> Dict[str, Dict[str, str]]:
        if self._tile_preview_map is not None:
            return self._tile_preview_map
        self._tile_preview_map = {}
        for manifest in self.tile_manifest_csvs:
            manifest_path = self._resolve_file_path(manifest)
            if not manifest_path.exists():
                continue
            with manifest_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("case_submitter_id") or "").strip()
                    if not key or key in self._tile_preview_map:
                        continue
                    output_png = (row.get("output_png") or "").strip()
                    output_png = self._resolve_output_png_path(output_png, manifest_path)
                    resolved = {
                        "output_png": output_png,
                        "x": row.get("x") or "",
                        "y": row.get("y") or "",
                    }
                    self._tile_preview_map[key] = resolved
        return self._tile_preview_map

    def _resolve_output_png_path(self, output_png: str, manifest_path: Path) -> str:
        value = (output_png or "").strip()
        if not value:
            return ""

        original = Path(value)
        if not original.is_absolute():
            return str((manifest_path.parent / original).resolve())

        # Prefer remapped in-repo paths when the CSV came from a different workspace.
        remapped = self._remap_legacy_workspace_path(original)
        if remapped is not None and remapped.exists():
            return str(remapped)
        return str(original)

    def _remap_legacy_workspace_path(self, source: Path) -> Optional[Path]:
        parts = list(source.parts)
        if "backend" not in parts:
            return None
        idx = parts.index("backend")
        if idx + 1 >= len(parts) or parts[idx + 1] != "diagnosticore-service":
            return None
        suffix = Path(*parts[idx:])
        candidate = (self.workspace_root / suffix).resolve()
        return candidate

    def _resolve_wsi_path(self, file_id: Optional[str], file_name: Optional[str]) -> Optional[str]:
        if not file_id or not file_name:
            return None
        root = self._resolve_file_path(self.wsi_download_dir)
        candidate = (root / file_id / file_name).resolve()
        if candidate.exists():
            return str(candidate)
        return None

    def _resolve_deepzoom_paths(self, case_key: str) -> tuple[Optional[str], Optional[str]]:
        base = self._resolve_file_path(self.deepzoom_dir)
        dzi = (base / f"{case_key}.dzi").resolve()
        tile_dir = (base / f"{case_key}_files").resolve()
        if dzi.exists() and tile_dir.exists():
            return str(dzi), str(tile_dir)
        return None, None

    @staticmethod
    def _resolve_file_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        return (Path(__file__).parent / path).resolve()

    @staticmethod
    def _extract_threshold(report: Dict) -> float:
        try:
            return float(report.get("locked_threshold", {}).get("value", 0.5))
        except Exception:
            return 0.5
