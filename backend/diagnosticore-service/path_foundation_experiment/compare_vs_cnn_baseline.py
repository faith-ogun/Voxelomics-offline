#!/usr/bin/env python3
"""
Compare case-level TP53 performance: CNN baseline vs Path Foundation head.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN and Path Foundation case predictions")
    parser.add_argument("--cnn-case-predictions", required=True)
    parser.add_argument("--pathfoundation-case-predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    if len(y_true) == 0:
        return {}
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, float] = {
        "n": float(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def evaluate_by_split(rows: List[Dict[str, str]], threshold: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for split in sorted({r.get("split", "") for r in rows}):
        subset = [r for r in rows if r.get("split", "") == split]
        if not subset:
            continue
        y = np.asarray([int(r["true_label"]) for r in subset], dtype=np.int64)
        p = np.asarray([float(r["pred_prob"]) for r in subset], dtype=np.float32)
        out[split] = metric_dict(y, p, threshold)
    return out


def align_rows(cnn_rows: List[Dict[str, str]], pf_rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    cnn_map = {(r.get("case_submitter_id", ""), r.get("split", "")): r for r in cnn_rows}
    pf_map = {(r.get("case_submitter_id", ""), r.get("split", "")): r for r in pf_rows}
    common_keys = sorted(set(cnn_map.keys()) & set(pf_map.keys()))
    return [cnn_map[k] for k in common_keys], [pf_map[k] for k in common_keys]


def subtract_metrics(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    # Returns b - a for overlapping keys.
    out: Dict[str, float] = {}
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        if isinstance(va, float) and isinstance(vb, float):
            out[k] = float(vb - va)
    return out


def main() -> None:
    args = parse_args()
    cnn_path = Path(args.cnn_case_predictions).resolve()
    pf_path = Path(args.pathfoundation_case_predictions).resolve()
    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cnn_rows = read_rows(cnn_path)
    pf_rows = read_rows(pf_path)
    cnn_rows, pf_rows = align_rows(cnn_rows, pf_rows)
    if not cnn_rows:
        raise RuntimeError("No overlapping case/split rows between CNN and Path Foundation files.")

    cnn_metrics = evaluate_by_split(cnn_rows, args.decision_threshold)
    pf_metrics = evaluate_by_split(pf_rows, args.decision_threshold)

    deltas: Dict[str, Dict[str, float]] = {}
    for split, metrics in cnn_metrics.items():
        if split in pf_metrics:
            deltas[split] = subtract_metrics(metrics, pf_metrics[split])

    payload = {
        "config": {
            "cnn_case_predictions": str(cnn_path),
            "pathfoundation_case_predictions": str(pf_path),
            "decision_threshold": float(args.decision_threshold),
        },
        "overlap": {
            "n_common_case_split_rows": int(len(cnn_rows)),
            "splits": sorted({r.get("split", "") for r in cnn_rows}),
        },
        "cnn_metrics_by_split": cnn_metrics,
        "pathfoundation_metrics_by_split": pf_metrics,
        "delta_pathfoundation_minus_cnn_by_split": deltas,
    }
    out_path.write_text(json.dumps(sanitize_json(payload), indent=2), encoding="utf-8")
    print(f"[compare] wrote {out_path}")


if __name__ == "__main__":
    main()
