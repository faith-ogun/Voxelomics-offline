#!/usr/bin/env python3
"""
Train/evaluate TP53 classifier from Path Foundation embeddings.

Outputs:
- <output-dir>/model.joblib
- <output-dir>/tile_predictions.csv
- <output-dir>/slide_predictions.csv
- <output-dir>/case_predictions.csv
- <output-dir>/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TP53 head on Path Foundation embeddings")
    parser.add_argument("--embeddings-npy", required=True)
    parser.add_argument("--embedding-rows-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-tiles", type=int, default=50000)
    parser.add_argument("--max-eval-tiles-per-split", type=int, default=15000)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def label_to_int(label: str) -> int:
    return 1 if str(label).strip().lower() == "tp53_mutated" else 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stratified_cap(indices: List[int], labels: np.ndarray, cap: int, seed: int) -> List[int]:
    if cap <= 0 or len(indices) <= cap:
        return indices

    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        by_label[int(labels[idx])].append(idx)

    all_labels = sorted(by_label.keys())
    per_label = max(1, cap // max(1, len(all_labels)))
    out: List[int] = []
    for k in all_labels:
        candidates = by_label[k]
        if len(candidates) <= per_label:
            out.extend(candidates)
        else:
            picked = rng.choice(candidates, size=per_label, replace=False).tolist()
            out.extend(int(x) for x in picked)

    if len(out) > cap:
        out = rng.choice(out, size=cap, replace=False).tolist()
    return out


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


def aggregate_predictions(
    rows: Sequence[Dict[str, str]],
    probs: np.ndarray,
    key: str,
    threshold: float,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, float]]]:
    grouped: Dict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for r, p in zip(rows, probs):
        gid = r.get(key, "")
        if not gid:
            continue
        split = r.get("split", "")
        y = label_to_int(r.get("label", ""))
        grouped[(split, gid)].append((y, float(p)))

    out_rows: List[Dict[str, str]] = []
    y_by_split: Dict[str, List[int]] = defaultdict(list)
    p_by_split: Dict[str, List[float]] = defaultdict(list)

    for (split, gid), values in grouped.items():
        true_label = int(round(sum(v[0] for v in values) / len(values)))
        pred_prob = float(sum(v[1] for v in values) / len(values))
        out_rows.append(
            {
                key: gid,
                "split": split,
                "true_label": str(true_label),
                "pred_prob": f"{pred_prob:.6f}",
                "pred_label": str(int(pred_prob >= threshold)),
                "n_tiles": str(len(values)),
            }
        )
        y_by_split[split].append(true_label)
        p_by_split[split].append(pred_prob)

    metrics_by_split: Dict[str, Dict[str, float]] = {}
    for split, ys in y_by_split.items():
        ps = p_by_split[split]
        metrics_by_split[split] = metric_dict(
            np.asarray(ys, dtype=np.int64),
            np.asarray(ps, dtype=np.float32),
            threshold=threshold,
        )
    return out_rows, metrics_by_split


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    embeddings_path = Path(args.embeddings_npy).resolve()
    rows_path = Path(args.embedding_rows_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(embeddings_path)
    rows = read_rows(rows_path)
    if X.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape={X.shape}")
    if X.shape[0] != len(rows):
        raise RuntimeError(f"Row count mismatch: embeddings={X.shape[0]}, rows={len(rows)}")

    y = np.asarray([label_to_int(r.get("label", "")) for r in rows], dtype=np.int64)
    split = np.asarray([r.get("split", "") for r in rows], dtype=object)

    train_idx = [i for i, s in enumerate(split) if s == "train"]
    val_idx = [i for i, s in enumerate(split) if s == "val"]
    test_idx = [i for i, s in enumerate(split) if s == "test"]

    if not train_idx:
        raise RuntimeError("No train rows found.")
    if len(set(y[train_idx].tolist())) < 2:
        raise RuntimeError("Train split must contain both classes.")

    train_idx = stratified_cap(train_idx, y, args.max_train_tiles, args.seed)
    val_idx = stratified_cap(val_idx, y, args.max_eval_tiles_per_split, args.seed + 1)
    test_idx = stratified_cap(test_idx, y, args.max_eval_tiles_per_split, args.seed + 2)
    eval_idx = val_idx + test_idx

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=args.c,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    clf.fit(X[train_idx], y[train_idx])

    train_prob = clf.predict_proba(X[train_idx])[:, 1].astype(np.float32)
    eval_prob = clf.predict_proba(X[eval_idx])[:, 1].astype(np.float32) if eval_idx else np.asarray([], dtype=np.float32)

    tile_rows: List[Dict[str, str]] = []
    tile_metrics: Dict[str, Dict[str, float]] = {}
    for split_name, idxs, probs in [
        ("train", train_idx, train_prob),
        ("eval", eval_idx, eval_prob),
    ]:
        if not idxs:
            continue
        ys = y[idxs]
        tile_metrics[split_name] = metric_dict(ys, probs, threshold=args.decision_threshold)
        for local_i, row_idx in enumerate(idxs):
            base = dict(rows[row_idx])
            base["true_label"] = str(int(y[row_idx]))
            p = float(probs[local_i])
            base["pred_prob"] = f"{p:.6f}"
            base["pred_label"] = str(int(p >= args.decision_threshold))
            tile_rows.append(base)

    # Build split-specific rows for aggregation.
    train_rows = [rows[i] for i in train_idx]
    eval_rows = [rows[i] for i in eval_idx]

    slide_train, slide_train_metrics = aggregate_predictions(
        train_rows, train_prob, key="file_id", threshold=args.decision_threshold
    )
    case_train, case_train_metrics = aggregate_predictions(
        train_rows, train_prob, key="case_submitter_id", threshold=args.decision_threshold
    )

    slide_eval, slide_eval_metrics = aggregate_predictions(
        eval_rows, eval_prob, key="file_id", threshold=args.decision_threshold
    )
    case_eval, case_eval_metrics = aggregate_predictions(
        eval_rows, eval_prob, key="case_submitter_id", threshold=args.decision_threshold
    )

    slide_rows = slide_train + slide_eval
    case_rows = case_train + case_eval

    # Overwrite eval pseudo-split with val/test from source rows where possible.
    split_lookup_slide: Dict[Tuple[str, str], str] = {}
    for r in eval_rows:
        split_lookup_slide[(r.get("file_id", ""), r.get("split", ""))] = r.get("split", "")
    for r in slide_rows:
        if r.get("split") == "eval":
            # not expected, but keep safe
            r["split"] = "unknown"

    split_lookup_case: Dict[str, str] = {}
    for r in eval_rows:
        cid = r.get("case_submitter_id", "")
        if cid and cid not in split_lookup_case:
            split_lookup_case[cid] = r.get("split", "")
    for r in case_rows:
        if r.get("case_submitter_id") in split_lookup_case and r.get("split") == "eval":
            r["split"] = split_lookup_case[r["case_submitter_id"]]

    # Recompute split-specific eval metrics (val/test) for case/slide from rows.
    def split_metrics_from_rows(rows_in: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for s in sorted({r.get("split", "") for r in rows_in}):
            subset = [r for r in rows_in if r.get("split", "") == s]
            if not subset:
                continue
            ys = np.asarray([int(r["true_label"]) for r in subset], dtype=np.int64)
            ps = np.asarray([float(r["pred_prob"]) for r in subset], dtype=np.float32)
            out[s] = metric_dict(ys, ps, threshold=args.decision_threshold)
        return out

    slide_metrics = split_metrics_from_rows(slide_rows)
    case_metrics = split_metrics_from_rows(case_rows)

    model_path = out_dir / "model.joblib"
    tile_path = out_dir / "tile_predictions.csv"
    slide_path = out_dir / "slide_predictions.csv"
    case_path = out_dir / "case_predictions.csv"
    metrics_path = out_dir / "metrics.json"

    joblib.dump(clf, model_path)
    write_csv(tile_path, tile_rows)
    write_csv(slide_path, slide_rows)
    write_csv(case_path, case_rows)

    metrics = {
        "config": {
            "embeddings_npy": str(embeddings_path),
            "embedding_rows_csv": str(rows_path),
            "seed": int(args.seed),
            "max_train_tiles": int(args.max_train_tiles),
            "max_eval_tiles_per_split": int(args.max_eval_tiles_per_split),
            "decision_threshold": float(args.decision_threshold),
            "logreg_c": float(args.c),
        },
        "dataset": {
            "n_rows_total": int(len(rows)),
            "n_features": int(X.shape[1]),
            "n_train_used": int(len(train_idx)),
            "n_eval_used": int(len(eval_idx)),
            "class_balance_train": float(y[train_idx].mean()) if train_idx else None,
        },
        "tile_metrics": tile_metrics,
        "slide_metrics_by_split": slide_metrics,
        "case_metrics_by_split": case_metrics,
        "compat_note": {
            "cnn_equivalent_files": [
                str(tile_path),
                str(slide_path),
                str(case_path),
                str(metrics_path),
            ]
        },
    }
    metrics_path.write_text(json.dumps(sanitize_json(metrics), indent=2), encoding="utf-8")

    print(f"[train] model: {model_path}")
    print(f"[train] tile predictions: {tile_path}")
    print(f"[train] slide predictions: {slide_path}")
    print(f"[train] case predictions: {case_path}")
    print(f"[train] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
