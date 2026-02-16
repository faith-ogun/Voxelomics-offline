#!/usr/bin/env python3
"""
Export MedASR to a frontend ONNX bundle used by the browser worker.

Outputs:
  - medasr.onnx
  - medasr_vocab.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCTC, AutoProcessor


class LasrFrontendWrapper(nn.Module):
    """
    Wraps LASR CTC model with log-mel extraction so browser ONNX can accept raw PCM.
    """

    def __init__(self, model: nn.Module, feature_extractor) -> None:
        super().__init__()
        self.model = model.eval()
        self.n_fft = int(feature_extractor.n_fft)
        self.hop_length = int(feature_extractor.hop_length)
        self.win_length = int(feature_extractor.win_length)
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length, periodic=False, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "mel_filters",
            feature_extractor.mel_filters.to(torch.float32),
            persistent=True,
        )

    def _extract_log_mel(self, input_values: torch.Tensor) -> torch.Tensor:
        # input_values: [B, T]
        if input_values.shape[-1] < self.win_length:
            pad = self.win_length - input_values.shape[-1]
            input_values = F.pad(input_values, (0, pad))

        frames = input_values.unfold(-1, self.win_length, self.hop_length)  # [B, frames, win]
        stft = torch.fft.rfft(self.window * frames, n=self.n_fft)
        power_spec = stft.real.pow(2) + stft.imag.pow(2)  # [B, frames, freq]
        mel_spec = torch.clamp(power_spec @ self.mel_filters, min=1e-5)  # [B, frames, mel]
        return torch.log(mel_spec)

    def _frame_mask(self, frame_count: int, input_lengths: torch.Tensor) -> torch.Tensor:
        frame_lengths = torch.clamp(
            (input_lengths - (self.win_length - 1) + (self.hop_length - 1)) // self.hop_length,
            min=1,
        )
        frame_idx = torch.arange(frame_count, device=input_lengths.device).unsqueeze(0)
        return frame_idx < frame_lengths.unsqueeze(1)

    def forward(self, input_values: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        input_values = input_values.to(torch.float32)
        input_lengths = input_lengths.to(torch.long)
        input_features = self._extract_log_mel(input_values)
        attention_mask = self._frame_mask(input_features.shape[1], input_lengths)
        logits = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
        ).logits
        return logits


def read_vocab(tokenizer_json_path: Path) -> list[str]:
    payload = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    model = payload.get("model") or {}
    vocab = model.get("vocab")
    if isinstance(vocab, list):
        out: list[str] = []
        for entry in vocab:
            if isinstance(entry, list) and entry:
                out.append(str(entry[0]))
            else:
                out.append(str(entry))
        return out
    raise RuntimeError(
        f"Unsupported tokenizer vocab format in {tokenizer_json_path}. "
        "Expected model.vocab as a list."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MedASR frontend ONNX bundle.")
    parser.add_argument(
        "--model-dir",
        default="../../models/medasr",
        help="Path to local MedASR HF model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../public/models",
        help="Where to write medasr.onnx + medasr_vocab.json.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=19,
        help="ONNX opset version (default: 19).",
    )
    parser.add_argument(
        "--dummy-seconds",
        type=int,
        default=20,
        help="Dummy input seconds used during export tracing.",
    )
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime environment guard
        raise SystemExit(
            "Missing ONNX export dependency. Install first:\n"
            "  python3 -m pip install onnx onnxscript\n"
            f"Detail: {exc}"
        ) from exc

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    print(f"[export] loading model from {model_dir}")
    model = AutoModelForCTC.from_pretrained(str(model_dir), local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    feature_extractor = processor.feature_extractor
    wrapper = LasrFrontendWrapper(model, feature_extractor).eval()

    sample_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
    dummy_samples = max(sample_rate, sample_rate * int(args.dummy_seconds))
    dummy_input = torch.zeros((1, dummy_samples), dtype=torch.float32)
    dummy_lengths = torch.tensor([dummy_samples], dtype=torch.int64)

    onnx_path = output_dir / "medasr.onnx"
    print(f"[export] writing ONNX: {onnx_path}")
    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_lengths),
        str(onnx_path),
        input_names=["input_values", "input_lengths"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {1: "samples"},
            "logits": {1: "frames"},
        },
        opset_version=int(args.opset),
        external_data=False,
        do_constant_folding=True,
    )

    # Some torch exporter paths may still emit external data sidecars for large weights.
    # Force-consolidate back to a single-file ONNX for browser runtime compatibility.
    import onnx

    model_proto = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(model_proto, str(onnx_path), save_as_external_data=False)

    # Remove stale external-data sidecar from prior exports; browser worker expects a single-file ONNX.
    sidecar = onnx_path.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()

    vocab = read_vocab(model_dir / "tokenizer.json")
    vocab_path = output_dir / "medasr_vocab.json"
    vocab_payload = {
        "id_to_token": vocab,
        "blank_id": 0,
        "sample_rate": sample_rate,
    }
    vocab_path.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] writing vocab: {vocab_path}")

    print("[export] done.")


if __name__ == "__main__":
    main()
