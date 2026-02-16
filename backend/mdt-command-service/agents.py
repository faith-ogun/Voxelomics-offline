"""
Voxelomics MDT Command Service - Agent Orchestration

Implements the 10-agent architecture:
1. MDTOrchestrator (system)
2. RadiologySynthesizer (parallel)
3. PathologySynthesizer (parallel)
4. GenomicsSynthesizer (parallel)
5. LiteratureAgent (parallel tool-backed)
6. TranscriptionAgent (concurrent MedASR adapter)
7. ConsensusSynthesizer (sequential)
8. SOAPGenerator (sequential)
9. HITLGatekeeper (system gate)

Execution mode:
- local (only): offline local inference and local storage
"""

from __future__ import annotations

import asyncio
import ast
import importlib
import json
import logging
import os
import re
import wave
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from case_repository import (
    CaseRepository,
    SqliteCaseRepository,
)
from clinical_reasoner import ClinicalReasoner
from diagnosticore_client import DiagnosticoreClient
from env_loader import load_service_env
from models import (
    AgentRunStatus,
    AgentTrace,
    ApprovalDecision,
    CaseStatus,
    ClinicalReasoningSummary,
    ConsensusRecommendation,
    DiagnosticorePrediction,
    GenomicsSummary,
    HITLGateOutput,
    LiteratureSummary,
    MDTCaseInput,
    MDTCaseRecord,
    PathologySummary,
    RadiologySummary,
    SOAPNote,
    StageOneOutput,
    TranscriptionResult,
)
from tools import (
    extract_gene_symbols,
    get_case_input,
    list_available_case_ids,
    parse_json_payload,
    search_literature_evidence,
)

logger = logging.getLogger(__name__)
TRANSFORMERS_IMPORT_LOCK = Lock()

# Load `.env` / `.env.local` for this service before reading os.environ.
load_service_env()

# Optional ADK imports. Service remains functional without ADK in `local` mode.
try:
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.function_tool import FunctionTool
    from google.genai import types

    ADK_AVAILABLE = True
except Exception:
    ADK_AVAILABLE = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _resolve_local_model_ref(ref: str) -> str:
    candidate = (ref or "").strip()
    if not candidate:
        return candidate
    direct = Path(candidate).expanduser()
    if direct.exists():
        return str(direct.resolve())
    anchored = (Path(__file__).resolve().parent / candidate).expanduser()
    if anchored.exists():
        return str(anchored.resolve())
    return candidate


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_nested(dct: Dict[str, object], path: List[str]) -> object:
    cur: object = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _extract_report_metric(report: Dict[str, object], candidate_paths: List[List[str]]) -> Optional[float]:
    for path in candidate_paths:
        val = _safe_float(_get_nested(report, path))
        if val is not None:
            return val
    return None


class MedASRAdapter:
    """
    MedASR adapter in strict local-only mode.
    """

    def __init__(self) -> None:
        self.mode = (os.getenv("MDT_MEDASR_MODE", "local") or "local").strip().lower()
        if self.mode != "local":
            raise RuntimeError(
                "Offline build requires MDT_MEDASR_MODE=local. "
                f"Received MDT_MEDASR_MODE={self.mode!r}."
            )

        default_engine = "medasr-local"
        self.engine_name = os.getenv(
            "MDT_TRANSCRIPTION_ENGINE",
            default_engine,
        )
        self.local_model_id = _resolve_local_model_ref(
            (os.getenv("MDT_MEDASR_LOCAL_MODEL_ID", "google/medasr") or "").strip()
        )
        self.local_device = (os.getenv("MDT_MEDASR_LOCAL_DEVICE", "auto") or "auto").strip().lower()
        self.local_chunk_length_s = max(1, _env_int("MDT_MEDASR_LOCAL_CHUNK_LENGTH_S", 25))
        self.local_stride_length_s = max(0, _env_int("MDT_MEDASR_LOCAL_STRIDE_LENGTH_S", 3))
        self.local_allow_text_fallback = (
            (os.getenv("MDT_MEDASR_LOCAL_ALLOW_TEXT_FALLBACK", "true") or "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.local_ctc_retry_enabled = (
            (os.getenv("MDT_MEDASR_LOCAL_CTC_RETRY", "false") or "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.local_transcript_max_chars = max(
            256, _env_int("MDT_MEDASR_LOCAL_MAX_TRANSCRIPT_CHARS", 2200)
        )
        self.timeout_seconds = max(5.0, _env_float("MDT_MEDASR_TIMEOUT_SECONDS", 45.0))
        self._local_pipe = None
        self._local_processor = None
        self._local_model = None
        self._local_torch = None
        self._local_device = "cpu"
        self._pipe_lock = Lock()
        self._infer_lock = Lock()

    async def transcribe(self, transcript_text: str, audio_uri: Optional[str] = None) -> TranscriptionResult:
        return await asyncio.to_thread(
            self._transcribe_local_sync,
            transcript_text,
            audio_uri,
        )

    def _transcribe_local_sync(
        self,
        transcript_text: str,
        audio_uri: Optional[str],
    ) -> TranscriptionResult:
        cleaned_text = " ".join((transcript_text or "").split())
        local_audio_path: Optional[str]
        try:
            local_audio_path = self._resolve_local_audio_path(audio_uri)
        except ValueError as exc:
            # Offline profile can receive legacy cloud URIs from seeded case files.
            # If transcript text is already present, keep execution local by using it.
            if cleaned_text:
                logger.warning(
                    "Ignoring non-local audio_uri in offline mode (%s). "
                    "Using provided transcript text instead.",
                    exc,
                )
                return TranscriptionResult(
                    engine=self.engine_name,
                    transcript=cleaned_text,
                    wer_estimate=None,
                    notes=(
                        "Using transcript text supplied by the client UI "
                        "(manual entry or on-device local transcription)."
                    ),
                )
            raise

        if not local_audio_path:
            if self.local_allow_text_fallback:
                if not cleaned_text:
                    raise ValueError("audio_uri is required for local MedASR transcription.")
                return TranscriptionResult(
                    engine=self.engine_name,
                    transcript=cleaned_text,
                    wer_estimate=None,
                    notes=(
                        "Using transcript text supplied by the client UI "
                        "(manual entry or on-device local transcription)."
                    ),
                )
            raise ValueError("audio_uri is required for MDT_MEDASR_MODE=local.")

        pipe = self._get_local_pipe()
        try:
            target_sr = None
            try:
                target_sr = int(getattr(pipe.feature_extractor, "sampling_rate", 0))  # type: ignore[attr-defined]
            except Exception:
                target_sr = None
            result = pipe(
                self._prepare_audio_input(local_audio_path, target_sampling_rate=target_sr),
                chunk_length_s=self.local_chunk_length_s,
                stride_length_s=self.local_stride_length_s,
            )
        except Exception as exc:
            if "ffmpeg was not found" in str(exc).lower():
                raise RuntimeError(
                    "ffmpeg is not installed. Use WAV input, or install ffmpeg for MP4/M4A transcription."
                ) from exc
            raise

        transcript = self._extract_transcript_from_obj(result)
        if not transcript:
            raise RuntimeError("Local MedASR response did not contain transcript text.")
        transcript = self._normalize_transcript_text(transcript)
        if self._looks_corrupted_transcript(transcript):
            repaired = self._repair_corrupted_transcript(transcript)
            if repaired and not self._looks_corrupted_transcript(repaired):
                transcript = repaired
            elif self.local_ctc_retry_enabled:
                logger.warning(
                    "MedASR pipeline output looked corrupted; retrying with direct CTC generate decode."
                )
                transcript = self._normalize_transcript_text(
                    self._transcribe_with_direct_ctc(local_audio_path)
                )
                if self._looks_corrupted_transcript(transcript):
                    repaired = self._repair_corrupted_transcript(transcript)
                    if repaired and not self._looks_corrupted_transcript(repaired):
                        transcript = repaired
                    elif self.local_allow_text_fallback and cleaned_text:
                        logger.warning(
                            "MedASR direct CTC decode still low quality; using provided transcript fallback."
                        )
                        transcript = cleaned_text
                    else:
                        raise RuntimeError(
                            "Local MedASR returned low-quality tokenized output "
                            "(epsilon/repetition artifacts)."
                        )
            elif self.local_allow_text_fallback and cleaned_text:
                logger.warning(
                    "MedASR output looked corrupted; using provided transcript fallback "
                    "(direct CTC retry disabled)."
                )
                transcript = cleaned_text
            else:
                raise RuntimeError(
                    "Local MedASR returned low-quality tokenized output "
                    "(epsilon/repetition artifacts)."
                )
        transcript = self._truncate_transcript(transcript)

        notes = f"Local MedASR transcription via model '{self.local_model_id}'."
        return TranscriptionResult(
            engine=self.engine_name,
            transcript=transcript,
            wer_estimate=self._extract_wer_from_obj(result),
            notes=notes,
        )

    def _resolve_local_audio_path(self, audio_uri: Optional[str]) -> Optional[str]:
        if not audio_uri:
            return None
        parsed = urlparse(audio_uri)

        if parsed.scheme in {"", "file"}:
            path = Path(parsed.path if parsed.scheme == "file" else audio_uri).expanduser().resolve()
            if path.exists() and path.is_file():
                return str(path)
            return None

        raise ValueError(
            f"Unsupported audio_uri scheme for offline MedASR: {parsed.scheme or 'unknown'}."
        )

    def _get_local_pipe(self):
        if self._local_pipe is not None:
            return self._local_pipe
        with self._pipe_lock:
            if self._local_pipe is not None:
                return self._local_pipe
            try:
                with TRANSFORMERS_IMPORT_LOCK:
                    import torch
                    transformers_mod = importlib.import_module("transformers")
                    pipeline_fn = getattr(transformers_mod, "pipeline", None)
                    if pipeline_fn is None:
                        pipelines_mod = importlib.import_module("transformers.pipelines")
                        pipeline_fn = getattr(pipelines_mod, "pipeline", None)
                    if pipeline_fn is None:
                        raise RuntimeError("transformers.pipeline is unavailable in this installation.")
            except Exception as exc:
                import sys
                raise RuntimeError(
                    "MDT_MEDASR_MODE=local could not import transformers/torch "
                    f"(python={sys.executable}): {exc}"
                ) from exc

            device = -1
            if self.local_device in {"cuda", "gpu"}:
                if not torch.cuda.is_available():
                    raise RuntimeError("MDT_MEDASR_LOCAL_DEVICE=cuda requested but CUDA is unavailable.")
                device = 0
            elif self.local_device == "auto" and torch.cuda.is_available():
                device = 0

            self._local_pipe = pipeline_fn(
                "automatic-speech-recognition",
                model=self.local_model_id,
                device=device,
            )
            # Some CPU environments hit mixed dtype errors during inference.
            # Force Float32 weights on CPU to match float32 waveform inputs.
            if device == -1:
                with suppress(Exception):
                    self._local_pipe.model.to(dtype=torch.float32)  # type: ignore[attr-defined]
                logger.info("MedASR local pipeline initialized on CPU with float32 weights.")
        return self._local_pipe

    def _get_local_ctc(self):
        if (
            self._local_processor is not None
            and self._local_model is not None
            and self._local_torch is not None
        ):
            return self._local_processor, self._local_model, self._local_torch, self._local_device

        with self._pipe_lock:
            if (
                self._local_processor is not None
                and self._local_model is not None
                and self._local_torch is not None
            ):
                return self._local_processor, self._local_model, self._local_torch, self._local_device

            try:
                with TRANSFORMERS_IMPORT_LOCK:
                    import torch
                    transformers_mod = importlib.import_module("transformers")
                    auto_processor_cls = getattr(transformers_mod, "AutoProcessor", None)
                    auto_model_ctc_cls = getattr(transformers_mod, "AutoModelForCTC", None)
                    if auto_processor_cls is None or auto_model_ctc_cls is None:
                        raise RuntimeError(
                            "AutoProcessor/AutoModelForCTC are unavailable in this transformers build."
                        )
            except Exception as exc:
                raise RuntimeError(
                    "Direct MedASR CTC decode initialization failed: "
                    f"{exc}"
                ) from exc

            device = "cpu"
            if self.local_device in {"cuda", "gpu"}:
                if not torch.cuda.is_available():
                    raise RuntimeError("MDT_MEDASR_LOCAL_DEVICE=cuda requested but CUDA is unavailable.")
                device = "cuda"
            elif self.local_device == "auto" and torch.cuda.is_available():
                device = "cuda"

            self._local_processor = auto_processor_cls.from_pretrained(self.local_model_id)
            self._local_model = auto_model_ctc_cls.from_pretrained(self.local_model_id)
            self._local_model = self._local_model.to(device)
            if device == "cpu":
                with suppress(Exception):
                    self._local_model = self._local_model.to(dtype=torch.float32)
            self._local_model.eval()
            self._local_torch = torch
            self._local_device = device

        return self._local_processor, self._local_model, self._local_torch, self._local_device

    def _transcribe_with_direct_ctc(self, local_audio_path: str) -> str:
        processor, model, torch, device = self._get_local_ctc()

        target_sr = 16000
        try:
            target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))  # type: ignore[attr-defined]
        except Exception:
            target_sr = 16000

        prepared = self._prepare_audio_input(local_audio_path, target_sampling_rate=target_sr)
        if not isinstance(prepared, dict):
            raise RuntimeError("Direct CTC decode currently supports WAV input.")
        raw = prepared.get("raw")
        sampling_rate = int(prepared.get("sampling_rate", target_sr))
        if raw is None:
            raise RuntimeError("Prepared audio did not contain waveform data.")

        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required for direct CTC MedASR decode.") from exc

        speech = np.asarray(raw, dtype=np.float32)
        if speech.ndim != 1 or speech.size == 0:
            raise RuntimeError("Invalid waveform shape for direct CTC MedASR decode.")

        chunk_samples = max(1, int(self.local_chunk_length_s * sampling_rate))
        stride_samples = int(self.local_stride_length_s * sampling_rate)
        step = max(1, chunk_samples - stride_samples)

        chunks: List[np.ndarray] = []
        if speech.size <= chunk_samples:
            chunks = [speech]
        else:
            start = 0
            while start < speech.size:
                end = min(speech.size, start + chunk_samples)
                chunks.append(speech[start:end])
                if end >= speech.size:
                    break
                start += step

        decoded_chunks: List[str] = []
        with self._infer_lock:
            for chunk in chunks:
                inputs = processor(
                    chunk,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(**inputs)
                text_list = processor.batch_decode(generated, skip_special_tokens=True)
                text = text_list[0] if text_list else ""
                cleaned = self._normalize_transcript_text(text)
                if cleaned:
                    decoded_chunks.append(cleaned)

        if not decoded_chunks:
            raise RuntimeError("Direct CTC MedASR decode returned no text.")
        return self._merge_decoded_chunks(decoded_chunks)

    @staticmethod
    def _merge_decoded_chunks(chunks: List[str]) -> str:
        if not chunks:
            return ""
        merged = chunks[0]
        for nxt in chunks[1:]:
            prev_words = merged.split()
            next_words = nxt.split()
            overlap = 0
            max_overlap = min(12, len(prev_words), len(next_words))
            for k in range(max_overlap, 0, -1):
                if prev_words[-k:] == next_words[:k]:
                    overlap = k
                    break
            merged = " ".join(prev_words + next_words[overlap:])
        return merged.strip()

    def _prepare_audio_input(
        self,
        local_audio_path: str,
        target_sampling_rate: Optional[int] = None,
    ) -> Any:
        """
        For WAV files, decode locally and pass raw waveform to avoid ffmpeg dependency.
        Non-WAV files are passed as paths and may require ffmpeg.
        """
        path = Path(local_audio_path)
        if path.suffix.lower() != ".wav":
            return local_audio_path

        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required for local WAV decoding.") from exc

        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            pcm_bytes = wav_file.readframes(frame_count)

        if sample_width == 1:
            waveform = (np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            waveform = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            waveform = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise RuntimeError(
                f"Unsupported WAV sample width: {sample_width} bytes. Use PCM WAV (8/16/32-bit)."
            )

        if channels > 1:
            # Keep float32 during downmix; np.mean defaults to float64.
            waveform = waveform.reshape(-1, channels).mean(axis=1, dtype=np.float32)

        effective_sr = sample_rate
        if (
            target_sampling_rate
            and target_sampling_rate > 0
            and sample_rate > 0
            and sample_rate != target_sampling_rate
            and waveform.size > 1
        ):
            duration = waveform.shape[0] / float(sample_rate)
            target_len = max(1, int(round(duration * target_sampling_rate)))
            source_x = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False, dtype=np.float64)
            target_x = np.linspace(0.0, duration, num=target_len, endpoint=False, dtype=np.float64)
            waveform = np.interp(target_x, source_x, waveform.astype(np.float64)).astype(np.float32)
            effective_sr = target_sampling_rate

        # Ensure ASR receives contiguous float32 data (avoid mixed dtype errors on CPU).
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        return {"raw": waveform, "sampling_rate": effective_sr}

    def _extract_transcript_from_obj(self, value: Any, depth: int = 0) -> str:
        if depth > 10:
            return ""
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else ""
        if isinstance(value, list):
            for item in value:
                extracted = self._extract_transcript_from_obj(item, depth + 1)
                if extracted:
                    return extracted
            return ""
        if not isinstance(value, dict):
            return ""

        preferred_keys = (
            "transcript",
            "text",
            "transcription",
            "output_text",
            "generated_text",
        )
        for key in preferred_keys:
            candidate = value.get(key)
            extracted = self._extract_transcript_from_obj(candidate, depth + 1)
            if extracted:
                return extracted

        # OpenAI/chat-style shape.
        content = value.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts)
        if isinstance(content, str) and content.strip():
            return content.strip()

        nested_priority_keys = (
            "predictions",
            "prediction",
            "choices",
            "candidates",
            "results",
            "result",
            "data",
            "response",
            "responses",
            "output",
            "outputs",
        )
        for key in nested_priority_keys:
            if key in value:
                extracted = self._extract_transcript_from_obj(value.get(key), depth + 1)
                if extracted:
                    return extracted

        for child in value.values():
            extracted = self._extract_transcript_from_obj(child, depth + 1)
            if extracted:
                return extracted
        return ""

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        # Remove common CTC/special tokens.
        cleaned = re.sub(r"</?s>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<epsilon>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<pad>", " ", cleaned, flags=re.IGNORECASE)
        # Remove any residual angle-bracket token.
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Collapse obvious duplicated words introduced by token merge artifacts.
        cleaned = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _truncate_transcript(self, text: str) -> str:
        cleaned = (text or "").strip()
        if len(cleaned) <= self.local_transcript_max_chars:
            return cleaned
        trimmed = cleaned[: self.local_transcript_max_chars]
        sentence_cut = max(trimmed.rfind(". "), trimmed.rfind("? "), trimmed.rfind("! "))
        if sentence_cut > int(self.local_transcript_max_chars * 0.6):
            trimmed = trimmed[: sentence_cut + 1]
        return trimmed.strip()

    @staticmethod
    def _destutter_token(token: str) -> str:
        value = (token or "").strip()
        if not value:
            return ""

        value = re.sub(r"[-_]{2,}", "-", value)
        # Collapse long repeated characters first, e.g. "EEEXAAMM" -> "EXAM".
        value = re.sub(r"([A-Za-z])\1{2,}", r"\1", value)
        # Collapse repeated short fragments, e.g. "chchestest" -> "chest".
        for n in (4, 3, 2):
            pattern = re.compile(rf"([A-Za-z]{{{n}}})\1+")
            while True:
                reduced = pattern.sub(r"\1", value)
                if reduced == value:
                    break
                value = reduced
        # LASR noisy outputs often duplicate each character once (e.g. "prottoccol").
        # For longer tokens this is usually artifact rather than a true spelling.
        if len(value) >= 6:
            value = re.sub(r"([A-Za-z])\1", r"\1", value)

        value = value.strip("-_")
        if re.fullmatch(r"([A-Za-z])\1+", value):
            value = value[:1]
        return value

    def _repair_corrupted_transcript(self, text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""

        # Normalize punctuation placeholders emitted by LASR token streams.
        placeholder_replacements = {
            r"\{\s*\{period(?:period)*\}\s*\}": ". ",
            r"\{\s*\{comma(?:comma)*\}\s*\}": ", ",
            r"\{\s*\{colon(?:colon)*\}\s*\}": ": ",
            r"\{\s*\{semicolon(?:semicolon)*\}\s*\}": "; ",
            r"\{\s*\{question(?:question)*\}\s*\}": "? ",
            r"\{\s*\{exclamation(?:exclamation)*\}\s*\}": "! ",
            r"\{\s*\{new\s*paragraph(?:new\s*paragraph)*\}\s*\}": ". ",
        }
        for pattern, replacement in placeholder_replacements.items():
            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

        value = re.sub(r"\[\s*\[", " ", value)
        value = re.sub(r"\]\s*\]", " ", value)
        value = re.sub(r"\{\s*\{", " ", value)
        value = re.sub(r"\}\s*\}", " ", value)

        token_pattern = re.compile(r"[A-Za-z0-9'+-]+|[.,;:!?]")
        tokens = token_pattern.findall(value)
        if not tokens:
            return ""

        repaired: List[str] = []
        prev_word = ""
        for idx, token in enumerate(tokens):
            if re.fullmatch(r"[.,;:!?]", token):
                if repaired and repaired[-1] in {".", ",", ";", ":", "?", "!"}:
                    repaired[-1] = token
                else:
                    repaired.append(token)
                prev_word = ""
                continue

            candidate = self._destutter_token(token)
            if not candidate:
                continue
            if len(candidate) == 1 and idx + 1 < len(tokens):
                nxt = tokens[idx + 1]
                if len(nxt) > 2 and nxt[0].lower() == candidate.lower():
                    continue
            if prev_word and candidate.lower() == prev_word.lower():
                continue
            repaired.append(candidate)
            prev_word = candidate

        merged = " ".join(repaired)
        merged = re.sub(r"\s+([.,;:!?])", r"\1", merged)
        merged = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", merged)
        merged = re.sub(r"\s+", " ", merged).strip()
        return merged

    @staticmethod
    def _looks_corrupted_transcript(text: str) -> bool:
        if not text:
            return True
        lowered = text.lower()
        # If special tokens survived normalization, transcription is unreliable.
        if "<epsilon>" in lowered or "<" in text or ">" in text:
            return True
        # Heavy character stutter patterns usually indicate bad CTC decode output.
        stutter_count = len(re.findall(r"(\b\w{2,})\1{1,}\b", lowered))
        if stutter_count >= 3:
            return True
        # Too little lexical diversity also indicates decode corruption.
        words = re.findall(r"[a-z0-9]+", lowered)
        if words:
            one_char_ratio = sum(1 for w in words if len(w) == 1) / float(len(words))
            if one_char_ratio > 0.34:
                return True
            short_ratio = sum(1 for w in words if len(w) <= 2) / float(len(words))
            if len(words) >= 30 and short_ratio > 0.52:
                return True
            one_char_runs = len(re.findall(r"(?:\b[a-z0-9]\b\s+){3,}\b[a-z0-9]\b", lowered))
            if one_char_runs >= 1:
                return True
        if len(words) >= 20:
            unique_ratio = len(set(words)) / float(len(words))
            if unique_ratio < 0.18:
                return True
        return False

    def _extract_wer_from_obj(self, value: Any, depth: int = 0) -> Optional[float]:
        if depth > 10:
            return None
        if isinstance(value, dict):
            for key in ("wer", "wer_estimate"):
                raw = value.get(key)
                if isinstance(raw, (float, int)):
                    return max(0.0, min(1.0, float(raw)))
                if isinstance(raw, str):
                    try:
                        parsed = float(raw.strip())
                    except ValueError:
                        parsed = None
                    if parsed is not None:
                        return max(0.0, min(1.0, parsed))
            for child in value.values():
                parsed = self._extract_wer_from_obj(child, depth + 1)
                if parsed is not None:
                    return parsed
            return None
        if isinstance(value, list):
            for item in value:
                parsed = self._extract_wer_from_obj(item, depth + 1)
                if parsed is not None:
                    return parsed
        return None


class MedGemmaEndpointAdapter:
    """
    Local MedGemma runtime adapter (offline-only).
    """

    def __init__(self, project_id: Optional[str], location: str) -> None:
        # kept for constructor compatibility; offline adapter does not use GCP routing
        _ = project_id
        _ = location
        self.model_id = _resolve_local_model_ref(
            (
            os.getenv("MDT_MEDGEMMA_LOCAL_MODEL_ID")
            or os.getenv("MDT_MEDGEMMA_MODEL_NAME")
            or "google/medgemma-4b-it"
            ).strip()
        )
        self.local_files_only = (
            (os.getenv("MDT_MEDGEMMA_LOCAL_FILES_ONLY", "true") or "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.device_pref = (os.getenv("MDT_MEDGEMMA_LOCAL_DEVICE", "auto") or "auto").strip().lower()
        self.timeout_seconds = max(5.0, _env_float("MDT_MEDGEMMA_TIMEOUT_SECONDS", 45.0))
        self.max_output_tokens = max(64, _env_int("MDT_MEDGEMMA_MAX_OUTPUT_TOKENS", 220))
        self.temperature = min(1.0, max(0.0, _env_float("MDT_MEDGEMMA_TEMPERATURE", 0.0)))
        self.max_input_chars = max(2000, _env_int("MDT_MEDGEMMA_MAX_INPUT_CHARS", 22000))
        self.strict_json_retries = max(
            0,
            min(2, _env_int("MDT_MEDGEMMA_STRICT_JSON_RETRIES", 1)),
        )
        self._processor = None
        self._model = None
        self._torch = None
        self._model_lock = Lock()
        self._infer_lock = Lock()

    @staticmethod
    def is_medgemma_model(model_name: str) -> bool:
        lowered = (model_name or "").strip().lower()
        return "medgemma" in lowered

    @property
    def configured(self) -> bool:
        return bool(self.model_id)

    def _resolve_torch_dtype(self, torch_module):
        dtype_name = (os.getenv("MDT_MEDGEMMA_LOCAL_DTYPE", "auto") or "auto").strip().lower()
        if dtype_name == "auto":
            return "auto"
        if dtype_name in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if dtype_name in {"float16", "fp16"}:
            return torch_module.float16
        if dtype_name in {"float32", "fp32"}:
            return torch_module.float32
        return "auto"

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        with self._model_lock:
            if self._model is not None and self._processor is not None and self._torch is not None:
                return

            try:
                with TRANSFORMERS_IMPORT_LOCK:
                    import torch
                    transformers_mod = importlib.import_module("transformers")
            except Exception as exc:
                raise RuntimeError(
                    "Local MedGemma import failed. Ensure compatible versions of "
                    f"transformers/torch/accelerate are installed. Detail: {exc}"
                ) from exc

            auto_processor_cls = getattr(transformers_mod, "AutoProcessor", None)
            model_cls = getattr(transformers_mod, "AutoModelForImageTextToText", None)
            if model_cls is None:
                model_cls = getattr(transformers_mod, "AutoModelForVision2Seq", None)
            if auto_processor_cls is None or model_cls is None:
                raise RuntimeError(
                    "Installed transformers build is missing MedGemma auto classes. "
                    "Expected AutoProcessor + (AutoModelForImageTextToText or AutoModelForVision2Seq)."
                )

            if self.device_pref == "cpu":
                device = "cpu"
            elif self.device_pref in {"cuda", "gpu"}:
                if not torch.cuda.is_available():
                    raise RuntimeError("MDT_MEDGEMMA_LOCAL_DEVICE=cuda requested but CUDA is unavailable.")
                device = "cuda"
            elif self.device_pref == "mps":
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MDT_MEDGEMMA_LOCAL_DEVICE=mps requested but MPS is unavailable.")
                device = "mps"
            else:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            model_kwargs: Dict[str, Any] = {
                "local_files_only": self.local_files_only,
                # Avoid meta-tensor/offload initialization path that can break later .to(...)
                # in some local runtime combinations.
                "low_cpu_mem_usage": False,
            }
            resolved_dtype = self._resolve_torch_dtype(torch)
            if resolved_dtype != "auto":
                model_kwargs["torch_dtype"] = resolved_dtype
            elif device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32

            self._processor = auto_processor_cls.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            self._model = model_cls.from_pretrained(
                self.model_id,
                **model_kwargs,
            )

            if any(getattr(param, "is_meta", False) for param in self._model.parameters()):
                raise RuntimeError(
                    "Local MedGemma loaded with meta tensors (weights not materialized). "
                    "Ensure the full model snapshot exists locally and retry."
                )

            try:
                self._model = self._model.to(device)
            except RuntimeError as exc:
                if "meta tensor" in str(exc).lower():
                    raise RuntimeError(
                        "Local MedGemma device move failed due to meta tensors. "
                        "This usually indicates incomplete local model files or an incompatible "
                        "torch/transformers runtime. Re-download model files and retry "
                        "with MDT_MEDGEMMA_LOCAL_DEVICE=cpu."
                    ) from exc
                raise
            self._model.eval()
            self._torch = torch

    def generate_json(
        self,
        *,
        agent_name: str,
        model_name: str,
        instruction: str,
        payload: Dict[str, Any],
        expected_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        _ = model_name  # kept for signature compatibility
        if not self.configured:
            raise RuntimeError("MDT_MEDGEMMA_LOCAL_MODEL_ID is not configured.")

        self._ensure_model()
        processor = self._processor
        model = self._model
        torch = self._torch

        payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if len(payload_json) > self.max_input_chars:
            compact_payload: Dict[str, Any] = {
                "case_id": payload.get("case_id"),
                "diagnosis": payload.get("diagnosis"),
                "input_truncated": True,
            }
            for key in ("stage_one", "consensus", "diagnosticore"):
                value = payload.get(key)
                if value is not None:
                    compact_payload[key] = value
            payload_json = json.dumps(compact_payload, ensure_ascii=True, separators=(",", ":"))
            if len(payload_json) > self.max_input_chars:
                payload_json = json.dumps(
                    {
                        "case_id": payload.get("case_id"),
                        "diagnosis": payload.get("diagnosis"),
                        "input_truncated": True,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )

        prompt = self._build_generation_prompt(
            agent_name=agent_name,
            instruction=instruction,
            payload_json=payload_json,
            expected_keys=expected_keys,
        )
        text = self._run_generation(
            processor=processor,
            model=model,
            torch=torch,
            agent_name=agent_name,
            user_prompt=prompt,
        )

        best_parsed = self._try_parse_json_object(text, expected_keys=expected_keys)
        if isinstance(best_parsed, dict):
            confidence_hint = self._extract_confidence_token(text)
            if confidence_hint is not None and "confidence" not in best_parsed:
                best_parsed["confidence"] = confidence_hint
        if self._has_required_keys(best_parsed, expected_keys):
            return best_parsed or {}

        # One repair pass significantly improves non-strict JSON outputs while
        # preserving deterministic generation behavior.
        repair_input = text
        for _ in range(self.strict_json_retries):
            repair_prompt = self._build_repair_prompt(
                agent_name=agent_name,
                instruction=instruction,
                payload_json=payload_json,
                expected_keys=expected_keys,
                raw_output=repair_input,
            )
            repaired_text = self._run_generation(
                processor=processor,
                model=model,
                torch=torch,
                agent_name=agent_name,
                user_prompt=repair_prompt,
            )
            repaired = self._try_parse_json_object(repaired_text, expected_keys=expected_keys)
            if isinstance(repaired, dict):
                confidence_hint = self._extract_confidence_token(repaired_text)
                if confidence_hint is not None and "confidence" not in repaired:
                    repaired["confidence"] = confidence_hint
            if self._has_required_keys(repaired, expected_keys):
                return repaired or {}
            if repaired is not None:
                best_parsed = repaired
            repair_input = repaired_text

        if best_parsed is not None:
            missing = self._missing_required_keys(best_parsed, expected_keys)
            if missing:
                logger.debug(
                    "Local MedGemma JSON missing keys after repair (agent=%s, missing=%s); applying field recovery.",
                    agent_name,
                    ",".join(missing),
                )
            recovered = self._recover_expected_fields(repair_input, expected_keys=expected_keys)
            if recovered:
                merged = dict(best_parsed)
                merged.update(recovered)
                if self._has_required_keys(merged, expected_keys):
                    return merged
                best_parsed = merged
            return best_parsed

        recovered = self._recover_expected_fields(repair_input, expected_keys=expected_keys)
        if recovered is not None:
            logger.debug(
                "Local MedGemma returned non-strict JSON; recovered partial payload with keys=%s.",
                sorted(recovered.keys()),
            )
            return recovered
        preview = (repair_input or "").strip()[:400]
        logger.warning(
            "Local MedGemma output was not valid JSON; returning empty payload. Preview: %s",
            preview,
        )
        return {}

    def _run_generation(
        self,
        *,
        processor: Any,
        model: Any,
        torch: Any,
        agent_name: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a medical AI assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]
        # HF generation is not reliably thread-safe on one shared model instance.
        # Serialize inference calls to avoid race conditions.
        with self._infer_lock:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens_for_agent(agent_name),
                    max_time=self.timeout_seconds,
                    # Deterministic decode improves strict JSON reliability.
                    do_sample=False,
                    temperature=None,
                )
            output_ids = generated[0][input_len:]
            return processor.decode(output_ids, skip_special_tokens=True)

    @staticmethod
    def _missing_required_keys(payload: Optional[Dict[str, Any]], expected_keys: Optional[List[str]]) -> List[str]:
        if not expected_keys:
            return []
        if not isinstance(payload, dict):
            return list(expected_keys)
        return [k for k in expected_keys if k not in payload]

    def _has_required_keys(self, payload: Optional[Dict[str, Any]], expected_keys: Optional[List[str]]) -> bool:
        return len(self._missing_required_keys(payload, expected_keys)) == 0

    @staticmethod
    def _build_schema_example(expected_keys: Optional[List[str]]) -> Optional[str]:
        if not expected_keys:
            return None
        list_like = {
            "action_items",
            "biomarkers",
            "risk_features",
            "actionable_mutations",
            "evidence",
            "red_flags",
            "key_risks",
            "recommended_actions",
            "confirmatory_actions",
            "evidence_links",
        }
        schema: Dict[str, Any] = {}
        for key in expected_keys:
            if key in list_like:
                schema[key] = []
            elif key == "confidence":
                schema[key] = None
            elif key in {"year"}:
                schema[key] = 2024
            else:
                schema[key] = ""
        return json.dumps(schema, ensure_ascii=True, separators=(",", ":"))

    def _build_generation_prompt(
        self,
        *,
        agent_name: str,
        instruction: str,
        payload_json: str,
        expected_keys: Optional[List[str]],
    ) -> str:
        schema_example = self._build_schema_example(expected_keys)
        strict_clause = "Return only one strict JSON object."
        if expected_keys:
            strict_clause += (
                " Required keys (all must be present): "
                + ", ".join(expected_keys)
                + "."
            )
        if schema_example:
            strict_clause += f" JSON schema example: {schema_example}"
        return (
            f"Agent:{agent_name}\n"
            f"Task:{instruction.strip()}\n"
            f"Input:{payload_json}\n"
            f"{strict_clause}\n"
            "If unsure, keep values conservative but valid JSON. No markdown. No commentary."
        )

    def _build_repair_prompt(
        self,
        *,
        agent_name: str,
        instruction: str,
        payload_json: str,
        expected_keys: Optional[List[str]],
        raw_output: str,
    ) -> str:
        schema_example = self._build_schema_example(expected_keys)
        required = ", ".join(expected_keys or [])
        return (
            f"Agent:{agent_name}\n"
            f"Original task:{instruction.strip()}\n"
            f"Original input:{payload_json}\n"
            f"Previous invalid output:{json.dumps(raw_output, ensure_ascii=True)}\n"
            "Rewrite the previous output as strict JSON only.\n"
            f"Required keys: {required if required else 'none specified'}.\n"
            f"Schema example: {schema_example or '{}'}\n"
            "Do not add markdown fences. Do not add commentary."
        )

    @staticmethod
    def _extract_confidence_token(text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(
            r"\bconfidence(?:_score)?\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?%?)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*%?\s*confidence\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _max_tokens_for_agent(self, agent_name: str) -> int:
        # Keep local generation bounded per stage to reduce long-running stalls.
        per_agent_caps = {
            "RadiologySynthesizer": 128,
            "PathologySynthesizer": 128,
            "GenomicsSynthesizer": 128,
            "LiteratureAgent": 160,
            "ConsensusSynthesizer": 220,
            "SOAPGenerator": 280,
            "ClinicalReasoner": 240,
        }
        return max(64, min(self.max_output_tokens, per_agent_caps.get(agent_name, self.max_output_tokens)))

    def _try_parse_json_object(
        self,
        text: str,
        expected_keys: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        candidates: List[str] = []
        cleaned = self._strip_markdown_fences(text)
        if cleaned:
            candidates.append(cleaned)
            if cleaned.startswith("Prompt:"):
                prompt_trimmed = cleaned.split("INPUT_JSON:", 1)
                if len(prompt_trimmed) == 2:
                    candidates.append(prompt_trimmed[1].strip())

        start = cleaned.find("{")
        if start != -1:
            object_tail = cleaned[start:].strip()
            candidates.append(object_tail)
            repaired = self._repair_truncated_json(object_tail)
            if repaired and repaired != object_tail:
                candidates.append(repaired)

        parsed_candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        decoder = json.JSONDecoder()

        def _add_candidate(obj: Dict[str, Any]) -> None:
            try:
                key = json.dumps(obj, sort_keys=True)
            except Exception:
                key = str(obj)
            if key in seen:
                return
            seen.add(key)
            parsed_candidates.append(obj)

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                _add_candidate(parsed)
            parsed_literal = self._safe_literal_eval_dict(candidate)
            if isinstance(parsed_literal, dict):
                _add_candidate(parsed_literal)
            parsed_jsonish = self._try_parse_jsonish_candidate(candidate)
            if isinstance(parsed_jsonish, dict):
                _add_candidate(parsed_jsonish)

            # Also scan for embedded JSON objects within mixed text.
            idx = 0
            while True:
                brace_idx = candidate.find("{", idx)
                if brace_idx == -1:
                    break
                try:
                    embedded, end_idx = decoder.raw_decode(candidate, brace_idx)
                except Exception:
                    idx = brace_idx + 1
                    continue
                if isinstance(embedded, dict):
                    _add_candidate(embedded)
                idx = max(brace_idx + 1, end_idx)

        if expected_keys:
            expected = set(expected_keys)
            for obj in parsed_candidates:
                if expected.issubset(set(obj.keys())):
                    return obj

        return parsed_candidates[0] if parsed_candidates else None

    @staticmethod
    def _safe_literal_eval_dict(candidate: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _try_parse_jsonish_candidate(self, candidate: str) -> Optional[Dict[str, Any]]:
        text = (candidate or "").strip()
        if not text or "{" not in text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        body = text[start : end + 1]
        # Quote unquoted keys: {key: ...} -> {"key": ...}
        body = re.sub(
            r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\- ]*)(\s*:)',
            lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}',
            body,
        )
        body = re.sub(r",(\s*[}\]])", r"\1", body)
        body = re.sub(r"\bTrue\b", "true", body)
        body = re.sub(r"\bFalse\b", "false", body)
        body = re.sub(r"\bNone\b", "null", body)
        parsed = self._safe_json_loads(body)
        if isinstance(parsed, dict):
            return parsed
        # Fallback: normalize single-quote payloads.
        if "'" in body:
            alt = body.replace("'", '"')
            parsed_alt = self._safe_json_loads(alt)
            if isinstance(parsed_alt, dict):
                return parsed_alt
        return None

    def _recover_expected_fields(
        self,
        text: str,
        expected_keys: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not text or not expected_keys:
            return None

        cleaned = self._strip_markdown_fences(text)
        recovered: Dict[str, Any] = {}
        for key in expected_keys:
            value = self._extract_key_value(cleaned, key)
            if value is not None:
                recovered[key] = value
        # Confidence is often present even when overall JSON is malformed.
        # Recover it from loose text patterns to avoid unnecessary fallback scores.
        if "confidence" in expected_keys and "confidence" not in recovered:
            match = re.search(
                r"\bconfidence(?:_score)?\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?%?)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if match:
                recovered["confidence"] = match.group(1)
        return recovered if recovered else None

    def _extract_key_value(self, text: str, key: str) -> Optional[Any]:
        key_pattern = re.compile(
            rf'(?:'
            rf'"{re.escape(key)}"'
            rf"|'{re.escape(key)}'"
            rf"|\b{re.escape(key)}\b"
            rf')\s*[:=]\s*',
            flags=re.IGNORECASE,
        )
        match = key_pattern.search(text)
        if not match:
            return None

        value_start = match.end()
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        if value_start >= len(text):
            return None

        ch = text[value_start]
        if ch == '"':
            string_value, _ = self._extract_json_string(text, value_start)
            return string_value
        if ch in "{[":
            segment = self._extract_balanced_segment(text, value_start)
            if not segment:
                return None
            parsed = self._safe_json_loads(segment)
            return parsed

        # primitives (number, bool, null) or unquoted token
        primitive_match = re.match(
            r'(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?%?|true|false|null)',
            text[value_start:],
            flags=re.IGNORECASE,
        )
        if primitive_match:
            raw = primitive_match.group(1)
            # Keep percentages as strings so downstream confidence normalization can handle them.
            if raw.endswith("%"):
                return raw
            parsed = self._safe_json_loads(raw.lower())
            if parsed is not None:
                return parsed

        end_idx = value_start
        while end_idx < len(text) and text[end_idx] not in ",}\n":
            end_idx += 1
        token = text[value_start:end_idx].strip()
        return token or None

    @staticmethod
    def _extract_json_string(text: str, start_idx: int) -> tuple[Optional[str], int]:
        # start_idx points at the opening quote.
        idx = start_idx + 1
        escaped = False
        buf: List[str] = []
        while idx < len(text):
            ch = text[idx]
            if escaped:
                buf.append(ch)
                escaped = False
                idx += 1
                continue
            if ch == "\\":
                buf.append(ch)
                escaped = True
                idx += 1
                continue
            if ch == '"':
                raw = "".join(buf)
                try:
                    return json.loads(f'"{raw}"'), idx + 1
                except Exception:
                    return raw, idx + 1
            buf.append(ch)
            idx += 1

        raw = "".join(buf)
        try:
            return json.loads(f'"{raw}"'), idx
        except Exception:
            return raw, idx

    @staticmethod
    def _extract_balanced_segment(text: str, start_idx: int) -> Optional[str]:
        if start_idx >= len(text):
            return None
        open_ch = text[start_idx]
        if open_ch not in "{[":
            return None

        close_for = {"{": "}", "[": "]"}
        stack: List[str] = [close_for[open_ch]]
        in_string = False
        escaped = False
        idx = start_idx + 1

        while idx < len(text):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                idx += 1
                continue

            if ch == '"':
                in_string = True
                idx += 1
                continue
            if ch in "{[":
                stack.append(close_for[ch])
                idx += 1
                continue
            if ch in "}]":
                if not stack:
                    idx += 1
                    continue
                expected = stack.pop()
                if ch != expected:
                    idx += 1
                    continue
                if not stack:
                    return text[start_idx : idx + 1]
            idx += 1

        return None

    @staticmethod
    def _safe_json_loads(candidate: str) -> Optional[Any]:
        try:
            return json.loads(candidate)
        except Exception:
            return None

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned.startswith("```"):
            return cleaned
        cleaned = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        return cleaned.strip()

    def _repair_truncated_json(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned
        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            if start == -1:
                return cleaned
            cleaned = cleaned[start:]

        in_string = False
        escaped = False
        stack: List[str] = []
        close_for = {"{": "}", "[": "]"}
        end_at: Optional[int] = None

        for idx, ch in enumerate(cleaned):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(close_for[ch])
                continue
            if ch in "}]":
                if stack:
                    expected = stack.pop()
                    if ch != expected:
                        # malformed close; keep scanning and let parser decide
                        continue
                if not stack:
                    end_at = idx + 1
                    break

        if end_at is not None:
            return cleaned[:end_at]

        repaired = cleaned
        if in_string:
            repaired += '"'
        while stack:
            repaired += stack.pop()
        return repaired


class MDTCommandOrchestrator:
    """
    Deterministic orchestrator coordinating the MDT agent pipeline.
    """

    def __init__(
        self,
        execution_mode: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        requested_execution_mode = (
            execution_mode or os.getenv("MDT_EXECUTION_MODE", "local")
        ).strip().lower()
        if requested_execution_mode not in {"local", "adk_local"}:
            raise ValueError(
                "Offline build supports MDT_EXECUTION_MODE in {'local','adk_local'}. "
                f"Received {requested_execution_mode!r}."
            )
        self.execution_mode = requested_execution_mode
        if self.execution_mode == "adk_local" and not ADK_AVAILABLE:
            raise RuntimeError(
                "MDT_EXECUTION_MODE=adk_local requested, but ADK dependencies are missing. "
                "Install `google-adk` and `litellm`, and ensure Ollama is running locally."
            )
        self.model_name = model_name or os.getenv("MDT_MODEL_NAME", "google/medgemma-4b-it")
        self.medgemma_model_name = os.getenv("MDT_MEDGEMMA_MODEL_NAME", "google/medgemma-4b-it")
        self.model_router_mode = (
            os.getenv("MDT_MODEL_ROUTER_MODE", "medgemma_only").strip().lower() or "medgemma_only"
        )
        if self.model_router_mode != "medgemma_only":
            raise ValueError(
                "Offline build supports only MDT_MODEL_ROUTER_MODE=medgemma_only."
            )
        self.low_confidence_threshold = min(
            0.99, max(0.0, _env_float("MDT_CONSENSUS_LOW_CONFIDENCE_THRESHOLD", 0.78))
        )
        self.min_evidence_items = max(1, _env_int("MDT_MIN_EVIDENCE_ITEMS", 2))
        self.red_flag_uncertainty_threshold = max(
            1, _env_int("MDT_RED_FLAG_UNCERTAINTY_THRESHOLD", 2)
        )
        self.literature_max_results = max(
            1, min(12, _env_int("MDT_LITERATURE_MAX_RESULTS", 8))
        )
        self.local_data_dir = Path(
            (os.getenv("MDT_LOCAL_DATA_DIR", "./local_data") or "./local_data").strip()
        ).expanduser().resolve()
        self.sqlite_db_path = (
            os.getenv("MDT_SQLITE_DB_PATH") or str(self.local_data_dir / "mdt_cases.sqlite3")
        ).strip()
        self.case_store_backend = (
            os.getenv("MDT_CASE_STORE_BACKEND", "sqlite").strip().lower() or "sqlite"
        )
        self.retrieval_mode = (os.getenv("MDT_RETRIEVAL_MODE", "local") or "local").strip().lower()
        if self.case_store_backend != "sqlite":
            raise ValueError(
                "Offline build supports only MDT_CASE_STORE_BACKEND=sqlite."
            )
        if self.retrieval_mode != "local":
            raise ValueError(
                "Offline build supports only MDT_RETRIEVAL_MODE=local."
            )
        self.case_repository: CaseRepository = self._build_case_repository()
        self.lock = asyncio.Lock()
        self.transcription_adapter = MedASRAdapter()
        self.diagnosticore_client = DiagnosticoreClient()
        self.clinical_reasoner = ClinicalReasoner()
        self.medgemma_endpoint = MedGemmaEndpointAdapter(
            project_id=None,
            location="local",
        )
        self.agent_call_timeout_seconds = max(
            5.0,
            _env_float("MDT_AGENT_CALL_TIMEOUT_SECONDS", 120.0),
        )
        self.agent_call_timeout_cap_seconds = max(
            10.0,
            _env_float("MDT_AGENT_CALL_TIMEOUT_CAP_SECONDS", 180.0),
        )
        self.disable_medgemma_on_timeout = (
            (os.getenv("MDT_DISABLE_MEDGEMMA_ON_TIMEOUT", "false") or "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._medgemma_disabled_reason: Optional[str] = None
        self.adk_model_provider = (
            os.getenv("MDT_ADK_MODEL_PROVIDER", "ollama_chat").strip() or "ollama_chat"
        )
        self.adk_model_name = (
            os.getenv("MDT_ADK_MODEL_NAME", "qwen2.5:7b-instruct").strip() or "qwen2.5:7b-instruct"
        )
        if "/" in self.adk_model_name:
            self.adk_model_ref = self.adk_model_name
        else:
            self.adk_model_ref = f"{self.adk_model_provider}/{self.adk_model_name}"
        self.session_service = None
        self.adk_model = None
        if self.execution_mode == "adk_local":
            self.session_service = InMemorySessionService()
            self.adk_model = LiteLlm(model=self.adk_model_ref)

        logger.info(
            "MDTCommandOrchestrator initialized | mode=%s | adk_available=%s | adk_model=%s | "
            "model=%s | medgemma=%s | router=%s | "
            "low_conf_threshold=%.2f | literature_max_results=%d | store=%s | retrieval=%s | medasr_mode=%s | diagnosticore_mode=%s | "
            "clinical_reasoner_mode=%s | agent_call_timeout=%.1fs",
            self.execution_mode,
            ADK_AVAILABLE,
            self.adk_model_ref if self.execution_mode == "adk_local" else "disabled",
            self.model_name,
            self.medgemma_model_name,
            self.model_router_mode,
            self.low_confidence_threshold,
            self.literature_max_results,
            self.case_store_backend,
            self.retrieval_mode,
            self.transcription_adapter.mode,
            self.diagnosticore_client.mode,
            self.clinical_reasoner.mode,
            self.agent_call_timeout_seconds,
        )

    async def _generate_medgemma_json(
        self,
        *,
        agent_name: str,
        model_name: str,
        instruction: str,
        payload: Dict[str, Any],
        expected_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self._medgemma_disabled_reason:
            logger.warning(
                "Skipping MedGemma call for %s because runtime is disabled: %s",
                agent_name,
                self._medgemma_disabled_reason,
            )
            return {}
        timeout_seconds = min(self.agent_call_timeout_seconds, self.agent_call_timeout_cap_seconds)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.medgemma_endpoint.generate_json,
                    agent_name=agent_name,
                    model_name=model_name,
                    instruction=instruction,
                    payload=payload,
                    expected_keys=expected_keys,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            reason = (
                f"Timed out after {timeout_seconds:.1f}s "
                f"during {agent_name} generation."
            )
            logger.error("Local MedGemma timeout: %s", reason)
            if self.disable_medgemma_on_timeout:
                self._medgemma_disabled_reason = reason
            return {}
        except Exception as exc:
            logger.warning("Local MedGemma call failed for %s: %s", agent_name, exc)
            return {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_case(
        self,
        case_id: Optional[str] = None,
        overrides: Optional[Dict[str, object]] = None,
    ) -> MDTCaseRecord:
        """
        Initialize case in CREATED state from local case dataset.
        """
        resolved_case_id = case_id or self._default_case_id()
        case_input = get_case_input(resolved_case_id)
        case_input = self._apply_input_overrides(case_input, overrides or {})
        autofetch_notes: Optional[str] = None

        if case_input.diagnosticore is None:
            case_key = case_input.diagnosticore_case_submitter_id or case_input.case_id
            fetched = self.diagnosticore_client.fetch_prediction(case_key)
            if fetched is not None:
                case_input.diagnosticore = fetched
                autofetch_notes = f"Auto-fetched DiagnostiCore payload using key={case_key}."

        record = MDTCaseRecord(
            case_id=case_input.case_id,
            patient_id=case_input.patient_id,
            patient_name=case_input.patient_name,
            diagnosis=case_input.diagnosis,
            status=CaseStatus.CREATED,
            input_payload=case_input,
        )
        record.artifacts.diagnosticore = case_input.diagnosticore
        if autofetch_notes:
            now = datetime.now(timezone.utc)
            self._append_trace(
                record,
                "DiagnosticoreHandoff",
                AgentRunStatus.COMPLETED,
                now,
                now,
                autofetch_notes,
            )
        return self.case_repository.save_case(record)

    async def analyze_case(self, case_id: str) -> MDTCaseRecord:
        """
        Runs full MDT orchestration:
          Stage 1 (parallel) + concurrent transcription
          Stage 2 (sequential) consensus -> SOAP
          Stage 3 HITL gate
        """
        record = self._get_case_or_raise(case_id)
        # Reset per-run MedGemma disable state; if a timeout happens during this run,
        # we degrade to baseline for remaining stages instead of cascading timeouts.
        self._medgemma_disabled_reason = None

        async with self.lock:
            record.status = CaseStatus.ANALYZING
            record.updated_at = datetime.now(timezone.utc)
            self.case_repository.save_case(record)

        try:
            # Concurrent transcription task.
            transcription_task = asyncio.create_task(
                self._run_transcription_agent(record, record.input_payload)
            )
            transcription_persist_task = asyncio.create_task(
                self._persist_transcription_on_ready(record, transcription_task)
            )

            stage_one = await self._run_stage_one(record, record.input_payload)
            consensus = await self._run_consensus_synthesizer(record, record.input_payload, stage_one)
            soap = await self._run_soap_generator(record, record.input_payload, stage_one, consensus)
            gate = await self._run_hitl_gatekeeper(record, record.input_payload, consensus, soap)
            clinical_reasoning = await self._run_clinical_reasoner(
                record=record,
                case_input=record.input_payload,
                stage_one=stage_one,
                consensus=consensus,
                soap=soap,
                gate=gate,
            )

            transcription = await transcription_task
            await transcription_persist_task

            async with self.lock:
                record.artifacts.stage_one = stage_one
                record.artifacts.consensus = consensus
                record.artifacts.soap_note = soap
                record.artifacts.clinical_reasoning = clinical_reasoning
                record.artifacts.hitl_gate = gate
                record.artifacts.diagnosticore = record.input_payload.diagnosticore
                if record.artifacts.transcription is None:
                    record.artifacts.transcription = transcription
                record.status = CaseStatus.PENDING_APPROVAL
                record.updated_at = datetime.now(timezone.utc)
                self.case_repository.save_case(record)

            return record
        except asyncio.CancelledError:
            if "transcription_persist_task" in locals() and not transcription_persist_task.done():
                transcription_persist_task.cancel()
                with suppress(asyncio.CancelledError):
                    await transcription_persist_task
            logger.info("MDT analysis cancelled for case %s.", case_id)
            async with self.lock:
                record.status = CaseStatus.ERROR
                record.updated_at = datetime.now(timezone.utc)
                self._append_trace(
                    record=record,
                    agent_name="MDTOrchestrator",
                    status=AgentRunStatus.FAILED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    notes="Pipeline cancelled (service shutdown or client disconnect).",
                )
                self.case_repository.save_case(record)
            raise
        except Exception as exc:
            if "transcription_persist_task" in locals() and not transcription_persist_task.done():
                transcription_persist_task.cancel()
                with suppress(asyncio.CancelledError):
                    await transcription_persist_task
            logger.exception("MDT analysis failed for case %s: %s", case_id, exc)
            async with self.lock:
                record.status = CaseStatus.ERROR
                record.updated_at = datetime.now(timezone.utc)
                self._append_trace(
                    record=record,
                    agent_name="MDTOrchestrator",
                    status=AgentRunStatus.FAILED,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    notes=f"Pipeline failure: {exc}",
                )
                self.case_repository.save_case(record)
            raise

    def _model_for_agent(self, agent_name: str) -> str:
        _ = agent_name
        return self.medgemma_model_name

    def _agent_uses_medgemma_endpoint(self, agent_name: str) -> bool:
        _ = agent_name
        return True

    def approve_case(
        self,
        case_id: str,
        decision: ApprovalDecision,
        clinician_name: str,
        notes: Optional[str] = None,
    ) -> MDTCaseRecord:
        record = self._get_case_or_raise(case_id)

        if record.status not in {CaseStatus.PENDING_APPROVAL, CaseStatus.REWORK_REQUIRED}:
            raise ValueError(f"Case {case_id} is not awaiting approval.")

        now = datetime.now(timezone.utc)
        if decision == ApprovalDecision.APPROVE:
            record.status = CaseStatus.APPROVED
            record.approved_by = clinician_name
            record.approved_at = now
            record.approval_notes = notes
            record.rework_reason = None
        else:
            record.status = CaseStatus.REWORK_REQUIRED
            record.approved_by = clinician_name
            record.approved_at = None
            record.approval_notes = notes
            record.rework_reason = notes or "Clinician requested revision."

        record.updated_at = now
        return self.case_repository.save_case(record)

    def get_case(self, case_id: str) -> MDTCaseRecord:
        return self._get_case_or_raise(case_id)

    def list_patient_case_history(
        self,
        patient_id: str,
        *,
        since: Optional[datetime] = None,
        limit: int = 100,
        include_error: bool = False,
    ) -> List[dict]:
        return self.case_repository.list_patient_case_history(
            patient_id=patient_id,
            since=since,
            limit=limit,
            include_error=include_error,
        )

    def get_case_history_snapshot(self, snapshot_id: int) -> tuple[datetime, MDTCaseRecord]:
        return self.case_repository.get_case_history_snapshot(snapshot_id)

    def delete_case_history_snapshot(self, snapshot_id: int) -> bool:
        return self.case_repository.delete_case_history_snapshot(snapshot_id)

    # -------------------------------------------------------------------------
    # Stage 1 - Parallel fan-out
    # -------------------------------------------------------------------------

    async def _run_stage_one(self, record: MDTCaseRecord, case_input: MDTCaseInput) -> StageOneOutput:
        started = datetime.now(timezone.utc)
        if self.execution_mode == "adk_local":
            stage_one = await self._run_stage_one_adk(case_input)
            notes = f"Stage-one completed with ADK local orchestration ({self.adk_model_ref})."
        else:
            stage_one = await self._run_stage_one_with_medgemma_routing(case_input)
            notes = "Stage-one completed with local MedGemma routing."
        self._append_trace(
            record,
            "ParallelFanOut",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            notes,
        )
        return stage_one

    async def _run_stage_one_adk(self, case_input: MDTCaseInput) -> StageOneOutput:
        if not ADK_AVAILABLE or self.session_service is None or self.adk_model is None:
            raise RuntimeError(
                "ADK local mode is unavailable. Install dependencies and run with MDT_EXECUTION_MODE=adk_local."
            )

        def fetch_radiology_context() -> str:
            parts = [
                f"CT: {case_input.imaging.ct_report or 'N/A'}",
                f"MRI: {case_input.imaging.mri_report or 'N/A'}",
                f"PET: {case_input.imaging.pet_report or 'N/A'}",
            ]
            return "\n".join(parts)

        def fetch_pathology_context() -> str:
            return (
                f"Biopsy: {case_input.pathology.biopsy_summary}\n"
                f"WSI: {case_input.pathology.wsi_summary or 'N/A'}\n"
                f"Receptor status: {case_input.pathology.receptor_status or 'N/A'}\n"
                f"Grade: {case_input.pathology.grade or 'N/A'}"
            )

        def fetch_genomics_context() -> str:
            mutations = ", ".join(
                [f"{m.gene} {m.variant}" for m in case_input.genomics.mutations]
            ) or "None"
            return (
                f"Summary: {case_input.genomics.report_summary}\n"
                f"Mutations: {mutations}\n"
                f"TMB: {case_input.genomics.tmb or 'N/A'}; MSI: {case_input.genomics.msi or 'N/A'}"
            )

        def fetch_literature_context() -> str:
            genes = extract_gene_symbols(case_input)
            evidence = search_literature_evidence(
                diagnosis=case_input.diagnosis,
                genes=genes,
                max_results=self.literature_max_results,
            )
            return json.dumps([e.model_dump() for e in evidence])

        def fetch_diagnosticore_context() -> str:
            if not case_input.diagnosticore:
                return json.dumps({"available": False})
            payload = case_input.diagnosticore.model_dump(mode="json")
            payload["available"] = True
            return json.dumps(payload)

        function_tools = {
            "fetch_radiology_context": FunctionTool(fetch_radiology_context),
            "fetch_pathology_context": FunctionTool(fetch_pathology_context),
            "fetch_genomics_context": FunctionTool(fetch_genomics_context),
            "fetch_literature_context": FunctionTool(fetch_literature_context),
            "fetch_diagnosticore_context": FunctionTool(fetch_diagnosticore_context),
        }

        radiology_agent = LlmAgent(
            name="RadiologySynthesizer",
            model=self.adk_model,
            instruction=(
                "You are RadiologySynthesizer. Use fetch_radiology_context tool, then return strict JSON only:\n"
                '{"findings":"...", "disease_burden":"low|moderate|high|indeterminate", "action_items":["..."]}'
            ),
            tools=[function_tools["fetch_radiology_context"]],
            output_key="radiology_summary",
        )
        pathology_agent = LlmAgent(
            name="PathologySynthesizer",
            model=self.adk_model,
            instruction=(
                "You are PathologySynthesizer. Use fetch_pathology_context tool, then return strict JSON only:\n"
                '{"diagnosis":"...", "biomarkers":["..."], "risk_features":["..."]}'
            ),
            tools=[function_tools["fetch_pathology_context"]],
            output_key="pathology_summary",
        )
        genomics_agent = LlmAgent(
            name="GenomicsSynthesizer",
            model=self.adk_model,
            instruction=(
                "You are GenomicsSynthesizer. Use fetch_genomics_context and fetch_diagnosticore_context tools, "
                "then return strict JSON only:\n"
                '{"actionable_mutations":["..."], "interpretation":"...", "molecular_risk":"low|intermediate|high"}'
            ),
            tools=[
                function_tools["fetch_genomics_context"],
                function_tools["fetch_diagnosticore_context"],
            ],
            output_key="genomics_summary",
        )
        literature_agent = LlmAgent(
            name="LiteratureAgent",
            model=self.adk_model,
            instruction=(
                "You are LiteratureAgent. Use fetch_literature_context tool, then return strict JSON only:\n"
                '{"highlights":"...", "evidence":[{"title":"...", "source":"...", "year":2024, '
                '"identifier":"PMID:123", "finding":"..."}]}'
            ),
            tools=[function_tools["fetch_literature_context"]],
            output_key="literature_summary",
        )

        parallel_stage = ParallelAgent(
            name="Stage1ParallelFanOut",
            sub_agents=[radiology_agent, pathology_agent, genomics_agent, literature_agent],
            description="Parallel synthesis across imaging, pathology, genomics, and literature",
        )
        formatter_agent = LlmAgent(
            name="Stage1Formatter",
            model=self.adk_model,
            instruction=(
                "Combine state values into one JSON object with keys: radiology, pathology, genomics, literature. "
                "Use state values from radiology_summary, pathology_summary, genomics_summary, literature_summary. "
                "Return strict JSON only without markdown fences."
            ),
            output_key="stage_one_output",
        )
        pipeline = SequentialAgent(
            name="MDTStageOnePipeline",
            sub_agents=[parallel_stage, formatter_agent],
            description="Parallel fan-out then formatter gather step",
        )

        runner = Runner(
            agent=pipeline,
            app_name="mdt-command-stage1",
            session_service=self.session_service,
        )
        session_id = f"stage1-{case_input.case_id}"
        await self._ensure_session("mdt-command-stage1", session_id)

        trigger_msg = types.Content(
            role="user",
            parts=[types.Part(text="Run stage one synthesis now.")],
        )
        raw_text = await self._run_runner_and_collect_text(runner, trigger_msg, session_id)
        payload = self._parse_adk_json_payload(raw_text)
        return self._validate_stage_one_payload(payload)

    async def _run_stage_one_with_medgemma_routing(self, case_input: MDTCaseInput) -> StageOneOutput:
        # Local MedGemma uses one shared model instance and serialized inference.
        # Running these sequentially avoids thread pileups when one call stalls.
        radiology = await self._run_stage_one_component_with_router("RadiologySynthesizer", case_input)
        pathology = await self._run_stage_one_component_with_router("PathologySynthesizer", case_input)
        genomics = await self._run_stage_one_component_with_router("GenomicsSynthesizer", case_input)
        literature = await self._run_stage_one_component_with_router("LiteratureAgent", case_input)
        return StageOneOutput(
            radiology=radiology,
            pathology=pathology,
            genomics=genomics,
            literature=literature,
        )

    async def _run_stage_one_component_with_router(
        self,
        agent_name: str,
        case_input: MDTCaseInput,
    ) -> object:
        if not self._agent_uses_medgemma_endpoint(agent_name):
            if agent_name == "RadiologySynthesizer":
                return self._baseline_radiology_summary(case_input)
            if agent_name == "PathologySynthesizer":
                return self._baseline_pathology_summary(case_input)
            if agent_name == "GenomicsSynthesizer":
                return self._baseline_genomics_summary(case_input)
            if agent_name == "LiteratureAgent":
                return self._baseline_literature_summary(case_input)
            raise ValueError(f"Unsupported stage-one agent: {agent_name}")

        model_name = self._model_for_agent(agent_name)
        payload = self._build_stage_one_payload(agent_name, case_input)
        instruction = self._stage_one_instruction(agent_name)
        expected_keys_map: Dict[str, List[str]] = {
            "RadiologySynthesizer": ["findings", "disease_burden", "action_items"],
            "PathologySynthesizer": ["diagnosis", "biomarkers", "risk_features"],
            "GenomicsSynthesizer": ["actionable_mutations", "interpretation", "molecular_risk"],
            "LiteratureAgent": ["highlights", "evidence"],
        }
        parsed = await self._generate_medgemma_json(
            agent_name=agent_name,
            model_name=model_name,
            instruction=instruction,
            payload=payload,
            expected_keys=expected_keys_map.get(agent_name),
        )
        parsed = self._normalize_stage_one_payload(agent_name, parsed)

        if agent_name == "RadiologySynthesizer":
            try:
                return RadiologySummary.model_validate(parsed)
            except Exception as exc:
                logger.warning("RadiologySynthesizer MedGemma payload invalid; using baseline. err=%s", exc)
                return self._baseline_radiology_summary(case_input)
        if agent_name == "PathologySynthesizer":
            try:
                return PathologySummary.model_validate(parsed)
            except Exception as exc:
                logger.warning("PathologySynthesizer MedGemma payload invalid; using baseline. err=%s", exc)
                return self._baseline_pathology_summary(case_input)
        if agent_name == "GenomicsSynthesizer":
            try:
                return GenomicsSummary.model_validate(parsed)
            except Exception as exc:
                logger.warning("GenomicsSynthesizer MedGemma payload invalid; using baseline. err=%s", exc)
                return self._baseline_genomics_summary(case_input)
        if agent_name == "LiteratureAgent":
            try:
                # Preserve retrieved evidence context even if model returns a shorter evidence list.
                payload_evidence = payload.get("evidence") if isinstance(payload, dict) else None
                if isinstance(payload_evidence, list):
                    merged: List[Dict[str, Any]] = []
                    seen_keys = set()
                    seen_titles = set()
                    for source_list in (parsed.get("evidence"), payload_evidence):
                        if not isinstance(source_list, list):
                            continue
                        for item in source_list:
                            if not isinstance(item, dict):
                                continue
                            raw_identifier = str(item.get("identifier", "")).strip() or None
                            normalized_identifier = raw_identifier
                            if raw_identifier:
                                pmid_match = re.match(r"PMID:(\d+)", raw_identifier, flags=re.IGNORECASE)
                                if pmid_match:
                                    normalized_identifier = f"PMID:{pmid_match.group(1)}"
                            normalized = {
                                "title": str(item.get("title", "")).strip() or "Untitled evidence",
                                "source": str(item.get("source", "Unknown")).strip() or "Unknown",
                                "year": item.get("year") if isinstance(item.get("year"), int) else None,
                                "identifier": normalized_identifier,
                                "finding": str(item.get("finding", "")).strip() or "No finding summary provided.",
                            }
                            title_key = normalized["title"].strip().lower()
                            if title_key in seen_titles:
                                # Prefer first seen title instance to avoid duplicate/truncated IDs
                                # from model-regenerated evidence rows.
                                continue
                            key = (normalized.get("identifier") or normalized["title"]).strip().lower()
                            if not key or key in seen_keys:
                                continue
                            seen_keys.add(key)
                            seen_titles.add(title_key)
                            merged.append(normalized)
                            if len(merged) >= self.literature_max_results:
                                break
                        if len(merged) >= self.literature_max_results:
                            break
                    parsed["evidence"] = merged
                    if not str(parsed.get("highlights", "")).strip() and merged:
                        parsed["highlights"] = (
                            f"Retrieved {len(merged)} literature evidence items for "
                            f"{case_input.diagnosis}."
                        )
                return LiteratureSummary.model_validate(parsed)
            except Exception as exc:
                logger.warning("LiteratureAgent MedGemma payload invalid; using baseline. err=%s", exc)
                return self._baseline_literature_summary(case_input)
        raise ValueError(f"Unsupported stage-one agent: {agent_name}")

    def _normalize_stage_one_payload(self, agent_name: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return {}

        def _dict_to_text(item: Dict[str, Any]) -> str:
            pairs: List[str] = []
            for k, v in item.items():
                key = str(k).strip()
                val = str(v).strip()
                if key and val:
                    pairs.append(f"{key}: {val}")
            return "; ".join(pairs).strip()

        def _decode_objectish(text: str) -> Optional[str]:
            raw = (text or "").strip()
            if not raw or not (raw.startswith("{") and raw.endswith("}")):
                return None
            try:
                parsed_obj = ast.literal_eval(raw)
            except Exception:
                return None
            if isinstance(parsed_obj, dict):
                return _dict_to_text(parsed_obj)
            return None

        def _listify(value: Any) -> List[str]:
            if isinstance(value, list):
                out: List[str] = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        decoded = _decode_objectish(item)
                        out.append(decoded if decoded else item.strip())
                    elif isinstance(item, dict):
                        text = _dict_to_text(item)
                        if text:
                            out.append(text)
                    elif item is not None:
                        text = str(item).strip()
                        if text:
                            decoded = _decode_objectish(text)
                            out.append(decoded if decoded else text)
                return out
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return []
                decoded = _decode_objectish(text)
                if decoded:
                    return [decoded]
                chunks = []
                for part in text.replace("\n", ",").split(","):
                    p = part.strip().lstrip("-").strip()
                    if p:
                        decoded_part = _decode_objectish(p)
                        if decoded_part:
                            chunks.append(decoded_part)
                            continue
                        chunks.append(p)
                return chunks
            if value is None:
                return []
            text = str(value).strip()
            decoded = _decode_objectish(text)
            if decoded:
                return [decoded]
            return [text] if text else []

        if agent_name == "RadiologySynthesizer":
            out = dict(parsed)
            out["findings"] = str(out.get("findings", "")).strip()
            burden = str(out.get("disease_burden", "indeterminate")).strip().lower()
            if burden not in {"low", "moderate", "high", "indeterminate"}:
                if "high" in burden:
                    burden = "high"
                elif "moderate" in burden:
                    burden = "moderate"
                elif "low" in burden:
                    burden = "low"
                else:
                    burden = "indeterminate"
            out["disease_burden"] = burden
            out["action_items"] = _listify(out.get("action_items"))
            return out

        if agent_name == "PathologySynthesizer":
            out = dict(parsed)
            out["diagnosis"] = str(out.get("diagnosis", "")).strip()
            out["biomarkers"] = _listify(out.get("biomarkers"))
            out["risk_features"] = _listify(out.get("risk_features"))
            return out

        if agent_name == "GenomicsSynthesizer":
            out = dict(parsed)
            out["actionable_mutations"] = _listify(out.get("actionable_mutations"))
            out["interpretation"] = str(out.get("interpretation", "")).strip()
            risk = str(out.get("molecular_risk", "intermediate")).strip().lower()
            if risk not in {"low", "intermediate", "high"}:
                if "high" in risk:
                    risk = "high"
                elif "intermediate" in risk or "medium" in risk or "moderate" in risk:
                    risk = "intermediate"
                elif "low" in risk:
                    risk = "low"
                else:
                    risk = "intermediate"
            out["molecular_risk"] = risk
            return out

        if agent_name == "LiteratureAgent":
            out = dict(parsed)
            out["highlights"] = str(out.get("highlights", "")).strip()
            evidence_raw = out.get("evidence")
            evidence_list: List[Dict[str, Any]] = []
            if isinstance(evidence_raw, list):
                for item in evidence_raw:
                    if not isinstance(item, dict):
                        continue
                    evidence_list.append(
                        {
                            "title": str(item.get("title", "")).strip() or "Untitled evidence",
                            "source": str(item.get("source", "Unknown")).strip() or "Unknown",
                            "year": item.get("year") if isinstance(item.get("year"), int) else None,
                            "identifier": str(item.get("identifier", "")).strip() or None,
                            "finding": str(item.get("finding", "")).strip() or "No finding summary provided.",
                        }
                    )
            out["evidence"] = evidence_list
            return out

        return parsed

    def _normalize_clinical_reasoning_payload(
        self,
        parsed: Dict[str, Any],
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
        consensus: ConsensusRecommendation,
        gate: HITLGateOutput,
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}

        # Common wrappers from model outputs.
        for wrapper_key in ("clinical_reasoning", "clinical_reasoning_summary", "result", "output"):
            wrapped = parsed.get(wrapper_key)
            if isinstance(wrapped, dict):
                parsed = wrapped
                break

        def _norm_key(value: Any) -> str:
            if value is None:
                return ""
            return re.sub(r"\s+", " ", str(value)).strip().lower()

        def _dict_to_text(item: Dict[str, Any]) -> str:
            pairs: List[str] = []
            for k, v in item.items():
                key = str(k).strip()
                val = str(v).strip()
                if key and val:
                    pairs.append(f"{key}: {val}")
            return "; ".join(pairs).strip()

        def _decode_objectish(text: str) -> Optional[str]:
            raw = (text or "").strip()
            if not raw or not (raw.startswith("{") and raw.endswith("}")):
                return None
            try:
                parsed_obj = ast.literal_eval(raw)
            except Exception:
                return None
            if isinstance(parsed_obj, dict):
                return _dict_to_text(parsed_obj)
            return None

        def _listify(value: Any) -> List[str]:
            if isinstance(value, list):
                out: List[str] = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        decoded = _decode_objectish(item)
                        out.append(decoded if decoded else item.strip())
                    elif isinstance(item, dict):
                        # evidence-like objects
                        ident = item.get("identifier") or item.get("id") or item.get("nct_id")
                        title = item.get("title")
                        if isinstance(ident, str) and ident.strip():
                            out.append(ident.strip())
                        elif isinstance(title, str) and title.strip():
                            out.append(title.strip())
                        else:
                            text = _dict_to_text(item)
                            if text:
                                out.append(text)
                    elif item is not None:
                        text = str(item).strip()
                        if text:
                            decoded = _decode_objectish(text)
                            out.append(decoded if decoded else text)
                return out
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return []
                decoded = _decode_objectish(text)
                if decoded:
                    return [decoded]
                parts: List[str] = []
                for part in text.replace("\n", ",").split(","):
                    normalized = part.strip().lstrip("-").strip()
                    if normalized:
                        decoded_part = _decode_objectish(normalized)
                        if decoded_part:
                            parts.append(decoded_part)
                            continue
                        parts.append(normalized)
                return parts
            if value is None:
                return []
            text = str(value).strip()
            decoded = _decode_objectish(text)
            if decoded:
                return [decoded]
            return [text] if text else []

        def _dedupe(items: List[str]) -> List[str]:
            out: List[str] = []
            seen: set[str] = set()
            for item in items:
                text = str(item).strip()
                if not text:
                    continue
                key = _norm_key(text)
                if key in seen:
                    continue
                seen.add(key)
                out.append(text)
            return out

        summary = self._dedupe_inline_citations(
            (
            str(parsed.get("summary") or parsed.get("assessment") or consensus.recommendation).strip()
            )
        )
        safety_flags = _dedupe(_listify(gate.safety_flags or consensus.red_flags))
        safety_norm = {_norm_key(flag) for flag in safety_flags if str(flag).strip()}

        key_risks = _dedupe(
            _listify(parsed.get("key_risks") or parsed.get("risks") or consensus.red_flags)
        )
        key_risks = [risk for risk in key_risks if _norm_key(risk) not in safety_norm]

        recommended_actions = _dedupe(
            _listify(
            parsed.get("recommended_actions") or parsed.get("recommendations")
            )
        )
        confirmatory_actions = _dedupe(_listify(
            parsed.get("confirmatory_actions") or parsed.get("confirmatory_steps")
        ))
        evidence_links = _dedupe(_listify(parsed.get("evidence_links") or parsed.get("evidence")))
        uncertainty_statement = self._dedupe_inline_citations(
            str(
            parsed.get("uncertainty_statement")
            or parsed.get("uncertainty")
            or f"Consensus confidence is {consensus.confidence:.2f}; clinician review required."
            ).strip()
        )

        blocked_action_norms = {_norm_key(summary), _norm_key(uncertainty_statement), *safety_norm}
        recommended_actions = [
            action for action in recommended_actions if _norm_key(action) not in blocked_action_norms
        ]
        risk_norm = {_norm_key(risk) for risk in key_risks if str(risk).strip()}
        recommended_actions = [
            action for action in recommended_actions if _norm_key(action) not in risk_norm
        ]

        if len(key_risks) < 2:
            derived_risks: List[str] = []
            derived_risks.extend(stage_one.pathology.risk_features[:3])
            if stage_one.genomics.actionable_mutations:
                derived_risks.append(
                    "Actionable genomic findings detected: "
                    + ", ".join(stage_one.genomics.actionable_mutations[:4])
                    + "."
                )
            if stage_one.genomics.molecular_risk == "high":
                derived_risks.append("Genomic profile indicates high molecular risk.")
            elif stage_one.genomics.molecular_risk == "intermediate":
                derived_risks.append("Genomic profile indicates intermediate molecular risk.")
            if stage_one.radiology.disease_burden == "indeterminate":
                derived_risks.append("Imaging burden remains indeterminate and needs clinician clarification.")
            key_risks = _dedupe(key_risks + [r.strip() for r in derived_risks if str(r).strip()])[:6]
        if not key_risks:
            key_risks = safety_flags[:6]

        if len(recommended_actions) < 2:
            derived_actions: List[str] = []
            derived_actions.extend(stage_one.radiology.action_items[:4])
            if stage_one.genomics.actionable_mutations:
                derived_actions.append(
                    "Review targeted therapy options for: "
                    + ", ".join(stage_one.genomics.actionable_mutations[:4])
                    + "."
                )
            if case_input.diagnosticore and not case_input.diagnosticore.is_confirmed_genomic_test:
                derived_actions.append(
                    "Plan confirmatory molecular testing for AI-inferred genomic signal before final treatment lock."
                )
            if consensus.recommendation.strip():
                derived_actions.append(consensus.recommendation.strip())
            recommended_actions = _dedupe(
                recommended_actions + [a.strip() for a in derived_actions if str(a).strip()]
            )[:8]
        # Avoid mirrored bullets across sections after derivation.
        key_risk_norm = {_norm_key(item) for item in key_risks}
        confirm_norm = {_norm_key(item) for item in confirmatory_actions}
        recommended_actions = [
            item
            for item in recommended_actions
            if _norm_key(item) not in key_risk_norm and _norm_key(item) not in confirm_norm
        ]
        if not recommended_actions and consensus.recommendation.strip():
            recommended_actions = [consensus.recommendation.strip()]
        if not confirmatory_actions and case_input.diagnosticore and not case_input.diagnosticore.is_confirmed_genomic_test:
            confirmatory_actions.append(
                "Confirm AI-inferred genomic findings with certified molecular assay before treatment lock."
            )
        if not evidence_links:
            for e in stage_one.literature.evidence[:4]:
                if e.identifier:
                    evidence_links.append(e.identifier)

        return {
            "summary": summary,
            "key_risks": _dedupe(key_risks),
            "recommended_actions": _dedupe(recommended_actions),
            "confirmatory_actions": _dedupe(confirmatory_actions),
            "evidence_links": _dedupe(evidence_links),
            "uncertainty_statement": uncertainty_statement,
            "model_route": parsed.get("model_route"),
            "generation_mode": parsed.get("generation_mode"),
        }

    def _normalize_consensus_payload(
        self,
        parsed: Dict[str, Any],
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}

        for wrapper_key in ("consensus", "consensus_json", "result", "output"):
            wrapped = parsed.get(wrapper_key)
            if isinstance(wrapped, dict):
                parsed = wrapped
                break

        def _listify(value: Any) -> List[str]:
            if isinstance(value, list):
                out: List[str] = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        out.append(item.strip())
                    elif item is not None:
                        text = str(item).strip()
                        if text:
                            out.append(text)
                return out
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return []
                parts: List[str] = []
                for part in text.replace("\n", ",").split(","):
                    normalized = part.strip().lstrip("-").strip()
                    if normalized:
                        parts.append(normalized)
                return parts
            if value is None:
                return []
            text = str(value).strip()
            return [text] if text else []

        def _to_confidence(value: Any) -> Optional[float]:
            if isinstance(value, (int, float)):
                num = float(value)
            elif isinstance(value, str):
                raw_text = value.strip()
                if not raw_text:
                    return None
                text = raw_text.replace("%", "")
                try:
                    num = float(text)
                except Exception:
                    match = re.search(r"-?\d+(?:\.\d+)?", text)
                    if not match:
                        return None
                    num = float(match.group(0))
            else:
                return None
            if num > 1.0 and num <= 100.0:
                num = num / 100.0
            return max(0.0, min(1.0, num))

        def _estimate_fallback_confidence() -> float:
            return self._estimate_consensus_confidence_from_signals(
                case_input=case_input,
                stage_one=stage_one,
            )

        inferred_actionable: List[str] = []
        if not stage_one.genomics.actionable_mutations:
            genomic_sources: List[str] = []
            if getattr(case_input, "genomics", None):
                genomic_sources.append(str(case_input.genomics.report_summary or ""))
                genomic_sources.extend(
                    f"{m.gene} {m.variant}" for m in (case_input.genomics.mutations or [])
                )
            genomic_blob = " ".join(genomic_sources)
            if genomic_blob.strip():
                inferred_actionable = list(
                    dict.fromkeys(
                        re.findall(r"\b[A-Z0-9]{2,12}\s+[A-Z]\d+[A-Z0-9]{0,4}\b", genomic_blob)
                    )
                )
        actionable_values = stage_one.genomics.actionable_mutations or inferred_actionable
        actionable = ", ".join(actionable_values) or "molecular profile pending confirmatory interpretation"
        fallback_recommendation = (
            f"MDT recommendation for {case_input.diagnosis}: align treatment strategy with imaging burden "
            f"({stage_one.radiology.disease_burden}) and molecular profile ({actionable})."
        )
        fallback_rationale = (
            f"Integrated from radiology, pathology, genomics, and "
            f"{len(stage_one.literature.evidence)} literature evidence items."
        )

        recommendation = str(
            parsed.get("recommendation")
            or parsed.get("assessment")
            or parsed.get("summary")
            or parsed.get("plan")
            or fallback_recommendation
        ).strip()
        rationale = str(
            parsed.get("rationale")
            or parsed.get("reasoning")
            or parsed.get("justification")
            or parsed.get("explanation")
            or parsed.get("plan")
            or fallback_rationale
        ).strip()
        confidence_raw: Any = None
        for key in ("confidence", "score", "confidence_score", "probability"):
            if key not in parsed:
                continue
            candidate = parsed.get(key)
            if candidate is None:
                continue
            if isinstance(candidate, str) and not candidate.strip():
                continue
            confidence_raw = candidate
            break
        if confidence_raw is None:
            confidence_match_sources = [
                str(parsed.get("rationale") or ""),
                str(parsed.get("recommendation") or ""),
                json.dumps(parsed, ensure_ascii=True),
            ]
            for source in confidence_match_sources:
                match = re.search(
                    r"\bconfidence(?:_score)?\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?%?)",
                    source,
                    flags=re.IGNORECASE,
                )
                if match:
                    confidence_raw = match.group(1)
                    break

        confidence = _to_confidence(confidence_raw)
        if confidence is None:
            confidence = _estimate_fallback_confidence()
            logger.debug(
                "Consensus confidence parse failed; using estimated fallback confidence=%.2f (case=%s).",
                confidence,
                case_input.case_id,
            )
        red_flags = _listify(
            parsed.get("red_flags")
            or parsed.get("risks")
            or parsed.get("risk_flags")
            or parsed.get("safety_flags")
        )

        return {
            "recommendation": recommendation,
            "rationale": rationale,
            "confidence": confidence,
            "red_flags": red_flags[:8],
        }

    def _normalize_soap_payload(
        self,
        parsed: Dict[str, Any],
        stage_one: StageOneOutput,
        consensus: ConsensusRecommendation,
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}

        for wrapper_key in ("soap", "soap_note", "soap_json", "result", "output"):
            wrapped = parsed.get(wrapper_key)
            if isinstance(wrapped, dict):
                parsed = wrapped
                break

        fallback_subjective = "Case reviewed in multidisciplinary tumor board with cross-specialty inputs."
        fallback_objective = (
            f"Imaging burden: {stage_one.radiology.disease_burden}. "
            f"Pathology: {stage_one.pathology.diagnosis}. "
            f"Molecular profile: {stage_one.genomics.interpretation}"
        )
        fallback_assessment = consensus.recommendation
        fallback_plan = (
            f"{consensus.rationale} Next: clinician-approved care plan."
        )

        return {
            "subjective": str(
                parsed.get("subjective")
                or parsed.get("s")
                or fallback_subjective
            ).strip(),
            "objective": str(
                parsed.get("objective")
                or parsed.get("o")
                or fallback_objective
            ).strip(),
            "assessment": str(
                parsed.get("assessment")
                or parsed.get("a")
                or parsed.get("summary")
                or fallback_assessment
            ).strip(),
            "plan": str(
                parsed.get("plan")
                or parsed.get("p")
                or parsed.get("recommendation")
                or fallback_plan
            ).strip(),
        }

    def _stage_one_instruction(self, agent_name: str) -> str:
        if agent_name == "RadiologySynthesizer":
            return (
                "Synthesize imaging into JSON keys: findings, disease_burden, action_items. "
                "disease_burden must be one of low|moderate|high|indeterminate."
            )
        if agent_name == "PathologySynthesizer":
            return (
                "Synthesize pathology into JSON keys: diagnosis, biomarkers, risk_features."
            )
        if agent_name == "GenomicsSynthesizer":
            return (
                "Synthesize genomics into JSON keys: actionable_mutations, interpretation, molecular_risk. "
                "molecular_risk must be one of low|intermediate|high. "
                "If DiagnostiCore TP53 inference exists, explicitly mark it as AI-inferred pending confirmation."
            )
        if agent_name == "LiteratureAgent":
            return (
                "Synthesize literature into JSON keys: highlights, evidence. "
                "evidence is an array of objects with keys title, source, year, identifier, finding."
            )
        raise ValueError(f"Unsupported stage-one agent: {agent_name}")

    def _build_stage_one_payload(self, agent_name: str, case_input: MDTCaseInput) -> Dict[str, Any]:
        if agent_name == "RadiologySynthesizer":
            return {
                "diagnosis": case_input.diagnosis,
                "ct_report": case_input.imaging.ct_report,
                "mri_report": case_input.imaging.mri_report,
                "pet_report": case_input.imaging.pet_report,
            }
        if agent_name == "PathologySynthesizer":
            return {
                "diagnosis": case_input.diagnosis,
                "biopsy_summary": case_input.pathology.biopsy_summary,
                "wsi_summary": case_input.pathology.wsi_summary,
                "receptor_status": case_input.pathology.receptor_status,
                "grade": case_input.pathology.grade,
            }
        if agent_name == "GenomicsSynthesizer":
            return {
                "diagnosis": case_input.diagnosis,
                "report_summary": case_input.genomics.report_summary,
                "mutations": [m.model_dump(mode="json") for m in case_input.genomics.mutations],
                "tmb": case_input.genomics.tmb,
                "msi": case_input.genomics.msi,
                "diagnosticore": case_input.diagnosticore.model_dump(mode="json")
                if case_input.diagnosticore
                else None,
            }
        if agent_name == "LiteratureAgent":
            genes = extract_gene_symbols(case_input)
            evidence = search_literature_evidence(
                case_input.diagnosis,
                genes,
                max_results=self.literature_max_results,
            )
            return {
                "diagnosis": case_input.diagnosis,
                "genes": genes,
                "evidence": [e.model_dump(mode="json") for e in evidence],
            }
        raise ValueError(f"Unsupported stage-one agent: {agent_name}")

    def _baseline_radiology_summary(self, case_input: MDTCaseInput) -> RadiologySummary:
        findings = " | ".join(
            [
                f"CT: {case_input.imaging.ct_report or 'N/A'}",
                f"MRI: {case_input.imaging.mri_report or 'N/A'}",
                f"PET: {case_input.imaging.pet_report or 'N/A'}",
            ]
        )
        disease_burden = "indeterminate"
        lower_text = findings.lower()
        if any(token in lower_text for token in ["metast", "multifocal", "nodal"]):
            disease_burden = "high"
        elif any(token in lower_text for token in ["localized", "single lesion"]):
            disease_burden = "moderate"
        return RadiologySummary(
            findings=findings,
            disease_burden=disease_burden,
            action_items=[
                "Correlate lesion burden with pathology subtype.",
                "Reconfirm baseline imaging before treatment transition.",
            ],
        )

    def _baseline_pathology_summary(self, case_input: MDTCaseInput) -> PathologySummary:
        biomarkers: List[str] = []
        receptor = case_input.pathology.receptor_status or ""
        if receptor:
            biomarkers.append(receptor)
        if case_input.pathology.grade:
            biomarkers.append(f"Grade {case_input.pathology.grade}")

        risk_features: List[str] = []
        text_blob = f"{case_input.pathology.biopsy_summary} {case_input.pathology.wsi_summary or ''}".lower()
        if any(token in text_blob for token in ["high-grade", "grade 3", "lymphovascular"]):
            risk_features.append("Aggressive histologic features")
        if "ki-67" in text_blob:
            risk_features.append("Elevated proliferative index")

        return PathologySummary(
            diagnosis=case_input.pathology.biopsy_summary,
            biomarkers=biomarkers,
            risk_features=risk_features,
        )

    def _baseline_genomics_summary(self, case_input: MDTCaseInput) -> GenomicsSummary:
        actionable: List[str] = []
        for mut in case_input.genomics.mutations:
            actionable.append(f"{mut.gene} {mut.variant}".strip())

        risk = "intermediate"
        summary_lower = case_input.genomics.report_summary.lower()
        if any(token in summary_lower for token in ["tp53", "brca1", "brca2"]):
            risk = "high"

        interpretation = case_input.genomics.report_summary
        diagnosticore = case_input.diagnosticore
        if diagnosticore:
            threshold = diagnosticore.threshold if diagnosticore.threshold is not None else 0.5
            inferred_label = diagnosticore.predicted_label or (
                "tp53_mutated" if diagnosticore.tp53_probability >= threshold else "tp53_wildtype"
            )
            interpretation += (
                " | DiagnostiCore inferred TP53 signal: "
                f"p={diagnosticore.tp53_probability:.3f}, threshold={threshold:.3f}, "
                f"label={inferred_label}. This is AI-inferred and not a confirmed molecular assay."
            )
            if diagnosticore.uncertainty_flags:
                interpretation += " Uncertainty: " + "; ".join(diagnosticore.uncertainty_flags)
            if inferred_label == "tp53_mutated":
                risk = "high"
                if "TP53 (AI-inferred, pending confirmation)" not in actionable:
                    actionable.append("TP53 (AI-inferred, pending confirmation)")
        return GenomicsSummary(
            actionable_mutations=actionable,
            interpretation=interpretation,
            molecular_risk=risk,
        )

    def _baseline_literature_summary(self, case_input: MDTCaseInput) -> LiteratureSummary:
        genes = extract_gene_symbols(case_input)
        evidence = search_literature_evidence(
            case_input.diagnosis,
            genes,
            max_results=self.literature_max_results,
        )
        highlights = (
            "Evidence supports molecularly informed treatment sequencing."
            if evidence
            else "No matched evidence found in mock set."
        )
        return LiteratureSummary(highlights=highlights, evidence=evidence)

    async def _run_radiology_synthesizer(
        self, record: MDTCaseRecord, case_input: MDTCaseInput
    ) -> RadiologySummary:
        started = datetime.now(timezone.utc)
        output = self._baseline_radiology_summary(case_input)
        self._append_trace(
            record,
            "RadiologySynthesizer",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            "Mock multimodal imaging synthesis",
        )
        return output

    async def _run_pathology_synthesizer(
        self, record: MDTCaseRecord, case_input: MDTCaseInput
    ) -> PathologySummary:
        started = datetime.now(timezone.utc)
        output = self._baseline_pathology_summary(case_input)
        self._append_trace(
            record,
            "PathologySynthesizer",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            "Mock biopsy/WSI synthesis",
        )
        return output

    async def _run_genomics_synthesizer(
        self, record: MDTCaseRecord, case_input: MDTCaseInput
    ) -> GenomicsSummary:
        started = datetime.now(timezone.utc)
        diagnosticore = case_input.diagnosticore
        output = self._baseline_genomics_summary(case_input)
        self._append_trace(
            record,
            "GenomicsSynthesizer",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            "Mock NGS interpretation"
            + (" + DiagnostiCore signal integration" if diagnosticore else ""),
        )
        return output

    async def _run_literature_agent(
        self, record: MDTCaseRecord, case_input: MDTCaseInput
    ) -> LiteratureSummary:
        started = datetime.now(timezone.utc)
        output = self._baseline_literature_summary(case_input)
        self._append_trace(
            record,
            "LiteratureAgent",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            f"Mock evidence retrieval, results={len(output.evidence)}",
        )
        return output

    async def _run_transcription_agent(
        self, record: MDTCaseRecord, case_input: MDTCaseInput
    ) -> TranscriptionResult:
        started = datetime.now(timezone.utc)
        transcript_text = case_input.transcript.raw_text
        audio_uri = case_input.transcript.audio_uri

        def _normalize_client_transcript(raw_text: str) -> str:
            normalized = self.transcription_adapter._normalize_transcript_text(" ".join((raw_text or "").split()))
            if self.transcription_adapter._looks_corrupted_transcript(normalized):
                repaired = self.transcription_adapter._repair_corrupted_transcript(normalized)
                if repaired and not self.transcription_adapter._looks_corrupted_transcript(repaired):
                    normalized = repaired
            return self.transcription_adapter._truncate_transcript(normalized)

        cleaned = _normalize_client_transcript(transcript_text)

        # Frontend ONNX path sends transcript text directly and intentionally clears audio_uri.
        # Treat this as a first-class successful path, not as a backend ASR fallback/error.
        if not (audio_uri or "").strip() and cleaned:
            result = TranscriptionResult(
                engine="frontend-transcript",
                transcript=cleaned,
                wer_estimate=None,
                notes=(
                    "Using transcript text supplied by the client UI "
                    "(manual entry or on-device local transcription)."
                ),
            )
            self._append_trace(
                record,
                "TranscriptionAgent",
                AgentRunStatus.COMPLETED,
                started,
                datetime.now(timezone.utc),
                "Engine=frontend-transcript",
            )
            return result

        try:
            result = await asyncio.wait_for(
                self.transcription_adapter.transcribe(
                    transcript_text=transcript_text,
                    audio_uri=audio_uri,
                ),
                timeout=self.transcription_adapter.timeout_seconds,
            )
        except Exception as exc:
            if cleaned:
                logger.warning(
                    "Transcription fallback engaged for case %s. err=%s",
                    case_input.case_id,
                    exc,
                )
                result = TranscriptionResult(
                    engine=self.transcription_adapter.engine_name,
                    transcript=cleaned,
                    wer_estimate=None,
                    notes=(
                        "Using transcript text supplied by the client UI "
                        f"after backend ASR error (audio_uri={audio_uri or 'none'})."
                    ),
                )
            else:
                raise
        self._append_trace(
            record,
            "TranscriptionAgent",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            f"Engine={result.engine}",
        )
        return result

    async def _persist_transcription_on_ready(
        self,
        record: MDTCaseRecord,
        transcription_task: "asyncio.Task[TranscriptionResult]",
    ) -> None:
        """
        Persists transcription as soon as it completes so the UI can poll draft
        artifacts and show near-real-time transcription updates while analyze is running.
        """
        try:
            transcription = await transcription_task
        except Exception:
            return

        async with self.lock:
            if record.status != CaseStatus.ANALYZING:
                return
            if record.artifacts.transcription is None:
                record.artifacts.transcription = transcription
                record.updated_at = datetime.now(timezone.utc)
                self.case_repository.save_case(record)

    def _sanitize_flag_list(self, flags: List[str], max_items: int = 5) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        noisy_markers = [
            "supports the need for molecularly informed sequencing",
            "which is common in",
            "this finding reinforces",
        ]
        empty_markers = {
            "none",
            "no",
            "n/a",
            "na",
            "no active safety flags",
            "no active safety flags.",
            "no safety flags",
            "no safety flags.",
        }
        for raw in flags:
            text = str(raw).strip().lstrip("-").lstrip("•").strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in empty_markers:
                continue
            if any(marker in lowered for marker in noisy_markers):
                continue
            # Keep safety signals concise and readable.
            if len(text) > 220:
                text = text[:217].rstrip() + "..."
            key = lowered
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _dedupe_inline_citations(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        pattern = re.compile(r"\[(PMID:\d+|DIAGNOSTICORE)\]", flags=re.IGNORECASE)
        citations: List[str] = []
        seen: set[str] = set()
        pmid_nums: List[str] = []
        for match in pattern.finditer(raw):
            token = match.group(1)
            normalized = token.upper()
            if normalized.startswith("PMID:"):
                pmid_num = normalized.split(":", 1)[1]
                pretty = f"[PMID:{pmid_num}]"
                pmid_nums.append(pmid_num)
            else:
                pretty = "[DIAGNOSTICORE]"
            if pretty in seen:
                continue
            seen.add(pretty)
            citations.append(pretty)

        if pmid_nums:
            longer_pmids = sorted(set(pmid_nums), key=len, reverse=True)
            filtered_citations: List[str] = []
            for cit in citations:
                m = re.match(r"\[PMID:(\d+)\]", cit)
                if not m:
                    filtered_citations.append(cit)
                    continue
                cur = m.group(1)
                if any(other != cur and other.startswith(cur) for other in longer_pmids):
                    # likely truncated artifact (e.g., PMID:3774 alongside PMID:37747019)
                    continue
                filtered_citations.append(cit)
            citations = filtered_citations

        body = pattern.sub("", raw)
        body = re.sub(r"\s+", " ", body).strip()
        if citations:
            return f"{body} {' '.join(citations)}".strip()
        return body

    def _clean_generated_text(self, text: str, max_len: int = 520) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        raw = raw.replace(" ,", ",")
        raw = re.sub(r"(,\s*){3,}", ", ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        raw = self._dedupe_inline_citations(raw)
        if len(raw) > max_len:
            clipped = raw[:max_len].rsplit(".", 1)[0].strip()
            if clipped and len(clipped) > 40:
                return clipped + "."
            return raw[:max_len].rstrip() + "..."
        return raw

    def _is_noisy_text(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return True
        if len(raw) > 420:
            return True
        if ", , ," in raw or " , , " in raw:
            return True
        if len(re.findall(r"\[[A-Z]+:?[\w-]+\]", raw)) > 8:
            return True
        return False

    def _postprocess_consensus_output(
        self,
        output: ConsensusRecommendation,
        stage_one: StageOneOutput,
    ) -> ConsensusRecommendation:
        evidence_tags = self._collect_evidence_tags(stage_one=stage_one)

        rec = self._clean_generated_text(output.recommendation.strip(), max_len=420)
        rationale = self._clean_generated_text(output.rationale.strip(), max_len=520)
        if evidence_tags:
            if "[PMID:" not in rec:
                rec = f"{rec} {' '.join(evidence_tags[:2])}".strip()
            if "[PMID:" not in rationale:
                rationale = f"{rationale} {' '.join(evidence_tags[:3])}".strip()
        rec = self._clean_generated_text(rec, max_len=420)
        rationale = self._clean_generated_text(rationale, max_len=520)

        red_flags = self._sanitize_flag_list(output.red_flags, max_items=5)
        confidence = round(max(0.0, min(1.0, float(output.confidence))), 2)
        return ConsensusRecommendation(
            recommendation=rec,
            rationale=rationale,
            confidence=confidence,
            red_flags=red_flags,
        )

    def _collect_evidence_tags(
        self,
        stage_one: StageOneOutput,
    ) -> List[str]:
        tags: List[str] = []
        for ev in stage_one.literature.evidence[:4]:
            if ev.identifier:
                tags.append(f"[{ev.identifier}]")
        return list(dict.fromkeys(tags))

    @staticmethod
    def _coerce_confidence(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            num = float(value)
        elif isinstance(value, str):
            text = value.strip().replace("%", "")
            if not text:
                return None
            try:
                num = float(text)
            except Exception:
                match = re.search(r"-?\d+(?:\.\d+)?", text)
                if not match:
                    return None
                num = float(match.group(0))
        else:
            return None
        if num > 1.0 and num <= 100.0:
            num = num / 100.0
        return max(0.0, min(1.0, num))

    @staticmethod
    def _estimate_consensus_confidence_from_signals(
        *,
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
    ) -> float:
        score = 0.58
        if stage_one.radiology.findings.strip():
            score += 0.06
        if stage_one.pathology.diagnosis.strip():
            score += 0.06
        if stage_one.pathology.biomarkers:
            score += 0.03
        if stage_one.genomics.actionable_mutations:
            score += min(0.12, 0.03 * len(stage_one.genomics.actionable_mutations))
        if stage_one.literature.evidence:
            score += min(0.12, 0.02 * len(stage_one.literature.evidence))
        if case_input.diagnosticore:
            score += 0.03
        if stage_one.radiology.disease_burden == "indeterminate":
            score -= 0.06
        if stage_one.genomics.molecular_risk == "high":
            score -= 0.04
        return round(max(0.50, min(0.92, score)), 2)

    async def _infer_consensus_confidence_with_llm(
        self,
        *,
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
        recommendation: str,
        rationale: str,
    ) -> Optional[float]:
        model_name = self._model_for_agent("ConsensusSynthesizer")
        payload = {
            "diagnosis": case_input.diagnosis,
            "recommendation": recommendation,
            "rationale": rationale,
            "stage_one": stage_one.model_dump(mode="json"),
            "diagnosticore_present": bool(case_input.diagnosticore),
        }
        parsed = await self._generate_medgemma_json(
            agent_name="ConsensusConfidenceCalibrator",
            model_name=model_name,
            instruction=(
                "Given the oncology MDT recommendation context, return strict JSON only with key: confidence. "
                "confidence must be a single numeric value between 0 and 1."
            ),
            payload=payload,
            expected_keys=["confidence"],
        )
        return self._coerce_confidence(parsed.get("confidence"))

    def _postprocess_soap_output(
        self,
        output: SOAPNote,
        stage_one: StageOneOutput,
    ) -> SOAPNote:
        evidence_tags = self._collect_evidence_tags(stage_one=stage_one)
        plan = self._clean_generated_text(output.plan.strip(), max_len=560)
        if evidence_tags and "[PMID:" not in plan:
            plan = f"{plan} {' '.join(evidence_tags[:3])}".strip()
        plan = self._clean_generated_text(plan, max_len=560)
        return SOAPNote(
            subjective=output.subjective.strip(),
            objective=output.objective.strip(),
            assessment=output.assessment.strip(),
            plan=plan,
        )

    @staticmethod
    def _dedupe_text_items(items: List[str], max_items: int) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = re.sub(r"\[[A-Z]+:?[\w-]+\]", "", text)
            key = re.sub(r"\s+", " ", key).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= max_items:
                break
        return out

    def _stabilize_clinical_reasoning_summary(
        self,
        *,
        output: ClinicalReasoningSummary,
        baseline: ClinicalReasoningSummary,
    ) -> ClinicalReasoningSummary:
        summary = output.summary.strip()
        if len(summary.split()) < 8:
            summary = baseline.summary.strip()

        key_risks = self._dedupe_text_items(output.key_risks + baseline.key_risks, max_items=6)
        recommended_actions = self._dedupe_text_items(
            output.recommended_actions + baseline.recommended_actions,
            max_items=8,
        )
        confirmatory_actions = self._dedupe_text_items(
            output.confirmatory_actions + baseline.confirmatory_actions,
            max_items=6,
        )
        key_risks = [item for item in key_risks if not self._is_noisy_text(item)]
        confirmatory_actions = [item for item in confirmatory_actions if not self._is_noisy_text(item)]
        risk_keys = {re.sub(r"\s+", " ", r).strip().lower() for r in key_risks}
        confirm_keys = {re.sub(r"\s+", " ", r).strip().lower() for r in confirmatory_actions}
        recommended_actions = [
            action
            for action in recommended_actions
            if re.sub(r"\s+", " ", action).strip().lower() not in risk_keys
            and re.sub(r"\s+", " ", action).strip().lower() not in confirm_keys
            and not self._is_noisy_text(action)
        ]
        if not recommended_actions:
            recommended_actions = self._dedupe_text_items(
                [baseline.summary] + baseline.recommended_actions,
                max_items=4,
            )

        evidence_links = self._dedupe_text_items(
            output.evidence_links + baseline.evidence_links,
            max_items=10,
        )
        uncertainty_statement = output.uncertainty_statement.strip() or baseline.uncertainty_statement.strip()

        return ClinicalReasoningSummary(
            summary=summary,
            key_risks=key_risks,
            recommended_actions=recommended_actions,
            confirmatory_actions=confirmatory_actions,
            evidence_links=evidence_links,
            uncertainty_statement=uncertainty_statement,
            model_route=output.model_route,
            generation_mode=output.generation_mode,
        )

    # -------------------------------------------------------------------------
    # Stage 2 - Sequential chain
    # -------------------------------------------------------------------------

    async def _run_consensus_synthesizer(
        self,
        record: MDTCaseRecord,
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
    ) -> ConsensusRecommendation:
        started = datetime.now(timezone.utc)
        model_name = self._model_for_agent("ConsensusSynthesizer")
        input_payload = {
            "diagnosis": case_input.diagnosis,
            "stage_one": stage_one.model_dump(mode="json"),
            "diagnosticore": case_input.diagnosticore.model_dump(mode="json")
            if case_input.diagnosticore
            else None,
            "rule": "If diagnosticore is present, treat as AI-inferred and mention confirmatory molecular testing.",
        }
        parsed = await self._generate_medgemma_json(
            agent_name="ConsensusSynthesizer",
            model_name=model_name,
            instruction=(
                "You are ConsensusSynthesizer for an oncology MDT. "
                "Return strict JSON only with keys: recommendation, rationale, confidence, red_flags. "
                "Confidence must be between 0 and 1. Red flags must be explicit safety concerns. "
                "In recommendation/rationale include source-tagged facts (e.g., [PMID:xxxx], [DIAGNOSTICORE]) "
                "whenever evidence is available."
            ),
            payload=input_payload,
            expected_keys=["recommendation", "rationale", "confidence", "red_flags"],
        )
        if "confidence" not in parsed or self._coerce_confidence(parsed.get("confidence")) is None:
            inferred_confidence = await self._infer_consensus_confidence_with_llm(
                case_input=case_input,
                stage_one=stage_one,
                recommendation=str(parsed.get("recommendation", "")).strip(),
                rationale=str(parsed.get("rationale", "")).strip(),
            )
            parsed = dict(parsed)
            if inferred_confidence is not None:
                parsed["confidence"] = inferred_confidence
            else:
                parsed["confidence"] = self._estimate_consensus_confidence_from_signals(
                    case_input=case_input,
                    stage_one=stage_one,
                )
        normalized = self._normalize_consensus_payload(
            parsed=parsed,
            case_input=case_input,
            stage_one=stage_one,
        )
        output = ConsensusRecommendation.model_validate(normalized)
        output = self._postprocess_consensus_output(output, stage_one=stage_one)
        self._append_trace(
            record,
            "ConsensusSynthesizer",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            f"Local MedGemma consensus with model={model_name}",
        )
        return output

    async def _run_soap_generator(
        self,
        record: MDTCaseRecord,
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
        consensus: ConsensusRecommendation,
    ) -> SOAPNote:
        started = datetime.now(timezone.utc)
        model_name = self._model_for_agent("SOAPGenerator")
        input_payload = {
            "case_id": case_input.case_id,
            "diagnosis": case_input.diagnosis,
            "stage_one": stage_one.model_dump(mode="json"),
            "consensus": consensus.model_dump(mode="json"),
            "safety_rule": "When genomic signal is AI-inferred, include confirmatory assay plan language.",
        }
        parsed = await self._generate_medgemma_json(
            agent_name="SOAPGenerator",
            model_name=model_name,
            instruction=(
                "You are SOAPGenerator for oncology documentation. "
                "Return strict JSON only with keys: subjective, objective, assessment, plan."
            ),
            payload=input_payload,
            expected_keys=["subjective", "objective", "assessment", "plan"],
        )
        normalized = self._normalize_soap_payload(
            parsed=parsed,
            stage_one=stage_one,
            consensus=consensus,
        )
        output = SOAPNote.model_validate(normalized)
        output = self._postprocess_soap_output(output, stage_one=stage_one)
        self._append_trace(
            record,
            "SOAPGenerator",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            f"Local MedGemma SOAP with model={model_name}",
        )
        return output

    async def _run_clinical_reasoner(
        self,
        record: MDTCaseRecord,
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
        consensus: ConsensusRecommendation,
        soap: SOAPNote,
        gate: HITLGateOutput,
    ) -> ClinicalReasoningSummary:
        started = datetime.now(timezone.utc)
        model_route = self._model_for_agent("ClinicalReasoner")
        payload = self.clinical_reasoner.llm_payload(
            case_input=case_input,
            stage_one=stage_one,
            consensus=consensus,
            soap=soap,
            gate=gate,
        )
        parsed = await self._generate_medgemma_json(
            agent_name="ClinicalReasoner",
            model_name=model_route,
            instruction=self.clinical_reasoner.llm_instruction(),
            payload=payload,
            expected_keys=[
                "summary",
                "key_risks",
                "recommended_actions",
                "confirmatory_actions",
                "evidence_links",
                "uncertainty_statement",
            ],
        )
        normalized = self._normalize_clinical_reasoning_payload(
            parsed=parsed,
            case_input=case_input,
            stage_one=stage_one,
            consensus=consensus,
            gate=gate,
        )
        output = ClinicalReasoningSummary.model_validate(normalized)
        baseline = self.clinical_reasoner.mock_summary(
            case_input=case_input,
            stage_one=stage_one,
            consensus=consensus,
            soap=soap,
            gate=gate,
            model_route=model_route,
        )
        output = self._stabilize_clinical_reasoning_summary(output=output, baseline=baseline)
        output.model_route = model_route
        output.generation_mode = "local_medgemma"
        self._append_trace(
            record,
            "ClinicalReasoner",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            f"Local MedGemma reasoning with model={model_route}",
        )
        return output

    # -------------------------------------------------------------------------
    # Stage 3 - HITL gate
    # -------------------------------------------------------------------------

    async def _run_hitl_gatekeeper(
        self,
        record: MDTCaseRecord,
        case_input: MDTCaseInput,
        consensus: ConsensusRecommendation,
        soap: SOAPNote,
    ) -> HITLGateOutput:
        started = datetime.now(timezone.utc)

        mandatory_confirmatory_flag = (
            "DiagnostiCore output is AI-inferred and requires confirmatory molecular testing."
        )
        safety_flags = self._sanitize_flag_list(list(consensus.red_flags), max_items=5)
        if not case_input.transcript.raw_text.strip() and not (case_input.transcript.audio_uri or "").strip():
            safety_flags.append("Missing MDT transcript")
        if len(soap.plan) < 30:
            safety_flags.append("Plan text unusually short")
        if consensus.confidence < self.low_confidence_threshold:
            safety_flags.append(
                "Low consensus confidence "
                f"({consensus.confidence:.2f} < {self.low_confidence_threshold:.2f})"
            )
        if len(consensus.red_flags) >= self.red_flag_uncertainty_threshold:
            safety_flags.append(
                f"High uncertainty due to multiple red flags ({len(consensus.red_flags)})"
            )
        if case_input.diagnosticore and not case_input.diagnosticore.is_confirmed_genomic_test:
            safety_flags.append(mandatory_confirmatory_flag)
        if case_input.diagnosticore and not case_input.diagnosticore.model_card:
            safety_flags.append("DiagnostiCore model card missing in handoff payload.")
        if case_input.diagnosticore and not case_input.diagnosticore.locked_threshold_report:
            safety_flags.append("DiagnostiCore locked-threshold report missing in handoff payload.")
        safety_flags = self._sanitize_flag_list(safety_flags, max_items=6)
        if case_input.diagnosticore and not case_input.diagnosticore.is_confirmed_genomic_test:
            if not any("requires confirmatory molecular testing" in f for f in safety_flags):
                safety_flags.insert(0, mandatory_confirmatory_flag)
                safety_flags = self._sanitize_flag_list(safety_flags, max_items=6)

        checklist = [
            "Confirm final radiology report sign-off",
            "Confirm pathology receptor and grade summary",
            "Confirm molecular panel provenance and date",
            "Approve or request revision of MDT recommendation",
        ]
        if case_input.diagnosticore:
            checklist.insert(
                3,
                "If using DiagnostiCore TP53 inference, document confirmatory sequencing/assay plan.",
            )
            checklist.insert(
                4,
                "Confirm case is within DiagnostiCore model card intended use and cohort constraints.",
            )
        if consensus.confidence < self.low_confidence_threshold:
            checklist.insert(0, "Second clinician review required due to low confidence")

        gate = HITLGateOutput(
            requires_clinician_approval=True,
            blocked_actions=[
                "publish_recommendation",
                "notify_patient",
                "export_final_report",
            ],
            approval_checklist=checklist,
            safety_flags=safety_flags,
            gate_notes="Human-in-the-loop checkpoint enforced before downstream actions.",
        )
        self._append_trace(
            record,
            "HITLGatekeeper",
            AgentRunStatus.COMPLETED,
            started,
            datetime.now(timezone.utc),
            "Case moved to pending clinician approval",
        )
        return gate

    def _parse_adk_json_payload(self, raw_text: str) -> Dict[str, Any]:
        try:
            parsed = parse_json_payload(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        cleaned = (raw_text or "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        raise ValueError("Unable to parse ADK JSON payload.")

    async def _ensure_session(self, app_name: str, session_id: str) -> None:
        if self.session_service is None:
            return
        try:
            await self.session_service.create_session(
                app_name=app_name,
                user_id="mdt-system",
                session_id=session_id,
            )
        except Exception as exc:
            logger.debug("Session creation note for %s: %s", session_id, exc)

    async def _run_runner_and_collect_text(
        self,
        runner: "Runner",
        trigger_msg: "types.Content",
        session_id: str,
    ) -> str:
        final_response_text = ""
        async for event in runner.run_async(
            new_message=trigger_msg,
            user_id="mdt-system",
            session_id=session_id,
        ):
            if not hasattr(event, "content") or not event.content:
                continue
            raw_content = event.content
            if isinstance(raw_content, str):
                text = raw_content
            elif hasattr(raw_content, "parts"):
                parts = [part.text for part in raw_content.parts if getattr(part, "text", None)]
                text = "".join(parts)
            else:
                text = ""
            if text.strip():
                final_response_text = text

        if not final_response_text:
            raise RuntimeError("ADK runner produced no textual output.")
        return final_response_text

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _default_case_id(self) -> str:
        case_ids = list_available_case_ids()
        if not case_ids:
            raise RuntimeError("No local cases available.")
        return case_ids[0]

    def _get_case_or_raise(self, case_id: str) -> MDTCaseRecord:
        return self.case_repository.get_case(case_id)

    def _build_case_repository(self) -> CaseRepository:
        if self.case_store_backend == "sqlite":
            return SqliteCaseRepository(db_path=self.sqlite_db_path)
        raise ValueError(
            f"Unsupported MDT_CASE_STORE_BACKEND='{self.case_store_backend}'. "
            "Allowed value in offline mode: sqlite."
        )

    def _apply_input_overrides(
        self,
        case_input: MDTCaseInput,
        overrides: Dict[str, object],
    ) -> MDTCaseInput:
        """
        Applies user-provided UI overrides to the loaded case input.
        """
        if not overrides:
            return case_input

        # Shallow copies of nested structures to keep override writes explicit.
        if "radiology_notes" in overrides:
            case_input.imaging.ct_report = str(overrides.get("radiology_notes") or "")
        if "pathology_notes" in overrides:
            case_input.pathology.biopsy_summary = str(overrides.get("pathology_notes") or "")
        if "genomics_notes" in overrides:
            case_input.genomics.report_summary = str(overrides.get("genomics_notes") or "")
        if "transcript_notes" in overrides:
            case_input.transcript.raw_text = str(overrides.get("transcript_notes") or "")
        if "transcript_audio_uri" in overrides:
            # Empty string explicitly clears inherited seeded URIs (for frontend-transcribed mode).
            case_input.transcript.audio_uri = str(overrides.get("transcript_audio_uri") or "").strip()
        if overrides.get("diagnosticore_case_submitter_id"):
            case_input.diagnosticore_case_submitter_id = str(
                overrides["diagnosticore_case_submitter_id"]
            )
        if overrides.get("diagnosticore"):
            case_input.diagnosticore = DiagnosticorePrediction.model_validate(
                overrides["diagnosticore"]
            )

        return case_input

    def _append_trace(
        self,
        record: MDTCaseRecord,
        agent_name: str,
        status: AgentRunStatus,
        started_at: datetime,
        completed_at: datetime,
        notes: Optional[str] = None,
    ) -> None:
        record.traces.append(
            AgentTrace(
                agent=agent_name,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                notes=notes,
            )
        )

    def _validate_stage_one_payload(self, payload: Dict) -> StageOneOutput:
        if not isinstance(payload, dict):
            raise ValueError("StageOne payload must be a JSON object.")

        for key in ("radiology", "pathology", "genomics", "literature"):
            if key not in payload:
                raise ValueError(f"StageOne payload missing required section: {key}")

        try:
            radiology = RadiologySummary.model_validate(payload["radiology"])
        except Exception as exc:
            raise ValueError(f"RadiologySynthesizer schema violation: {exc}") from exc
        try:
            pathology = PathologySummary.model_validate(payload["pathology"])
        except Exception as exc:
            raise ValueError(f"PathologySynthesizer schema violation: {exc}") from exc
        try:
            genomics = GenomicsSummary.model_validate(payload["genomics"])
        except Exception as exc:
            raise ValueError(f"GenomicsSynthesizer schema violation: {exc}") from exc
        try:
            literature = LiteratureSummary.model_validate(payload["literature"])
        except Exception as exc:
            raise ValueError(f"LiteratureAgent schema violation: {exc}") from exc

        return StageOneOutput(
            radiology=radiology,
            pathology=pathology,
            genomics=genomics,
            literature=literature,
        )


# Service-level singleton
orchestrator_agent = MDTCommandOrchestrator()
