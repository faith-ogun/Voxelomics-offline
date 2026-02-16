"""
Voxelomics MDT Command Service - FastAPI Application

Endpoints:
  POST /mdt/start
  POST /mdt/{case_id}/analyze
  GET  /mdt/{case_id}/draft
  POST /mdt/{case_id}/approve
  GET  /mdt/{case_id}/status
  GET  /health
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agents import ADK_AVAILABLE, orchestrator_agent
from models import (
    AnalyzeCaseResponse,
    AudioUploadResponse,
    CaseHistorySnapshotResponse,
    DeleteCaseHistorySnapshotResponse,
    ApproveCaseRequest,
    ApproveCaseResponse,
    CaseDraftResponse,
    CaseStatusResponse,
    EvidenceSyncRequest,
    EvidenceSyncResponse,
    EvidenceSyncStatusResponse,
    HealthResponse,
    PatientCaseHistoryItem,
    PatientCaseHistoryResponse,
    StartCaseRequest,
    StartCaseResponse,
)
from tools import extract_gene_symbols, get_case_input, get_local_evidence_sync_status, sync_local_evidence_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voxelomics MDT Command Service",
    description="Offline-local MDT orchestration service with HITL safety gating",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp4", ".m4a"}
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "audio/mp4",
    "video/mp4",
    "audio/x-m4a",
}


def _debug_error_enabled() -> bool:
    return (os.getenv("MDT_EXPOSE_ERRORS", "true") or "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _error_detail(prefix: str, exc: Exception) -> str:
    if not _debug_error_enabled():
        return prefix
    detail = str(exc).strip() or exc.__class__.__name__
    # Keep payload concise for UI.
    if len(detail) > 500:
        detail = detail[:500] + "..."
    return f"{prefix} {detail}"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


ANALYZE_TIMEOUT_SECONDS = max(30.0, _env_float("MDT_ANALYZE_TIMEOUT_SECONDS", 900.0))


def _guess_upload_extension(filename: str, content_type: str) -> str:
    ext = Path(filename).suffix.lower().strip()
    if ext:
        return ext
    mime_map = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/vnd.wave": ".wav",
        "audio/mp4": ".m4a",
        "video/mp4": ".mp4",
        "audio/x-m4a": ".m4a",
    }
    return mime_map.get((content_type or "").lower(), ".wav")


def _is_supported_audio_upload(filename: str, content_type: str) -> bool:
    ext = Path(filename or "").suffix.lower().strip()
    mime = (content_type or "").lower().strip()
    return ext in SUPPORTED_AUDIO_EXTENSIONS or mime in SUPPORTED_AUDIO_MIME_TYPES


def _resolve_local_audio_dir() -> Path:
    explicit = (os.getenv("MDT_LOCAL_AUDIO_DIR") or "").strip()
    if explicit:
        target = Path(explicit).expanduser().resolve()
    else:
        base = Path(
            (os.getenv("MDT_LOCAL_DATA_DIR", "./local_data") or "./local_data").strip()
        ).expanduser().resolve()
        target = base / "audio"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _upload_audio_local(file_bytes: bytes, filename: str, content_type: str) -> str:
    audio_dir = _resolve_local_audio_dir()
    ext = _guess_upload_extension(filename, content_type)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    file_path = audio_dir / f"{timestamp}-{uuid4().hex}{ext}"
    file_path.write_bytes(file_bytes)
    return file_path.as_uri()


def _upload_audio(file_bytes: bytes, filename: str, content_type: str) -> str:
    backend = (os.getenv("MDT_AUDIO_UPLOAD_BACKEND", "local") or "local").strip().lower()
    if backend != "local":
        raise RuntimeError(
            "Offline build supports only MDT_AUDIO_UPLOAD_BACKEND=local."
        )
    return _upload_audio_local(file_bytes=file_bytes, filename=filename, content_type=content_type)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="mdt-command-service",
        execution_mode=orchestrator_agent.execution_mode,
        case_store_backend=orchestrator_agent.case_store_backend,
        retrieval_mode=orchestrator_agent.retrieval_mode,
        adk_available=ADK_AVAILABLE,
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/mdt/audio/upload", response_model=AudioUploadResponse)
async def upload_mdt_audio(file: UploadFile = File(...)) -> AudioUploadResponse:
    try:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        if len(payload) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file exceeds 25MB upload limit.")

        filename = file.filename or "recording.wav"
        content_type = file.content_type or "application/octet-stream"
        if not _is_supported_audio_upload(filename, content_type):
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format. Upload WAV, MP4, or M4A for MedASR.",
            )
        audio_uri = _upload_audio(
            file_bytes=payload,
            filename=filename,
            content_type=content_type,
        )
        return AudioUploadResponse(
            success=True,
            # Response field kept as `gcs_uri` for frontend compatibility.
            gcs_uri=audio_uri,
            content_type=content_type,
            bytes_uploaded=len(payload),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to upload MDT audio: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to upload MDT audio.") from exc


@app.post("/mdt/start", response_model=StartCaseResponse)
async def start_case(request: StartCaseRequest) -> StartCaseResponse:
    try:
        overrides = request.overrides.model_dump(exclude_none=True) if request.overrides else None
        case = orchestrator_agent.start_case(case_id=request.case_id, overrides=overrides)
        msg = "MDT case initialized."
        if overrides:
            msg = "MDT case initialized with input overrides."
        return StartCaseResponse(
            success=True,
            case_id=case.case_id,
            status=case.status,
            message=msg,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to start case: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to start case.") from exc


@app.post("/mdt/{case_id}/analyze", response_model=AnalyzeCaseResponse)
async def analyze_case(case_id: str) -> AnalyzeCaseResponse:
    try:
        case = await asyncio.wait_for(
            orchestrator_agent.analyze_case(case_id),
            timeout=ANALYZE_TIMEOUT_SECONDS,
        )
        return AnalyzeCaseResponse(
            success=True,
            case_id=case.case_id,
            status=case.status,
            message="MDT analysis completed and routed to HITL gate.",
            consensus=case.artifacts.consensus,
            hitl_gate=case.artifacts.hitl_gate,
        )
    except asyncio.TimeoutError as exc:
        timeout_seconds = max(1, int(round(ANALYZE_TIMEOUT_SECONDS)))
        logger.error(
            "Analyze case timed out after %.1fs for case %s",
            ANALYZE_TIMEOUT_SECONDS,
            case_id,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Failed to analyze case. Timed out after {timeout_seconds}s.",
        ) from exc
    except asyncio.CancelledError:
        logger.info(
            "Analyze request cancelled for case %s (shutdown or client disconnect).",
            case_id,
        )
        raise HTTPException(status_code=499, detail="Analyze request cancelled.")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to analyze case %s: %s", case_id, exc)
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Failed to analyze case.", exc),
        ) from exc


@app.get("/")
async def root() -> dict:
    return {
        "service": "mdt-command-service",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/mdt/{case_id}/draft", response_model=CaseDraftResponse)
async def get_draft(case_id: str) -> CaseDraftResponse:
    try:
        case = orchestrator_agent.get_case(case_id)
        return CaseDraftResponse(
            success=True,
            case_id=case.case_id,
            status=case.status,
            artifacts=case.artifacts,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/mdt/{case_id}/diagnosticore/tile-preview")
async def get_diagnosticore_tile_preview(case_id: str) -> FileResponse:
    try:
        case = orchestrator_agent.get_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    diagnosticore = case.artifacts.diagnosticore or case.input_payload.diagnosticore
    tile_path = (diagnosticore.tile_preview_png if diagnosticore else None) or ""
    path = Path(tile_path).resolve() if tile_path else None

    if path and path.exists() and path.is_file():
        media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        return FileResponse(path=str(path), media_type=media_type)

    # Fallback: if explicit preview path is stale/missing, serve one DeepZoom tile.
    tile_dir = (diagnosticore.deepzoom_tile_dir if diagnosticore else None) or ""
    if tile_dir:
        preview = _resolve_deepzoom_preview_tile(Path(tile_dir))
        if preview is not None:
            media_type = "image/jpeg" if preview.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            return FileResponse(path=str(preview), media_type=media_type)

    missing = str(path) if path else "(unset)"
    raise HTTPException(status_code=404, detail=f"DiagnostiCore tile preview not found: {missing}")


def _resolve_deepzoom_preview_tile(tile_dir: Path) -> Optional[Path]:
    base = tile_dir.resolve()
    if not base.exists() or not base.is_dir():
        return None

    # Pick the highest-detail level, then first tile deterministically.
    level_dirs = sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    for level in reversed(level_dirs):
        tiles = sorted(
            p
            for p in level.iterdir()
            if p.is_file() and re.fullmatch(r"\d+_\d+\.(jpg|jpeg|png)", p.name, flags=re.IGNORECASE)
        )
        if tiles:
            return tiles[0]
    return None


@app.get("/mdt/{case_id}/diagnosticore/deepzoom.dzi")
async def get_diagnosticore_deepzoom_dzi(case_id: str, request: Request) -> Response:
    try:
        case = orchestrator_agent.get_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    diagnosticore = case.artifacts.diagnosticore or case.input_payload.diagnosticore
    dzi_path = (diagnosticore.deepzoom_dzi_path if diagnosticore else None) or ""
    if not dzi_path:
        raise HTTPException(status_code=404, detail="No DiagnostiCore DeepZoom pyramid is available for this case.")

    path = Path(dzi_path).resolve()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"DiagnostiCore DZI file not found: {path}")

    content = path.read_text(encoding="utf-8")
    tile_url = f"{str(request.base_url).rstrip('/')}/mdt/{case_id}/diagnosticore/deepzoom_tiles/"
    if re.search(r'Url="[^"]+"', content):
        content = re.sub(r'Url="[^"]+"', f'Url="{tile_url}"', content, count=1)
    else:
        content = re.sub(r"<Image\s+", f'<Image Url="{tile_url}" ', content, count=1)
    return Response(content=content, media_type="application/xml")


@app.get("/mdt/{case_id}/diagnosticore/deepzoom_tiles/{level}/{tile_name}")
async def get_diagnosticore_deepzoom_tile(case_id: str, level: str, tile_name: str) -> FileResponse:
    if not re.fullmatch(r"\d+", level):
        raise HTTPException(status_code=400, detail="Invalid DeepZoom level.")
    if not re.fullmatch(r"\d+_\d+\.(jpg|jpeg|png)", tile_name, flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid DeepZoom tile name.")

    try:
        case = orchestrator_agent.get_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    diagnosticore = case.artifacts.diagnosticore or case.input_payload.diagnosticore
    tile_dir = (diagnosticore.deepzoom_tile_dir if diagnosticore else None) or ""
    if not tile_dir:
        raise HTTPException(status_code=404, detail="No DiagnostiCore DeepZoom tiles are available for this case.")

    level_dir = (Path(tile_dir).resolve() / level).resolve()
    path = (level_dir / tile_name).resolve()
    if not path.exists() or not path.is_file():
        # Some generators/storefronts disagree on jpg vs jpeg extension.
        # Recover by matching same tile stem with any supported image extension.
        tile_stem = Path(tile_name).stem
        recovered = None
        if level_dir.exists() and level_dir.is_dir():
            for candidate in sorted(level_dir.glob(f"{tile_stem}.*")):
                if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    recovered = candidate.resolve()
                    break
        if recovered is None:
            raise HTTPException(status_code=404, detail=f"DiagnostiCore DeepZoom tile not found: {path}")
        path = recovered

    media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path=str(path), media_type=media_type)


@app.post("/mdt/{case_id}/approve", response_model=ApproveCaseResponse)
async def approve_case(case_id: str, request: ApproveCaseRequest) -> ApproveCaseResponse:
    try:
        case = orchestrator_agent.approve_case(
            case_id=case_id,
            decision=request.decision,
            clinician_name=request.clinician_name,
            notes=request.notes,
        )
        msg = "Case approved and unlocked for downstream actions."
        if case.status.value == "rework_required":
            msg = "Case marked for rework."
        return ApproveCaseResponse(
            success=True,
            case_id=case.case_id,
            status=case.status,
            message=msg,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to process approval for case %s: %s", case_id, exc)
        raise HTTPException(status_code=500, detail="Failed to process approval.") from exc


@app.get("/mdt/evidence/status", response_model=EvidenceSyncStatusResponse)
async def get_evidence_sync_status() -> EvidenceSyncStatusResponse:
    status = get_local_evidence_sync_status()
    raw_last_synced = status.get("last_synced_at")
    parsed_last_synced = None
    if isinstance(raw_last_synced, str) and raw_last_synced.strip():
        try:
            parsed_last_synced = datetime.fromisoformat(raw_last_synced.strip())
        except Exception:
            parsed_last_synced = None
    return EvidenceSyncStatusResponse(
        success=True,
        retrieval_mode=orchestrator_agent.retrieval_mode,
        literature_count=int(status.get("literature_count", 0)),
        literature_path=str(status.get("literature_path", "")),
        last_synced_at=parsed_last_synced,
    )


@app.post("/mdt/evidence/sync", response_model=EvidenceSyncResponse)
async def sync_evidence_cache(request: EvidenceSyncRequest) -> EvidenceSyncResponse:
    try:
        diagnosis: Optional[str] = None
        genes: list[str] = []

        if request.case_id:
            try:
                case = orchestrator_agent.get_case(request.case_id)
                diagnosis = case.input_payload.diagnosis
                genes = extract_gene_symbols(case.input_payload)
            except KeyError:
                seeded = get_case_input(request.case_id)
                diagnosis = seeded.diagnosis
                genes = extract_gene_symbols(seeded)
        else:
            diagnosis = (request.diagnosis or "").strip() or None
            genes = [g.strip().upper() for g in request.genes if str(g).strip()]

        if not diagnosis:
            raise HTTPException(
                status_code=400,
                detail="Provide case_id or diagnosis to sync evidence.",
            )

        result = sync_local_evidence_cache(
            diagnosis=diagnosis,
            genes=genes,
            max_results=request.max_results,
        )
        synced_at = datetime.fromisoformat(str(result["last_synced_at"]))
        warnings = result.get("warnings") or []
        message = "Evidence cache synced from online sources and stored for offline use."
        if warnings:
            message = (
                "Evidence cache sync completed with partial sources. "
                "Some providers were unavailable."
            )
        return EvidenceSyncResponse(
            success=True,
            message=message,
            diagnosis=result["diagnosis"],
            genes=result["genes"],
            literature_count=result["literature_count"],
            literature_path=result["literature_path"],
            last_synced_at=synced_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to sync local evidence cache: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Failed to sync evidence cache.", exc),
        ) from exc


@app.get("/mdt/{case_id}/status", response_model=CaseStatusResponse)
async def get_status(case_id: str) -> CaseStatusResponse:
    try:
        case = orchestrator_agent.get_case(case_id)
        requires_approval = bool(
            case.artifacts.hitl_gate and case.artifacts.hitl_gate.requires_clinician_approval
        )
        return CaseStatusResponse(
            success=True,
            case_id=case.case_id,
            status=case.status,
            updated_at=case.updated_at,
            requires_approval=requires_approval,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/mdt/patients/{patient_id}/cases", response_model=PatientCaseHistoryResponse)
async def list_patient_cases(
    patient_id: str,
    since: Optional[str] = None,
    limit: int = 100,
    include_error: bool = False,
) -> PatientCaseHistoryResponse:
    try:
        parsed_since: Optional[datetime] = None
        if since and since.strip():
            raw = since.strip().replace("Z", "+00:00")
            parsed_since = datetime.fromisoformat(raw)
        rows = orchestrator_agent.list_patient_case_history(
            patient_id=patient_id,
            since=parsed_since,
            limit=max(1, min(500, int(limit))),
            include_error=include_error,
        )
        return PatientCaseHistoryResponse(
            success=True,
            patient_id=patient_id,
            count=len(rows),
            cases=[PatientCaseHistoryItem.model_validate(row) for row in rows],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to list patient case history for %s: %s", patient_id, exc)
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Failed to list patient case history.", exc),
        ) from exc


@app.get("/mdt/cases/history/{snapshot_id}", response_model=CaseHistorySnapshotResponse)
async def get_case_history_snapshot(snapshot_id: int) -> CaseHistorySnapshotResponse:
    try:
        saved_at, record = orchestrator_agent.get_case_history_snapshot(snapshot_id)
        return CaseHistorySnapshotResponse(
            success=True,
            snapshot_id=snapshot_id,
            saved_at=saved_at,
            case=record,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load case history snapshot %s: %s", snapshot_id, exc)
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Failed to load case history snapshot.", exc),
        ) from exc


@app.delete("/mdt/cases/history/{snapshot_id}", response_model=DeleteCaseHistorySnapshotResponse)
async def delete_case_history_snapshot(snapshot_id: int) -> DeleteCaseHistorySnapshotResponse:
    try:
        deleted = orchestrator_agent.delete_case_history_snapshot(snapshot_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Case history snapshot not found: {snapshot_id}.")
        return DeleteCaseHistorySnapshotResponse(
            success=True,
            snapshot_id=snapshot_id,
            message=f"Deleted case history snapshot #{snapshot_id}.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to delete case history snapshot %s: %s", snapshot_id, exc)
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Failed to delete case history snapshot.", exc),
        ) from exc
