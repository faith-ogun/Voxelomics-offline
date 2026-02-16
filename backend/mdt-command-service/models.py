"""
Voxelomics MDT Command Service - Data Models

Pydantic contracts for:
- Case lifecycle
- Specialist agent outputs
- HITL gate and approval workflow
- API request/response payloads
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# ENUMS
# =============================================================================


class CaseStatus(str, Enum):
    CREATED = "created"
    ANALYZING = "analyzing"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REWORK_REQUIRED = "rework_required"
    ERROR = "error"


class AgentRunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REWORK = "rework"


# =============================================================================
# CORE CLINICAL INPUT MODELS
# =============================================================================


class GenomicAlteration(BaseModel):
    gene: str
    variant: str
    tier: Optional[str] = None
    notes: Optional[str] = None


class DiagnosticoreEvidenceItem(BaseModel):
    slide_file_id: Optional[str] = None
    tile_path: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    note: Optional[str] = None


class DiagnosticoreModelCard(BaseModel):
    cohort: Optional[str] = None
    intended_use: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class DiagnosticorePrediction(BaseModel):
    source_service: str = "diagnosticore-service"
    target: str = "tp53_mutation"
    case_submitter_id: Optional[str] = None
    tp53_probability: float = Field(ge=0.0, le=1.0)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    predicted_label: Optional[str] = None
    uncertainty_flags: List[str] = Field(default_factory=list)
    model_version: Optional[str] = None
    data_version: Optional[str] = None
    prediction_time_utc: Optional[datetime] = None
    evidence: List[DiagnosticoreEvidenceItem] = Field(default_factory=list)
    is_confirmed_genomic_test: bool = False
    model_card: Optional[DiagnosticoreModelCard] = None
    locked_threshold_report: Optional[Dict[str, Any]] = None
    locked_threshold_report_uri: Optional[str] = None
    source_split: Optional[str] = None
    n_tiles: Optional[int] = None
    raw_pred_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    calibration_method: Optional[str] = None
    wsi_project_id: Optional[str] = None
    wsi_file_id: Optional[str] = None
    wsi_file_name: Optional[str] = None
    wsi_local_path: Optional[str] = None
    tile_preview_png: Optional[str] = None
    tile_preview_x: Optional[int] = None
    tile_preview_y: Optional[int] = None
    cohort_relation: Optional[str] = None
    deepzoom_dzi_path: Optional[str] = None
    deepzoom_tile_dir: Optional[str] = None


class ImagingInput(BaseModel):
    ct_report: Optional[str] = None
    mri_report: Optional[str] = None
    pet_report: Optional[str] = None


class PathologyInput(BaseModel):
    biopsy_summary: str
    wsi_summary: Optional[str] = None
    receptor_status: Optional[str] = None
    grade: Optional[str] = None


class GenomicsInput(BaseModel):
    report_summary: str
    mutations: List[GenomicAlteration] = Field(default_factory=list)
    tmb: Optional[str] = None
    msi: Optional[str] = None


class TranscriptInput(BaseModel):
    raw_text: str
    audio_uri: Optional[str] = None


class MDTCaseInput(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    diagnosis: str
    stage: Optional[str] = None
    diagnosticore_case_submitter_id: Optional[str] = None
    imaging: ImagingInput
    pathology: PathologyInput
    genomics: GenomicsInput
    transcript: TranscriptInput
    diagnosticore: Optional[DiagnosticorePrediction] = None


# =============================================================================
# AGENT OUTPUT MODELS
# =============================================================================


class RadiologySummary(BaseModel):
    findings: str
    disease_burden: str
    action_items: List[str] = Field(default_factory=list)


class PathologySummary(BaseModel):
    diagnosis: str
    biomarkers: List[str] = Field(default_factory=list)
    risk_features: List[str] = Field(default_factory=list)


class GenomicsSummary(BaseModel):
    actionable_mutations: List[str] = Field(default_factory=list)
    interpretation: str
    molecular_risk: str


class EvidenceItem(BaseModel):
    title: str
    source: str
    year: Optional[int] = None
    identifier: Optional[str] = None
    finding: str


class LiteratureSummary(BaseModel):
    highlights: str
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ConsensusRecommendation(BaseModel):
    recommendation: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    red_flags: List[str] = Field(default_factory=list)


class SOAPNote(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class ClinicalReasoningSummary(BaseModel):
    summary: str
    key_risks: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    confirmatory_actions: List[str] = Field(default_factory=list)
    evidence_links: List[str] = Field(default_factory=list)
    uncertainty_statement: str
    model_route: Optional[str] = None
    generation_mode: Optional[str] = None


class TranscriptionResult(BaseModel):
    engine: str
    transcript: str
    wer_estimate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = None


class HITLGateOutput(BaseModel):
    requires_clinician_approval: bool = True
    blocked_actions: List[str] = Field(default_factory=list)
    approval_checklist: List[str] = Field(default_factory=list)
    safety_flags: List[str] = Field(default_factory=list)
    gate_notes: Optional[str] = None


class StageOneOutput(BaseModel):
    radiology: RadiologySummary
    pathology: PathologySummary
    genomics: GenomicsSummary
    literature: LiteratureSummary


class MDTCaseArtifacts(BaseModel):
    diagnosticore: Optional[DiagnosticorePrediction] = None
    stage_one: Optional[StageOneOutput] = None
    consensus: Optional[ConsensusRecommendation] = None
    soap_note: Optional[SOAPNote] = None
    clinical_reasoning: Optional[ClinicalReasoningSummary] = None
    transcription: Optional[TranscriptionResult] = None
    hitl_gate: Optional[HITLGateOutput] = None


class AgentTrace(BaseModel):
    agent: str
    status: AgentRunStatus
    started_at: datetime
    completed_at: datetime
    notes: Optional[str] = None


class MDTCaseRecord(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    diagnosis: str
    status: CaseStatus
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approval_notes: Optional[str] = None
    rework_reason: Optional[str] = None

    input_payload: MDTCaseInput
    artifacts: MDTCaseArtifacts = Field(default_factory=MDTCaseArtifacts)
    traces: List[AgentTrace] = Field(default_factory=list)


# =============================================================================
# API MODELS
# =============================================================================


class StartCaseRequest(BaseModel):
    case_id: Optional[str] = None
    overrides: Optional["StartCaseOverrides"] = None


class StartCaseOverrides(BaseModel):
    radiology_notes: Optional[str] = None
    pathology_notes: Optional[str] = None
    genomics_notes: Optional[str] = None
    transcript_notes: Optional[str] = None
    transcript_audio_uri: Optional[str] = None
    diagnosticore_case_submitter_id: Optional[str] = None
    diagnosticore: Optional[DiagnosticorePrediction] = None


class StartCaseResponse(BaseModel):
    success: bool
    case_id: str
    status: CaseStatus
    message: str


class AnalyzeCaseResponse(BaseModel):
    success: bool
    case_id: str
    status: CaseStatus
    message: str
    consensus: Optional[ConsensusRecommendation] = None
    hitl_gate: Optional[HITLGateOutput] = None


class CaseDraftResponse(BaseModel):
    success: bool
    case_id: str
    status: CaseStatus
    artifacts: MDTCaseArtifacts


class ApproveCaseRequest(BaseModel):
    decision: ApprovalDecision
    clinician_name: str
    notes: Optional[str] = None


class ApproveCaseResponse(BaseModel):
    success: bool
    case_id: str
    status: CaseStatus
    message: str


class CaseStatusResponse(BaseModel):
    success: bool
    case_id: str
    status: CaseStatus
    updated_at: datetime
    requires_approval: bool


class HealthResponse(BaseModel):
    status: str
    service: str
    execution_mode: str
    case_store_backend: str
    retrieval_mode: str
    adk_available: bool
    timestamp: datetime = Field(default_factory=utc_now)


class AudioUploadResponse(BaseModel):
    success: bool
    gcs_uri: str
    content_type: Optional[str] = None
    bytes_uploaded: int


class EvidenceSyncRequest(BaseModel):
    case_id: Optional[str] = None
    diagnosis: Optional[str] = None
    genes: List[str] = Field(default_factory=list)
    max_results: int = Field(default=8, ge=1, le=20)


class EvidenceSyncResponse(BaseModel):
    success: bool
    message: str
    diagnosis: str
    genes: List[str] = Field(default_factory=list)
    literature_count: int
    literature_path: str
    last_synced_at: datetime


class EvidenceSyncStatusResponse(BaseModel):
    success: bool
    retrieval_mode: str
    literature_count: int
    literature_path: str
    last_synced_at: Optional[datetime] = None


class PatientCaseHistoryItem(BaseModel):
    snapshot_id: int
    case_id: str
    patient_id: str
    patient_name: str
    diagnosis: str
    status: CaseStatus
    saved_at: datetime
    updated_at: datetime


class PatientCaseHistoryResponse(BaseModel):
    success: bool
    patient_id: str
    count: int
    cases: List[PatientCaseHistoryItem] = Field(default_factory=list)


class CaseHistorySnapshotResponse(BaseModel):
    success: bool
    snapshot_id: int
    saved_at: datetime
    case: MDTCaseRecord


class DeleteCaseHistorySnapshotResponse(BaseModel):
    success: bool
    snapshot_id: int
    message: str


# =============================================================================
# SCHEMA CHECK HELPERS
# =============================================================================


def validate_structured_output(model_cls: Any, payload: Dict[str, Any]) -> Any:
    """
    Strict schema gate used after any LLM/tool output.
    Raises pydantic ValidationError on mismatches.
    """
    return model_cls.model_validate(payload)
