"""
Clinical reasoning helper for MDT artifacts.

Provides:
- structured prompt/context payload builders for MedGemma-backed generation
"""

from __future__ import annotations

from typing import Dict, List

from models import (
    ClinicalReasoningSummary,
    ConsensusRecommendation,
    HITLGateOutput,
    MDTCaseInput,
    SOAPNote,
    StageOneOutput,
)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _normalize_tag(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        return text
    return f"[{text}]"


class ClinicalReasoner:
    def __init__(self) -> None:
        self.mode = "local"
        self.allow_mock_fallback = False

    def resolve_mode(self, execution_mode: str, adk_available: bool) -> str:
        _ = execution_mode
        _ = adk_available
        return "local"

    def llm_instruction(self) -> str:
        return (
            "You are ClinicalReasoner for an oncology MDT decision-support workflow. "
            "Return strict JSON only with keys: "
            "summary, key_risks, recommended_actions, confirmatory_actions, evidence_links, "
            "uncertainty_statement. "
            "Do not add markdown or extra keys. Keep wording clinically safe and explicit about "
            "AI-inferred genomic signals requiring confirmatory testing. Include source-tagged facts "
            "in summary and actions, using tags like [PMID:xxxx], [DIAGNOSTICORE]. "
            "Avoid repetition. Keep key_risks/recommended_actions concise and non-duplicative."
        )

    def llm_payload(
        self,
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
        consensus: ConsensusRecommendation,
        soap: SOAPNote,
        gate: HITLGateOutput,
    ) -> Dict[str, object]:
        return {
            "case": {
                "case_id": case_input.case_id,
                "diagnosis": case_input.diagnosis,
                "stage": case_input.stage,
            },
            "stage_one": stage_one.model_dump(mode="json"),
            "consensus": consensus.model_dump(mode="json"),
            "soap_note": soap.model_dump(mode="json"),
            "hitl_gate": gate.model_dump(mode="json"),
            "diagnosticore": (
                case_input.diagnosticore.model_dump(mode="json") if case_input.diagnosticore else None
            ),
            "rules": [
                "Anchor statements to provided evidence fields only.",
                "When diagnosticore is AI-inferred, include confirmatory molecular testing action.",
                "Preserve uncertainty and model-limit language from safety artifacts.",
            ],
        }

    def mock_summary(
        self,
        case_input: MDTCaseInput,
        stage_one: StageOneOutput,
        consensus: ConsensusRecommendation,
        soap: SOAPNote,
        gate: HITLGateOutput,
        model_route: str,
    ) -> ClinicalReasoningSummary:
        evidence_links: List[str] = []
        for e in stage_one.literature.evidence[:4]:
            if e.identifier:
                evidence_links.append(_normalize_tag(e.identifier))
            elif e.title:
                evidence_links.append(_normalize_tag(f"SRC:{e.title[:60]}"))
        if case_input.diagnosticore and case_input.diagnosticore.locked_threshold_report_uri:
            evidence_links.append(_normalize_tag("DIAGNOSTICORE"))

        route_note = "primary-route synthesis"
        if "medgemma" in (model_route or "").lower():
            route_note = "medgemma-routed synthesis"

        key_risks = _dedupe_keep_order(consensus.red_flags + gate.safety_flags)[:6]
        recommended_actions = [
            "Proceed only after clinician sign-off through HITL checklist.",
            f"{consensus.recommendation} {' '.join(evidence_links[:2])}".strip(),
            f"{soap.plan} {' '.join(evidence_links[:2])}".strip(),
        ]
        recommended_actions = _dedupe_keep_order(recommended_actions)[:5]

        confirmatory_actions: List[str] = []
        if case_input.diagnosticore and not case_input.diagnosticore.is_confirmed_genomic_test:
            confirmatory_actions.append(
                "Order confirmatory molecular assay for TP53 status before irreversible treatment decisions."
            )
            confirmatory_actions.append(
                "Document that DiagnostiCore output is AI-inferred and not standalone diagnosis."
            )
        if stage_one.radiology.disease_burden == "high":
            confirmatory_actions.append("Reconfirm baseline imaging interpretation prior to regimen finalization.")
        confirmatory_actions = _dedupe_keep_order(confirmatory_actions)

        uncertainty = (
            f"Consensus confidence is {consensus.confidence:.2f}. "
            f"{len(key_risks)} safety signal(s) are active; maintain human-in-the-loop review. "
            f"Generation path: {route_note}."
        )

        summary = (
            f"{case_input.diagnosis} case synthesis integrates imaging burden "
            f"({stage_one.radiology.disease_burden}), pathology/genomics signals, and "
            f"{len(stage_one.literature.evidence)} literature item(s). "
            f"Recommendation is decision-support only and must remain clinician-verified. "
            f"{' '.join(evidence_links[:3])}"
        )

        return ClinicalReasoningSummary(
            summary=summary,
            key_risks=key_risks,
            recommended_actions=recommended_actions,
            confirmatory_actions=confirmatory_actions,
            evidence_links=_dedupe_keep_order(evidence_links),
            uncertainty_statement=uncertainty,
            model_route=model_route,
            generation_mode="mock",
        )
