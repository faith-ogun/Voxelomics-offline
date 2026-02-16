# Voxelomics Offline - 3 Minute Demo Script

## Scene 1 (0:00-0:20) Problem
On screen: Home page + case context

Say:
"Tumor boards are high-stakes and time-limited. Clinicians have to combine pathology, genomics, imaging, and dictation quickly, with clear traceability and safety checks. Voxelomics is an offline-first MDT system built for that exact workflow."

## Scene 2 (0:20-0:45) Input and Transcription
On screen: Clinical Dictation tab, upload/record flow

Say:
"We start with real clinical dictation. MedASR transcribes locally, and the transcript becomes structured input for the board workflow. This keeps sensitive audio local while still giving usable text."

## Scene 3 (0:45-1:20) Agentic Workflow
On screen: MDT Board Prep, pipeline stages and status updates

Say:
"This is not a single prompt. We run a multi-agent pipeline: domain fan-out, consensus synthesis, SOAP generation, then a clinician safety gate. MedGemma handles clinical synthesis and reasoning, while the workflow enforces human approval before final recommendation lock."

## Scene 4 (1:20-1:55) DiagnostiCore Pathology
On screen: DiagnostiCore tab with DeepZoom WSI viewer

Say:
"For pathology, we ingest whole-slide imaging, generate embeddings with Google Path Foundation, and run a TP53 prediction head. The system surfaces probability, validation metrics, and provenance so the board can review AI output as decision support, not autonomous diagnosis."

## Scene 5 (1:55-2:25) Benchmark Transparency
On screen: Fusion Inference Pipeline benchmark panel

Say:
"We benchmarked Path Foundation against our CNN baseline on the same split. Path Foundation improved recall in this configuration, while we also display tradeoffs across other metrics depending on threshold and calibration. We show this transparently inside the product."

## Scene 6 (2:25-2:45) HITL and Case Memory
On screen: HITL checklist + Patient Cases snapshot history

Say:
"Every run passes through a clinician approval checklist. We also save local patient case snapshots, so teams can reopen prior analyses without rerunning the full pipeline."

## Scene 7 (2:45-3:00) Close
On screen: Technology or Board Prep summary view

Say:
"Voxelomics combines MedASR, MedGemma, and Path Foundation in a practical agentic workflow for MDTs: local-first, transparent, and clinician-gated."
