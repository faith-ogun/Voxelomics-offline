# Project name
Voxelomics Offline: Agentic MDT Co-Pilot for Clinician-Gated Tumor Board Decisions

# Your team
Faith Ogundimu - project lead, clinical domain framing, system design, implementation, evaluation, and demo production.

# Problem statement
Multidisciplinary tumor boards must synthesize radiology, pathology, genomics, literature, and clinical dictation under severe time pressure. In practice, this process is fragmented, hard to audit, and difficult to deploy in privacy-constrained environments.

Voxelomics Offline addresses this by providing an offline-first, clinician-centered workflow that reduces manual synthesis burden while preserving human oversight.

Primary users are MDT clinicians who need faster, traceable, and safer case preparation, without requiring always-on cloud inference.

# Overall solution
Voxelomics Offline combines three Google Health AI Developer Foundations models in one agentic workflow:

1. MedASR for local speech-to-text clinical dictation.
2. MedGemma for structured multimodal synthesis and recommendation drafting.
3. Path Foundation embeddings with a TP53 prediction head for pathology-side risk support from WSI.

The orchestration flow uses specialized agents for domain fan-out, consensus synthesis, SOAP generation, and a mandatory HITL safety gate before final recommendation lock.

This is designed as decision support, not autonomous diagnosis. Outputs include uncertainty framing, confidence context, and explicit prompts for confirmatory testing.

# Technical details
System architecture:

- Frontend: React/TypeScript MDT workspace with dictation, reasoning, pathology view, and patient case snapshots.
- Backend: local-first orchestrator service with timeout guards and draft polling.
- Storage: local SQLite for case persistence/history.
- Speech: browser-local MedASR worker path with fallback-safe behavior.
- Pathology: DeepZoom whole-slide viewing + Path Foundation TP53 head outputs.

Agentic workflow design:

- Parallel specialists (radiology, pathology, genomics, literature, transcription)
- Sequential synthesis (consensus + SOAP)
- Human safety gate with clinician approval checklist

Pathology modeling and benchmark context:

- Path Foundation TP53 head was benchmarked against CNN baseline on the same split.
- In that benchmark configuration, Path Foundation improved recall (with metric tradeoffs depending on threshold/calibration).
- Metrics and model artifacts are surfaced in-product for transparency.

Feasibility and deployment:

- Built for offline-first local operation.
- Provides deterministic local case replay via saved snapshots.
- Includes explicit model limitations and confirmatory-testing guardrails.

Links:

- Required video (<=3 min): [ADD LINK]
- Required public code repository: [ADD LINK]
- Bonus interactive demo: [ADD LINK]
- Bonus Hugging Face artifact/model card: [ADD LINK]
