import React from 'react';
import {
  Activity,
  Brain,
  Database,
  HardDrive,
  LockKeyhole,
  Mic,
  ServerCog,
  Shield,
  Sparkles,
} from 'lucide-react';

const modules = [
  {
    title: 'MedASR (On-Device Dictation)',
    icon: Mic,
    desc: 'Frontend ONNX worker transcribes local audio with runtime providers set to WebGPU/WASM. Backend local MedASR remains available as fallback.',
    tag: 'Speech Layer',
  },
  {
    title: 'MedGemma (Local Clinical Synthesis)',
    icon: Brain,
    desc: 'MDT orchestration routes specialist drafting and synthesis through local MedGemma (`google/medgemma-4b-it`) loaded from local model files.',
    tag: 'Reasoning Layer',
  },
  {
    title: 'Local Retrieval Cache',
    icon: Database,
    desc: 'Evidence lookup runs in local mode against local snapshots, so tumor board drafting remains functional without live internet retrieval.',
    tag: 'Evidence Layer',
  },
  {
    title: 'Local Persistence',
    icon: HardDrive,
    desc: 'Case state is persisted in SQLite and uploaded audio is stored in local directories to keep runtime data anchored to the machine.',
    tag: 'Data Layer',
  },
  {
    title: 'DiagnostiCore Handoff',
    icon: Activity,
    desc: 'DiagnostiCore now uses Google Path Foundation embeddings with a TP53 prediction head, then hands off calibrated case-level outputs through local files.',
    tag: 'Diagnostic Layer',
  },
  {
    title: 'HITL Safety Gate',
    icon: Shield,
    desc: 'Human approval checkpoints remain mandatory before generated outputs are finalized or exported.',
    tag: 'Safety Layer',
  },
];

const specs = [
  { label: 'Runtime Profile', value: 'Offline', note: 'Service defaults to `MDT_EXECUTION_MODE=local`.' },
  { label: 'Agent Runtime', value: 'Modular', note: 'Local orchestrator by default; optional `adk_local` stage-one pipeline.' },
  { label: 'Retrieval Mode', value: 'Local Cache', note: '`MDT_RETRIEVAL_MODE=local` with local evidence snapshot files.' },
  { label: 'Reasoning Model', value: 'MedGemma 4B', note: '`google/medgemma-4b-it` loaded from local model directories.' },
  { label: 'Pathology Model', value: 'Path Foundation + TP53 Head', note: '`google/path-foundation` embeddings + local logistic TP53 classifier head.' },
  { label: 'Speech Stack', value: 'MedASR', note: 'Frontend ONNX worker + backend local fallback (`medasr-local`).' },
  { label: 'Case Store', value: 'SQLite', note: 'Case and workflow state persisted to local SQLite paths.' },
  { label: 'Service URL', value: '127.0.0.1:8084', note: 'Frontend technology flow targets localhost MDT command service.' },
  { label: 'WSI Viewer', value: 'DeepZoom Local', note: 'OpenSeadragon serves local DZI + tile pyramids for interactive slide review.' },
];

const MODEL_SPEC_LABELS = new Set(['Speech Stack', 'Reasoning Model', 'Pathology Model']);

const agenticStages = [
  {
    title: 'Stage 1 Fan-Out',
    detail:
      'Radiology, pathology, genomics, and literature synthesis run as modular specialist steps. In `adk_local`, these execute as ADK sub-agents.',
    icon: Activity,
  },
  {
    title: 'Concurrent MedASR',
    detail:
      'Transcription runs in parallel with stage-one synthesis, keeping dictation and reasoning on the same offline case timeline.',
    icon: Mic,
  },
  {
    title: 'Stage 2 Chain',
    detail:
      'ConsensusSynthesizer and SOAPGenerator run in sequence to produce structured draft recommendations.',
    icon: Brain,
  },
  {
    title: 'HITL Final Gate',
    detail:
      'HITLGatekeeper enforces clinician approval before final output, with uncertainty framing surfaced for safer decision review.',
    icon: Shield,
  },
];

const callableTools = [
  'fetch_radiology_context',
  'fetch_pathology_context',
  'fetch_genomics_context',
  'fetch_diagnosticore_context',
  'fetch_literature_context',
  'search_literature_evidence',
];

export const Technology: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#060b2e] text-white pt-28 pb-24 px-4">
      <div className="max-w-6xl mx-auto">
        <section className="text-center">
          <p className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[#2f3a73] text-[#57D4F4] text-xs font-bold uppercase tracking-[0.2em]">
            <Sparkles className="w-3.5 h-3.5" />
            Engineering & Performance
          </p>
          <h1
            className="mt-7 text-[3rem] md:text-[5rem] leading-[0.9] font-black tracking-tight"
            style={{ fontFamily: '"Inter", "Manrope", "Segoe UI", sans-serif' }}
          >
            Local Native.
            <span className="block text-[#8A78FF]">Clinically Intelligent.</span>
          </h1>
          <p className="mt-6 text-xl text-[#a5b2de] max-w-3xl mx-auto">
            This technology stack is intentionally offline-at-runtime: local dictation with MedASR, local synthesis with
            MedGemma, local retrieval cache, and clinician-in-the-loop gating for final outputs.
          </p>
        </section>

        <section className="mt-16 grid grid-cols-1 lg:grid-cols-[1.05fr_0.95fr] gap-6">
          <article className="rounded-[28px] border border-[#27315f] bg-gradient-to-br from-[#0b123f] to-[#060a24] p-8">
            <div className="w-12 h-12 rounded-xl bg-[#57D4F4] text-[#031425] flex items-center justify-center">
              <Brain className="w-6 h-6" />
            </div>
            <h2 className="mt-6 text-4xl font-extrabold">Offline Clinical Runtime</h2>
            <p className="mt-4 text-[#b6c0e3] text-lg leading-relaxed">
              The MDT command workflow runs with local services, local models, and local persistence by default.
              No cloud relay is required for baseline tumor board analysis flow.
            </p>
            <div className="mt-6 flex flex-wrap gap-3 text-xs font-bold uppercase tracking-[0.12em]">
              <span className="px-3 py-1.5 rounded-full border border-[#2e3a73] text-[#57D4F4]">Local MedASR</span>
              <span className="px-3 py-1.5 rounded-full border border-[#2e3a73] text-[#57D4F4]">Local MedGemma</span>
              <span className="px-3 py-1.5 rounded-full border border-[#2e3a73] text-[#57D4F4]">Path Foundation + TP53 Head</span>
              <span className="px-3 py-1.5 rounded-full border border-[#2e3a73] text-[#9cb0f4]">SQLite + Local Files</span>
              <span className="px-3 py-1.5 rounded-full border border-[#2e3a73] text-[#9cb0f4]">HITL Safety Gate</span>
            </div>
          </article>

          <div className="grid gap-6">
            <article className="rounded-[24px] border border-[#4450a9] bg-[#8A78FF] text-[#f4f1ff] p-7">
              <div className="w-10 h-10 rounded-lg bg-black text-[#8A78FF] flex items-center justify-center">
                <LockKeyhole className="w-5 h-5" />
              </div>
              <h3 className="mt-4 text-3xl font-extrabold">Privacy Vault</h3>
              <p className="mt-3 text-[#f4f1ff] text-base">
                Runtime data paths are local-first: audio uploads, case state, and evidence snapshots remain on local
                storage unless a deployment explicitly changes this profile.
              </p>
            </article>
            <article className="rounded-[24px] border border-[#27315f] bg-[#070d2f] p-7">
              <div className="w-10 h-10 rounded-lg bg-[#2553df] text-white flex items-center justify-center">
                <ServerCog className="w-5 h-5" />
              </div>
              <h3 className="mt-4 text-3xl font-extrabold">No Cloud Dependency</h3>
              <p className="mt-3 text-[#b6c0e3] text-base">
                The frontend points to localhost (`127.0.0.1:8084`) and the backend enforces local runtime modes for
                retrieval, MedASR, and MedGemma execution.
              </p>
            </article>
          </div>
        </section>

        <section className="mt-20">
          <h2 className="text-center text-4xl font-black tracking-tight text-[#9aa6d1]">System Specifications</h2>
          <div className="mt-7 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px rounded-2xl overflow-hidden border border-[#27315f] bg-[#27315f]">
            {specs.map((spec) => {
              const isModelSpec = MODEL_SPEC_LABELS.has(spec.label);
              return (
              <article
                key={spec.label}
                className={
                  isModelSpec
                    ? 'bg-gradient-to-br from-[#08133f] to-[#070d2f] p-7 transition-all duration-200 shadow-[inset_0_0_0_1px_rgba(87,212,244,0.28)] hover:shadow-[inset_0_0_0_1px_rgba(87,212,244,0.75),0_0_26px_rgba(87,212,244,0.24)] hover:-translate-y-0.5'
                    : 'bg-[#070d2f] p-7 transition-all duration-200'
                }
              >
                <p className={`text-xs uppercase tracking-[0.25em] font-bold ${isModelSpec ? 'text-[#57D4F4]' : 'text-[#7f8cb6]'}`}>
                  {spec.label}
                </p>
                <p className="mt-2 text-4xl font-extrabold">{spec.value}</p>
                <p className="mt-2 text-[#9eabd6]">{spec.note}</p>
              </article>
            )})}
          </div>
        </section>

        <section className="mt-16 rounded-[28px] border border-[#27315f] bg-gradient-to-br from-[#0b123f] to-[#070d2f] p-8">
          <div className="max-w-4xl">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-[#57D4F4]">Agentic Workflow</p>
            <h2 className="mt-3 text-4xl font-black tracking-tight">Callable Tools, Offline Orchestration</h2>
            <p className="mt-4 text-[#b6c0e3] text-lg">
              This stack qualifies as agentic by design: it decomposes work into specialized modules, executes staged
              orchestration, and invokes callable tools for context retrieval and evidence grounding. Cloud Gemini is not
              required for this behavior. `adk_local` can still run locally with a local model backend (for example,
              Qwen via Ollama) while staying offline.
            </p>
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-5">
            {agenticStages.map(({ title, detail, icon: Icon }) => (
              <article key={title} className="rounded-2xl border border-[#2a3668] bg-[#090f35] p-6">
                <div className="w-10 h-10 rounded-lg bg-[#121a4a] text-[#57D4F4] flex items-center justify-center">
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="mt-4 text-2xl font-extrabold">{title}</h3>
                <p className="mt-2 text-[#9eabd6]">{detail}</p>
              </article>
            ))}
          </div>

          <div className="mt-8 rounded-2xl border border-[#2a3668] bg-[#090f35] p-6">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-[#7f8cb6]">Callable Tool Surface</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {callableTools.map((tool) => (
                <span
                  key={tool}
                  className="px-3 py-1.5 rounded-full border border-[#33417a] bg-[#070d2f] text-[#b8c5ef] text-xs font-semibold"
                >
                  {tool}
                </span>
              ))}
            </div>
          </div>
        </section>

        <section className="mt-16 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {modules.map(({ title, icon: Icon, desc, tag }) => (
            <article key={title} className="rounded-2xl border border-[#27315f] bg-[#070d2f] p-6">
              <div className="flex items-center justify-between">
                <div className="w-11 h-11 rounded-xl bg-[#121a4a] text-[#57D4F4] flex items-center justify-center">
                  <Icon className="w-5 h-5" />
                </div>
                <span className="text-xs font-bold uppercase tracking-[0.2em] text-[#7f8cb6]">{tag}</span>
              </div>
              <h3 className="mt-4 text-2xl font-extrabold">{title}</h3>
              <p className="mt-2 text-[#9eabd6]">{desc}</p>
            </article>
          ))}
        </section>
      </div>
    </div>
  );
};
