import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { BrainCircuit, CheckCircle, Plus, ShieldCheck, Stethoscope, Microscope, Upload, Users } from 'lucide-react';
import { HeroScrollDemo } from './ui/demo';
import { RotatingWireframeCube } from './ui/RotatingWireframeCube';

const HeroBackground = () => {
  return (
    <div className="absolute inset-0 overflow-hidden w-full h-full pointer-events-none bg-[#f7f9ff]">
      <div className="absolute inset-0 bg-gradient-to-b from-white via-[#f7f9ff] to-[#eff3ff]" />
      <motion.div
        className="absolute -top-24 -left-24 w-[380px] h-[380px] rounded-full bg-cyan-200/45 blur-[70px]"
        animate={{ x: [0, 20, 0], y: [0, 30, 0] }}
        transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute -top-16 right-0 w-[360px] h-[360px] rounded-full bg-indigo-200/45 blur-[75px]"
        animate={{ x: [0, -25, 0], y: [0, 18, 0] }}
        transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute -bottom-20 left-1/3 w-[420px] h-[420px] rounded-full bg-[#8A78FF]/25 blur-[85px]"
        animate={{ y: [0, -18, 0] }}
        transition={{ duration: 11, repeat: Infinity, ease: 'easeInOut' }}
      />
      <div className="absolute inset-0 opacity-[0.05] mix-blend-multiply" style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 200 200\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noiseFilter\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.75\' numOctaves=\'2\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noiseFilter)\'/%3E%3C/svg%3E")' }}></div>
    </div>
  );
};

const InfinityRibbon = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;
    let rafId = 0;

    const resizeCanvas = () => {
      const container = canvas.parentElement;
      if (!container) return;
      const dpr = window.devicePixelRatio || 1;
      const width = container.clientWidth;
      const height = container.clientHeight;
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const drawHelix = () => {
      const width = canvas.width / (window.devicePixelRatio || 1);
      const height = canvas.height / (window.devicePixelRatio || 1);
      ctx.clearRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;
      const helixWidth = width * 0.95;
      const amplitude = 85;
      const numPoints = 40;
      const spacing = helixWidth / numPoints;
      const startX = centerX - helixWidth / 2;

      // Helix ladder connectors
      ctx.strokeStyle = 'rgba(59, 76, 202, 0.15)';
      ctx.lineWidth = 4;

      for (let i = 0; i < numPoints; i++) {
        const x = startX + i * spacing;
        const phase = (i / numPoints) * Math.PI * 4 + time;
        const y1 = centerY + Math.sin(phase) * amplitude;
        const y2 = centerY + Math.sin(phase + Math.PI) * amplitude;

        if (i % 3 === 0) {
          ctx.beginPath();
          ctx.moveTo(x, y1);
          ctx.lineTo(x, y2);
          ctx.stroke();
        }
      }

      // Strand 1 with depth alpha
      ctx.beginPath();
      ctx.strokeStyle = '#3B4CCA';
      ctx.lineWidth = 3;
      for (let i = 0; i < numPoints; i++) {
        const x = startX + i * spacing;
        const phase = (i / numPoints) * Math.PI * 4 + time;
        const y = centerY + Math.sin(phase) * amplitude;
        const depth = Math.cos(phase);

        ctx.globalAlpha = 0.4 + depth * 0.3;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Strand 2 with depth alpha
      ctx.beginPath();
      ctx.strokeStyle = '#6B7CFF';
      for (let i = 0; i < numPoints; i++) {
        const x = startX + i * spacing;
        const phase = (i / numPoints) * Math.PI * 4 + time + Math.PI;
        const y = centerY + Math.sin(phase) * amplitude;
        const depth = Math.cos(phase);

        ctx.globalAlpha = 0.4 + depth * 0.3;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Base pair nodes
      ctx.globalAlpha = 1;
      for (let i = 0; i < numPoints; i++) {
        const x = startX + i * spacing;
        const phase = (i / numPoints) * Math.PI * 4 + time;
        const depth1 = Math.cos(phase);
        const depth2 = Math.cos(phase + Math.PI);
        const y1 = centerY + Math.sin(phase) * amplitude;
        const y2 = centerY + Math.sin(phase + Math.PI) * amplitude;

        ctx.beginPath();
        ctx.fillStyle = depth1 > 0 ? '#3B4CCA' : '#2A3A99';
        ctx.arc(x, y1, 4 + depth1 * 2, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = depth2 > 0 ? '#6B7CFF' : '#3B4CCA';
        ctx.arc(x, y2, 4 + depth2 * 2, 0, Math.PI * 2);
        ctx.fill();
      }

      time += 0.015;
      rafId = requestAnimationFrame(drawHelix);
    };

    resizeCanvas();
    drawHelix();
    window.addEventListener('resize', resizeCanvas);
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  return (
    <div className="relative left-1/2 -translate-x-1/2 w-screen h-[180px] sm:h-[220px] md:h-[255px]">
      <canvas ref={canvasRef} className="w-full h-full" aria-hidden="true" />
    </div>
  );
};

interface HomeProps {
  onNavigate: (section: string) => void;
  isLoggedIn?: boolean;
  userRole?: 'patient' | 'clinician';
}

export const Home: React.FC<HomeProps> = ({ onNavigate }) => {
  const stripItems = ['MDT READY WORKFLOWS', 'CLINICAL SAFETY GATES', 'PATHOLOGY + IMAGING SYNTHESIS', 'EVIDENCE-BACKED REASONING'];

  return (
    <div className="w-full">
      <section className="relative min-h-[92vh] flex items-center overflow-hidden pt-28 pb-16 px-4">
        <HeroBackground />
        <div className="relative z-10 max-w-6xl mx-auto w-full">
          <motion.div
            initial={{ opacity: 0, y: 26 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="relative -mt-2"
          >
            <InfinityRibbon />
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-12 items-start mt-10">
            <div>
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="inline-flex items-center gap-3 mb-7">
                <span className="px-3 py-1 rounded-full bg-blue-600 text-white text-xs font-semibold">Main Track</span>
                <span className="text-[#5f688a] text-sm font-medium">MedGemma Impact Challenge 2026</span>
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-[2.75rem] sm:text-6xl md:text-[4.2rem] font-display font-semibold text-[#121424] leading-[0.95] tracking-tight mb-6"
              >
                Pathology + Imaging + Context
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-[#0B0D7D] via-[#3D53D8] to-[#8A78FF]">
                  into one MDT-ready view
                </span>
              </motion.h1>
              <p className="text-[#4e587d] text-lg max-w-xl">
                Voxelomics turns fragmented oncology signals into one clinician-gated decision surface with traceable
                reasoning and faster board preparation.
              </p>
            </div>

            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="relative"
            >
              <div className="relative bg-white/90 border border-[#dbe2fb] rounded-3xl p-7 shadow-[0_20px_42px_rgba(28,45,117,0.12)]">
                <p className="text-[#4e587d] leading-relaxed text-[17px] max-w-md mb-7">
                  Start directly in MDT Command for live board preparation without any sign-in step.
                </p>
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    onClick={() => onNavigate('mdt-command')}
                    className="px-5 py-2.5 rounded-full bg-blue-600 text-white font-semibold text-sm hover:bg-blue-700 transition-colors"
                  >
                    Open MDT Command
                  </button>
                  <button
                    onClick={() => onNavigate('technology')}
                    className="px-5 py-2.5 rounded-full border border-blue-300 text-blue-700 bg-white font-semibold text-sm hover:bg-blue-50 transition-colors"
                  >
                    See Technology
                  </button>
                </div>
                <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm font-medium text-[#475279]">
                  {['MedGemma Routing', 'Structured Clinical Reasoning', 'Human-In-The-Loop'].map((item) => (
                    <span key={item} className="inline-flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-blue-600" />
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <section className="py-8 bg-[#07085F] text-[#57D4F4] overflow-hidden border-y border-[#2a3a99]">
        <div className="whitespace-nowrap flex items-center gap-8 animate-marquee px-6">
          {[...stripItems, ...stripItems].map((item, i) => (
            <div key={`${item}-${i}`} className="inline-flex items-center gap-4">
              <span className="text-4xl font-black tracking-tight">{item}</span>
              <div className="w-10 h-10 rounded-full border-2 border-[#57D4F4] flex items-center justify-center">
                <Plus className="w-5 h-5" />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="py-24 px-4 bg-[#eef2ff]">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-center text-4xl md:text-6xl font-black tracking-tight text-[#11173d]">
            Who is Voxelomics for?
          </h2>
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <article className="rounded-3xl border-2 border-black bg-[#e7f4ff] p-8 shadow-[6px_6px_0_#000] transform-gpu transition-all duration-300 ease-out hover:-rotate-[1.5deg] hover:scale-[1.03] hover:-translate-y-1 hover:shadow-[10px_10px_0_#000]">
              <div className="w-12 h-12 rounded-full bg-black text-[#57D4F4] flex items-center justify-center">
                <Microscope className="w-5 h-5" />
              </div>
              <h3 className="mt-6 text-4xl font-black">Pathologists</h3>
              <p className="mt-4 text-[#37475c] text-lg">
                Surface morphology and risk context faster for multidisciplinary review.
              </p>
            </article>
            <article className="rounded-3xl border-2 border-black bg-[#edf0ff] p-8 shadow-[6px_6px_0_#000] transform-gpu transition-all duration-300 ease-out hover:-rotate-[1.5deg] hover:scale-[1.03] hover:-translate-y-1 hover:shadow-[10px_10px_0_#000]">
              <div className="w-12 h-12 rounded-full bg-black text-[#8A78FF] flex items-center justify-center">
                <Stethoscope className="w-5 h-5" />
              </div>
              <h3 className="mt-6 text-4xl font-black">Oncologists</h3>
              <p className="mt-4 text-[#37475c] text-lg">
                Get structured recommendations with explicit uncertainty and action trails.
              </p>
            </article>
            <article className="rounded-3xl border-2 border-black bg-[#e3e8ff] p-8 shadow-[6px_6px_0_#000] transform-gpu transition-all duration-300 ease-out hover:-rotate-[1.5deg] hover:scale-[1.03] hover:-translate-y-1 hover:shadow-[10px_10px_0_#000]">
              <div className="w-12 h-12 rounded-full bg-black text-[#57D4F4] flex items-center justify-center">
                <Users className="w-5 h-5" />
              </div>
              <h3 className="mt-6 text-4xl font-black">MDT Boards</h3>
              <p className="mt-4 text-[#37475c] text-lg">
                Align pathology, imaging, and evidence into one board-ready clinical summary.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="py-24 px-4 bg-[#2663eb] text-center">
          <h2
            className="text-5xl md:text-7xl font-black text-white leading-[0.9]"
            style={{ fontFamily: '"Inter", "Manrope", "Segoe UI", sans-serif' }}
          >
            Clinical Signal.
            <span className="block text-[#dce8ff]">Actionable Confidence.</span>
          </h2>
        </section>

      <section className="py-24 px-4 bg-[#050918] text-white">
          <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
            <div>
            <span className="inline-flex px-4 py-1.5 rounded-full bg-[#2663eb] text-white font-bold text-sm">Clinician-Gated MDT Workflow</span>
            <h2 className="mt-6 text-5xl font-black leading-[0.95]">
              Your Data.
              <span className="block">Your Voxel.</span>
            </h2>
            <p className="mt-6 text-[#a9b3dc] text-2xl leading-relaxed">
              Voxelomics keeps the clinical workflow transparent, auditable, and centered on clinician approval before
              recommendation finalization.
            </p>
            <button
              onClick={() => onNavigate('technology')}
              className="mt-8 px-8 py-4 bg-white text-black rounded-full text-2xl font-black hover:bg-slate-200 transition-colors"
            >
              See How It Works
            </button>
          </div>
          <div className="rounded-3xl border border-[#2b355f] bg-[#060b1f] p-4">
            <RotatingWireframeCube />
          </div>
        </div>
      </section>

      <section className="relative py-24 px-4 bg-[#f8fbff] overflow-hidden">
        <div className="pointer-events-none absolute -top-20 -left-16 h-72 w-72 rounded-full bg-[#9ac4ff]/35 blur-3xl" />
        <div className="pointer-events-none absolute top-12 right-10 h-56 w-56 rounded-full bg-[#b7a6ff]/35 blur-3xl" />
        <div className="pointer-events-none absolute bottom-4 left-1/3 h-64 w-64 rounded-full bg-[#7fe0ff]/30 blur-3xl" />
        <div className="pointer-events-none absolute inset-0 opacity-[0.06] mix-blend-multiply" style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 200 200\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noiseFilter2\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.8\' numOctaves=\'2\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noiseFilter2)\'/%3E%3C/svg%3E")' }} />
        <div className="max-w-6xl mx-auto relative z-10">
          <div className="text-center max-w-4xl mx-auto">
            <span className="inline-flex px-4 py-1.5 rounded-full border border-[#a9c4ff] bg-white text-[#1f3f95] text-sm font-bold">
              Google Health AI Developer Foundations
            </span>
            <h2 className="mt-6 text-5xl md:text-6xl font-black tracking-tight text-[#11173d]">
              Three Models.
              <span className="block text-[#2e5ce3]">One Clinical Workflow.</span>
            </h2>
            <p className="mt-5 text-xl text-[#44517a]">
              We combine speech, reasoning, and pathology understanding so the MDT team can move from raw inputs to
              clinician-reviewed decisions in one place.
            </p>
          </div>

          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <article className="rounded-3xl border border-[#c8d8ff] bg-white/95 p-8 shadow-[0_16px_36px_rgba(28,45,117,0.12)] transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_26px_48px_rgba(28,45,117,0.2)]">
              <div className="w-12 h-12 rounded-full bg-[#eef3ff] text-[#2663eb] flex items-center justify-center">
                <Upload className="w-5 h-5" />
              </div>
              <h3 className="mt-6 text-3xl font-black text-[#121a3f]">MedASR</h3>
              <p className="mt-3 text-[#4d5a85] text-lg">
                Transcribes uploaded or recorded board audio into editable clinical dictation for the case workspace.
              </p>
            </article>

            <article className="rounded-3xl border border-[#c8d8ff] bg-white/95 p-8 shadow-[0_16px_36px_rgba(28,45,117,0.12)] transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_26px_48px_rgba(28,45,117,0.2)]">
              <div className="w-12 h-12 rounded-full bg-[#eef3ff] text-[#2663eb] flex items-center justify-center">
                <BrainCircuit className="w-5 h-5" />
              </div>
              <h3 className="mt-6 text-3xl font-black text-[#121a3f]">MedGemma 4B</h3>
              <p className="mt-3 text-[#4d5a85] text-lg">
                Synthesizes radiology, pathology, genomics, and evidence into board-ready recommendations with safety checks.
              </p>
            </article>

            <article className="rounded-3xl border border-[#c8d8ff] bg-white/95 p-8 shadow-[0_16px_36px_rgba(28,45,117,0.12)] transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_26px_48px_rgba(28,45,117,0.2)]">
              <div className="w-12 h-12 rounded-full bg-[#eef3ff] text-[#2663eb] flex items-center justify-center">
                <Microscope className="w-5 h-5" />
              </div>
              <h3 className="mt-6 text-3xl font-black text-[#121a3f]">Path Foundation</h3>
              <p className="mt-3 text-[#4d5a85] text-lg">
                Generates pathology slide embeddings that power the TP53 prediction head used by DiagnostiCore.
              </p>
            </article>
          </div>

          <div className="mt-10 flex justify-center">
            <button
              onClick={() => onNavigate('technology')}
              className="px-7 py-3 rounded-full bg-[#2663eb] text-white text-lg font-bold hover:bg-[#1f53c7] transition-colors"
            >
              Check out Technology for more
            </button>
          </div>
        </div>
      </section>

      <section className="py-24 px-4 bg-[#f2f6ff] overflow-hidden relative">
        <div className="max-w-6xl mx-auto relative z-10">
          <HeroScrollDemo />
        </div>
      </section>

      <section className="py-24 px-4 bg-gradient-to-br from-[#2663eb] via-[#4a7ef1] to-[#7aa2f7] text-white relative overflow-hidden">
        <div className="pointer-events-none absolute -top-24 -left-16 h-64 w-64 rounded-full bg-white/25 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-28 right-0 h-72 w-72 rounded-full bg-[#2663eb]/25 blur-3xl" />
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-4xl mx-auto">
            <h2 className="text-5xl md:text-7xl font-black tracking-tight text-white">
              The Workflow.
            </h2>
            <p className="mt-4 text-xl md:text-2xl font-semibold text-[#e6efff]">
              Three clear steps. One clinician-gated path to board-ready decisions.
            </p>
          </div>

          <div className="mt-14 grid grid-cols-1 lg:grid-cols-3 gap-6">
            <article className="group relative rounded-[34px] border border-[#c7d8ff] bg-[#f5f8ff] p-8 text-[#10285e] shadow-[0_16px_30px_rgba(16,40,94,0.24)] transform-gpu transition-all duration-300 ease-out hover:scale-[1.03] hover:-translate-y-1 hover:shadow-[0_28px_48px_rgba(0,0,0,0.35)]">
              <div className="pointer-events-none absolute inset-0 rounded-[34px] bg-black/0 transition-colors duration-300 group-hover:bg-black/10" />
              <div className="relative z-10 w-20 h-20 rounded-3xl bg-[#2663eb] text-white flex items-center justify-center transition-transform duration-300 group-hover:scale-110">
                <Upload className="w-9 h-9" />
              </div>
              <p className="relative z-10 absolute top-8 right-8 text-7xl font-black text-[#2663eb]/25">01</p>
              <h3 className="relative z-10 mt-9 text-4xl font-black leading-tight">Load the Case</h3>
              <p className="relative z-10 mt-5 text-2xl text-[#3a5188] leading-relaxed">
                Bring pathology, imaging, and patient context into one shared MDT workspace.
              </p>
            </article>

            <article className="group relative rounded-[34px] border border-[#b4c9ff] bg-[#eaf1ff] p-8 text-[#10285e] shadow-[0_16px_30px_rgba(16,40,94,0.24)] transform-gpu transition-all duration-300 ease-out hover:scale-[1.03] hover:-translate-y-1 hover:shadow-[0_28px_48px_rgba(0,0,0,0.35)]">
              <div className="pointer-events-none absolute inset-0 rounded-[34px] bg-black/0 transition-colors duration-300 group-hover:bg-black/10" />
              <div className="relative z-10 w-20 h-20 rounded-3xl bg-[#2663eb] text-white flex items-center justify-center transition-transform duration-300 group-hover:scale-110">
                <BrainCircuit className="w-9 h-9" />
              </div>
              <p className="relative z-10 absolute top-8 right-8 text-7xl font-black text-[#2663eb]/25">02</p>
              <h3 className="relative z-10 mt-9 text-4xl font-black leading-tight">AI Builds Context</h3>
              <p className="relative z-10 mt-5 text-2xl text-[#3a5188] leading-relaxed">
                DiagnostiCore and PathGenomicPredictor synthesize signals with evidence-linked reasoning.
              </p>
            </article>

            <article className="group relative rounded-[34px] border border-[#c7d8ff] bg-[#f5f8ff] p-8 text-[#10285e] shadow-[0_16px_30px_rgba(16,40,94,0.24)] transform-gpu transition-all duration-300 ease-out hover:scale-[1.03] hover:-translate-y-1 hover:shadow-[0_28px_48px_rgba(0,0,0,0.35)]">
              <div className="pointer-events-none absolute inset-0 rounded-[34px] bg-black/0 transition-colors duration-300 group-hover:bg-black/10" />
              <div className="relative z-10 w-20 h-20 rounded-3xl bg-[#2663eb] text-white flex items-center justify-center transition-transform duration-300 group-hover:scale-110">
                <ShieldCheck className="w-9 h-9" />
              </div>
              <p className="relative z-10 absolute top-8 right-8 text-7xl font-black text-[#2663eb]/25">03</p>
              <h3 className="relative z-10 mt-9 text-4xl font-black leading-tight">Clinician Signs Off</h3>
              <p className="relative z-10 mt-5 text-2xl text-[#3a5188] leading-relaxed">
                MDT Command keeps humans in control before any recommendation is finalized.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="py-24 px-4 bg-[#e6ecff] text-black">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-center text-5xl md:text-6xl font-black tracking-tight">Why Voxelomics?</h2>
          <p className="text-center text-2xl mt-4">Traditional workflows are fragmented. We built for MDT velocity and safety.</p>
          <div className="mt-10 rounded-[30px] border-2 border-black overflow-hidden shadow-[8px_8px_0_#000]">
            <table className="w-full text-left bg-[#edf1ff] text-xl">
              <thead>
                <tr className="bg-white">
                  <th className="p-6 border-b border-black">Feature</th>
                  <th className="p-6 border-b border-black">Traditional Workflow</th>
                  <th className="p-6 border-b border-black bg-black text-[#57D4F4]">Voxelomics</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Decision Assembly', 'Manual across systems', 'Unified MDT workspace'],
                  ['Reasoning Trace', 'Scattered notes', 'Structured clinical rationale'],
                  ['Safety Gate', 'Implicit', 'Explicit HITL approval'],
                  ['Evidence Context', 'Ad hoc retrieval', 'Evidence-linked recommendation'],
                  ['Tumor Board Readiness', 'Variable', 'Board-ready output'],
                  ['Turnaround Confidence', 'Inconsistent', 'Consistent clinician-facing flow'],
                ].map((row) => (
                  <tr key={row[0]} className="border-b border-black/10">
                    <td className="p-5 font-semibold">{row[0]}</td>
                    <td className="p-5 text-[#59647f]">{row[1]}</td>
                    <td className="p-5 font-bold">{row[2]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
};
