import React from 'react';
import { BrainCircuit, Microscope, ShieldCheck } from 'lucide-react';
import { ContainerScroll } from './container-scroll-animation';

export function HeroScrollDemo() {
  const heroImageSrc = `${import.meta.env.BASE_URL}mdt_command_submission.png`;

  return (
    <div className="flex flex-col overflow-hidden pb-8 pt-8 md:pt-12">
      <ContainerScroll
        titleComponent={
          <>
            <p className="text-sm font-bold text-[#6573a9] uppercase tracking-[0.32em] mb-3">
              How Voxelomics Works
            </p>
            <h2 className="text-3xl md:text-5xl font-display font-semibold text-[#11173d] leading-tight">
              Pathology + Imaging + Context
              <span className="block text-[#1d3cff]">into one MDT-ready view</span>
            </h2>
            <p className="text-[#526091] mt-4 max-w-3xl mx-auto">
              DiagnostiCore interprets whole-slide and 3D imaging, PathGenomicPredictor adds genomic risk signals,
              and MDT Command turns all of it into clinician-gated decisions.
            </p>
          </>
        }
      >
        <div className="relative h-full w-full">
          <img
            src={heroImageSrc}
            alt="Clinician reviewing oncology imaging and pathology results"
            className="mx-auto h-full w-full object-cover object-center"
            draggable={false}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#050a2a] via-[#0a1246]/55 to-transparent" />

          <div className="absolute left-4 right-4 bottom-4 grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="rounded-xl border border-cyan-200/20 bg-[#0d1a5c]/85 backdrop-blur p-3 text-left text-white">
              <div className="flex items-center gap-2 text-cyan-200 mb-1">
                <Microscope className="w-4 h-4" />
                <span className="text-xs font-bold uppercase tracking-[0.2em]">DiagnostiCore</span>
              </div>
              <p className="text-sm text-blue-100/90">WSI + 3D scan interpretation with morphology signals.</p>
            </div>

            <div className="rounded-xl border border-indigo-200/20 bg-[#1a1f70]/85 backdrop-blur p-3 text-left text-white">
              <div className="flex items-center gap-2 text-indigo-200 mb-1">
                <BrainCircuit className="w-4 h-4" />
                <span className="text-xs font-bold uppercase tracking-[0.2em]">PathGenomicPredictor</span>
              </div>
              <p className="text-sm text-blue-100/90">Mutation likelihood scoring directly from histopathology.</p>
            </div>

            <div className="rounded-xl border border-emerald-200/25 bg-[#0f2360]/85 backdrop-blur p-3 text-left text-white">
              <div className="flex items-center gap-2 text-emerald-200 mb-1">
                <ShieldCheck className="w-4 h-4" />
                <span className="text-xs font-bold uppercase tracking-[0.2em]">MDT Command</span>
              </div>
              <p className="text-sm text-blue-100/90">Human-in-the-loop outputs before final recommendations.</p>
            </div>
          </div>
        </div>
      </ContainerScroll>
    </div>
  );
}
