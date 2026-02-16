import React from 'react';
import { HeartHandshake, ShieldCheck, GaugeCircle, BrainCircuit } from 'lucide-react';

const principles = [
  {
    title: 'Privacy First',
    body: 'Clinical context is handled with strict access control patterns and audit-ready workflows.',
    icon: ShieldCheck,
  },
  {
    title: 'Speed Matters',
    body: 'Decision support must appear in time for MDT discussion, not hours later.',
    icon: GaugeCircle,
  },
  {
    title: 'Human Oversight',
    body: 'Recommendations remain clinician-reviewed before action, always.',
    icon: HeartHandshake,
  },
];

export const Mission: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#e7eeff] pt-28 pb-24 px-4 text-[#071321]">
      <div className="max-w-6xl mx-auto">
        <section className="text-center">
          <h1
            className="text-[3rem] md:text-[5rem] leading-[0.9] font-black tracking-tight"
            style={{ fontFamily: '"Inter", "Manrope", "Segoe UI", sans-serif' }}
          >
            Oncology needs
            <span className="block text-[#3d53d8]">Clinical Clarity.</span>
          </h1>
          <p className="mt-6 text-xl text-[#2f3f4f] max-w-3xl mx-auto">
            We believe high-stakes cancer decisions should be faster, safer, and easier to review across pathology,
            imaging, and multidisciplinary workflows.
          </p>
        </section>

        <section className="mt-16 grid grid-cols-1 lg:grid-cols-[1.05fr_0.95fr] gap-6">
          <div className="rounded-[30px] border-2 border-black bg-[#0a1036] text-white p-9 shadow-[8px_8px_0_#000]">
            <BrainCircuit className="w-9 h-9 text-[#57D4F4]" />
            <h2 className="mt-8 text-4xl font-extrabold">The Clinical Gap</h2>
            <p className="mt-5 text-lg text-slate-200 leading-relaxed">
              Tumor board decisions rely on fragmented evidence from different systems. Voxelomics unifies those signals
              into one review surface with explicit human approval gates before final recommendations.
            </p>
          </div>

          <div className="grid grid-rows-2 gap-6">
            <div className="rounded-3xl border-2 border-black bg-white p-7 shadow-[6px_6px_0_#000]">
              <h3 className="text-2xl font-extrabold">Empathy First</h3>
              <p className="mt-3 text-[#314252] text-lg">
                Built for real clinicians working under pressure, not benchmark demos.
              </p>
            </div>
            <div className="rounded-3xl border-2 border-black bg-[#8A78FF] p-7 shadow-[6px_6px_0_#000]">
              <h3 className="text-2xl font-extrabold">Expression, Not Overload</h3>
              <p className="mt-3 text-white text-lg">
                Structured summaries and evidence traces reduce cognitive load and speed team consensus.
              </p>
            </div>
          </div>
        </section>

        <section className="mt-20">
          <h2 className="text-center text-4xl font-black tracking-tight">Our Principles</h2>
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-5">
            {principles.map(({ title, body, icon: Icon }) => (
              <article key={title} className="rounded-2xl border-2 border-black bg-[#f3f6ff] p-6 shadow-[5px_5px_0_#000]">
                <div className="w-11 h-11 rounded-xl bg-black text-[#57D4F4] flex items-center justify-center">
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="mt-4 text-2xl font-extrabold">{title}</h3>
                <p className="mt-3 text-[#314252]">{body}</p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};
