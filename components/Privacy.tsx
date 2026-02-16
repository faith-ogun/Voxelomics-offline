import React from 'react';

export const Privacy: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-50 pt-28 pb-20 px-4">
      <div className="max-w-4xl mx-auto bg-white border border-slate-200 rounded-2xl p-8 md:p-10 shadow-sm">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-900">Privacy</h1>
        <p className="mt-3 text-sm text-slate-500">Last updated: February 13, 2026</p>

        <div className="mt-8 space-y-8 text-slate-700 leading-relaxed">
          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">1. What We Collect</h2>
            <p>
              Voxelomics demo pages may process profile details, clinical notes, and interaction logs that you enter
              during testing. For hackathon evaluation, this site is intended as a demonstration environment and should
              not be used for real patient-identifiable data.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">2. How Data Is Used</h2>
            <p>
              Data is used to power demo workflows, generate AI summaries, and display mock or test outputs in the UI.
              Data is not sold and is not used for advertising.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">3. Data Handling and Security</h2>
            <p>
              The project uses role-based access patterns, encrypted transport, and cloud-managed infrastructure. Despite
              these controls, this demo is not presented as a production clinical deployment.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">4. Medical Disclaimer</h2>
            <p>
              Voxelomics is an AI-assisted decision support prototype. It does not replace licensed clinical judgment,
              diagnosis, or treatment planning.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">5. Contact</h2>
            <p>
              For project questions, use the LinkedIn or GitHub links in the footer.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};
