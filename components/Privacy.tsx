import React from 'react';

export const Privacy: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-50 pt-28 pb-20 px-4">
      <div className="max-w-4xl mx-auto bg-white border border-slate-200 rounded-2xl p-8 md:p-10 shadow-sm">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-900">Privacy</h1>
        <p className="mt-3 text-sm text-slate-500">Last updated: February 17, 2026</p>

        <div className="mt-8 space-y-8 text-slate-700 leading-relaxed">
          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">1. What We Collect</h2>
            <p>
              Voxelomics Offline may process clinical notes, transcript text, uploaded audio, and case interaction logs
              entered during demo use. This is a hackathon/research prototype and should not be used with real
              patient-identifiable data.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">2. How Data Is Used</h2>
            <p>
              Data is used to run local MDT workflow orchestration, generate draft AI-supported summaries, and display
              demo outputs in the UI. Data is not sold and is not used for advertising.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">3. Data Handling and Security</h2>
            <p>
              The default profile is local-first: runtime case state is stored in local SQLite and local files on the
              device running the app. Cloud services are not required for baseline offline workflow operation. If you
              configure external services manually, you are responsible for those service security settings.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">4. Retention and Deletion</h2>
            <p>
              Demo case history snapshots can be removed from the Patient Cases view. Local runtime files can also be
              deleted directly from local storage by the user.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">5. Medical Disclaimer</h2>
            <p>
              Voxelomics is an AI-assisted decision support prototype. It does not replace licensed clinical judgment,
              diagnosis, or treatment planning.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">6. Contact</h2>
            <p>
              For project questions, use the LinkedIn or GitHub links in the footer.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};
