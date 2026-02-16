import React from 'react';

export const Terms: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-50 pt-28 pb-20 px-4">
      <div className="max-w-4xl mx-auto bg-white border border-slate-200 rounded-2xl p-8 md:p-10 shadow-sm">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-900">Terms</h1>
        <p className="mt-3 text-sm text-slate-500">Last updated: February 13, 2026</p>

        <div className="mt-8 space-y-8 text-slate-700 leading-relaxed">
          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">1. Demo Use Only</h2>
            <p>
              This website is provided as a hackathon and research demonstration. It is not a regulated clinical
              software product and is not intended for real-world medical decision making.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">2. User Responsibility</h2>
            <p>
              You agree not to upload sensitive personal data that you are not authorized to share. You are responsible
              for your own use of this demo and any outputs you generate.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">3. AI Output Boundaries</h2>
            <p>
              AI-generated summaries, suggestions, and workflow outputs are informational drafts. Human review is
              required before any clinical or operational action.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">4. Availability and Changes</h2>
            <p>
              The service may be modified, paused, or removed at any time. Features can change without notice as part
              of rapid prototyping during the competition window.
            </p>
          </section>

          <section>
            <h2 className="text-xl font-bold text-slate-900 mb-2">5. Competition Context</h2>
            <p>
              Submission and participation are also governed by the official hackathon and platform rules, including the
              stated judging requirements and submission constraints.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};
