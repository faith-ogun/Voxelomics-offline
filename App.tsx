import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { Home } from './components/Home';
import { Mission } from './components/Mission';
import { Technology } from './components/Technology';
import { Privacy } from './components/Privacy';
import { Terms } from './components/Terms';
import { MDTCommand } from './components/MDTCommand';

type Section = 'home' | 'mission' | 'technology' | 'privacy' | 'terms' | 'mdt-command';

const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState<Section>('home');

  const handleNavigate = (section: string) => {
    const allowed: Section[] = ['home', 'mission', 'technology', 'privacy', 'terms', 'mdt-command'];
    const normalized = allowed.includes(section as Section) ? (section as Section) : 'home';
    setActiveSection(normalized);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const showChrome = activeSection !== 'mdt-command';

  return (
    <div className="min-h-screen bg-voxelomics-cloud flex flex-col font-sans relative text-voxelomics-ink">
      <Toaster position="bottom-right" />

      {showChrome && (
        <Header
          activeSection={activeSection}
          onNavigate={handleNavigate}
          currentUser={null}
          userProfile={null}
          onGetStartedClick={() => setActiveSection('mdt-command')}
        />
      )}

      <main className="flex-grow w-full">
        {activeSection === 'home' && <Home onNavigate={handleNavigate} />}
        {activeSection === 'mission' && (
          <div className="animate-fade-in-up">
            <Mission />
          </div>
        )}
        {activeSection === 'technology' && (
          <div className="animate-fade-in-up">
            <Technology />
          </div>
        )}
        {activeSection === 'privacy' && (
          <div className="animate-fade-in-up">
            <Privacy />
          </div>
        )}
        {activeSection === 'terms' && (
          <div className="animate-fade-in-up">
            <Terms />
          </div>
        )}
        {activeSection === 'mdt-command' && (
          <div className="animate-fade-in-up">
            <MDTCommand onNavigateHome={() => setActiveSection('home')} />
          </div>
        )}
      </main>

      {showChrome && <Footer onNavigate={handleNavigate} />}
    </div>
  );
};

export default App;
