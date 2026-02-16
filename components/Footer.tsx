import React from 'react';
import { Github, Linkedin } from 'lucide-react';
import { VoxelomicsLogo } from './Logo';

interface FooterProps {
  onNavigate: (section: string) => void;
}

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  return (
    <footer className="bg-[#0a054c] border-t border-[#1b126f] py-8 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
          <div>
            <button onClick={() => onNavigate('home')} className="text-left">
              <VoxelomicsLogo tone="light" />
            </button>
            <p className="mt-2 text-sm text-blue-100/75">
              Human-centered oncology intelligence for real clinical decisions.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-5 text-sm font-semibold text-blue-100/85">
            <button onClick={() => onNavigate('privacy')} className="hover:text-white transition-colors">
              Privacy
            </button>
            <button onClick={() => onNavigate('terms')} className="hover:text-white transition-colors">
              Terms
            </button>
            <a
              href="https://www.linkedin.com/in/faith-ogundimu"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="LinkedIn"
              className="text-blue-100/80 hover:text-white transition-colors"
            >
              <Linkedin className="w-4 h-4" />
            </a>
            <a
              href="https://github.com/faith-ogun/Voxelomics"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub"
              className="text-blue-100/80 hover:text-white transition-colors"
            >
              <Github className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};
