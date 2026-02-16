import React, { useState, useRef, useEffect } from 'react';
import { User } from 'firebase/auth';
import { LogOut, ChevronDown, User as UserIcon, Settings } from 'lucide-react';
import { VoxelomicsLogo } from './Logo';

type HeaderUserProfile = {
  displayName?: string;
  email?: string;
  role?: 'patient' | 'clinician' | string;
};

interface HeaderProps {
  activeSection: string;
  onNavigate: (section: string) => void;
  currentUser?: User | null;
  userProfile?: HeaderUserProfile | null;
  onLogout?: () => void;
  onGetStartedClick?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeSection,
  onNavigate,
  currentUser,
  userProfile,
  onLogout,
  onGetStartedClick
}) => {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const getButtonClass = (section: string) => {
    const isActive = activeSection === section;
    return `px-1 py-2 text-base font-semibold transition-colors ${isActive
      ? 'text-[#0b1c62]'
      : 'text-[#1f316f] hover:text-[#0b1c62]'
      }`;
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const isHome = activeSection === 'home';

  return (
    <header className="fixed top-0 left-0 right-0 z-50 transition-all duration-300">
      <div className={`absolute inset-0 ${isHome ? 'bg-transparent border-b border-transparent' : 'bg-white/85 backdrop-blur-xl border-b border-[#dbe4ff]'}`}></div>
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-[84px] grid grid-cols-[1fr_auto_1fr] items-center">
        <div className="flex justify-start">
          <button
            className="flex items-center gap-0 group focus:outline-none"
            onClick={() => onNavigate('home')}
          >
            <VoxelomicsLogo tone="dark" />
          </button>
        </div>

        {/* Navigation (centered) */}
        <nav className="hidden md:flex items-center justify-center gap-10">
          <button
            className={getButtonClass('mission')}
            onClick={() => onNavigate('mission')}
          >
            Our Mission
          </button>
          <button
            className={getButtonClass('technology')}
            onClick={() => onNavigate('technology')}
          >
            Technology
          </button>
        </nav>

        <div className="flex justify-end items-center gap-4">
          {/* Auth Section */}
          {currentUser && userProfile ? (
            // Logged in - Show user menu
            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 pl-2 pr-3 py-1.5 rounded-full bg-white border border-slate-200 hover:border-slate-300 transition-all"
              >
                {/* Avatar */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold ${userProfile.role === 'patient'
                  ? 'bg-gradient-to-br from-voxelomics-cyan to-voxelomics-lilac'
                  : 'bg-gradient-to-br from-voxelomics-lilac to-voxelomics-midnight'
                  }`}>
                  {userProfile.displayName ? getInitials(userProfile.displayName) : <UserIcon className="w-4 h-4" />}
                </div>
                <span className="text-sm font-medium text-slate-900 hidden sm:block max-w-[120px] truncate">
                  {userProfile.displayName || 'User'}
                </span>
                <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              {showUserMenu && (
                <div className="absolute right-0 top-full mt-2 w-64 bg-white rounded-xl border border-slate-200 shadow-xl py-2 z-50">
                  {/* User Info */}
                  <div className="px-4 py-3 border-b border-slate-100">
                    <p className="font-bold text-slate-900 truncate">{userProfile.displayName}</p>
                    <p className="text-sm text-slate-500 truncate">{userProfile.email}</p>
                    <span className={`inline-block mt-2 px-2 py-0.5 rounded-full text-xs font-bold ${userProfile.role === 'patient'
                      ? 'bg-cyan-100 text-cyan-800'
                      : 'bg-indigo-100 text-indigo-800'
                      }`}>
                      {userProfile.role === 'patient' ? 'Patient' : 'Clinician'}
                    </span>
                  </div>

                  {/* Menu Items */}
                  <div className="py-2">
                    <button
                      onClick={() => {
                        setShowUserMenu(false);
                        onNavigate(userProfile.role === 'patient' ? 'patients' : 'clinicians');
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-3"
                    >
                      <UserIcon className="w-4 h-4 text-slate-400" />
                      My Dashboard
                    </button>
                    <button
                      onClick={() => setShowUserMenu(false)}
                      className="w-full px-4 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-3"
                    >
                      <Settings className="w-4 h-4 text-slate-400" />
                      Settings
                    </button>
                  </div>

                  {/* Logout */}
                  <div className="border-t border-slate-100 pt-2">
                    <button
                      onClick={() => {
                        setShowUserMenu(false);
                        onLogout?.();
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-3"
                    >
                      <LogOut className="w-4 h-4" />
                      Log Out
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={onGetStartedClick}
              className="px-5 py-2.5 rounded-full bg-[#2663eb] text-white text-base font-semibold hover:bg-[#1f57d4] transition-colors"
            >
              Get Started
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
