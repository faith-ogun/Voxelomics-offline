import React from 'react';

export const VoxelomicsLogo: React.FC<{
  className?: string,
  showTagline?: boolean,
  tone?: 'light' | 'dark'
}> = ({
  className = "",
  showTagline = false,
  tone = 'dark'
}) => {
  const textTone = tone === 'light' ? 'text-white' : 'text-voxelomics-ink';
  const subTone = tone === 'light' ? 'text-white/70' : 'text-voxelomics-ink/60';

  return (
    <div className={`inline-flex flex-col leading-none ${className}`}>
      <p
        className={`${textTone} tracking-tight`}
        style={{
          fontFamily: '"Inter", "Manrope", "Segoe UI", sans-serif',
          fontWeight: 900,
          fontSize: 'clamp(1.35rem,2.2vw,2.35rem)',
          letterSpacing: '-0.03em',
        }}
      >
        Voxelomics
      </p>
      {showTagline && (
        <p className={`mt-1 text-[10px] uppercase tracking-[0.3em] font-semibold ${subTone}`}>
          Clinical AI, Human-Reviewed
        </p>
      )}
    </div>
  );
};

// Keep these exports for backwards compatibility if anything uses legacy names.
export const OncydraLogo = VoxelomicsLogo;
export const OncydraIcon = VoxelomicsLogo;
