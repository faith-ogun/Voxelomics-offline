import React, { useEffect, useRef } from 'react';

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

const VERTICES: Vec3[] = [
  { x: -1, y: -1, z: -1 },
  { x: 1, y: -1, z: -1 },
  { x: 1, y: 1, z: -1 },
  { x: -1, y: 1, z: -1 },
  { x: -1, y: -1, z: 1 },
  { x: 1, y: -1, z: 1 },
  { x: 1, y: 1, z: 1 },
  { x: -1, y: 1, z: 1 },
];

const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

const rotatePoint = (p: Vec3, rx: number, ry: number): Vec3 => {
  const cosy = Math.cos(ry);
  const siny = Math.sin(ry);
  const cosx = Math.cos(rx);
  const sinx = Math.sin(rx);

  const xzX = p.x * cosy - p.z * siny;
  const xzZ = p.x * siny + p.z * cosy;

  const yzY = p.y * cosx - xzZ * sinx;
  const yzZ = p.y * sinx + xzZ * cosx;

  return { x: xzX, y: yzY, z: yzZ };
};

export const RotatingWireframeCube: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let t = 0;
    let lastFrame = 0;
    let renderDpr = 1;
    let isVisible = true;
    const targetFps = 30;
    const frameInterval = 1000 / targetFps;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      renderDpr = dpr;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (!isVisible) return;
      if (now - lastFrame < frameInterval) return;
      const delta = Math.min(2, (now - lastFrame) / 16.67 || 1);
      lastFrame = now;

      const w = canvas.width / renderDpr;
      const h = canvas.height / renderDpr;
      ctx.clearRect(0, 0, w, h);

      const cx = w * 0.5 + Math.sin(t * 0.51) * 28;
      const cy = h * 0.5 + Math.cos(t * 0.39) * 16;
      const scale = Math.min(w, h) * 0.24;
      const perspective = 3.8;

      const rx = t * 0.58;
      const ry = t * 0.91;
      const rz = Math.sin(t * 0.45) * 0.35;

      const projected = VERTICES.map((v) => {
        const rxy = rotatePoint(v, rx, ry);
        const cosz = Math.cos(rz);
        const sinz = Math.sin(rz);
        const r = {
          x: rxy.x * cosz - rxy.y * sinz,
          y: rxy.x * sinz + rxy.y * cosz,
          z: rxy.z,
        };
        const depth = perspective / (perspective - r.z);
        return {
          x: cx + r.x * scale * depth,
          y: cy + r.y * scale * depth,
          z: r.z,
        };
      });

      ctx.strokeStyle = '#d7e8ff';
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';

      for (const [a, b] of EDGES) {
        const pa = projected[a];
        const pb = projected[b];
        const meanZ = (pa.z + pb.z) * 0.5;
        const glow = Math.max(0.12, (meanZ + 1) / 2);
        ctx.setLineDash(meanZ < 0 ? [4, 5] : []);
        ctx.globalAlpha = meanZ < 0 ? 0.22 : 0.45 + glow * 0.5;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }

      // Inner guide lines for richer "observed while panning" effect
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(projected[0].x, projected[0].y);
      ctx.lineTo(projected[6].x, projected[6].y);
      ctx.moveTo(projected[1].x, projected[1].y);
      ctx.lineTo(projected[7].x, projected[7].y);
      ctx.moveTo(projected[3].x, projected[3].y);
      ctx.lineTo(projected[5].x, projected[5].y);
      ctx.stroke();

      ctx.globalAlpha = 1;
      t += 0.0115 * delta;
    };

    resize();
    const observer = new IntersectionObserver(
      (entries) => {
        isVisible = entries.some((entry) => entry.isIntersecting);
      },
      { threshold: 0.1 }
    );
    observer.observe(canvas);
    raf = requestAnimationFrame(draw);
    window.addEventListener('resize', resize);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="w-full h-[300px] md:h-[360px]" aria-hidden="true" />;
};
