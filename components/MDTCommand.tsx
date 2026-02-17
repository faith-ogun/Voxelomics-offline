import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Brain,
  CalendarDays,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Download,
  ExternalLink,
  FileText,
  FlaskConical,
  LayoutGrid,
  Mic,
  MessageSquare,
  Microscope,
  Minimize2,
  Home,
  Plus,
  Minus,
  Maximize2,
  Play,
  RefreshCw,
  ShieldCheck,
  Stethoscope,
  Trash2,
  Users
} from 'lucide-react';
import { jsPDF } from 'jspdf';

type PipelineState = 'idle' | 'running' | 'done';
type ApprovalState = 'pending' | 'approved' | 'rework';
type WorkspaceTab = 'board' | 'reasoning' | 'dictation' | 'diagnosticore' | 'patients' | 'calendar';
type DictationMode = 'select' | 'demo' | 'live' | 'manual';
type TranscriptSource = 'manual' | 'live' | 'upload';

interface MockCase {
  id: string;
  patientId: string;
  patientName: string;
  diagnosis: string;
  stage: string;
  age: number;
  priority: 'Routine' | 'High';
  radiology: string;
  pathology: string;
  genomics: string;
  transcript: string;
  transcriptAudioUri: string;
}

interface CalendarAttendee {
  id: string;
  name: string;
  role: string;
  confirmed: boolean;
}

interface AnalyzeResponse {
  success: boolean;
  case_id: string;
  status: string;
  message: string;
  consensus?: {
    recommendation: string;
    confidence: number;
    red_flags: string[];
  };
  hitl_gate?: {
    approval_checklist: string[];
    safety_flags: string[];
  };
}

interface DraftResponse {
  success: boolean;
  case_id: string;
  status: string;
  artifacts: {
    diagnosticore?: {
      case_submitter_id?: string;
      tp53_probability?: number;
      threshold?: number;
      predicted_label?: string;
      source_split?: string;
      n_tiles?: number;
      raw_pred_prob?: number;
      validation_metrics?: Record<string, number>;
      calibration_method?: string;
      wsi_project_id?: string;
      wsi_file_id?: string;
      wsi_file_name?: string;
      wsi_local_path?: string;
      tile_preview_png?: string;
      tile_preview_x?: number;
      tile_preview_y?: number;
      cohort_relation?: string;
      deepzoom_dzi_path?: string;
      deepzoom_tile_dir?: string;
      model_version?: string;
      data_version?: string;
      model_card?: {
        cohort?: string;
        intended_use?: string;
        limitations?: string[];
      };
      locked_threshold_report?: {
        selected_threshold?: number;
        test_ece_10?: number;
        test_recall_ci_low?: number;
      };
    };
    stage_one?: {
      literature?: {
        highlights?: string;
        evidence?: Array<{
          title: string;
          source?: string;
          year?: number;
          identifier?: string;
          finding?: string;
        }>;
      };
    };
    consensus?: {
      recommendation: string;
      confidence: number;
      red_flags: string[];
    };
    soap_note?: {
      subjective?: string;
      objective?: string;
      assessment?: string;
      plan: string;
    };
    clinical_reasoning?: {
      summary?: string;
      key_risks?: string[];
      recommended_actions?: string[];
      confirmatory_actions?: string[];
      evidence_links?: string[];
      uncertainty_statement?: string;
      model_route?: string;
      generation_mode?: string;
    };
    hitl_gate?: {
      approval_checklist: string[];
      safety_flags: string[];
    };
    transcription?: {
      engine?: string;
      transcript?: string;
      wer_estimate?: number;
      notes?: string;
    };
  };
}

interface EvidenceReferenceItem {
  label: string;
  identifier: string;
  source?: string;
  year?: number;
  url?: string;
}

type OpenSeadragonViewer = {
  destroy: () => void;
  addHandler: (eventName: string, handler: (...args: unknown[]) => void) => void;
  isFullPage?: () => boolean;
  setFullPage?: (fullPage: boolean) => void;
  viewport?: {
    zoomBy: (factor: number) => void;
    applyConstraints: () => void;
    goHome: () => void;
  };
};

type OpenSeadragonFactory = (options: {
  element: HTMLElement;
  tileSources: string;
  prefixUrl: string;
  navImages?: Record<string, unknown>;
  showNavigator?: boolean;
  showNavigationControl?: boolean;
  navigatorPosition?: string;
  animationTime?: number;
  blendTime?: number;
  minZoomImageRatio?: number;
  maxZoomPixelRatio?: number;
  visibilityRatio?: number;
  constrainDuringPan?: boolean;
}) => OpenSeadragonViewer;

interface DeepZoomDescriptor {
  tileSize: number;
  overlap: number;
  format: string;
  width: number;
  height: number;
  maxLevel: number;
}

interface NativeDeepZoomViewerProps {
  dziUrl: string;
  onReadyChange?: (ready: boolean) => void;
  onStatusChange?: (status: string) => void;
}

const clampNumber = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

const parseDeepZoomDescriptor = (xmlText: string): DeepZoomDescriptor | null => {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xmlText, 'application/xml');
  const imageNode = doc.querySelector('Image');
  const sizeNode = doc.querySelector('Size');
  if (!imageNode || !sizeNode) return null;

  const tileSize = Number(imageNode.getAttribute('TileSize') || '256');
  const overlap = Number(imageNode.getAttribute('Overlap') || '0');
  const format = (imageNode.getAttribute('Format') || 'jpeg').toLowerCase();
  const width = Number(sizeNode.getAttribute('Width') || '0');
  const height = Number(sizeNode.getAttribute('Height') || '0');
  if (!Number.isFinite(tileSize) || tileSize <= 0) return null;
  if (!Number.isFinite(width) || width <= 0) return null;
  if (!Number.isFinite(height) || height <= 0) return null;

  const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));
  return {
    tileSize,
    overlap: Number.isFinite(overlap) && overlap >= 0 ? overlap : 0,
    format: format === 'jpg' ? 'jpg' : format === 'png' ? 'png' : 'jpeg',
    width,
    height,
    maxLevel
  };
};

const NativeDeepZoomViewer: React.FC<NativeDeepZoomViewerProps> = ({
  dziUrl,
  onReadyChange,
  onStatusChange
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{
    dragging: boolean;
    startX: number;
    startY: number;
    offsetX: number;
    offsetY: number;
  }>({
    dragging: false,
    startX: 0,
    startY: 0,
    offsetX: 0,
    offsetY: 0
  });
  const [descriptor, setDescriptor] = useState<DeepZoomDescriptor | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [fitScale, setFitScale] = useState(1);

  const tileBaseUrl = useMemo(() => {
    if (!dziUrl) return '';
    return dziUrl.replace(/\/deepzoom\.dzi(\?.*)?$/i, '/deepzoom_tiles');
  }, [dziUrl]);

  useEffect(() => {
    let cancelled = false;
    setDescriptor(null);
    onReadyChange?.(false);
    onStatusChange?.('Loading offline DeepZoom descriptor...');

    const load = async () => {
      try {
        const response = await fetch(dziUrl, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`Descriptor request failed (${response.status})`);
        }
        const xml = await response.text();
        const parsed = parseDeepZoomDescriptor(xml);
        if (!parsed) throw new Error('DeepZoom descriptor parse failed.');
        if (cancelled) return;
        setDescriptor(parsed);
        onStatusChange?.('Offline DeepZoom viewer ready.');
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : 'Unknown DeepZoom error.';
        onStatusChange?.(`Offline DeepZoom failed: ${message}`);
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [dziUrl, onReadyChange, onStatusChange]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;

    const updateSize = () => {
      const rect = node.getBoundingClientRect();
      setContainerSize({
        width: Math.max(1, Math.floor(rect.width)),
        height: Math.max(1, Math.floor(rect.height))
      });
    };

    updateSize();
    const observer = new ResizeObserver(() => updateSize());
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!descriptor || containerSize.width <= 0 || containerSize.height <= 0) return;
    const nextFitScale = Math.min(
      containerSize.width / descriptor.width,
      containerSize.height / descriptor.height
    );
    const safeFit = Number.isFinite(nextFitScale) && nextFitScale > 0 ? nextFitScale : 1;
    setFitScale(safeFit);
    setScale(safeFit);
    setOffset({
      x: (containerSize.width - descriptor.width * safeFit) / 2,
      y: (containerSize.height - descriptor.height * safeFit) / 2
    });
    onReadyChange?.(true);
  }, [descriptor, containerSize.width, containerSize.height, onReadyChange]);

  const normalizeOffset = (
    nextScale: number,
    candidateX: number,
    candidateY: number
  ): { x: number; y: number } => {
    if (!descriptor) return { x: candidateX, y: candidateY };
    const imageW = descriptor.width * nextScale;
    const imageH = descriptor.height * nextScale;

    let x = candidateX;
    let y = candidateY;

    if (imageW <= containerSize.width) {
      x = (containerSize.width - imageW) / 2;
    } else {
      x = clampNumber(x, containerSize.width - imageW, 0);
    }
    if (imageH <= containerSize.height) {
      y = (containerSize.height - imageH) / 2;
    } else {
      y = clampNumber(y, containerSize.height - imageH, 0);
    }
    return { x, y };
  };

  const handleWheel: React.WheelEventHandler<HTMLDivElement> = (event) => {
    if (!descriptor) return;
    event.preventDefault();

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;
    const zoomStep = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    const minScale = Math.max(fitScale * 0.5, 0.02);
    const maxScale = Math.max(fitScale * 24, 1);
    const nextScale = clampNumber(scale * zoomStep, minScale, maxScale);

    const worldX = (pointerX - offset.x) / scale;
    const worldY = (pointerY - offset.y) / scale;
    const candidateX = pointerX - worldX * nextScale;
    const candidateY = pointerY - worldY * nextScale;
    const nextOffset = normalizeOffset(nextScale, candidateX, candidateY);

    setScale(nextScale);
    setOffset(nextOffset);
  };

  const handlePointerDown: React.PointerEventHandler<HTMLDivElement> = (event) => {
    if (!descriptor) return;
    dragRef.current = {
      dragging: true,
      startX: event.clientX,
      startY: event.clientY,
      offsetX: offset.x,
      offsetY: offset.y
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove: React.PointerEventHandler<HTMLDivElement> = (event) => {
    if (!dragRef.current.dragging || !descriptor) return;
    const dx = event.clientX - dragRef.current.startX;
    const dy = event.clientY - dragRef.current.startY;
    const candidateX = dragRef.current.offsetX + dx;
    const candidateY = dragRef.current.offsetY + dy;
    setOffset(normalizeOffset(scale, candidateX, candidateY));
  };

  const handlePointerUp: React.PointerEventHandler<HTMLDivElement> = (event) => {
    dragRef.current.dragging = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const visibleTiles = useMemo(() => {
    if (!descriptor || containerSize.width <= 0 || containerSize.height <= 0) return [];

    const level = clampNumber(
      Math.round(descriptor.maxLevel + Math.log2(Math.max(scale, 1e-6))),
      0,
      descriptor.maxLevel
    );
    const levelScale = Math.pow(2, level - descriptor.maxLevel);
    const levelWidth = Math.max(1, Math.ceil(descriptor.width * levelScale));
    const levelHeight = Math.max(1, Math.ceil(descriptor.height * levelScale));
    const tileSize = descriptor.tileSize;
    const cols = Math.max(1, Math.ceil(levelWidth / tileSize));
    const rows = Math.max(1, Math.ceil(levelHeight / tileSize));

    const worldXMin = clampNumber((0 - offset.x) / scale, 0, descriptor.width);
    const worldYMin = clampNumber((0 - offset.y) / scale, 0, descriptor.height);
    const worldXMax = clampNumber((containerSize.width - offset.x) / scale, 0, descriptor.width);
    const worldYMax = clampNumber((containerSize.height - offset.y) / scale, 0, descriptor.height);

    const levelXMin = worldXMin * levelScale;
    const levelYMin = worldYMin * levelScale;
    const levelXMax = worldXMax * levelScale;
    const levelYMax = worldYMax * levelScale;

    const tileXMin = clampNumber(Math.floor(levelXMin / tileSize) - 1, 0, cols - 1);
    const tileYMin = clampNumber(Math.floor(levelYMin / tileSize) - 1, 0, rows - 1);
    const tileXMax = clampNumber(Math.floor(levelXMax / tileSize) + 1, 0, cols - 1);
    const tileYMax = clampNumber(Math.floor(levelYMax / tileSize) + 1, 0, rows - 1);

    const items: Array<{
      key: string;
      src: string;
      left: number;
      top: number;
      width: number;
      height: number;
    }> = [];

    for (let ty = tileYMin; ty <= tileYMax; ty += 1) {
      for (let tx = tileXMin; tx <= tileXMax; tx += 1) {
        const tileX = tx * tileSize;
        const tileY = ty * tileSize;
        const tileW = Math.min(tileSize, levelWidth - tileX);
        const tileH = Math.min(tileSize, levelHeight - tileY);

        const worldX = tileX / levelScale;
        const worldY = tileY / levelScale;
        const worldW = tileW / levelScale;
        const worldH = tileH / levelScale;

        const left = worldX * scale + offset.x;
        const top = worldY * scale + offset.y;
        const drawW = worldW * scale;
        const drawH = worldH * scale;
        const src = `${tileBaseUrl}/${level}/${tx}_${ty}.${descriptor.format}`;
        items.push({
          key: `${level}-${tx}-${ty}`,
          src,
          left,
          top,
          width: drawW,
          height: drawH
        });
      }
    }

    return items;
  }, [containerSize.height, containerSize.width, descriptor, offset.x, offset.y, scale, tileBaseUrl]);

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className="relative w-full h-[360px] rounded-lg border border-[#3b4f9e] bg-[#09133a] overflow-hidden touch-none"
    >
      {descriptor ? (
        <>
          {visibleTiles.map((tile) => (
            <img
              key={tile.key}
              src={tile.src}
              alt=""
              draggable={false}
              className="absolute select-none pointer-events-none"
              style={{
                left: `${tile.left}px`,
                top: `${tile.top}px`,
                width: `${tile.width}px`,
                height: `${tile.height}px`
              }}
            />
          ))}
        </>
      ) : (
        <div className="absolute inset-0 grid place-items-center text-xs text-[#c4d0ff]">
          Preparing offline DeepZoom viewer...
        </div>
      )}
    </div>
  );
};

declare global {
  interface VoxelomicsDesktopBridge {
    getAppInfo?: () => Promise<{
      backendUrl: string;
      ollamaUrl: string;
      dataDir: string;
      executionMode: string;
      packaged: boolean;
      appVersion: string;
    }>;
    chooseDataDirectory?: () => Promise<{
      changed: boolean;
      dataDir?: string;
      backendReady?: boolean;
    }>;
    saveReport?: (payload: {
      format: 'txt' | 'pdf';
      defaultFileName: string;
      textContent?: string;
      binaryBytes?: number[];
    }) => Promise<{
      success: boolean;
      canceled: boolean;
      path?: string;
    }>;
  }

  interface Window {
    OpenSeadragon?: OpenSeadragonFactory;
    __openSeadragonLoader?: Promise<OpenSeadragonFactory>;
    voxelomicsDesktop?: VoxelomicsDesktopBridge;
  }
}

interface ApproveResponse {
  success: boolean;
  case_id: string;
  status: string;
  message: string;
}

interface HealthResponse {
  status: string;
  service: string;
  execution_mode: string;
  case_store_backend: string;
  retrieval_mode: string;
  adk_available: boolean;
  timestamp: string;
}

interface EvidenceSyncStatusResponse {
  success: boolean;
  retrieval_mode: string;
  literature_count: number;
  literature_path: string;
  last_synced_at?: string | null;
}

interface EvidenceSyncResponse {
  success: boolean;
  message: string;
  diagnosis: string;
  genes: string[];
  literature_count: number;
  literature_path: string;
  last_synced_at: string;
}

interface PatientCaseHistoryItem {
  snapshot_id: number;
  case_id: string;
  patient_id: string;
  patient_name: string;
  diagnosis: string;
  status: string;
  saved_at: string;
  updated_at: string;
}

interface PatientCaseHistoryResponse {
  success: boolean;
  patient_id: string;
  count: number;
  cases: PatientCaseHistoryItem[];
}

interface CaseHistorySnapshotResponse {
  success: boolean;
  snapshot_id: number;
  saved_at: string;
  case: {
    case_id: string;
    patient_id: string;
    patient_name: string;
    diagnosis: string;
    status: string;
    input_payload: {
      stage?: string;
      imaging?: { ct_report?: string; mri_report?: string; pet_report?: string };
      pathology?: { biopsy_summary?: string };
      genomics?: { report_summary?: string };
      transcript?: { raw_text?: string; audio_uri?: string };
    };
    artifacts: DraftResponse['artifacts'];
  };
}

interface DeleteCaseHistorySnapshotResponse {
  success: boolean;
  snapshot_id: number;
  message: string;
}

const parseEnvNumber = (rawValue: string | undefined, defaultValue: number, minValue = 0): number => {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) return defaultValue;
  return Math.max(minValue, parsed);
};

const API_BASE_URL =
  import.meta.env.VITE_MDT_COMMAND_SERVICE_URL || 'http://127.0.0.1:8084';
const FRONTEND_MEDASR_ENABLED =
  (import.meta.env.VITE_MEDASR_FRONTEND_ONNX || 'true').toLowerCase() === 'true';
const FRONTEND_MEDASR_MODEL_URL =
  import.meta.env.VITE_MEDASR_ONNX_URL || './models/medasr.onnx';
const FRONTEND_MEDASR_VOCAB_URL =
  import.meta.env.VITE_MEDASR_VOCAB_URL || './models/medasr_vocab.json';
const FRONTEND_MEDASR_ORT_URL =
  import.meta.env.VITE_MEDASR_ORT_URL || './vendor/onnxruntime-web/ort.min.js';
const FRONTEND_MEDASR_WORKER_PATH = './workers/medasr-onnx.worker.js';
const FRONTEND_MEDASR_EXECUTION_PROVIDERS = (
  import.meta.env.VITE_MEDASR_EXECUTION_PROVIDERS || 'webgpu,wasm'
)
  .split(',')
  .map((value: string) => value.trim().toLowerCase())
  .filter((value: string) => value.length > 0);
const FRONTEND_MEDASR_CHUNK_SECONDS = parseEnvNumber(import.meta.env.VITE_MEDASR_CHUNK_SECONDS, 10, 1);
const FRONTEND_MEDASR_OVERLAP_SECONDS = parseEnvNumber(import.meta.env.VITE_MEDASR_OVERLAP_SECONDS, 1, 0);
const ANALYZE_REQUEST_TIMEOUT_MS = parseEnvNumber(import.meta.env.VITE_MDT_ANALYZE_TIMEOUT_MS, 900000, 10000);

const resolveFrontendAssetUrl = (assetPath: string): string => {
  const value = String(assetPath || '').trim();
  if (!value) return value;
  if (/^(https?:|file:|blob:|data:)/i.test(value)) return value;
  if (value.startsWith('/')) {
    // In Electron file:// mode, absolute paths like "/models/..." resolve to disk root.
    // Normalize to relative-from-index for cross-mode compatibility.
    return new URL(`.${value}`, window.location.href).toString();
  }
  return new URL(value, window.location.href).toString();
};

const DEFAULT_APPROVAL_CHECKLIST = [
  'Confirm final radiology report sign-off',
  'Confirm pathology receptor and grade summary',
  'Confirm molecular panel provenance and date',
  'Approve or request revision of MDT recommendation'
];

const MOCK_CASES: MockCase[] = [
  {
    id: 'MDT-001',
    patientId: 'P-1001',
    patientName: 'Sarah Johnson',
    diagnosis: 'Invasive Ductal Carcinoma',
    stage: 'Stage IIB',
    age: 48,
    priority: 'High',
    radiology:
      'MRI breast: 2.8 cm irregular enhancing lesion in upper outer quadrant, no chest wall invasion. Axillary node with cortical thickening 5 mm.',
    pathology:
      'Core biopsy: Grade 3 IDC. ER 10%, PR negative, HER2 3+ by IHC. Ki-67: 22%.',
    genomics:
      'NGS panel: PIK3CA H1047R detected, TP53 R248Q detected. TMB low. No MSI.',
    transcript:
      '48-year-old woman with stage IIB invasive ductal carcinoma, presenting with two positive axillary lymph nodes. ' +
      'Breast MRI shows a 2.8 cm irregular enhancing lesion in the upper outer quadrant, with no evidence of chest wall invasion. ' +
      'There is one suspicious ipsilateral axillary lymph node. Core biopsy confirms grade 3 invasive ductal carcinoma. ' +
      'Oestrogen receptor is low positive at 10 percent, progesterone receptor is negative, HER2 is 3 positive, and Ki-67 is 22 percent. ' +
      'Molecular panel reports a PIK3CA H1047R mutation and a TP53 R248Q mutation. Tumour mutational burden is low, and microsatellite status is stable. ' +
      'Given the HER2 positivity and nodal disease, the recommendation is to proceed with neoadjuvant anti-HER2 systemic therapy prior to surgery. ' +
      'The team agrees with a neoadjuvant approach, followed by restaging and definitive surgical planning. ' +
      'Consensus is to proceed with neoadjuvant anti-HER2 regimen, reassess response, and then perform definitive surgery.',
    transcriptAudioUri: ''
  }
];

const DEFAULT_CALENDAR_ATTENDEES: CalendarAttendee[] = [
  { id: 'oncology', name: 'Dr. Patel', role: 'Medical Oncology', confirmed: true },
  { id: 'surgery', name: 'Dr. Nguyen', role: 'Surgical Oncology', confirmed: false },
  { id: 'radiology', name: 'Dr. Romero', role: 'Radiology', confirmed: true },
  { id: 'pathology', name: 'Dr. Chen', role: 'Pathology', confirmed: false },
  { id: 'nurse', name: 'Nurse Alvarez', role: 'MDT Nurse Coordinator', confirmed: true }
];

const FUTURE_DRIVER_OUTPUTS = [
  { gene: 'TP53', disease: 'Breast (current)', active: true },
  { gene: 'BRCA1', disease: 'Breast', active: false },
  { gene: 'BRCA2', disease: 'Breast', active: false },
  { gene: 'EGFR', disease: 'Lung', active: false },
  { gene: 'KRAS', disease: 'Lung/Colorectal', active: false },
  { gene: 'BRAF', disease: 'Colorectal', active: false },
  { gene: 'CTNNB1', disease: 'Liver', active: false },
] as const;

const PATH_FOUNDATION_VS_CNN_BENCHMARK = {
  dataset: 'TCGA-BRCA primary-slide test split',
  n: 50,
  threshold: 0.5,
  source: 'backend/diagnosticore-service/output/pathfoundation_tp53_200/comparison_vs_cnn.json',
  cnn: {
    accuracy: 0.7,
    recall: 0.3,
    f1: 0.4444444444444444
  },
  pathFoundation: {
    accuracy: 0.72,
    recall: 0.65,
    f1: 0.65
  }
} as const;

const TOUR_STEPS_BY_TAB = {
  board: [
    {
      id: 'tour-run',
      title: 'Run Pipeline',
      body: 'Start the full MDT orchestration for this case. This sends notes + transcript to backend services.',
      hint: 'This is the primary judge flow.'
    },
    {
      id: 'tour-refresh',
      title: 'Refresh Outputs',
      body: 'Pull latest generated artifacts from the backend draft endpoint.',
      hint: 'Use after a run or if network hiccups.'
    },
    {
      id: 'tour-inputs',
      title: 'Review Inputs',
      body: 'These fields are the exact radiology, pathology, genomics, and transcript context used for analysis.',
      hint: 'Keep this concise and clinically grounded.'
    },
    {
      id: 'tour-clinical-workspace',
      title: 'Clinical Review Workspace',
      body: 'This is the synthesized clinical view: safety signals, risks, actions, evidence, uncertainty, and confidence.',
      hint: 'Most board decisions happen from this panel.'
    },
    {
      id: 'tour-safety-signals',
      title: 'Safety Signals',
      body: 'Potential safety issues and escalation markers surfaced by the pipeline.',
      hint: 'Review this before approval.'
    },
    {
      id: 'tour-key-risks',
      title: 'Key Risks',
      body: 'Case-level risk factors extracted from pathology, genomics, imaging, and literature context.',
      hint: 'Use this to frame risk conversation.'
    },
    {
      id: 'tour-recommended-actions',
      title: 'Recommended Actions',
      body: 'Structured clinical actions proposed for follow-up and decision support.',
      hint: 'Treat as a checklist, not autopilot.'
    },
    {
      id: 'tour-evidence-links',
      title: 'Evidence Links',
      body: 'Grounding references with title + PMID so clinicians can prioritize reading quickly.',
      hint: 'Open sources directly from here.'
    },
    {
      id: 'tour-uncertainty-confidence',
      title: 'Uncertainty & Confidence',
      body: 'Shows uncertainty narrative and confidence score for fast trust calibration.',
      hint: 'High confidence still requires clinician judgment.'
    },
    {
      id: 'tour-pipeline',
      title: 'Monitor Stages',
      body: 'Track fan-out, synthesis, SOAP generation, and HITL gate completion.',
      hint: 'Watch progress before sign-off.'
    },
    {
      id: 'tour-approval',
      title: 'Approve or Rework',
      body: 'Final outputs remain blocked until clinician approval.',
      hint: 'This is your human safety gate.'
    }
  ],
  reasoning: [
    {
      id: 'tour-reasoning-summary',
      title: 'Clinical Summary',
      body: 'This is the consolidated reasoning summary generated from the latest run.',
      hint: 'Start here during review.'
    },
    {
      id: 'tour-reasoning-model',
      title: 'Model Route & Mode',
      body: 'Shows the routing and generation mode used for this output.',
      hint: 'Useful for auditability.'
    },
    {
      id: 'tour-reasoning-risks',
      title: 'Key Risks',
      body: 'Highlights risk factors extracted from multimodal signals.',
      hint: 'Bring these into board discussion.'
    },
    {
      id: 'tour-reasoning-actions',
      title: 'Recommended Actions',
      body: 'Action list proposed for follow-up and decision support.',
      hint: 'Use as structured checklist input.'
    }
  ],
  dictation: [
    {
      id: 'tour-dictation-modes',
      title: 'Dictation Controls',
      body: 'Choose demo, live record, or manual dictation flow from this control row.',
      hint: 'Demo mode is fastest for judging.'
    },
    {
      id: 'tour-dictation-record',
      title: 'Live Recording',
      body: 'Start, stop/upload, or upload audio for MedASR-driven transcript generation.',
      hint: 'Only visible when Live mode is active.'
    },
    {
      id: 'tour-dictation-transcript',
      title: 'Transcript Editor',
      body: 'Edit clinician dictation before sending it into board prep.',
      hint: 'This is the primary text input.'
    },
    {
      id: 'tour-dictation-uri',
      title: 'Audio URI',
      body: 'Optional: set a local file URI (for example `file:///...`) only if you want backend-side transcription.',
      hint: 'Used by MedASR on pipeline run.'
    }
  ],
  diagnosticore: [
    {
      id: 'tour-diagno-wsi',
      title: 'Whole-Slide Input',
      body: 'This panel displays real WSI DeepZoom tiles and patch scanning context for the selected case.',
      hint: 'Shows where tile extraction starts.'
    },
    {
      id: 'tour-diagno-pipeline',
      title: 'Feature Pipeline',
      body: 'Path Foundation tile embeddings feed a TP53 prediction head for case-level risk support.',
      hint: 'Path Foundation + TP53 head is shown here.'
    },
    {
      id: 'tour-diagno-prediction',
      title: 'TP53 Prediction',
      body: 'Case-level probability and confidence interval are surfaced for board prep.',
      hint: 'Use this to communicate mutation likelihood.'
    },
    {
      id: 'tour-diagno-metrics',
      title: 'Validation Metrics',
      body: 'Precision, recall, F1, and AUROC summarize current model performance.',
      hint: 'Keep this transparent for trust.'
    }
  ],
  patients: [
    {
      id: 'tour-patients-list',
      title: 'Patient Case History',
      body: 'Review locally saved case snapshots for this patient and reopen prior outputs without re-running the pipeline.',
      hint: 'History is filtered from your local baseline timestamp.'
    }
  ],
  calendar: [
    {
      id: 'tour-calendar-schedule',
      title: 'MDT Calendar',
      body: 'Schedule board meetings and track clinician attendance readiness in one place.',
      hint: 'This panel is currently a UI mock for workflow planning.'
    }
  ]
} as const;

const PIPELINE_STAGES = [
  {
    title: 'Parallel Fan-Out',
    subtitle: 'Radiology, Pathology, Genomics, Literature',
    icon: Activity
  },
  {
    title: 'Consensus Synthesis',
    subtitle: 'Cross-domain recommendation merge',
    icon: Brain
  },
  {
    title: 'SOAP Generation',
    subtitle: 'Structured draft for board summary',
    icon: FileText
  },
  {
    title: 'HITL Safety Gate',
    subtitle: 'Clinician approval required',
    icon: ShieldCheck
  }
];

const mapStatusToApproval = (status: string): ApprovalState => {
  if (status === 'approved') return 'approved';
  if (status === 'rework_required') return 'rework';
  return 'pending';
};

const mapStatusLabel = (status: string): string => {
  if (!status) return 'PENDING_REVIEW';
  return status.toUpperCase();
};

const getLocalDateForInput = (): string => {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const HISTORY_LAST_SEEN_SNAPSHOT_STORAGE_KEY = 'voxelomics_patient_history_last_seen_snapshot_v1';

const getHistoryLastSeenSnapshotId = (): number => {
  if (typeof window === 'undefined') return 0;
  const raw = window.localStorage.getItem(HISTORY_LAST_SEEN_SNAPSHOT_STORAGE_KEY);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return Math.floor(parsed);
};

const persistHistoryLastSeenSnapshotId = (value: number): void => {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(
    HISTORY_LAST_SEEN_SNAPSHOT_STORAGE_KEY,
    String(Math.max(0, Math.floor(value)))
  );
};

const getApiError = async (res: Response): Promise<string> => {
  try {
    const data = await res.json();
    if (typeof data?.detail === 'string') return data.detail;
    if (typeof data?.message === 'string') return data.message;
  } catch {
    // ignore json parse errors
  }
  return `Request failed (${res.status})`;
};

const OPEN_SEADRAGON_LOCAL_SCRIPT_SRC = resolveFrontendAssetUrl(
  './vendor/openseadragon/openseadragon.min.js'
);
const OPEN_SEADRAGON_CDN_SCRIPT_SRC =
  'https://cdn.jsdelivr.net/npm/openseadragon@5.0.1/build/openseadragon/openseadragon.min.js';
const OPEN_SEADRAGON_IMAGES_PREFIX = resolveFrontendAssetUrl('./vendor/openseadragon/images/');
const OPEN_SEADRAGON_NAV_IMAGES = {
  zoomIn: {
    REST: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomin_rest.png'),
    GROUP: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomin_grouphover.png'),
    HOVER: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomin_hover.png'),
    DOWN: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomin_pressed.png'),
  },
  zoomOut: {
    REST: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomout_rest.png'),
    GROUP: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomout_grouphover.png'),
    HOVER: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomout_hover.png'),
    DOWN: resolveFrontendAssetUrl('./vendor/openseadragon/images/zoomout_pressed.png'),
  },
  home: {
    REST: resolveFrontendAssetUrl('./vendor/openseadragon/images/home_rest.png'),
    GROUP: resolveFrontendAssetUrl('./vendor/openseadragon/images/home_grouphover.png'),
    HOVER: resolveFrontendAssetUrl('./vendor/openseadragon/images/home_hover.png'),
    DOWN: resolveFrontendAssetUrl('./vendor/openseadragon/images/home_pressed.png'),
  },
  fullpage: {
    REST: resolveFrontendAssetUrl('./vendor/openseadragon/images/fullpage_rest.png'),
    GROUP: resolveFrontendAssetUrl('./vendor/openseadragon/images/fullpage_grouphover.png'),
    HOVER: resolveFrontendAssetUrl('./vendor/openseadragon/images/fullpage_hover.png'),
    DOWN: resolveFrontendAssetUrl('./vendor/openseadragon/images/fullpage_pressed.png'),
  },
};

const loadScriptBySrc = (src: string): Promise<void> =>
  new Promise<void>((resolve, reject) => {
    const existingScript = document.querySelector<HTMLScriptElement>(
      `script[data-openseadragon-src="${src}"]`
    );
    if (existingScript) {
      if ((existingScript as HTMLScriptElement).dataset.loaded === 'true') {
        resolve();
        return;
      }
      existingScript.addEventListener('load', () => resolve(), { once: true });
      existingScript.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), {
        once: true,
      });
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.dataset.openseadragonLoader = 'true';
    script.dataset.openseadragonSrc = src;
    script.onload = () => {
      script.dataset.loaded = 'true';
      resolve();
    };
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });

const loadOpenSeadragon = async (): Promise<OpenSeadragonFactory> => {
  if (typeof window === 'undefined') throw new Error('OpenSeadragon requires a browser environment.');
  if (window.OpenSeadragon) return window.OpenSeadragon;
  if (window.__openSeadragonLoader) return window.__openSeadragonLoader;

  window.__openSeadragonLoader = new Promise<OpenSeadragonFactory>(async (resolve, reject) => {
    try {
      await loadScriptBySrc(OPEN_SEADRAGON_LOCAL_SCRIPT_SRC);
      if (window.OpenSeadragon) {
        resolve(window.OpenSeadragon);
        return;
      }
    } catch {
      // Continue to module import / CDN fallback.
    }

    try {
      const mod = await import('openseadragon');
      const resolved = (
        (mod as { default?: OpenSeadragonFactory; OpenSeadragon?: OpenSeadragonFactory }).default ||
        (mod as { default?: OpenSeadragonFactory; OpenSeadragon?: OpenSeadragonFactory }).OpenSeadragon
      );
      if (resolved) {
        window.OpenSeadragon = resolved;
        resolve(resolved);
        return;
      }
    } catch {
      // Fallback to CDN when local vendor and module import are unavailable.
    }

    try {
      await loadScriptBySrc(OPEN_SEADRAGON_CDN_SCRIPT_SRC);
      if (window.OpenSeadragon) {
        resolve(window.OpenSeadragon);
        return;
      }
      reject(new Error('OpenSeadragon loaded but global object was not found.'));
    } catch {
      reject(new Error('Failed to load OpenSeadragon from local vendor and CDN.'));
    }
  });

  return window.__openSeadragonLoader;
};

interface MDTCommandProps {
  onNavigateHome?: () => void;
}

export const MDTCommand: React.FC<MDTCommandProps> = ({ onNavigateHome }) => {
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>('board');
  const [dictationMode, setDictationMode] = useState<DictationMode>('select');
  const [pipelineState, setPipelineState] = useState<PipelineState>('idle');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [completedStages, setCompletedStages] = useState(0);
  const [approvalState, setApprovalState] = useState<ApprovalState>('pending');
  const [trace, setTrace] = useState<string[]>([]);
  const [tourOpen, setTourOpen] = useState(false);
  const [tourStep, setTourStep] = useState(0);
  const [tourRect, setTourRect] = useState<DOMRect | null>(null);
  const [tourSeen, setTourSeen] = useState(false);
  const [tourTab, setTourTab] = useState<WorkspaceTab>('board');
  const [selectedCase, setSelectedCase] = useState<MockCase>(MOCK_CASES[0]);
  const [historyLastSeenSnapshotId, setHistoryLastSeenSnapshotId] = useState<number>(() =>
    getHistoryLastSeenSnapshotId()
  );
  const [patientCaseHistory, setPatientCaseHistory] = useState<PatientCaseHistoryItem[]>([]);
  const [isLoadingPatientCaseHistory, setIsLoadingPatientCaseHistory] = useState(false);
  const [loadingSnapshotId, setLoadingSnapshotId] = useState<number | null>(null);
  const [deletingSnapshotIds, setDeletingSnapshotIds] = useState<number[]>([]);
  const [patientCaseHistoryError, setPatientCaseHistoryError] = useState('');

  const [backendCaseStatus, setBackendCaseStatus] = useState('created');
  const [consensusOutput, setConsensusOutput] = useState('');
  const [soapOutput, setSoapOutput] = useState('');
  const [soapSubjective, setSoapSubjective] = useState('');
  const [soapObjective, setSoapObjective] = useState('');
  const [soapAssessment, setSoapAssessment] = useState('');
  const [soapPlan, setSoapPlan] = useState('');
  const [evidenceOutput, setEvidenceOutput] = useState('');
  const [transcriptionOutput, setTranscriptionOutput] = useState('');
  const [transcriptionMeta, setTranscriptionMeta] = useState('');
  const [safetyFlags, setSafetyFlags] = useState<string[]>([]);
  const [approvalChecklist, setApprovalChecklist] = useState<string[]>(DEFAULT_APPROVAL_CHECKLIST);
  const [clinicalReasoningSummary, setClinicalReasoningSummary] = useState('');
  const [clinicalReasoningUncertainty, setClinicalReasoningUncertainty] = useState('');
  const [clinicalReasoningModelRoute, setClinicalReasoningModelRoute] = useState('');
  const [clinicalReasoningGenerationMode, setClinicalReasoningGenerationMode] = useState('');
  const [consensusConfidence, setConsensusConfidence] = useState<number | null>(null);
  const [clinicalReasoningKeyRisks, setClinicalReasoningKeyRisks] = useState<string[]>([]);
  const [clinicalReasoningRecommendedActions, setClinicalReasoningRecommendedActions] = useState<string[]>([]);
  const [clinicalReasoningConfirmatoryActions, setClinicalReasoningConfirmatoryActions] = useState<string[]>([]);
  const [clinicalReasoningEvidenceLinks, setClinicalReasoningEvidenceLinks] = useState<string[]>([]);
  const [evidenceReferences, setEvidenceReferences] = useState<EvidenceReferenceItem[]>([]);
  const [diagnosticoreArtifact, setDiagnosticoreArtifact] = useState<DraftResponse['artifacts']['diagnosticore'] | null>(null);
  const [deepZoomStatus, setDeepZoomStatus] = useState('');
  const [deepZoomReady, setDeepZoomReady] = useState(false);
  const [deepZoomIsFullPage, setDeepZoomIsFullPage] = useState(false);
  const [useNativeDeepZoom, setUseNativeDeepZoom] = useState(false);
  const [transcriptSource, setTranscriptSource] = useState<TranscriptSource>('manual');

  const [apiStatusMessage, setApiStatusMessage] = useState('Connected to backend-ready workspace.');
  const [apiError, setApiError] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isApproving, setIsApproving] = useState(false);
  const [isSyncingEvidence, setIsSyncingEvidence] = useState(false);
  const [isRecordingAudio, setIsRecordingAudio] = useState(false);
  const [isUploadingAudio, setIsUploadingAudio] = useState(false);
  const [audioRecorderLabel, setAudioRecorderLabel] = useState('Ready to record');
  const [healthLabel, setHealthLabel] = useState('Checking health...');
  const [healthBadgeClass, setHealthBadgeClass] = useState(
    'border-slate-200 bg-slate-50 text-slate-700'
  );
  const [evidenceSyncLabel, setEvidenceSyncLabel] = useState(
    'Evidence cache: not synced yet.'
  );

  const [radiologyInput, setRadiologyInput] = useState(selectedCase.radiology);
  const [pathologyInput, setPathologyInput] = useState(selectedCase.pathology);
  const [genomicsInput, setGenomicsInput] = useState(selectedCase.genomics);
  const [transcriptInput, setTranscriptInput] = useState(selectedCase.transcript);
  const [transcriptAudioUriInput, setTranscriptAudioUriInput] = useState(selectedCase.transcriptAudioUri);
  const [meetingDate, setMeetingDate] = useState(getLocalDateForInput);
  const [meetingTime, setMeetingTime] = useState('09:30');
  const [meetingLocation, setMeetingLocation] = useState('Tumor Board Room A');
  const [meetingAgenda, setMeetingAgenda] = useState(
    'Review high-priority oncology cases and lock multidisciplinary treatment sequencing.'
  );
  const [calendarAttendees, setCalendarAttendees] = useState<CalendarAttendee[]>(DEFAULT_CALENDAR_ATTENDEES);

  const draftPollInFlightRef = useRef(false);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioProcessorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const recordingActiveRef = useRef(false);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const pcmSampleRateRef = useRef<number>(16000);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const deepZoomViewerContainerRef = useRef<HTMLDivElement | null>(null);
  const deepZoomViewerRef = useRef<OpenSeadragonViewer | null>(null);

  const progressPercent = Math.round((completedStages / PIPELINE_STAGES.length) * 100);
  const diagnosticoreMetrics = useMemo(() => {
    const raw = diagnosticoreArtifact?.validation_metrics || {};
    return [
      { label: 'Accuracy', value: raw.accuracy },
      { label: 'Precision', value: raw.precision },
      { label: 'Recall', value: raw.recall },
      { label: 'F1', value: raw.f1 },
      { label: 'AUROC', value: raw.roc_auc }
    ].filter((metric): metric is { label: string; value: number } => typeof metric.value === 'number');
  }, [diagnosticoreArtifact]);
  const diagnosticoreBenchmarkRows = useMemo(
    () => [
      {
        label: 'Recall',
        pathFoundation: PATH_FOUNDATION_VS_CNN_BENCHMARK.pathFoundation.recall,
        cnn: PATH_FOUNDATION_VS_CNN_BENCHMARK.cnn.recall
      },
      {
        label: 'F1',
        pathFoundation: PATH_FOUNDATION_VS_CNN_BENCHMARK.pathFoundation.f1,
        cnn: PATH_FOUNDATION_VS_CNN_BENCHMARK.cnn.f1
      },
      {
        label: 'Accuracy',
        pathFoundation: PATH_FOUNDATION_VS_CNN_BENCHMARK.pathFoundation.accuracy,
        cnn: PATH_FOUNDATION_VS_CNN_BENCHMARK.cnn.accuracy
      }
    ],
    []
  );
  const diagnosticoreProbability = diagnosticoreArtifact?.tp53_probability;
  const diagnosticoreProbabilityPct =
    typeof diagnosticoreProbability === 'number' ? Math.round(diagnosticoreProbability * 100) : null;
  const diagnosticoreThreshold =
    typeof diagnosticoreArtifact?.threshold === 'number'
      ? diagnosticoreArtifact.threshold
      : diagnosticoreArtifact?.locked_threshold_report?.selected_threshold;
  const diagnosticoreTilePreviewUrl = diagnosticoreArtifact
    ? `${API_BASE_URL}/mdt/${selectedCase.id}/diagnosticore/tile-preview`
    : '';
  const diagnosticoreDeepZoomUrl = diagnosticoreArtifact?.deepzoom_dzi_path
    ? `${API_BASE_URL}/mdt/${selectedCase.id}/diagnosticore/deepzoom.dzi`
    : '';
  const confirmedAttendeeCount = useMemo(
    () => calendarAttendees.filter((attendee) => attendee.confirmed).length,
    [calendarAttendees]
  );
  const transcriptSourceLabel = useMemo(() => {
    if (transcriptSource === 'live') return 'MedASR (Live Record)';
    if (transcriptSource === 'upload') return 'MedASR (Uploaded Audio)';
    return 'Manual Transcript';
  }, [transcriptSource]);
  const unseenPatientCaseCount = useMemo(() => {
    if (!patientCaseHistory.length) return 0;
    return patientCaseHistory.filter((snapshot) => snapshot.snapshot_id > historyLastSeenSnapshotId).length;
  }, [patientCaseHistory, historyLastSeenSnapshotId]);
  const activeSafetyFlags = useMemo(() => {
    return safetyFlags.filter((flag) => {
      const normalized = String(flag || '')
        .trim()
        .toLowerCase()
        .replace(/^[-•\s]+/, '')
        .replace(/[.!]+$/, '')
        .trim();
      if (!normalized) return false;
      return !/^(none|no|na|n\/a|no active safety flags|no safety flags)$/i.test(normalized);
    });
  }, [safetyFlags]);
  const transcriptSourceUsesMedAsr = transcriptSource !== 'manual';
  const transcriptTextareaEditable = transcriptSource === 'manual';

  const stageStatus = (idx: number): 'done' | 'active' | 'pending' => {
    if (pipelineState === 'done' || completedStages > idx) return 'done';
    if (pipelineState === 'running' && completedStages === idx) return 'active';
    return 'pending';
  };

  const isReadyForApproval =
    pipelineState === 'done' &&
    backendCaseStatus === 'pending_approval' &&
    !isApproving;

  const pushTrace = (message: string) => {
    const now = new Date();
    const stamp = `${now.getHours().toString().padStart(2, '0')}:${now
      .getMinutes()
      .toString()
      .padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    setTrace((prev) => [`${stamp}  ${message}`, ...prev].slice(0, 20));
  };

  const normalizeConfidence = (value: unknown): number | null => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      if (value > 1 && value <= 100) return Math.max(0, Math.min(1, value / 100));
      return Math.max(0, Math.min(1, value));
    }
    if (typeof value === 'string') {
      const text = value.trim();
      if (!text) return null;
      const numeric = Number(text.replace('%', ''));
      if (Number.isFinite(numeric)) {
        if (text.includes('%') || (numeric > 1 && numeric <= 100)) {
          return Math.max(0, Math.min(1, numeric / 100));
        }
        return Math.max(0, Math.min(1, numeric));
      }
    }
    return null;
  };

  const hydrateFromDraft = (draft: DraftResponse) => {
    setBackendCaseStatus(draft.status);
    setApprovalState(mapStatusToApproval(draft.status));

    const consensus = draft.artifacts.consensus?.recommendation || '';
    const soap = draft.artifacts.soap_note;
    const soapS = soap?.subjective || '';
    const soapO = soap?.objective || '';
    const soapA = soap?.assessment || '';
    const soapP = soap?.plan || '';
    const evidence = draft.artifacts.stage_one?.literature?.highlights || '';
    const directConfidence = normalizeConfidence(draft.artifacts.consensus?.confidence);
    const transcriptionText = draft.artifacts.transcription?.transcript || '';
    const transcriptionEngine = draft.artifacts.transcription?.engine || '';
    const transcriptionWer = draft.artifacts.transcription?.wer_estimate;
    const rawTranscriptionNotes = draft.artifacts.transcription?.notes || '';
    const transcriptionNotes =
      /no local audio path was available|audio_uri=none|backend ASR error|non-local/i.test(
        rawTranscriptionNotes
      ) && transcriptSourceUsesMedAsr
        ? 'Using on-device MedASR transcript from frontend upload/recording.'
        : rawTranscriptionNotes;
    const clinicalReasoning = draft.artifacts.clinical_reasoning;
    const metaParts: string[] = [];
    if (transcriptionEngine) metaParts.push(`Engine: ${transcriptionEngine}`);
    if (typeof transcriptionWer === 'number') metaParts.push(`WER: ${transcriptionWer.toFixed(3)}`);
    if (transcriptionNotes) metaParts.push(transcriptionNotes);

    setConsensusOutput(consensus);
    setSoapOutput(soapP);
    setSoapSubjective(soapS);
    setSoapObjective(soapO);
    setSoapAssessment(soapA);
    setSoapPlan(soapP);
    setEvidenceOutput(evidence);
    setTranscriptionOutput(transcriptionText);
    setTranscriptionMeta(metaParts.join(' | '));
    setSafetyFlags(draft.artifacts.hitl_gate?.safety_flags || []);
    setApprovalChecklist(draft.artifacts.hitl_gate?.approval_checklist || DEFAULT_APPROVAL_CHECKLIST);
    setClinicalReasoningSummary(clinicalReasoning?.summary || '');
    setClinicalReasoningUncertainty(clinicalReasoning?.uncertainty_statement || '');
    setClinicalReasoningModelRoute(clinicalReasoning?.model_route || '');
    setClinicalReasoningGenerationMode(clinicalReasoning?.generation_mode || '');
    setConsensusConfidence(directConfidence);
    setClinicalReasoningKeyRisks(clinicalReasoning?.key_risks || []);
    setClinicalReasoningRecommendedActions(clinicalReasoning?.recommended_actions || []);
    setClinicalReasoningConfirmatoryActions(clinicalReasoning?.confirmatory_actions || []);
    setClinicalReasoningEvidenceLinks(clinicalReasoning?.evidence_links || []);
    setDiagnosticoreArtifact(draft.artifacts.diagnosticore || null);

    const canonicalizeIdentifier = (identifier: string): string => {
      const id = identifier.replace(/^\[|\]$/g, '').trim();
      const pmidMatch = id.match(/PMID[:\s-]*(\d+)/i);
      if (pmidMatch) return `PMID:${pmidMatch[1]}`;
      return id;
    };

    const buildEvidenceUrl = (identifier: string): string | undefined => {
      const id = canonicalizeIdentifier(identifier);
      const pmidMatch = id.match(/^PMID:(\d+)$/i);
      if (pmidMatch) return `https://pubmed.ncbi.nlm.nih.gov/${pmidMatch[1]}/`;
      return undefined;
    };

    const isPlaceholderLabel = (label: string, identifier: string): boolean => {
      const normalizedLabel = label.trim().toLowerCase();
      const normalizedIdentifier = identifier.trim().toLowerCase();
      return (
        !normalizedLabel ||
        normalizedLabel === normalizedIdentifier ||
        normalizedLabel === 'untitled evidence'
      );
    };

    const referencesByIdentifier = new Map<string, EvidenceReferenceItem>();
    const pushReference = (item: EvidenceReferenceItem) => {
      const identifier = canonicalizeIdentifier(item.identifier);
      if (!identifier) return;

      const candidate: EvidenceReferenceItem = {
        ...item,
        identifier,
        label: item.label?.trim() || identifier,
        url: item.url || buildEvidenceUrl(identifier),
      };
      const existing = referencesByIdentifier.get(identifier);
      if (!existing) {
        referencesByIdentifier.set(identifier, candidate);
        return;
      }

      const merged: EvidenceReferenceItem = {
        label:
          isPlaceholderLabel(existing.label, existing.identifier) &&
          !isPlaceholderLabel(candidate.label, candidate.identifier)
            ? candidate.label
            : existing.label,
        identifier,
        source: existing.source || candidate.source,
        year: existing.year ?? candidate.year,
        url: existing.url || candidate.url,
      };

      if (isPlaceholderLabel(merged.label, merged.identifier) && !isPlaceholderLabel(candidate.label, candidate.identifier)) {
        merged.label = candidate.label;
      }
      referencesByIdentifier.set(identifier, merged);
    };

    const literatureEvidence = draft.artifacts.stage_one?.literature?.evidence || [];
    for (const evidence of literatureEvidence) {
      const identifier = evidence.identifier?.trim() || '';
      if (!identifier) continue;
      pushReference({
        label: evidence.title || 'Untitled evidence',
        identifier,
        source: evidence.source,
        year: evidence.year,
        url: buildEvidenceUrl(identifier),
      });
    }

    for (const ref of clinicalReasoning?.evidence_links || []) {
      const normalized = canonicalizeIdentifier(ref);
      if (!normalized) continue;
      if (!/^PMID:/i.test(normalized)) continue;
      pushReference({
        label: normalized,
        identifier: normalized,
        url: buildEvidenceUrl(normalized),
      });
    }

    setEvidenceReferences(Array.from(referencesByIdentifier.values()));

    if (directConfidence === null) {
      const uncertaintyText = clinicalReasoning?.uncertainty_statement || '';
      const match = uncertaintyText.match(/confidence(?: in (?:the )?assessment)?(?: is|:)?\s*([0-9]+(?:\.[0-9]+)?%?)/i);
      setConsensusConfidence(match ? normalizeConfidence(match[1]) : null);
    }
  };

  const buildReportMarkdown = (): string => {
    const lines: string[] = [];
    lines.push(`# MDT Case Report - ${selectedCase.id}`);
    lines.push('');
    lines.push(`- Status: ${mapStatusLabel(backendCaseStatus)}`);
    lines.push(`- Model Route: ${clinicalReasoningModelRoute || 'n/a'}`);
    lines.push(`- Generation Mode: ${clinicalReasoningGenerationMode || 'n/a'}`);
    lines.push(`- Confidence: ${consensusConfidence !== null ? consensusConfidence.toFixed(2) : 'n/a'}`);
    lines.push(`- Generated At: ${new Date().toISOString()}`);
    lines.push('');
    lines.push('## Consensus');
    lines.push(consensusOutput || 'No consensus output.');
    lines.push('');
    lines.push('## SOAP');
    lines.push(`- Subjective: ${soapSubjective || 'n/a'}`);
    lines.push(`- Objective: ${soapObjective || 'n/a'}`);
    lines.push(`- Assessment: ${soapAssessment || 'n/a'}`);
    lines.push(`- Plan: ${soapPlan || soapOutput || 'n/a'}`);
    lines.push('');
    lines.push('## Safety Flags');
    if (safetyFlags.length > 0) {
      for (const flag of safetyFlags) lines.push(`- ${flag}`);
    } else {
      lines.push('- None');
    }
    lines.push('');
    lines.push('## Key Risks');
    if (clinicalReasoningKeyRisks.length > 0) {
      for (const risk of clinicalReasoningKeyRisks) lines.push(`- ${risk}`);
    } else {
      lines.push('- None');
    }
    lines.push('');
    lines.push('## Recommended Actions');
    if (clinicalReasoningRecommendedActions.length > 0) {
      for (const action of clinicalReasoningRecommendedActions) lines.push(`- ${action}`);
    } else {
      lines.push('- None');
    }
    lines.push('');
    lines.push('## Evidence References');
    if (evidenceReferences.length > 0) {
      for (const ref of evidenceReferences) {
        const meta = [ref.source, ref.year ? String(ref.year) : ''].filter(Boolean).join(' | ');
        lines.push(`- ${ref.label} (${ref.identifier})${meta ? ` - ${meta}` : ''}${ref.url ? ` - ${ref.url}` : ''}`);
      }
    } else {
      lines.push('- None');
    }
    lines.push('');
    lines.push('## Uncertainty');
    lines.push(clinicalReasoningUncertainty || 'No uncertainty statement.');
    lines.push('');
    return lines.join('\n');
  };

  const buildPdfDocument = (markdown: string): jsPDF => {
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const margin = 48;
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const maxWidth = pageWidth - margin * 2;
    const lineHeight = 15;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10.5);

    const lines = doc.splitTextToSize(
      markdown
        .replace(/^# /gm, '')
        .replace(/^## /gm, '')
        .replace(/\*\*/g, ''),
      maxWidth
    ) as string[];

    let y = margin;
    for (const line of lines) {
      if (y > pageHeight - margin) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += lineHeight;
    }
    return doc;
  };

  const buildTxtContent = (markdown: string): string =>
    markdown
      .replace(/^# /gm, '')
      .replace(/^## /gm, '')
      .replace(/`/g, '');

  const downloadReport = async (format: 'txt' | 'pdf') => {
    const markdown = buildReportMarkdown();
    if (format === 'pdf') {
      const doc = buildPdfDocument(markdown);
      doc.save(`${selectedCase.id}_mdt_report.pdf`);
      pushTrace('Downloaded PDF report');
      return;
    }

    const content = buildTxtContent(markdown);
    const mime = 'text/plain;charset=utf-8';
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedCase.id}_mdt_report.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    pushTrace("Downloaded TXT report");
  };

  const saveReportToFolder = async (format: 'txt' | 'pdf') => {
    const bridge = window.voxelomicsDesktop;
    if (!bridge?.saveReport) {
      setApiStatusMessage('Save to folder is available in the desktop app. Downloading instead.');
      await downloadReport(format);
      return;
    }

    try {
      const markdown = buildReportMarkdown();
      const defaultFileName = `${selectedCase.id}_mdt_report.${format}`;
      if (format === 'txt') {
        const result = await bridge.saveReport({
          format: 'txt',
          defaultFileName,
          textContent: buildTxtContent(markdown),
        });
        if (result?.canceled) {
          setApiStatusMessage('Save canceled.');
          return;
        }
        if (result?.success) {
          setApiStatusMessage(`Saved TXT report to ${result.path}`);
          pushTrace(`Saved TXT report: ${result.path}`);
        }
        return;
      }

      const doc = buildPdfDocument(markdown);
      const arrayBuffer = doc.output('arraybuffer') as ArrayBuffer;
      const bytes = Array.from(new Uint8Array(arrayBuffer));
      const result = await bridge.saveReport({
        format: 'pdf',
        defaultFileName,
        binaryBytes: bytes,
      });
      if (result?.canceled) {
        setApiStatusMessage('Save canceled.');
        return;
      }
      if (result?.success) {
        setApiStatusMessage(`Saved PDF report to ${result.path}`);
        pushTrace(`Saved PDF report: ${result.path}`);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Save failed.';
      setApiError(msg);
      pushTrace(`Save report failed: ${msg}`);
    }
  };

  const formatSyncTimestamp = (value?: string | null): string => {
    if (!value) return 'never';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString();
  };

  const formatHistoryTimestamp = (value?: string | null): string => {
    if (!value) return 'n/a';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString();
  };

  const fetchEvidenceSyncStatus = async (silent = true) => {
    try {
      const res = await fetch(`${API_BASE_URL}/mdt/evidence/status`);
      if (!res.ok) throw new Error(await getApiError(res));
      const payload = (await res.json()) as EvidenceSyncStatusResponse;
      const syncedAt = formatSyncTimestamp(payload.last_synced_at);
      setEvidenceSyncLabel(
        `Evidence cache: ${payload.literature_count} refs · Last sync: ${syncedAt}`
      );
    } catch (error) {
      if (silent) return;
      const msg = error instanceof Error ? error.message : 'Failed to fetch evidence sync status.';
      setApiError(msg);
    }
  };

  const syncEvidenceCache = async () => {
    setIsSyncingEvidence(true);
    setApiError('');
    try {
      pushTrace(`POST /mdt/evidence/sync (${selectedCase.id})`);
      const res = await fetch(`${API_BASE_URL}/mdt/evidence/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: selectedCase.id,
          max_results: 8,
        }),
      });
      if (!res.ok) throw new Error(await getApiError(res));
      const payload = (await res.json()) as EvidenceSyncResponse;
      const syncedAt = formatSyncTimestamp(payload.last_synced_at);
      setApiStatusMessage(
        `${payload.message} ${payload.literature_count} refs cached locally (as of ${syncedAt}).`
      );
      pushTrace(
        `Evidence synced for ${payload.diagnosis} (${payload.genes.join(', ') || 'no genes'}).`
      );
      await fetchEvidenceSyncStatus(true);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Evidence sync failed.';
      setApiError(msg);
      pushTrace(`Evidence sync failed: ${msg}`);
    } finally {
      setIsSyncingEvidence(false);
    }
  };

  const fetchPatientCaseHistory = async (silent = false) => {
    if (!silent) setIsLoadingPatientCaseHistory(true);
    if (!silent) setPatientCaseHistoryError('');
    try {
      const params = new URLSearchParams({
        include_error: 'false',
        limit: '200',
      });
      const res = await fetch(
        `${API_BASE_URL}/mdt/patients/${encodeURIComponent(selectedCase.patientId)}/cases?${params.toString()}`
      );
      if (!res.ok) throw new Error(await getApiError(res));
      const payload = (await res.json()) as PatientCaseHistoryResponse;
      setPatientCaseHistory(Array.isArray(payload.cases) ? payload.cases : []);
      if (!silent) {
        setApiStatusMessage(
          `Loaded ${payload.count} saved case snapshot(s) for ${selectedCase.patientName}.`
        );
        pushTrace(`Loaded patient history (${payload.count} snapshot(s))`);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Failed to load patient case history.';
      setPatientCaseHistoryError(msg);
      if (!silent) {
        setApiError(msg);
        pushTrace(`Patient case history failed: ${msg}`);
      }
    } finally {
      if (!silent) setIsLoadingPatientCaseHistory(false);
    }
  };

  const deleteCaseHistorySnapshot = async (snapshotId: number) => {
    const confirmed = window.confirm(
      `Delete saved snapshot #${snapshotId}? This removes it from local case history only.`
    );
    if (!confirmed) return;
    setDeletingSnapshotIds((prev) => (prev.includes(snapshotId) ? prev : [...prev, snapshotId]));
    setPatientCaseHistoryError('');
    setApiError('');
    try {
      const res = await fetch(`${API_BASE_URL}/mdt/cases/history/${snapshotId}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(await getApiError(res));
      const payload = (await res.json()) as DeleteCaseHistorySnapshotResponse;
      setPatientCaseHistory((prev) => prev.filter((row) => row.snapshot_id !== snapshotId));
      setApiStatusMessage(payload.message || `Deleted snapshot #${snapshotId}.`);
      pushTrace(`Deleted case snapshot #${snapshotId}`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Failed to delete case snapshot.';
      setPatientCaseHistoryError(msg);
      setApiError(msg);
      pushTrace(`Case snapshot delete failed: ${msg}`);
    } finally {
      setDeletingSnapshotIds((prev) => prev.filter((id) => id !== snapshotId));
    }
  };

  const loadCaseHistorySnapshot = async (snapshotId: number) => {
    setLoadingSnapshotId(snapshotId);
    setPatientCaseHistoryError('');
    setApiError('');
    try {
      const res = await fetch(`${API_BASE_URL}/mdt/cases/history/${snapshotId}`);
      if (!res.ok) throw new Error(await getApiError(res));
      const payload = (await res.json()) as CaseHistorySnapshotResponse;
      const snapshot = payload.case;
      const input = snapshot.input_payload || {};
      const imaging = input.imaging || {};
      const pathology = input.pathology || {};
      const genomics = input.genomics || {};
      const transcript = input.transcript || {};

      const nextCase: MockCase = {
        id: snapshot.case_id,
        patientId: snapshot.patient_id || selectedCase.patientId,
        patientName: snapshot.patient_name || selectedCase.patientName,
        diagnosis: snapshot.diagnosis || selectedCase.diagnosis,
        stage: input.stage || selectedCase.stage,
        age: selectedCase.age,
        priority: selectedCase.priority,
        radiology: imaging.ct_report || imaging.mri_report || imaging.pet_report || '',
        pathology: pathology.biopsy_summary || '',
        genomics: genomics.report_summary || '',
        transcript: transcript.raw_text || '',
        transcriptAudioUri: transcript.audio_uri || '',
      };

      setSelectedCase(nextCase);
      setRadiologyInput(nextCase.radiology);
      setPathologyInput(nextCase.pathology);
      setGenomicsInput(nextCase.genomics);
      setTranscriptInput(nextCase.transcript);
      setTranscriptAudioUriInput(nextCase.transcriptAudioUri);

      const draftLike: DraftResponse = {
        success: true,
        case_id: snapshot.case_id,
        status: snapshot.status,
        artifacts: snapshot.artifacts || {},
      };
      hydrateFromDraft(draftLike);
      if (snapshot.artifacts?.consensus) {
        setPipelineState('done');
        setCompletedStages(4);
      } else {
        setPipelineState('idle');
        setCompletedStages(0);
      }
      setWorkspaceTab('board');
      setApiStatusMessage(
        `Loaded saved snapshot #${snapshotId} for case ${snapshot.case_id} (${formatHistoryTimestamp(payload.saved_at)}).`
      );
      pushTrace(`Loaded case snapshot #${snapshotId} (${snapshot.case_id})`);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Failed to load case snapshot.';
      setPatientCaseHistoryError(msg);
      setApiError(msg);
      pushTrace(`Case snapshot load failed: ${msg}`);
    } finally {
      setLoadingSnapshotId(null);
    }
  };

  const fetchDraft = async (caseId: string, silent = false, withTrace = true) => {
    if (!silent) setIsRefreshing(true);
    try {
      const hadTranscript = Boolean(transcriptionOutput);
      const res = await fetch(`${API_BASE_URL}/mdt/${caseId}/draft`);
      if (!res.ok) throw new Error(await getApiError(res));
      const draft = (await res.json()) as DraftResponse;
      hydrateFromDraft(draft);
      if (draft.artifacts.consensus) {
        setPipelineState('done');
        setCompletedStages(4);
      }
      if (withTrace) {
        pushTrace('Draft artifacts synced from backend');
      }
      if (!hadTranscript && draft.artifacts.transcription?.transcript) {
        pushTrace('Live MedASR transcript available during run');
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Failed to fetch draft.';
      setApiError(msg);
      if (withTrace) {
        pushTrace(`Draft sync failed: ${msg}`);
      }
    } finally {
      if (!silent) setIsRefreshing(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (!res.ok) throw new Error(`Health check failed (${res.status})`);
        const payload = (await res.json()) as HealthResponse;
        if (cancelled) return;
        setHealthLabel(
          `Connected: ${payload.service} (${payload.execution_mode} / ${payload.case_store_backend})`
        );
        setHealthBadgeClass('border-blue-200 bg-blue-50 text-blue-700');
      } catch {
        if (cancelled) return;
        setHealthLabel('Health check unavailable');
        setHealthBadgeClass('border-red-200 bg-red-50 text-red-700');
      }
    };

    fetchHealth();
    const timer = window.setInterval(fetchHealth, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    void fetchEvidenceSyncStatus(true);
    const timer = window.setInterval(() => {
      void fetchEvidenceSyncStatus(true);
    }, 60000);
    return () => {
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (workspaceTab !== 'patients') return;
    void fetchPatientCaseHistory(false);
  }, [workspaceTab, selectedCase.patientId]);

  useEffect(() => {
    void fetchPatientCaseHistory(true);
    const timer = window.setInterval(() => {
      void fetchPatientCaseHistory(true);
    }, 30000);
    return () => {
      window.clearInterval(timer);
    };
  }, [selectedCase.patientId]);

  useEffect(() => {
    if (workspaceTab !== 'patients') return;
    const maxSnapshotId = patientCaseHistory.reduce((maxVal, row) => Math.max(maxVal, row.snapshot_id), 0);
    if (maxSnapshotId <= historyLastSeenSnapshotId) return;
    setHistoryLastSeenSnapshotId(maxSnapshotId);
    persistHistoryLastSeenSnapshotId(maxSnapshotId);
  }, [workspaceTab, patientCaseHistory, historyLastSeenSnapshotId]);

  useEffect(() => {
    if (workspaceTab !== 'diagnosticore' || !diagnosticoreDeepZoomUrl) {
      setDeepZoomReady(false);
      setDeepZoomIsFullPage(false);
      setUseNativeDeepZoom(false);
      if (deepZoomViewerRef.current) {
        deepZoomViewerRef.current.destroy();
        deepZoomViewerRef.current = null;
      }
      return;
    }

    let cancelled = false;
    setDeepZoomReady(false);
    setUseNativeDeepZoom(false);
    setDeepZoomStatus('Loading interactive whole-slide viewer...');

    const mountViewer = async () => {
      try {
        const OpenSeadragon = await loadOpenSeadragon();
        if (cancelled) return;
        if (!deepZoomViewerContainerRef.current) {
          setUseNativeDeepZoom(true);
          setDeepZoomStatus('Using built-in offline DeepZoom viewer.');
          return;
        }

        if (deepZoomViewerRef.current) {
          deepZoomViewerRef.current.destroy();
          deepZoomViewerRef.current = null;
        }

        const viewer = OpenSeadragon({
          element: deepZoomViewerContainerRef.current,
          tileSources: diagnosticoreDeepZoomUrl,
          prefixUrl: OPEN_SEADRAGON_IMAGES_PREFIX,
          navImages: OPEN_SEADRAGON_NAV_IMAGES,
          showNavigator: true,
          showNavigationControl: false,
          navigatorPosition: 'BOTTOM_RIGHT',
          animationTime: 0.7,
          blendTime: 0.1,
          minZoomImageRatio: 0.8,
          maxZoomPixelRatio: 2,
          visibilityRatio: 1,
          constrainDuringPan: true,
        });
        deepZoomViewerRef.current = viewer;

        let hasOpened = false;
        let tileFailureCount = 0;

        viewer.addHandler('open', () => {
          if (cancelled) return;
          hasOpened = true;
          setDeepZoomReady(true);
          setDeepZoomStatus('Interactive whole-slide viewer loaded from real WSI pyramid.');
        });
        viewer.addHandler('open-failed', () => {
          if (cancelled) return;
          setDeepZoomReady(false);
          setDeepZoomStatus('Failed to open DeepZoom descriptor from backend.');
        });
        viewer.addHandler('tile-load-failed', () => {
          if (cancelled) return;
          tileFailureCount += 1;
          if (!hasOpened) {
            setDeepZoomReady(false);
            setDeepZoomStatus('DeepZoom tiles are unavailable for this case.');
            return;
          }
          setDeepZoomReady(true);
          setDeepZoomStatus(
            `Interactive viewer loaded with ${tileFailureCount} missing tile${tileFailureCount === 1 ? '' : 's'}.`
          );
        });
        viewer.addHandler('full-page', (event: unknown) => {
          if (cancelled) return;
          const fullPage = Boolean((event as { fullPage?: boolean } | null)?.fullPage);
          setDeepZoomIsFullPage(fullPage);
        });
      } catch {
        if (cancelled) return;
        setUseNativeDeepZoom(true);
        setDeepZoomStatus('OpenSeadragon unavailable. Using built-in offline DeepZoom viewer.');
      }
    };

    void mountViewer();

    return () => {
      cancelled = true;
      if (deepZoomViewerRef.current) {
        deepZoomViewerRef.current.destroy();
        deepZoomViewerRef.current = null;
      }
    };
  }, [workspaceTab, diagnosticoreDeepZoomUrl]);

  const handleDeepZoomZoomIn = () => {
    const viewer = deepZoomViewerRef.current;
    if (!viewer?.viewport) return;
    viewer.viewport.zoomBy(1.2);
    viewer.viewport.applyConstraints();
  };

  const handleDeepZoomZoomOut = () => {
    const viewer = deepZoomViewerRef.current;
    if (!viewer?.viewport) return;
    viewer.viewport.zoomBy(1 / 1.2);
    viewer.viewport.applyConstraints();
  };

  const handleDeepZoomHome = () => {
    deepZoomViewerRef.current?.viewport?.goHome();
  };

  const handleDeepZoomToggleFullPage = () => {
    const viewer = deepZoomViewerRef.current;
    if (!viewer?.setFullPage) return;
    const next = !(viewer.isFullPage?.() ?? deepZoomIsFullPage);
    viewer.setFullPage(next);
    setDeepZoomIsFullPage(next);
  };

  useEffect(() => {
    if (transcriptSource !== 'manual' && transcriptionOutput.trim()) {
      setTranscriptInput(transcriptionOutput);
    }
  }, [transcriptSource, transcriptionOutput]);

  const mergePcmChunks = (chunks: Float32Array[]): Float32Array => {
    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  };

  const pcmToWavBuffer = (samples: Float32Array, sampleRate: number): ArrayBuffer => {
    const bytesPerSample = 2;
    const numChannels = 1;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset: number, value: string) => {
      for (let i = 0; i < value.length; i += 1) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i += 1) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return buffer;
  };

  const resampleLinearPcm = (
    input: Float32Array,
    inputSampleRate: number,
    targetSampleRate: number
  ): Float32Array => {
    if (inputSampleRate === targetSampleRate) return input;
    const ratio = targetSampleRate / inputSampleRate;
    const outputLength = Math.max(1, Math.floor(input.length * ratio));
    const output = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i += 1) {
      const position = i / ratio;
      const index = Math.floor(position);
      const nextIndex = Math.min(index + 1, input.length - 1);
      const frac = position - index;
      output[i] = input[index] * (1 - frac) + input[nextIndex] * frac;
    }
    return output;
  };

  const decodeAudioBlobToPcm16k = async (audioBlob: Blob): Promise<Float32Array> => {
    const encoded = await audioBlob.arrayBuffer();
    const context = new AudioContext();
    try {
      const decoded = await context.decodeAudioData(encoded.slice(0));
      const channelCount = decoded.numberOfChannels || 1;
      const mono = new Float32Array(decoded.length);
      for (let ch = 0; ch < channelCount; ch += 1) {
        const data = decoded.getChannelData(ch);
        for (let i = 0; i < decoded.length; i += 1) {
          mono[i] += data[i] / channelCount;
        }
      }
      return resampleLinearPcm(mono, decoded.sampleRate, 16000);
    } finally {
      await context.close();
    }
  };

  const transcribeAudioLocally = async (
    audioBlob: Blob
  ): Promise<{ transcript: string; engine: string }> => {
    if (!FRONTEND_MEDASR_ENABLED) {
      throw new Error(
        'Frontend MedASR ONNX is disabled. Set VITE_MEDASR_FRONTEND_ONNX=true to enable local worker transcription.'
      );
    }

    const pcm16k = await decodeAudioBlobToPcm16k(audioBlob);
    if (!pcm16k.length) {
      throw new Error('Decoded audio is empty.');
    }

    const workerUrl = new URL(FRONTEND_MEDASR_WORKER_PATH, window.location.href);
    const worker = new Worker(workerUrl);
    const requestId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    return await new Promise<{ transcript: string; engine: string }>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        worker.terminate();
        reject(
          new Error(
            'Local ONNX transcription timed out. Verify MedASR ONNX assets and onnxruntime-web are available.'
          )
        );
      }, 180000);

      worker.onerror = () => {
        window.clearTimeout(timeout);
        worker.terminate();
        reject(
          new Error(
            'Local ONNX worker failed to start. Verify /public/workers/medasr-onnx.worker.js is packaged.'
          )
        );
      };

      worker.onmessage = (event: MessageEvent) => {
        const payload = event.data || {};
        if (payload.id !== requestId) return;
        window.clearTimeout(timeout);
        worker.terminate();
        if (payload.type === 'transcribe:ok') {
          const transcript = String(payload.transcript || '').trim();
          const engine = String(payload.engine || 'medasr-webgpu');
          if (!transcript) {
            reject(new Error('Local ONNX transcription returned an empty transcript.'));
            return;
          }
          resolve({ transcript, engine });
          return;
        }
        const err = String(payload.error || 'Local ONNX transcription failed.');
        reject(new Error(err));
      };

      worker.postMessage(
        {
          type: 'transcribe',
          id: requestId,
          audio_pcm: pcm16k,
          sample_rate: 16000,
          model_url: resolveFrontendAssetUrl(FRONTEND_MEDASR_MODEL_URL),
          vocab_url: resolveFrontendAssetUrl(FRONTEND_MEDASR_VOCAB_URL),
          ort_url: resolveFrontendAssetUrl(FRONTEND_MEDASR_ORT_URL),
          execution_providers: FRONTEND_MEDASR_EXECUTION_PROVIDERS,
          chunk_seconds: FRONTEND_MEDASR_CHUNK_SECONDS,
          overlap_seconds: FRONTEND_MEDASR_OVERLAP_SECONDS,
        },
        [pcm16k.buffer]
      );
    });
  };

  const isSupportedMedAsrAudioFile = (file: File): boolean => {
    const fileName = (file.name || '').toLowerCase();
    const contentType = (file.type || '').toLowerCase();
    if (fileName.endsWith('.wav') || fileName.endsWith('.mp4') || fileName.endsWith('.m4a')) {
      return true;
    }
    return (
      contentType === 'audio/wav' ||
      contentType === 'audio/x-wav' ||
      contentType === 'audio/wave' ||
      contentType === 'audio/mp4' ||
      contentType === 'video/mp4' ||
      contentType === 'audio/x-m4a'
    );
  };

  const startAudioRecording = async () => {
    if (isRecordingAudio || isUploadingAudio) return;
    try {
      setTranscriptSource('live');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      pcmChunksRef.current = [];
      recordingActiveRef.current = true;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      pcmSampleRateRef.current = audioContext.sampleRate;

      const sourceNode = audioContext.createMediaStreamSource(stream);
      const processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      audioSourceNodeRef.current = sourceNode;
      audioProcessorNodeRef.current = processorNode;

      processorNode.onaudioprocess = (event) => {
        if (!recordingActiveRef.current) return;
        const input = event.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(input));
      };

      sourceNode.connect(processorNode);
      processorNode.connect(audioContext.destination);

      setIsRecordingAudio(true);
      setAudioRecorderLabel('Recording (WAV capture)...');
      pushTrace('Microphone recording started (WAV capture)');
    } catch (error) {
      recordingActiveRef.current = false;
      const msg = error instanceof Error ? error.message : 'Unable to access microphone.';
      setApiError(msg);
      setAudioRecorderLabel('Microphone access failed');
      pushTrace(`Recording failed: ${msg}`);
    }
  };

  const stopAudioRecordingAndUpload = async () => {
    if (!isRecordingAudio || isUploadingAudio) return;
    recordingActiveRef.current = false;
    setIsUploadingAudio(true);
    setAudioRecorderLabel('Encoding WAV and transcribing locally...');

    try {
      const processorNode = audioProcessorNodeRef.current;
      const sourceNode = audioSourceNodeRef.current;
      const audioContext = audioContextRef.current;

      if (processorNode) processorNode.disconnect();
      if (sourceNode) sourceNode.disconnect();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioContext) {
        await audioContext.close();
      }

      const mergedPcm = mergePcmChunks(pcmChunksRef.current);
      const wavBuffer = pcmToWavBuffer(mergedPcm, pcmSampleRateRef.current);
      const audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });

      if (audioBlob.size === 0) {
        throw new Error('Recording is empty. Please try again.');
      }

      const { transcript, engine } = await transcribeAudioLocally(audioBlob);
      setTranscriptAudioUriInput('');
      setAudioRecorderLabel('Recording transcribed locally');
      setTranscriptSource('live');
      setTranscriptionOutput(transcript);
      setTranscriptInput(transcript);
      setTranscriptionMeta(`Engine: ${engine} | Local ONNX transcription on-device.`);
      pushTrace('Live audio transcribed locally with MedASR ONNX worker');
      setApiStatusMessage(
        'Live dictation transcribed locally in browser. Backend ASR is bypassed for pipeline runs.'
      );
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Local transcription failed.';
      setApiError(msg);
      setAudioRecorderLabel('Transcription failed');
      pushTrace(`Live transcription failed: ${msg}`);
    } finally {
      setIsRecordingAudio(false);
      setIsUploadingAudio(false);
      recordingActiveRef.current = false;
      mediaStreamRef.current = null;
      audioContextRef.current = null;
      audioSourceNodeRef.current = null;
      audioProcessorNodeRef.current = null;
      pcmChunksRef.current = [];
    }
  };

  const uploadAudioFile = async (file: File) => {
    if (!file || isUploadingAudio) return;
    if (!isSupportedMedAsrAudioFile(file)) {
      const msg = 'Upload WAV, MP4, or M4A audio for MedASR compatibility.';
      setApiError(msg);
      setAudioRecorderLabel('Unsupported audio format');
      pushTrace(`Audio upload blocked: ${msg}`);
      return;
    }
    setIsUploadingAudio(true);
    setAudioRecorderLabel('Transcribing selected file locally...');
    try {
      const { transcript, engine } = await transcribeAudioLocally(file);
      setTranscriptAudioUriInput('');
      setTranscriptSource('upload');
      setTranscriptionOutput(transcript);
      setTranscriptInput(transcript);
      setTranscriptionMeta(`Engine: ${engine} | Local ONNX transcription on-device.`);
      setAudioRecorderLabel(`File transcribed: ${file.name}`);
      pushTrace(`Audio file transcribed locally: ${file.name}`);
      setApiStatusMessage(
        'Uploaded audio transcribed locally in browser. Backend ASR is bypassed for pipeline runs.'
      );
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'File transcription failed.';
      setApiError(msg);
      setAudioRecorderLabel('File transcription failed');
      pushTrace(`Audio file transcription failed: ${msg}`);
    } finally {
      setIsUploadingAudio(false);
    }
  };

  const runLivePipeline = async () => {
    setPipelineState('running');
    setCompletedStages(0);
    setApprovalState('pending');
    setApiError('');
    setApiStatusMessage('Running live MDT orchestration and streaming draft updates...');
    setTrace([]);

    let draftPollTimer: ReturnType<typeof window.setInterval> | null = null;
    try {
      const trimmedAudioUri = transcriptAudioUriInput.trim();
      const localTranscriptText = (transcriptionOutput || transcriptInput).trim();
      if (transcriptSourceUsesMedAsr) {
        if (!localTranscriptText) {
          throw new Error(
            'MedASR local transcript is empty. Record or upload audio and wait for local transcription first.'
          );
        }
      }

      const transcriptNotesOverride = transcriptSourceUsesMedAsr
        ? localTranscriptText
        : transcriptInput;
      const transcriptAudioUriOverride = transcriptSourceUsesMedAsr
        ? ''
        : trimmedAudioUri;

      pushTrace(`POST /mdt/start (${selectedCase.id})`);
      const startRes = await fetch(`${API_BASE_URL}/mdt/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: selectedCase.id,
          overrides: {
            radiology_notes: radiologyInput,
            pathology_notes: pathologyInput,
            genomics_notes: genomicsInput,
            transcript_notes: transcriptNotesOverride,
            transcript_audio_uri: transcriptAudioUriOverride
          }
        })
      });
      if (!startRes.ok) throw new Error(await getApiError(startRes));
      pushTrace('Input overrides submitted from MDT workspace');
      setCompletedStages(1);
      pushTrace('Case initialized');
      pushTrace('Draft polling started (1.5s) for near-real-time transcription');

      draftPollTimer = window.setInterval(async () => {
        if (draftPollInFlightRef.current) return;
        draftPollInFlightRef.current = true;
        try {
          await fetchDraft(selectedCase.id, true, false);
        } finally {
          draftPollInFlightRef.current = false;
        }
      }, 1500);

      pushTrace(`POST /mdt/${selectedCase.id}/analyze`);
      const analyzeController = new AbortController();
      const analyzeTimeout = window.setTimeout(() => {
        analyzeController.abort();
      }, ANALYZE_REQUEST_TIMEOUT_MS);
      const analyzeRes = await fetch(`${API_BASE_URL}/mdt/${selectedCase.id}/analyze`, {
        method: 'POST',
        signal: analyzeController.signal
      }).finally(() => window.clearTimeout(analyzeTimeout));
      if (!analyzeRes.ok) throw new Error(await getApiError(analyzeRes));
      const analyzePayload = (await analyzeRes.json()) as AnalyzeResponse;

      setCompletedStages(3);
      setBackendCaseStatus(analyzePayload.status);
      setApprovalState(mapStatusToApproval(analyzePayload.status));
      setConsensusOutput(analyzePayload.consensus?.recommendation || '');
      setSafetyFlags(analyzePayload.hitl_gate?.safety_flags || []);
      setApprovalChecklist(analyzePayload.hitl_gate?.approval_checklist || DEFAULT_APPROVAL_CHECKLIST);

      pushTrace('Analysis completed and HITL gate applied');
      await fetchDraft(selectedCase.id, true);
      await fetchPatientCaseHistory(true);
      setCompletedStages(4);
      setPipelineState('done');
      setApiStatusMessage('Backend pipeline completed. Case is ready for clinician review.');
    } catch (error) {
      const msg =
        error instanceof DOMException && error.name === 'AbortError'
          ? `Analyze request timed out after ${Math.round(ANALYZE_REQUEST_TIMEOUT_MS / 1000)}s.`
          : error instanceof Error
            ? error.message
            : 'Unexpected backend error.';
      setApiError(msg);
      setApiStatusMessage('Pipeline stopped due to backend error.');
      setPipelineState('idle');
      setCompletedStages(0);
      setBackendCaseStatus('error');
      await fetchDraft(selectedCase.id, true).catch(() => undefined);
      pushTrace(`Pipeline failed: ${msg}`);
    } finally {
      if (draftPollTimer) {
        window.clearInterval(draftPollTimer);
      }
      draftPollInFlightRef.current = false;
    }
  };

  const loadDemoDictation = () => {
    setDictationMode('demo');
    setTranscriptSource('manual');
    setTranscriptInput(selectedCase.transcript);
    setTranscriptAudioUriInput(selectedCase.transcriptAudioUri);
    setTranscriptionOutput(selectedCase.transcript);
    setAudioRecorderLabel('Demo transcript loaded (manual mode)');
    setTranscriptionMeta('Demo dictation loaded from seeded MDT case transcript.');
    setApiStatusMessage('Demo dictation loaded. Switch to MDT Board Prep to run full pipeline.');
    pushTrace('Demo dictation prepared');
  };

  const useDictationForBoardPrep = () => {
    setWorkspaceTab('board');
    setApiStatusMessage('Dictation inputs copied to board prep. Run MDT pipeline when ready.');
  };

  const toggleCalendarAttendee = (attendeeId: string) => {
    setCalendarAttendees((prev) =>
      prev.map((attendee) =>
        attendee.id === attendeeId ? { ...attendee, confirmed: !attendee.confirmed } : attendee
      )
    );
  };

  const saveMockCalendar = () => {
    setApiStatusMessage(
      `Mock MDT meeting scheduled for ${meetingDate} at ${meetingTime} (${meetingLocation}).`
    );
    pushTrace(
      `MDT calendar (mock): ${meetingDate} ${meetingTime} | ${confirmedAttendeeCount}/${calendarAttendees.length} confirmed`
    );
  };

  const handleApproval = async (decision: 'approve' | 'rework') => {
    if (!isReadyForApproval) return;
    setIsApproving(true);
    setApiError('');
    try {
      pushTrace(`POST /mdt/${selectedCase.id}/approve (${decision})`);
      const res = await fetch(`${API_BASE_URL}/mdt/${selectedCase.id}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision,
          clinician_name: 'Dr. Faith',
          notes: decision === 'approve' ? 'Approved in MDT Command UI.' : 'Returned for rework in MDT Command UI.'
        })
      });
      if (!res.ok) throw new Error(await getApiError(res));
      const payload = (await res.json()) as ApproveResponse;
      setBackendCaseStatus(payload.status);
      setApprovalState(mapStatusToApproval(payload.status));
      setApiStatusMessage(payload.message);
      pushTrace(`Approval action completed: ${payload.status}`);
      await fetchDraft(selectedCase.id, true);
      await fetchPatientCaseHistory(true);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Approval request failed.';
      setApiError(msg);
      pushTrace(`Approval failed: ${msg}`);
    } finally {
      setIsApproving(false);
    }
  };

  const activeTourSteps = TOUR_STEPS_BY_TAB[tourTab];
  const activeTourStep = activeTourSteps[tourStep];

  const startTour = () => {
    setTourTab(workspaceTab);
    if (workspaceTab === 'dictation') {
      setDictationMode('live');
    }
    setTourStep(0);
    setTourOpen(true);
    setTourSeen(true);
  };

  const closeTour = () => {
    setTourOpen(false);
    setTourRect(null);
  };

  const nextTourStep = () => {
    if (tourStep >= activeTourSteps.length - 1) {
      closeTour();
      return;
    }
    setTourStep((prev) => prev + 1);
  };

  useEffect(() => {
    if (workspaceTab !== tourTab && tourOpen) {
      closeTour();
    }
  }, [workspaceTab, tourOpen, tourTab]);

  useEffect(() => {
    if (!tourOpen) return;
    const updateRect = () => {
      const el = document.querySelector(`[data-tour-id="${activeTourStep.id}"]`) as HTMLElement | null;
      if (!el) {
        setTourRect(null);
        return;
      }
      setTourRect(el.getBoundingClientRect());
    };
    updateRect();
    window.addEventListener('resize', updateRect);
    window.addEventListener('scroll', updateRect, true);
    return () => {
      window.removeEventListener('resize', updateRect);
      window.removeEventListener('scroll', updateRect, true);
    };
  }, [tourOpen, activeTourStep, workspaceTab, dictationMode]);

  const tourCardStyle = useMemo<React.CSSProperties>(() => {
    if (!tourRect || typeof window === 'undefined') {
      return {
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
      };
    }

    const cardWidth = 380;
    const cardHeight = 250;
    const margin = 16;
    let left = tourRect.left + tourRect.width / 2 - cardWidth / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - cardWidth - margin));

    let top = tourRect.bottom + 14;
    if (top + cardHeight > window.innerHeight - margin) {
      top = Math.max(margin, tourRect.top - cardHeight - 16);
    }

    return { left, top };
  }, [tourRect]);

  const isTourTarget = (id: string) =>
    tourOpen && activeTourStep.id === id;

  return (
    <div className="min-h-screen pt-5 pb-12 bg-gradient-to-b from-[#eef3ff] via-[#f7f9ff] to-white">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <section>
          <button
            type="button"
            onClick={() => onNavigateHome?.()}
            className="inline-flex items-center gap-3 text-[#11173d] hover:opacity-80 transition-opacity"
          >
            <ArrowLeft className="w-5 h-5" />
            <span
              className="text-[1.6rem] leading-none tracking-tight font-black"
              style={{ fontFamily: '"Inter", "Manrope", "Segoe UI", sans-serif' }}
            >
              Voxelomics
            </span>
          </button>
        </section>

        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp4,.m4a,audio/wav,audio/x-wav,audio/mp4,video/mp4,audio/x-m4a"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void uploadAudioFile(file);
            e.currentTarget.value = '';
          }}
        />

        <div className={`flex flex-col lg:flex-row lg:items-start ${sidebarOpen ? 'gap-6' : 'gap-0'}`}>
          <aside
            className={`shrink-0 overflow-hidden transition-all duration-300 ease-in-out ${
              sidebarOpen ? 'w-full lg:w-[260px] opacity-100 translate-x-0' : 'w-0 opacity-0 -translate-x-6 pointer-events-none'
            } self-start`}
          >
            <div className="rounded-2xl border border-[#ced8ff] bg-white p-4 shadow-sm h-fit">
              <div className="pb-4 border-b border-slate-100">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.18em] text-[#6a74a6] font-bold">Case Overview</p>
                    <p className="mt-1 text-sm font-semibold text-[#11173d]">Case {selectedCase.id}</p>
                    <p className="text-xs text-slate-500">{selectedCase.patientName} · {selectedCase.diagnosis}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSidebarOpen(false)}
                    className="inline-flex items-center gap-1 rounded-md border border-[#d4dcff] bg-[#f5f7ff] px-2 py-1 text-[11px] font-semibold text-[#2f4399]"
                    aria-label="Hide sidebar"
                  >
                    <ChevronLeft className="h-3.5 w-3.5" />
                    Hide
                  </button>
                </div>
              </div>

              <nav className="pt-4 space-y-2">
                <button
                  type="button"
                  onClick={() => setWorkspaceTab('board')}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    workspaceTab === 'board'
                      ? 'bg-[#2663eb] text-white'
                      : 'text-[#24306f] hover:bg-[#eef2ff]'
                  }`}
                >
                  <LayoutGrid className="w-4 h-4" />
                  MDT Board Prep
                </button>
                <button
                  type="button"
                  onClick={() => setWorkspaceTab('reasoning')}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    workspaceTab === 'reasoning'
                      ? 'bg-[#2663eb] text-white'
                      : 'text-[#24306f] hover:bg-[#eef2ff]'
                  }`}
                >
                  <Brain className="w-4 h-4" />
                  Clinical Reasoning
                </button>
                <button
                  type="button"
                  onClick={() => setWorkspaceTab('dictation')}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    workspaceTab === 'dictation'
                      ? 'bg-[#2663eb] text-white'
                      : 'text-[#24306f] hover:bg-[#eef2ff]'
                  }`}
                >
                  <MessageSquare className="w-4 h-4" />
                  Clinical Dictation
                </button>
                <button
                  type="button"
                  onClick={() => setWorkspaceTab('diagnosticore')}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    workspaceTab === 'diagnosticore'
                      ? 'bg-[#2663eb] text-white'
                      : 'text-[#24306f] hover:bg-[#eef2ff]'
                  }`}
                >
                  <Microscope className="w-4 h-4" />
                  DiagnostiCore
                </button>
                <button
                  type="button"
                  onClick={() => setWorkspaceTab('patients')}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    workspaceTab === 'patients'
                      ? 'bg-[#2663eb] text-white'
                      : 'text-[#24306f] hover:bg-[#eef2ff]'
                  }`}
                >
                  <Users className="w-4 h-4" />
                  <span className="inline-flex items-center gap-1.5">
                    Patient Cases
                    {unseenPatientCaseCount > 0 && (
                      <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-[#ff4d4f] px-1.5 text-[10px] font-bold leading-none text-white">
                        {unseenPatientCaseCount}
                      </span>
                    )}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => setWorkspaceTab('calendar')}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    workspaceTab === 'calendar'
                      ? 'bg-[#2663eb] text-white'
                      : 'text-[#24306f] hover:bg-[#eef2ff]'
                  }`}
                >
                  <CalendarDays className="w-4 h-4" />
                  Calendar
                </button>
              </nav>

              <div className="mt-5 p-3 rounded-xl bg-[#f5f7ff] border border-[#dfe6ff]">
                <p className="text-[11px] font-bold uppercase tracking-wide text-[#6a74a6]">Status</p>
                <p className="mt-1 text-xs text-[#24306f]">{mapStatusLabel(backendCaseStatus)}</p>
              </div>
            </div>
          </aside>

          <main className="flex-1 min-w-0 w-full space-y-4">
            <div className="flex items-center justify-between gap-2">
              {!sidebarOpen && (
                <button
                  type="button"
                  onClick={() => setSidebarOpen(true)}
                  className="inline-flex items-center gap-1 rounded-lg border border-[#c9d5ff] bg-white px-3 py-1.5 text-xs font-semibold text-[#2f4399]"
                  aria-label="Show sidebar"
                >
                  <ChevronRight className="h-3.5 w-3.5" />
                  Show Sidebar
                </button>
              )}
              <span className={`px-3 py-1.5 rounded-full border text-xs font-semibold ${healthBadgeClass}`}>
                {healthLabel}
              </span>
            </div>

            {workspaceTab === 'board' && (
              <>
                <section className="rounded-2xl border border-[#ced8ff] bg-white p-4 md:p-5 shadow-sm">
                  <div className="flex items-center justify-between gap-3">
                    <h2 className="text-base font-bold text-[#11173d]">Quick Actions</h2>
                  </div>
                  <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                    <button
                      data-tour-id="tour-run"
                      onClick={runLivePipeline}
                      disabled={pipelineState === 'running'}
                      className={`rounded-xl border p-3 text-left transition-colors ${
                        isTourTarget('tour-run') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                      } border-[#d8e0ff] bg-[#f6f8ff] hover:bg-[#eaf0ff] disabled:opacity-60`}
                    >
                      <Play className="w-4 h-4 text-[#2663eb]" />
                      <p className="mt-2 text-sm font-semibold text-[#11173d]">
                        {pipelineState === 'running' ? 'Running...' : 'Run MDT Pipeline'}
                      </p>
                    </button>

                    <button
                      data-tour-id="tour-refresh"
                      onClick={() => fetchDraft(selectedCase.id)}
                      disabled={isRefreshing || pipelineState === 'running'}
                      className={`rounded-xl border p-3 text-left transition-colors ${
                        isTourTarget('tour-refresh') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                      } border-[#d8e0ff] bg-[#f6f8ff] hover:bg-[#eaf0ff] disabled:opacity-60`}
                    >
                      <RefreshCw className={`w-4 h-4 text-[#2663eb] ${isRefreshing ? 'animate-spin' : ''}`} />
                      <p className="mt-2 text-sm font-semibold text-[#11173d]">Refresh Outputs</p>
                    </button>

                    <button
                      onClick={loadDemoDictation}
                      className="rounded-xl border p-3 text-left border-[#d8e0ff] bg-[#f6f8ff] hover:bg-[#eaf0ff] transition-colors"
                    >
                      <Mic className="w-4 h-4 text-[#2663eb]" />
                      <p className="mt-2 text-sm font-semibold text-[#11173d]">Load Demo Dictation</p>
                    </button>

                    <button
                      onClick={() => setWorkspaceTab('dictation')}
                      className="rounded-xl border p-3 text-left border-[#d8e0ff] bg-[#f6f8ff] hover:bg-[#eaf0ff] transition-colors"
                    >
                      <MessageSquare className="w-4 h-4 text-[#2663eb]" />
                      <p className="mt-2 text-sm font-semibold text-[#11173d]">Open Dictation View</p>
                    </button>

                    <button
                      onClick={syncEvidenceCache}
                      disabled={isSyncingEvidence || pipelineState === 'running'}
                      className="rounded-xl border p-3 text-left border-[#d8e0ff] bg-[#f6f8ff] hover:bg-[#eaf0ff] transition-colors disabled:opacity-60"
                    >
                      <RefreshCw className={`w-4 h-4 text-[#2663eb] ${isSyncingEvidence ? 'animate-spin' : ''}`} />
                      <p className="mt-2 text-sm font-semibold text-[#11173d]">
                        {isSyncingEvidence ? 'Syncing...' : 'Sync Evidence'}
                      </p>
                    </button>
                  </div>
                  <p className="mt-3 text-xs text-slate-600">{evidenceSyncLabel}</p>
                </section>

                {apiError && (
                  <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                    {apiError}
                  </div>
                )}
                <div className="rounded-xl border border-[#cfe0ff] bg-[#f2f7ff] px-3 py-2 text-xs text-[#2a4a9f]">
                  {apiStatusMessage}
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-[1.3fr_0.7fr] gap-4 items-start">
                  <section
                    data-tour-id="tour-inputs"
                    className={`rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 md:p-5 space-y-4 ${
                      isTourTarget('tour-inputs') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                    }`}
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <label className="block">
                        <span className="text-xs font-bold text-[#5c6698] inline-flex items-center gap-1.5">
                          <Stethoscope className="w-3.5 h-3.5" />
                          Radiology
                        </span>
                        <textarea
                          value={radiologyInput}
                          onChange={(e) => setRadiologyInput(e.target.value)}
                          className="mt-1.5 w-full h-28 rounded-lg border border-[#d7dffb] bg-[#f9faff] p-3 text-sm text-slate-700 focus:outline-none focus:border-[#2663eb]"
                        />
                      </label>
                      <label className="block">
                        <span className="text-xs font-bold text-[#5c6698] inline-flex items-center gap-1.5">
                          <FileText className="w-3.5 h-3.5" />
                          Pathology
                        </span>
                        <textarea
                          value={pathologyInput}
                          onChange={(e) => setPathologyInput(e.target.value)}
                          className="mt-1.5 w-full h-28 rounded-lg border border-[#d7dffb] bg-[#f9faff] p-3 text-sm text-slate-700 focus:outline-none focus:border-[#2663eb]"
                        />
                      </label>
                      <label className="block">
                        <span className="text-xs font-bold text-[#5c6698] inline-flex items-center gap-1.5">
                          <FlaskConical className="w-3.5 h-3.5" />
                          Genomics
                        </span>
                        <textarea
                          value={genomicsInput}
                          onChange={(e) => setGenomicsInput(e.target.value)}
                          className="mt-1.5 w-full h-28 rounded-lg border border-[#d7dffb] bg-[#f9faff] p-3 text-sm text-slate-700 focus:outline-none focus:border-[#2663eb]"
                        />
                      </label>
                      <label className="block">
                        <span className="text-xs font-bold text-[#5c6698] inline-flex items-center gap-1.5">
                          <Mic className="w-3.5 h-3.5" />
                          Transcript
                        </span>
                        <textarea
                          value={transcriptInput}
                          onChange={(e) => setTranscriptInput(e.target.value)}
                          className="mt-1.5 w-full h-28 rounded-lg border border-[#d7dffb] bg-[#f9faff] p-3 text-sm text-slate-700 focus:outline-none focus:border-[#2663eb]"
                        />
                      </label>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-2 items-end">
                      <label className="block">
                        <span className="text-xs font-bold text-[#5c6698]">Transcript Audio URI</span>
                        <input
                          value={transcriptAudioUriInput}
                          onChange={(e) => setTranscriptAudioUriInput(e.target.value)}
                          className="mt-1.5 w-full rounded-lg border border-[#d7dffb] bg-[#f9faff] p-2 text-xs text-slate-700 focus:outline-none focus:border-[#2663eb]"
                        />
                      </label>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          disabled={isRecordingAudio || isUploadingAudio}
                          className="px-3 py-2 rounded-lg border border-[#c7d3ff] text-xs font-semibold disabled:opacity-60"
                        >
                          Upload Audio File
                        </button>
                        <button
                          type="button"
                          onClick={startAudioRecording}
                          disabled={isRecordingAudio || isUploadingAudio}
                          className="px-3 py-2 rounded-lg bg-[#2663eb] text-white text-xs font-semibold disabled:opacity-60"
                        >
                          Record
                        </button>
                        <button
                          type="button"
                          onClick={stopAudioRecordingAndUpload}
                          disabled={!isRecordingAudio || isUploadingAudio}
                          className="px-3 py-2 rounded-lg border border-[#c7d3ff] text-xs font-semibold disabled:opacity-60"
                        >
                          {isUploadingAudio ? 'Uploading' : 'Stop & Upload'}
                        </button>
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-500">
                      {audioRecorderLabel} Upload WAV/MP4/M4A or record live.
                    </p>

                    <section
                      data-tour-id="tour-clinical-workspace"
                      className={`rounded-xl border border-[#d7e2ff] bg-[#f8fbff] p-4 ${
                        isTourTarget('tour-clinical-workspace') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="text-sm font-bold text-[#11173d]">Clinical Review Workspace</h3>
                        <span className="text-[10px] px-2 py-1 rounded-full border border-[#d3dcff] bg-white text-[#3b4d9f]">
                          {clinicalReasoningGenerationMode ? `Mode: ${clinicalReasoningGenerationMode}` : 'Mode: pending'}
                        </span>
                      </div>
                      <p className="mt-2 text-xs text-slate-700">
                        {clinicalReasoningSummary || consensusOutput || 'Run pipeline to populate risk synthesis and grounded recommendations.'}
                      </p>

                      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                        <article
                          data-tour-id="tour-safety-signals"
                          className={`rounded-lg border border-[#f4d1d1] bg-[#fff7f7] p-3 ${
                            isTourTarget('tour-safety-signals') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                          }`}
                        >
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-[#9a3030]">Safety Signals</p>
                          <ul className="mt-2 space-y-1 text-xs text-[#8a2f2f] max-h-32 overflow-auto">
                            {(activeSafetyFlags.length > 0 ? activeSafetyFlags : ['No active safety flags.']).map((flag, idx) => (
                              <li key={`safety-${idx}-${flag}`}>• {flag}</li>
                            ))}
                          </ul>
                        </article>

                        <article
                          data-tour-id="tour-key-risks"
                          className={`rounded-lg border border-[#d9e2ff] bg-white p-3 ${
                            isTourTarget('tour-key-risks') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                          }`}
                        >
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-[#4c5a97]">Key Risks</p>
                          <ul className="mt-2 space-y-1 text-xs text-slate-700 max-h-32 overflow-auto">
                            {(clinicalReasoningKeyRisks.length > 0 ? clinicalReasoningKeyRisks : ['No key risks yet.']).map((risk, idx) => (
                              <li key={`risk-${idx}-${risk}`}>• {risk}</li>
                            ))}
                          </ul>
                        </article>

                        <article
                          data-tour-id="tour-recommended-actions"
                          className={`rounded-lg border border-[#d9e2ff] bg-white p-3 md:col-span-2 ${
                            isTourTarget('tour-recommended-actions') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                          }`}
                        >
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-[#4c5a97]">Recommended Actions</p>
                          <ul className="mt-2 space-y-1 text-xs text-slate-700 max-h-36 overflow-auto">
                            {(clinicalReasoningRecommendedActions.length > 0
                              ? clinicalReasoningRecommendedActions
                              : ['No recommended actions yet.']).map((action, idx) => (
                              <li key={`action-${idx}-${action}`}>• {action}</li>
                            ))}
                          </ul>
                        </article>

                        <article
                          data-tour-id="tour-evidence-links"
                          className={`rounded-lg border border-[#d9e2ff] bg-white p-3 md:col-span-2 ${
                            isTourTarget('tour-evidence-links') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-[11px] font-semibold uppercase tracking-wide text-[#4c5a97]">
                              Evidence Links
                            </p>
                            {evidenceReferences.length > 0 && (
                              <span className="text-[10px] px-2 py-1 rounded-full border border-[#d8e0ff] bg-[#f5f7ff] text-[#4c5a97]">
                                {evidenceReferences.length} references
                              </span>
                            )}
                          </div>
                          {evidenceReferences.length > 0 ? (
                            <ul className="mt-2 space-y-2 text-xs text-slate-700 max-h-44 overflow-auto pr-1">
                              {evidenceReferences.map((ref, idx) => (
                                <li key={`${ref.identifier}-${ref.label}-${idx}`} className="rounded-md border border-[#e2e7ff] bg-[#f8faff] p-2">
                                  <p className="font-semibold text-slate-800">{ref.label}</p>
                                  <p className="mt-0.5 text-[11px] text-slate-600">
                                    {ref.identifier}
                                    {ref.source ? ` | ${ref.source}` : ''}
                                    {ref.year ? ` (${ref.year})` : ''}
                                  </p>
                                  {ref.url && (
                                    <a
                                      href={ref.url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="mt-1 inline-flex items-center gap-1 text-[11px] font-semibold text-[#2f4399] hover:underline"
                                    >
                                      Open source
                                      <ExternalLink className="w-3 h-3" />
                                    </a>
                                  )}
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <ul className="mt-2 space-y-1 text-xs text-slate-700 max-h-28 overflow-auto">
                              {(clinicalReasoningEvidenceLinks.length > 0
                                ? clinicalReasoningEvidenceLinks
                                : ['No linked evidence yet.']).map((link, idx) => (
                                <li key={`evidence-link-${idx}-${link}`}>• {link}</li>
                              ))}
                            </ul>
                          )}
                        </article>

                        <article
                          data-tour-id="tour-uncertainty-confidence"
                          className={`rounded-lg border border-[#d9e2ff] bg-white p-3 ${
                            isTourTarget('tour-uncertainty-confidence') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                          }`}
                        >
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-[#4c5a97]">Uncertainty</p>
                          <p className="mt-2 text-xs text-slate-700">
                            {clinicalReasoningUncertainty || 'Uncertainty statement will appear after analysis.'}
                          </p>
                        </article>

                        <article
                          className={`rounded-lg border border-[#cfe0ff] bg-[#f6f9ff] p-3 ${
                            isTourTarget('tour-uncertainty-confidence') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                          }`}
                        >
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-[#4c5a97]">Confidence</p>
                          <div className="mt-2 flex items-end gap-2">
                            <p className="text-3xl font-black text-[#0B0D7D]">
                              {consensusConfidence !== null ? consensusConfidence.toFixed(2) : '--'}
                            </p>
                            {consensusConfidence !== null && (
                              <span className="mb-1 inline-flex rounded-full border border-[#bcd0ff] bg-white px-2 py-0.5 text-[10px] font-semibold text-[#2f4399]">
                                {(consensusConfidence * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                          <p className="mt-2 text-[11px] text-slate-600">
                            {consensusConfidence !== null
                              ? 'Model confidence surfaced from consensus output.'
                              : 'Confidence will appear after analysis.'}
                          </p>
                        </article>
                      </div>
                    </section>
                  </section>

                  <aside className="space-y-4">
                    <section
                      data-tour-id="tour-pipeline"
                      className={`rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 ${
                        isTourTarget('tour-pipeline') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                      }`}
                    >
                      <h3 className="text-sm font-bold text-[#11173d]">Pipeline</h3>
                      <div className="mt-2 w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-[#2663eb] to-[#2663eb]" style={{ width: `${progressPercent}%` }} />
                      </div>
                      <p className="mt-2 text-xs text-slate-500">{progressPercent}% complete</p>
                      <div className="mt-3 space-y-2">
                        {PIPELINE_STAGES.map((stage, idx) => {
                          const Icon = stage.icon;
                          const status = stageStatus(idx);
                          return (
                            <div key={stage.title} className="rounded-lg border border-slate-200 p-2.5">
                              <div className="flex items-center gap-2">
                                <Icon className="w-4 h-4 text-[#3d53d8]" />
                                <p className="text-xs font-semibold text-slate-800">{stage.title}</p>
                                <span className="ml-auto text-[10px] text-slate-500 uppercase">{status}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </section>

                    <section
                      data-tour-id="tour-approval"
                      className={`rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 ${
                        isTourTarget('tour-approval') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                      }`}
                    >
                      <h3 className="text-sm font-bold text-[#11173d]">HITL Approval</h3>
                      <ul className="mt-3 space-y-1.5 text-xs text-slate-700">
                        {approvalChecklist.map((item, idx) => (
                          <li key={`approval-${idx}-${item}`} className="flex items-start gap-1.5">
                            <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 text-[#3d53d8]" />
                            {item}
                          </li>
                        ))}
                      </ul>
                      {activeSafetyFlags.length > 0 && (
                        <p className="mt-2 text-[11px] text-slate-500">
                          {activeSafetyFlags.length} safety signal(s) listed in Clinical Review Workspace.
                        </p>
                      )}
                      <div className="mt-3 grid grid-cols-2 gap-2">
                        <button
                          disabled={!isReadyForApproval}
                          onClick={() => handleApproval('approve')}
                          className="px-3 py-2 rounded-lg bg-[#2663eb] text-white text-xs font-semibold disabled:opacity-50"
                        >
                          {isApproving ? 'Submitting...' : 'Approve'}
                        </button>
                        <button
                          disabled={!isReadyForApproval}
                          onClick={() => handleApproval('rework')}
                          className="px-3 py-2 rounded-lg bg-[#edf1ff] text-[#2d3c8d] text-xs font-semibold disabled:opacity-50"
                        >
                          Request Rework
                        </button>
                      </div>
                      <p className="mt-2 text-[11px] font-semibold text-slate-500">Status: {mapStatusLabel(backendCaseStatus)}</p>
                    </section>

                    <section className="rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4">
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="text-sm font-bold text-[#11173d]">Outputs</h3>
                        <div className="flex flex-wrap items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => {
                              void downloadReport('pdf');
                            }}
                            className="inline-flex items-center gap-1 rounded-md border border-[#c9d5ff] bg-white px-2 py-1 text-[11px] font-semibold text-[#2f4399] hover:bg-[#eef2ff]"
                            title="Download PDF report"
                          >
                            <Download className="w-3 h-3" />
                            PDF
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              void saveReportToFolder('pdf');
                            }}
                            className="inline-flex items-center gap-1 rounded-md border border-[#c9d5ff] bg-white px-2 py-1 text-[11px] font-semibold text-[#2f4399] hover:bg-[#eef2ff]"
                            title="Save PDF to a chosen folder"
                          >
                            Save PDF...
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              void downloadReport('txt');
                            }}
                            className="inline-flex items-center gap-1 rounded-md border border-[#c9d5ff] bg-white px-2 py-1 text-[11px] font-semibold text-[#2f4399] hover:bg-[#eef2ff]"
                            title="Download text report"
                          >
                            <Download className="w-3 h-3" />
                            TXT
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              void saveReportToFolder('txt');
                            }}
                            className="inline-flex items-center gap-1 rounded-md border border-[#c9d5ff] bg-white px-2 py-1 text-[11px] font-semibold text-[#2f4399] hover:bg-[#eef2ff]"
                            title="Save TXT to a chosen folder"
                          >
                            Save TXT...
                          </button>
                        </div>
                      </div>
                      <p className="mt-2 text-xs text-slate-700">{consensusOutput || 'Consensus output appears here.'}</p>
                      <div className="mt-3 rounded-lg border border-[#d8e0ff] bg-[#f8faff] p-3">
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-[#4c5a97]">SOAP Note</p>
                        <div className="mt-2 grid grid-cols-1 gap-2">
                          <div>
                            <p className="text-[11px] font-semibold text-[#2f4399]">Subjective</p>
                            <p className="text-xs text-slate-700">{soapSubjective || 'No subjective note yet.'}</p>
                          </div>
                          <div>
                            <p className="text-[11px] font-semibold text-[#2f4399]">Objective</p>
                            <p className="text-xs text-slate-700">{soapObjective || 'No objective note yet.'}</p>
                          </div>
                          <div>
                            <p className="text-[11px] font-semibold text-[#2f4399]">Assessment</p>
                            <p className="text-xs text-slate-700">{soapAssessment || 'No assessment note yet.'}</p>
                          </div>
                          <div>
                            <p className="text-[11px] font-semibold text-[#2f4399]">Plan</p>
                            <p className="text-xs text-slate-700">{soapPlan || soapOutput || 'No SOAP plan yet.'}</p>
                          </div>
                        </div>
                      </div>
                      <div className="mt-2 max-h-24 overflow-auto rounded-lg bg-[#0f1730] p-2">
                        {trace.length > 0 ? trace.map((line, idx) => (
                          <p key={`trace-${idx}-${line}`} className="font-mono text-[10px] text-slate-200">{line}</p>
                        )) : (
                          <p className="text-[10px] text-slate-400">No trace yet.</p>
                        )}
                      </div>
                    </section>
                  </aside>
                </div>
              </>
            )}

            {workspaceTab === 'reasoning' && (
              <section className="rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 md:p-5 space-y-4">
                <article data-tour-id="tour-reasoning-summary" className="rounded-xl border border-[#d8e0ff] bg-[#f5f7ff] p-4">
                  <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">Clinical Summary</p>
                  <p className="mt-2 text-sm text-slate-800">
                    {clinicalReasoningSummary || 'Run pipeline and refresh to load structured reasoning.'}
                  </p>
                  <div data-tour-id="tour-reasoning-model" className="mt-2 flex flex-wrap gap-2 text-[11px]">
                    <span className="px-2 py-1 rounded border border-slate-200 bg-white text-slate-700">
                      Model: {clinicalReasoningModelRoute || 'n/a'}
                    </span>
                    <span className="px-2 py-1 rounded border border-slate-200 bg-white text-slate-700">
                      Mode: {clinicalReasoningGenerationMode || 'n/a'}
                    </span>
                  </div>
                </article>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <article data-tour-id="tour-reasoning-risks" className="rounded-xl border border-[#d8e0ff] p-4">
                    <p className="text-xs font-semibold text-[#5c6698] uppercase tracking-wide">Key Risks</p>
                    <ul className="mt-2 text-sm text-slate-700 space-y-1">
                      {(clinicalReasoningKeyRisks.length > 0 ? clinicalReasoningKeyRisks : ['No key risks yet.']).map((r, idx) => (
                        <li key={`reasoning-risk-${idx}-${r}`}>• {r}</li>
                      ))}
                    </ul>
                  </article>
                  <article data-tour-id="tour-reasoning-actions" className="rounded-xl border border-[#d8e0ff] p-4">
                    <p className="text-xs font-semibold text-[#5c6698] uppercase tracking-wide">Recommended Actions</p>
                    <ul className="mt-2 text-sm text-slate-700 space-y-1">
                      {(clinicalReasoningRecommendedActions.length > 0 ? clinicalReasoningRecommendedActions : ['No actions yet.']).map((r, idx) => (
                        <li key={`reasoning-action-${idx}-${r}`}>• {r}</li>
                      ))}
                    </ul>
                  </article>
                  <article className="rounded-xl border border-[#d8e0ff] p-4 md:col-span-2">
                    <p className="text-xs font-semibold text-[#5c6698] uppercase tracking-wide">Evidence Grounding</p>
                    {evidenceReferences.length > 0 ? (
                      <ul className="mt-2 text-sm text-slate-700 space-y-2">
                        {evidenceReferences.map((ref, idx) => (
                          <li key={`reasoning-${ref.identifier}-${ref.label}-${idx}`} className="rounded-md border border-[#e2e7ff] bg-[#f8faff] p-2.5">
                            <p className="font-semibold text-slate-800">{ref.label}</p>
                            <p className="text-xs text-slate-600">
                              {ref.identifier}
                              {ref.source ? ` | ${ref.source}` : ''}
                              {ref.year ? ` (${ref.year})` : ''}
                            </p>
                            {ref.url && (
                              <a
                                href={ref.url}
                                target="_blank"
                                rel="noreferrer"
                                className="mt-1 inline-flex items-center gap-1 text-xs font-semibold text-[#2f4399] hover:underline"
                              >
                                Open source
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            )}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <ul className="mt-2 text-sm text-slate-700 space-y-1">
                        {(clinicalReasoningEvidenceLinks.length > 0 ? clinicalReasoningEvidenceLinks : ['No linked evidence yet.']).map((r, idx) => (
                          <li key={`reasoning-evidence-link-${idx}-${r}`}>• {r}</li>
                        ))}
                      </ul>
                    )}
                  </article>
                </div>
              </section>
            )}

            {workspaceTab === 'dictation' && (
              <section className="rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 md:p-5 space-y-4">
                <div data-tour-id="tour-dictation-modes" className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={loadDemoDictation}
                    className="px-3 py-2 rounded-lg bg-[#2663eb] text-white text-xs font-semibold"
                  >
                    Load Demo
                  </button>
                  <button
                    type="button"
                    onClick={() => setDictationMode('live')}
                    className="px-3 py-2 rounded-lg border border-[#c7d3ff] text-xs font-semibold text-[#2d3c8d]"
                  >
                    Live Record Mode
                  </button>
                  <button
                    type="button"
                    onClick={() => setDictationMode('manual')}
                    className="px-3 py-2 rounded-lg border border-[#c7d3ff] text-xs font-semibold text-[#2d3c8d]"
                  >
                    Manual Mode
                  </button>
                  <button
                    type="button"
                    onClick={useDictationForBoardPrep}
                    className="px-3 py-2 rounded-lg border border-[#c7d3ff] text-xs font-semibold text-[#2d3c8d]"
                  >
                    Send to Board Prep
                  </button>
                </div>

                {dictationMode === 'live' && (
                  <div data-tour-id="tour-dictation-record" className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={startAudioRecording}
                      disabled={isRecordingAudio || isUploadingAudio}
                      className="px-3 py-2 rounded-lg bg-[#2663eb] text-white text-xs font-semibold disabled:opacity-60"
                    >
                      Start Recording
                    </button>
                    <button
                      type="button"
                      onClick={stopAudioRecordingAndUpload}
                      disabled={!isRecordingAudio || isUploadingAudio}
                      className="px-3 py-2 rounded-lg border border-[#c7d3ff] text-xs font-semibold disabled:opacity-60"
                    >
                      {isUploadingAudio ? 'Uploading...' : 'Stop & Upload'}
                    </button>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isRecordingAudio || isUploadingAudio}
                      className="px-3 py-2 rounded-lg border border-[#c7d3ff] text-xs font-semibold disabled:opacity-60"
                    >
                      Upload Audio
                    </button>
                  </div>
                )}

                <label data-tour-id="tour-dictation-transcript" className="block">
                  <span className="text-xs font-bold text-[#5c6698]">Dictation Transcript</span>
                  <textarea
                    value={transcriptInput}
                    onChange={(e) => setTranscriptInput(e.target.value)}
                    className="mt-1.5 w-full h-40 rounded-lg border border-[#d7dffb] bg-[#f9faff] p-3 text-sm text-slate-700 focus:outline-none focus:border-[#2663eb]"
                  />
                </label>
                <label data-tour-id="tour-dictation-uri" className="block">
                  <span className="text-xs font-bold text-[#5c6698]">Transcript Audio URI</span>
                  <input
                    value={transcriptAudioUriInput}
                    onChange={(e) => setTranscriptAudioUriInput(e.target.value)}
                    className="mt-1.5 w-full rounded-lg border border-[#d7dffb] bg-[#f9faff] p-2 text-xs text-slate-700 focus:outline-none focus:border-[#2663eb]"
                  />
                </label>
                <p className="text-[11px] text-slate-500">{audioRecorderLabel}</p>
                {transcriptionMeta && <p className="text-[11px] text-slate-500">{transcriptionMeta}</p>}
              </section>
            )}

            {workspaceTab === 'diagnosticore' && (
              <section className="rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 md:p-5 space-y-4">
                <div className="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr] gap-4">
                  <article
                    data-tour-id="tour-diagno-wsi"
                    className={`rounded-xl border border-[#d8e0ff] bg-[#f5f7ff] p-4 ${
                      isTourTarget('tour-diagno-wsi') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">WSI Ingestion</p>
                      <span className="inline-flex items-center gap-1 rounded-full border border-[#d4dcff] bg-white px-2 py-1 text-[11px] text-[#33458f]">
                        <Microscope className="w-3.5 h-3.5" />
                        DiagnostiCore
                      </span>
                    </div>
                    <div className="mt-3 rounded-xl border border-[#1f2f6f] bg-[#060c24] p-3">
                      {diagnosticoreDeepZoomUrl ? (
                        <div className="space-y-2">
                          {useNativeDeepZoom ? (
                            <NativeDeepZoomViewer
                              dziUrl={diagnosticoreDeepZoomUrl}
                              onReadyChange={setDeepZoomReady}
                              onStatusChange={setDeepZoomStatus}
                            />
                          ) : (
                            <div className="relative">
                              <div className="absolute left-2 top-2 z-10 flex items-center gap-1.5 rounded-lg border border-[#3b4f9e] bg-[#0a1335]/90 p-1">
                                <button
                                  type="button"
                                  onClick={handleDeepZoomZoomIn}
                                  aria-label="Zoom in"
                                  title="Zoom in"
                                  className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#3b4f9e] bg-[#122461] text-[#dce4ff] hover:bg-[#1a2f79]"
                                >
                                  <Plus className="h-4 w-4" />
                                </button>
                                <button
                                  type="button"
                                  onClick={handleDeepZoomZoomOut}
                                  aria-label="Zoom out"
                                  title="Zoom out"
                                  className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#3b4f9e] bg-[#122461] text-[#dce4ff] hover:bg-[#1a2f79]"
                                >
                                  <Minus className="h-4 w-4" />
                                </button>
                                <button
                                  type="button"
                                  onClick={handleDeepZoomHome}
                                  aria-label="Go home"
                                  title="Go home"
                                  className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#3b4f9e] bg-[#122461] text-[#dce4ff] hover:bg-[#1a2f79]"
                                >
                                  <Home className="h-4 w-4" />
                                </button>
                                <button
                                  type="button"
                                  onClick={handleDeepZoomToggleFullPage}
                                  aria-label={deepZoomIsFullPage ? 'Exit full page' : 'Toggle full page'}
                                  title={deepZoomIsFullPage ? 'Exit full page' : 'Toggle full page'}
                                  className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#3b4f9e] bg-[#122461] text-[#dce4ff] hover:bg-[#1a2f79]"
                                >
                                  {deepZoomIsFullPage ? (
                                    <Minimize2 className="h-4 w-4" />
                                  ) : (
                                    <Maximize2 className="h-4 w-4" />
                                  )}
                                </button>
                              </div>
                              <div
                                ref={deepZoomViewerContainerRef}
                                className="w-full h-[360px] rounded-lg border border-[#3b4f9e] bg-[#09133a] overflow-hidden"
                              />
                            </div>
                          )}
                          <p className="text-[10px] text-[#c4d0ff]">{deepZoomStatus}</p>
                          {!deepZoomReady && diagnosticoreTilePreviewUrl ? (
                            <img
                              src={diagnosticoreTilePreviewUrl}
                              alt="DiagnostiCore tile preview fallback"
                              className="w-full max-w-[220px] aspect-square rounded-md border border-[#3b4f9e] object-cover bg-[#09133a]"
                            />
                          ) : null}
                        </div>
                      ) : diagnosticoreTilePreviewUrl ? (
                        <div className="space-y-2">
                          <img
                            src={diagnosticoreTilePreviewUrl}
                            alt="DiagnostiCore tile preview"
                            className="w-full max-w-[340px] aspect-square rounded-lg border border-[#3b4f9e] object-cover bg-[#09133a]"
                          />
                          <p className="text-[10px] text-[#c4d0ff]">
                            Real tile patch from generated manifest
                            {typeof diagnosticoreArtifact?.tile_preview_x === 'number' &&
                            typeof diagnosticoreArtifact?.tile_preview_y === 'number'
                              ? ` (x=${diagnosticoreArtifact.tile_preview_x}, y=${diagnosticoreArtifact.tile_preview_y})`
                              : ''}
                          </p>
                        </div>
                      ) : (
                        <div className="rounded-lg border border-[#3b4f9e] bg-[#09133a] p-3 text-[11px] text-[#c4d0ff]">
                          No tile preview is available for this case yet.
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-slate-600">
                      {diagnosticoreArtifact?.wsi_file_name
                        ? `Whole-slide source: ${diagnosticoreArtifact.wsi_file_name}`
                        : 'Whole-slide tissue is segmented into inferable patches before feature extraction.'}
                    </p>
                  </article>

                  <article
                    data-tour-id="tour-diagno-pipeline"
                    className={`rounded-xl border border-[#d8e0ff] p-4 ${
                      isTourTarget('tour-diagno-pipeline') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                    }`}
                  >
                    <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">Fusion Inference Pipeline</p>
                    <div className="mt-3 space-y-2.5">
                      {[
                        {
                          title: 'Path Foundation Encoder (ViT)',
                          detail: 'Extracts foundation morphology embeddings from WSI tiles.',
                          icon: Microscope
                        },
                        { title: 'MedGemma Clinical Context', detail: 'Injects structured context for case-level reasoning.', icon: Brain },
                        { title: 'Fusion Head + Calibration', detail: 'Produces TP53 probability with calibrated confidence.', icon: Cpu }
                      ].map((step) => {
                        const Icon = step.icon;
                        return (
                          <div key={step.title} className="rounded-lg border border-[#dbe3ff] bg-[#f8faff] p-3">
                            <div className="flex items-start gap-2">
                              <span className="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-md bg-[#2663eb] text-white">
                                <Icon className="h-3.5 w-3.5" />
                              </span>
                              <div>
                                <p className="text-sm font-semibold text-[#11173d]">{step.title}</p>
                                <p className="mt-0.5 text-xs text-slate-600">{step.detail}</p>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="mt-3 rounded-lg border border-[#d8e0ff] bg-[#f8faff] p-3">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <p className="text-[11px] font-semibold uppercase tracking-wide text-[#5c6698]">
                            Benchmark Snapshot (Same Test Split)
                          </p>
                          <p className="mt-1 text-xs text-slate-700">
                            Path Foundation TP53 head vs CNN baseline on {PATH_FOUNDATION_VS_CNN_BENCHMARK.dataset} (
                            n={PATH_FOUNDATION_VS_CNN_BENCHMARK.n}, threshold={PATH_FOUNDATION_VS_CNN_BENCHMARK.threshold}).
                          </p>
                        </div>
                        <span className="inline-flex rounded-full bg-[#e6ecff] px-2 py-1 text-[11px] font-semibold text-[#2f4399]">
                          Recall +{Math.round((PATH_FOUNDATION_VS_CNN_BENCHMARK.pathFoundation.recall - PATH_FOUNDATION_VS_CNN_BENCHMARK.cnn.recall) * 100)} pts
                        </span>
                      </div>
                      <div className="mt-2.5 space-y-2">
                        {diagnosticoreBenchmarkRows.map((metric) => (
                          <div key={metric.label} className="rounded-md border border-[#dbe3ff] bg-white p-2">
                            <div className="flex items-center justify-between gap-2 text-[11px]">
                              <span className="font-semibold text-[#24306f]">{metric.label}</span>
                              <span className="text-[#5c6698]">
                                Delta{' '}
                                <span className="font-bold text-[#1f3fa7]">
                                  {(metric.pathFoundation - metric.cnn >= 0 ? '+' : '') +
                                    `${Math.round((metric.pathFoundation - metric.cnn) * 100)} pts`}
                                </span>
                              </span>
                            </div>
                            <div className="mt-1.5 grid grid-cols-2 gap-2 text-[11px]">
                              <div className="rounded-md border border-[#ced8ff] bg-[#f4f7ff] px-2 py-1.5">
                                <p className="text-[#5c6698]">Path Foundation</p>
                                <p className="font-bold text-[#11173d]">{Math.round(metric.pathFoundation * 100)}%</p>
                              </div>
                              <div className="rounded-md border border-[#d8def2] bg-[#f9fbff] px-2 py-1.5">
                                <p className="text-[#6c789f]">CNN Baseline</p>
                                <p className="font-bold text-[#2b3356]">{Math.round(metric.cnn * 100)}%</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="mt-2 text-[10px] text-slate-600">
                        Source: {PATH_FOUNDATION_VS_CNN_BENCHMARK.source}
                      </p>
                    </div>
                  </article>
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-[0.95fr_1.05fr] gap-4">
                  <article
                    data-tour-id="tour-diagno-prediction"
                    className={`rounded-xl border border-[#d8e0ff] p-4 ${
                      isTourTarget('tour-diagno-prediction') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">Case-Level Mutation Output</p>
                      <span className="inline-flex items-center gap-1 rounded-full bg-[#ebefff] px-2 py-1 text-[11px] font-semibold text-[#2f4399]">
                        {(diagnosticoreArtifact?.predicted_label || 'unknown').replaceAll('_', ' ').toUpperCase()}
                      </span>
                    </div>
                    <p className="mt-3 text-3xl font-black text-[#11173d]">
                      {typeof diagnosticoreProbability === 'number' ? diagnosticoreProbability.toFixed(3) : 'N/A'}
                    </p>
                    <p className="text-xs text-slate-600">
                      {typeof diagnosticoreThreshold === 'number'
                        ? `Posterior TP53 mutation probability · threshold ${diagnosticoreThreshold.toFixed(3)}`
                        : 'Posterior TP53 mutation probability'}
                    </p>
                    <div className="mt-3 h-2 rounded-full bg-[#e8ecff] overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-[#3D53D8] to-[#2663eb]"
                        style={{ width: `${diagnosticoreProbabilityPct ?? 0}%` }}
                      />
                    </div>
                    <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                      <div className="rounded-md border border-[#d7e1ff] bg-[#f8faff] p-2">
                        <p className="text-[#5c6698]">Tiles Processed</p>
                        <p className="mt-0.5 font-bold text-[#11173d]">
                          {typeof diagnosticoreArtifact?.n_tiles === 'number'
                            ? diagnosticoreArtifact.n_tiles.toLocaleString()
                            : 'N/A'}
                        </p>
                      </div>
                      <div className="rounded-md border border-[#d7e1ff] bg-[#f8faff] p-2">
                        <p className="text-[#5c6698]">Source Split</p>
                        <p className="mt-0.5 font-bold text-[#11173d]">
                          {(diagnosticoreArtifact?.source_split || 'unknown').toUpperCase()}
                        </p>
                      </div>
                      <div className="rounded-md border border-[#d7e1ff] bg-[#f8faff] p-2">
                        <p className="text-[#5c6698]">Calibration</p>
                        <p className="mt-0.5 font-bold text-[#11173d]">
                          {(diagnosticoreArtifact?.calibration_method || 'unknown').toUpperCase()}
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 rounded-lg border border-[#d9e2ff] bg-[#f7f9ff] p-2.5">
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-[#5c6698]">Clinical Honesty Note</p>
                      <p className="mt-1 text-[11px] text-slate-700">
                        DiagnostiCore estimates TP53 mutation risk from morphology (WSI). This is AI-inferred decision
                        support and requires confirmatory molecular testing before irreversible care decisions.
                      </p>
                      <p className="mt-1 text-[11px] text-slate-600">
                        Cohort relation:{' '}
                        {diagnosticoreArtifact?.cohort_relation === 'same_cohort_external_split'
                          ? 'External split within TCGA-BRCA (not cross-cohort).'
                          : diagnosticoreArtifact?.cohort_relation === 'cross_cohort_external'
                          ? 'Cross-cohort external sample.'
                          : diagnosticoreArtifact?.cohort_relation?.replaceAll('_', ' ') || 'Not provided.'}
                      </p>
                    </div>
                    <div className="mt-3 rounded-lg border border-[#d9e2ff] bg-[#f7f9ff] p-2.5">
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-[#5c6698]">Future Work</p>
                      <p className="mt-1 text-[11px] text-slate-700">
                        Extend beyond TP53 to canonical cancer-specific driver mutation risk outputs (disease-aligned),
                        e.g., BRCA1/2 for breast, EGFR/KRAS for lung, BRAF/KRAS for colorectal, CTNNB1 for liver.
                      </p>
                      <p className="mt-1 text-[11px] text-slate-600">
                        Example gene sets would be sourced from guideline-aligned disease panels and common clinical NGS
                        panel conventions (for transparency and clinical relevance).
                      </p>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {FUTURE_DRIVER_OUTPUTS.map((item) => (
                          <button
                            key={`${item.gene}-${item.disease}`}
                            type="button"
                            disabled={!item.active}
                            title={
                              item.active
                                ? `${item.gene} is active in the current model output.`
                                : `${item.gene} (${item.disease}) is planned future work and is not available yet.`
                            }
                            className={`rounded-full border px-2 py-1 text-[10px] font-semibold ${
                              item.active
                                ? 'border-[#c7d3ff] bg-white text-[#24306f]'
                                : 'border-[#dde5ff] bg-[#edf1ff] text-[#6d7ab0] cursor-not-allowed'
                            }`}
                          >
                            {item.gene}
                          </button>
                        ))}
                      </div>
                    </div>
                  </article>

                  <article
                    data-tour-id="tour-diagno-metrics"
                    className={`rounded-xl border border-[#d8e0ff] p-4 ${
                      isTourTarget('tour-diagno-metrics') ? 'relative z-[85] ring-4 ring-[#2663eb]' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">Validation Metrics</p>
                      <BarChart3 className="h-4 w-4 text-[#2f4399]" />
                    </div>
                    <div className="mt-3 space-y-2.5">
                      {diagnosticoreMetrics.length > 0 ? (
                        diagnosticoreMetrics.map((metric) => (
                          <div key={metric.label}>
                            <div className="mb-1 flex items-center justify-between text-xs">
                              <span className="font-semibold text-[#24306f]">{metric.label}</span>
                              <span className="font-bold text-[#11173d]">{(metric.value * 100).toFixed(0)}%</span>
                            </div>
                            <div className="h-2 rounded-full bg-[#e8ecff] overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-[#2663eb] to-[#57D4F4]"
                                style={{ width: `${Math.round(metric.value * 100)}%` }}
                              />
                            </div>
                          </div>
                        ))
                      ) : (
                        <p className="text-xs text-slate-600">Validation metrics are unavailable for this handoff payload.</p>
                      )}
                    </div>
                    <div className="mt-4 rounded-lg border border-[#d9e2ff] bg-[#f7f9ff] p-2.5">
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-[#5c6698]">Source Provenance</p>
                      <p className="mt-1 text-[11px] text-slate-700">
                        Case key: {diagnosticoreArtifact?.case_submitter_id || 'N/A'}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-700">
                        Project: {diagnosticoreArtifact?.wsi_project_id || 'N/A'}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-700 break-all">
                        WSI file: {diagnosticoreArtifact?.wsi_file_name || 'N/A'}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-700 break-all">
                        Model card cohort: {diagnosticoreArtifact?.model_card?.cohort || 'N/A'}
                      </p>
                    </div>
                  </article>
                </div>
              </section>
            )}

            {workspaceTab === 'patients' && (
              <section className="rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 md:p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">Local SQLite History</p>
                    <h2 className="mt-1 text-base font-bold text-[#11173d]">Patient Case Snapshots</h2>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void fetchPatientCaseHistory(false)}
                      disabled={isLoadingPatientCaseHistory}
                      className="inline-flex items-center gap-1 rounded-lg border border-[#c7d3ff] bg-[#f7f9ff] px-3 py-1.5 text-xs font-semibold text-[#2f4399] disabled:opacity-60"
                    >
                      <RefreshCw className={`h-3.5 w-3.5 ${isLoadingPatientCaseHistory ? 'animate-spin' : ''}`} />
                      Refresh
                    </button>
                    <span className="inline-flex items-center gap-1 rounded-full border border-[#d4dcff] bg-[#f4f7ff] px-2 py-1 text-[11px] font-semibold text-[#2f4399]">
                      <Users className="h-3.5 w-3.5" />
                      {patientCaseHistory.length} saved
                    </span>
                  </div>
                </div>

                <p className="text-[11px] text-slate-600">
                  Showing all snapshots saved locally (errors hidden).
                </p>

                {patientCaseHistoryError && (
                  <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                    {patientCaseHistoryError}
                  </div>
                )}

                <div data-tour-id="tour-patients-list" className="overflow-x-auto rounded-xl border border-[#d8e0ff]">
                  <table className="min-w-full text-sm">
                    <thead className="bg-[#f5f8ff] text-[#33458f]">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-bold uppercase tracking-wide">Saved</th>
                        <th className="px-3 py-2 text-left text-xs font-bold uppercase tracking-wide">Case</th>
                        <th className="px-3 py-2 text-left text-xs font-bold uppercase tracking-wide">Diagnosis</th>
                        <th className="px-3 py-2 text-left text-xs font-bold uppercase tracking-wide">Status</th>
                        <th className="px-3 py-2 text-left text-xs font-bold uppercase tracking-wide">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#e2e8ff] bg-white">
                      {patientCaseHistory.length > 0 ? (
                        patientCaseHistory.map((snapshot) => (
                          <tr key={snapshot.snapshot_id} className="hover:bg-[#f9fbff]">
                            <td className="px-3 py-2 text-xs text-slate-700">
                              {formatHistoryTimestamp(snapshot.saved_at)}
                            </td>
                            <td className="px-3 py-2">
                              <p className="font-semibold text-[#11173d]">{snapshot.case_id}</p>
                              <p className="text-[11px] text-slate-500">#{snapshot.snapshot_id}</p>
                            </td>
                            <td className="px-3 py-2 text-xs text-slate-700">{snapshot.diagnosis}</td>
                            <td className="px-3 py-2 text-xs font-semibold text-[#2f4399]">
                              {snapshot.status.toUpperCase()}
                            </td>
                            <td className="px-3 py-2">
                              <div className="flex items-center gap-2">
                                <button
                                  type="button"
                                  onClick={() => void loadCaseHistorySnapshot(snapshot.snapshot_id)}
                                  disabled={loadingSnapshotId === snapshot.snapshot_id || deletingSnapshotIds.includes(snapshot.snapshot_id)}
                                  className="rounded-lg border border-[#c7d3ff] bg-[#f7f9ff] px-2.5 py-1.5 text-xs font-semibold text-[#2f4399] disabled:opacity-60"
                                >
                                  {loadingSnapshotId === snapshot.snapshot_id ? 'Loading...' : 'Open Snapshot'}
                                </button>
                                <button
                                  type="button"
                                  onClick={() => void deleteCaseHistorySnapshot(snapshot.snapshot_id)}
                                  disabled={deletingSnapshotIds.includes(snapshot.snapshot_id) || loadingSnapshotId === snapshot.snapshot_id}
                                  className="inline-flex items-center gap-1 rounded-lg border border-red-200 bg-red-50 px-2.5 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:opacity-60"
                                  title={`Delete snapshot #${snapshot.snapshot_id}`}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                  {deletingSnapshotIds.includes(snapshot.snapshot_id) ? 'Deleting...' : 'Delete'}
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={5} className="px-3 py-6 text-center text-xs text-slate-600">
                            No saved snapshots found yet for this patient in the selected local range.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {workspaceTab === 'calendar' && (
              <section className="rounded-2xl border border-[#ced8ff] bg-white shadow-sm p-4 md:p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">MDT Coordination</p>
                    <h2 className="mt-1 text-base font-bold text-[#11173d]">Calendar & Attendance</h2>
                  </div>
                  <span className="inline-flex items-center gap-1 rounded-full border border-[#d4dcff] bg-[#f4f7ff] px-2 py-1 text-[11px] font-semibold text-[#2f4399]">
                    {confirmedAttendeeCount}/{calendarAttendees.length} confirmed
                  </span>
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-4">
                  <article data-tour-id="tour-calendar-schedule" className="rounded-xl border border-[#d8e0ff] bg-[#f8faff] p-4 space-y-3">
                    <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">Schedule MDT Meeting</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <label className="block">
                        <span className="text-[11px] font-bold text-[#5c6698]">Date</span>
                        <input
                          type="date"
                          value={meetingDate}
                          onChange={(e) => setMeetingDate(e.target.value)}
                          className="mt-1.5 w-full rounded-lg border border-[#d7dffb] bg-white p-2 text-xs text-slate-700 focus:outline-none focus:border-[#2663eb]"
                        />
                      </label>
                      <label className="block">
                        <span className="text-[11px] font-bold text-[#5c6698]">Time</span>
                        <input
                          type="time"
                          value={meetingTime}
                          onChange={(e) => setMeetingTime(e.target.value)}
                          className="mt-1.5 w-full rounded-lg border border-[#d7dffb] bg-white p-2 text-xs text-slate-700 focus:outline-none focus:border-[#2663eb]"
                        />
                      </label>
                    </div>
                    <label className="block">
                      <span className="text-[11px] font-bold text-[#5c6698]">Location / Call Link</span>
                      <input
                        value={meetingLocation}
                        onChange={(e) => setMeetingLocation(e.target.value)}
                        className="mt-1.5 w-full rounded-lg border border-[#d7dffb] bg-white p-2 text-xs text-slate-700 focus:outline-none focus:border-[#2663eb]"
                      />
                    </label>
                    <label className="block">
                      <span className="text-[11px] font-bold text-[#5c6698]">Agenda</span>
                      <textarea
                        value={meetingAgenda}
                        onChange={(e) => setMeetingAgenda(e.target.value)}
                        className="mt-1.5 w-full h-24 rounded-lg border border-[#d7dffb] bg-white p-2 text-xs text-slate-700 focus:outline-none focus:border-[#2663eb]"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={saveMockCalendar}
                      className="rounded-lg bg-[#2663eb] px-3 py-2 text-xs font-semibold text-white"
                    >
                      Save Schedule (Mock)
                    </button>
                  </article>

                  <article className="rounded-xl border border-[#d8e0ff] p-4">
                    <p className="text-xs uppercase tracking-wide text-[#5c6698] font-semibold">Team Readiness</p>
                    <div className="mt-3 space-y-2.5">
                      {calendarAttendees.map((attendee) => (
                        <label
                          key={attendee.id}
                          className="flex items-center justify-between gap-2 rounded-lg border border-[#dbe3ff] bg-[#f8faff] px-3 py-2 cursor-pointer"
                        >
                          <div>
                            <p className="text-sm font-semibold text-[#11173d]">{attendee.name}</p>
                            <p className="text-xs text-slate-600">{attendee.role}</p>
                          </div>
                          <div className="flex items-center gap-2">
                            <span
                              className={`text-[11px] font-semibold ${
                                attendee.confirmed ? 'text-[#14823c]' : 'text-[#986f18]'
                              }`}
                            >
                              {attendee.confirmed ? 'Confirmed' : 'Pending'}
                            </span>
                            <input
                              type="checkbox"
                              checked={attendee.confirmed}
                              onChange={() => toggleCalendarAttendee(attendee.id)}
                              className="h-4 w-4 accent-[#2663eb]"
                            />
                          </div>
                        </label>
                      ))}
                    </div>
                    <div className="mt-3 rounded-lg border border-[#d9e2ff] bg-[#f7f9ff] p-2.5 text-[11px] text-slate-700">
                      Coordination status is currently mocked for workflow demo. Calendar provider integration comes in scale-up.
                    </div>
                  </article>
                </div>
              </section>
            )}
          </main>
        </div>
      </div>

      {typeof document !== 'undefined' &&
        createPortal(
          <>
            <button
              type="button"
              onClick={startTour}
              className="fixed left-5 bottom-5 z-[95] px-4 py-2 rounded-xl border-2 border-black bg-[#f2f6ff] text-[#11173d] text-sm font-semibold shadow-[4px_4px_0_#000]"
            >
              {tourSeen ? 'Retake Tour' : 'Show Tour'}
            </button>

            {tourOpen && (
              <>
                <div className="fixed inset-0 z-[70] bg-black/55" />
                {tourRect && (
                  <div
                    className="fixed z-[80] rounded-xl pointer-events-none"
                    style={{
                      top: tourRect.top - 6,
                      left: tourRect.left - 6,
                      width: tourRect.width + 12,
                      height: tourRect.height + 12,
                      border: '3px solid #2663eb',
                      boxShadow: '0 0 0 9999px rgba(0,0,0,0.35)',
                    }}
                  />
                )}
                <div
                  className="fixed z-[90] w-[380px] max-w-[calc(100vw-2rem)] rounded-2xl border-2 border-black bg-white shadow-[10px_10px_0_#000] overflow-hidden"
                  style={tourCardStyle}
                >
                  <div className="px-4 py-3 bg-[#2663eb] text-white">
                    <p className="text-[11px] uppercase tracking-[0.2em] font-bold">
                      Step {tourStep + 1} of {activeTourSteps.length}
                    </p>
                    <h3 className="text-xl font-black mt-0.5">{activeTourStep.title}</h3>
                  </div>
                  <div className="p-4">
                    <p className="text-base text-slate-800">{activeTourStep.body}</p>
                    <p className="mt-3 text-xs rounded-md bg-[#eef2ff] px-3 py-2 text-[#33458f] font-semibold">
                      {activeTourStep.hint}
                    </p>
                    <div className="mt-4 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                      <div
                        className="h-full bg-[#2663eb]"
                        style={{ width: `${((tourStep + 1) / activeTourSteps.length) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="px-4 py-3 border-t border-slate-200 flex items-center justify-between">
                    <button onClick={closeTour} className="text-sm font-semibold text-slate-500 hover:text-slate-700">
                      Skip Tour
                    </button>
                    <button
                      onClick={nextTourStep}
                      className="px-4 py-2 rounded-lg bg-[#2663eb] text-white text-sm font-semibold"
                    >
                      {tourStep === activeTourSteps.length - 1 ? 'Finish' : 'Next'}
                    </button>
                  </div>
                </div>
              </>
            )}
          </>,
          document.body
        )}
    </div>
  );
};
