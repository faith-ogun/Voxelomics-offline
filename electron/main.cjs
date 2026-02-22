const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const BACKEND_HOST = "127.0.0.1";
const BACKEND_PORT = Number(process.env.MDT_BACKEND_PORT || "8084");
const OLLAMA_HOST = "127.0.0.1";
const OLLAMA_PORT = Number(process.env.OLLAMA_PORT || "11434");

let mainWindow = null;
let splashWindow = null;
let backendProcess = null;
let ollamaProcess = null;
let currentDataDir = null;
let appQuitting = false;
let forcedExecutionMode = null;
let cachedPythonBin = null;

function appIconPath() {
  const candidate = path.join(appRootDir(), "electron", "assets", "voxelomics.png");
  if (fs.existsSync(candidate)) {
    return candidate;
  }
  return null;
}

function applyAppIcon() {
  const iconPath = appIconPath();
  if (!iconPath) {
    return;
  }
  if (process.platform === "darwin" && app.dock && typeof app.dock.setIcon === "function") {
    app.dock.setIcon(iconPath);
  }
}

function appRootDir() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "app.asar");
  }
  return path.resolve(__dirname, "..");
}

function backendServiceDir() {
  if (app.isPackaged) {
    const unpackedPath = path.join(
      process.resourcesPath,
      "app.asar.unpacked",
      "backend",
      "mdt-command-service"
    );
    if (fs.existsSync(unpackedPath)) {
      return unpackedPath;
    }
  }
  return path.join(appRootDir(), "backend", "mdt-command-service");
}

function desktopSettingsPath() {
  return path.join(app.getPath("userData"), "desktop-settings.json");
}

function loadDesktopSettings() {
  const file = desktopSettingsPath();
  try {
    const raw = fs.readFileSync(file, "utf-8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveDesktopSettings(settings) {
  const file = desktopSettingsPath();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(settings, null, 2), "utf-8");
}

function defaultDataDir() {
  return path.join(app.getPath("documents"), "VoxelomicsOfflineData");
}

async function chooseDataDirectory(defaultPathValue) {
  const result = await dialog.showOpenDialog({
    title: "Choose Voxelomics data folder",
    defaultPath: defaultPathValue || defaultDataDir(),
    buttonLabel: "Use Folder",
    properties: ["openDirectory", "createDirectory", "promptToCreate"],
  });
  if (result.canceled || !result.filePaths.length) {
    return null;
  }
  return result.filePaths[0];
}

function getSavedDataDirectory() {
  const settings = loadDesktopSettings();
  const saved = typeof settings.dataDir === "string" ? settings.dataDir : "";
  if (saved) {
    fs.mkdirSync(saved, { recursive: true });
    return saved;
  }
  return null;
}

function resolveDataDirectory() {
  const saved = getSavedDataDirectory();
  if (saved) {
    return { dataDir: saved, hasPersistedChoice: true };
  }

  const dataDir = defaultDataDir();
  fs.mkdirSync(dataDir, { recursive: true });
  return { dataDir, hasPersistedChoice: false };
}

function pythonCommand() {
  if (cachedPythonBin) return cachedPythonBin;
  if (process.env.VOXELOMICS_PYTHON_BIN) {
    cachedPythonBin = process.env.VOXELOMICS_PYTHON_BIN;
    return cachedPythonBin;
  }

  const serviceDir = backendServiceDir();
  const candidates = [
    path.join(serviceDir, ".venv", "bin", "python"),
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    process.platform === "win32" ? "python" : "python3",
    "python",
  ];

  const deduped = [...new Set(candidates)];
  for (const candidate of deduped) {
    if (!candidate) continue;
    if (path.isAbsolute(candidate) && !fs.existsSync(candidate)) continue;
    const probe = spawnSync(
      candidate,
      [
        "-c",
        "import torch, transformers, accelerate; print('ok')",
      ],
      {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 15000,
      }
    );
    if (probe.status === 0) {
      cachedPythonBin = candidate;
      console.log(`[backend] using python runtime: ${candidate}`);
      return cachedPythonBin;
    }
  }

  cachedPythonBin = process.platform === "win32" ? "python" : "python3";
  console.warn(
    `[backend] no verified python runtime found with torch/transformers/accelerate; falling back to ${cachedPythonBin}`
  );
  return cachedPythonBin;
}

function ollamaCommand() {
  if (process.env.VOXELOMICS_OLLAMA_BIN) return process.env.VOXELOMICS_OLLAMA_BIN;
  return "ollama";
}

function configuredExecutionMode() {
  return (
    forcedExecutionMode ||
    process.env.MDT_EXECUTION_MODE ||
    "local"
  )
    .trim()
    .toLowerCase();
}

function shouldUseAdkLocalMode() {
  return configuredExecutionMode() === "adk_local";
}

function findModelDirByWalkingUp(startDir, modelFolderName, maxDepth = 8) {
  if (!startDir || !modelFolderName) return null;
  let current = path.resolve(startDir);
  for (let i = 0; i < maxDepth; i += 1) {
    const candidate = path.join(current, "models", modelFolderName);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return null;
}

function resolveLocalModelRef(rawRef, dataDir, defaultFolderName) {
  const trimmed = (rawRef || "").trim();
  const tryPaths = [];

  if (trimmed) {
    if (path.isAbsolute(trimmed)) {
      tryPaths.push(trimmed);
    } else {
      tryPaths.push(path.resolve(backendServiceDir(), trimmed));
      tryPaths.push(path.resolve(appRootDir(), trimmed));
      tryPaths.push(path.resolve(process.cwd(), trimmed));
      if (dataDir) {
        tryPaths.push(path.resolve(dataDir, trimmed));
      }
    }
  }

  if (dataDir && defaultFolderName) {
    tryPaths.push(path.join(dataDir, "models", defaultFolderName));
  }
  if (defaultFolderName) {
    tryPaths.push(path.join(appRootDir(), "models", defaultFolderName));
    tryPaths.push(path.join(process.cwd(), "models", defaultFolderName));
  }

  for (const p of tryPaths) {
    if (p && fs.existsSync(p)) {
      return path.resolve(p);
    }
  }

  if (defaultFolderName) {
    const walked =
      findModelDirByWalkingUp(backendServiceDir(), defaultFolderName) ||
      findModelDirByWalkingUp(appRootDir(), defaultFolderName) ||
      findModelDirByWalkingUp(process.cwd(), defaultFolderName);
    if (walked) {
      return walked;
    }
  }

  return trimmed;
}

function backendEnv(dataDir) {
  const env = { ...process.env };
  const audioDir = path.join(dataDir, "audio");
  const sqlitePath = path.join(dataDir, "mdt_cases.sqlite3");
  const evidenceDir = path.join(dataDir, "evidence_cache");
  const diagnosticoreRoot = path.join(path.dirname(backendServiceDir()), "diagnosticore-service");

  fs.mkdirSync(audioDir, { recursive: true });
  fs.mkdirSync(evidenceDir, { recursive: true });

  env.MDT_EXECUTION_MODE = configuredExecutionMode();
  env.MDT_CASE_STORE_BACKEND = env.MDT_CASE_STORE_BACKEND || "sqlite";
  env.MDT_SQLITE_DB_PATH = env.MDT_SQLITE_DB_PATH || sqlitePath;
  env.MDT_LOCAL_DATA_DIR = env.MDT_LOCAL_DATA_DIR || dataDir;
  env.MDT_AUDIO_UPLOAD_BACKEND = env.MDT_AUDIO_UPLOAD_BACKEND || "local";
  env.MDT_LOCAL_AUDIO_DIR = env.MDT_LOCAL_AUDIO_DIR || audioDir;
  env.MDT_RETRIEVAL_MODE = env.MDT_RETRIEVAL_MODE || "local";
  env.MDT_LOCAL_EVIDENCE_DIR = env.MDT_LOCAL_EVIDENCE_DIR || evidenceDir;
  env.MDT_DIAGNOSTICORE_FETCH_MODE = env.MDT_DIAGNOSTICORE_FETCH_MODE || "file";
  env.MDT_DIAGNOSTICORE_ALLOW_FALLBACK = env.MDT_DIAGNOSTICORE_ALLOW_FALLBACK || "true";
  env.MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV =
    env.MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV ||
    path.join(
      diagnosticoreRoot,
      "output",
      "pathfoundation_tp53_200",
      "case_predictions_calibrated_platt.csv"
    );
  env.MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON =
    env.MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON ||
    path.join(
      diagnosticoreRoot,
      "output",
      "pathfoundation_tp53_200",
      "tp53_clinical_report_pathfoundation_platt.json"
    );
  env.MDT_DIAGNOSTICORE_WSI_METADATA_CSV =
    env.MDT_DIAGNOSTICORE_WSI_METADATA_CSV ||
    path.join(diagnosticoreRoot, "output", "tcga_brca_tp53_wsi_primary_slide.csv");
  env.MDT_DIAGNOSTICORE_WSI_DOWNLOAD_DIR =
    env.MDT_DIAGNOSTICORE_WSI_DOWNLOAD_DIR || path.join(diagnosticoreRoot, "gdc_wsi");
  env.MDT_DIAGNOSTICORE_DEEPZOOM_DIR =
    env.MDT_DIAGNOSTICORE_DEEPZOOM_DIR || path.join(diagnosticoreRoot, "output", "deepzoom");
  env.MDT_DIAGNOSTICORE_TILE_MANIFEST_CSVS =
    env.MDT_DIAGNOSTICORE_TILE_MANIFEST_CSVS ||
    [
      path.join(diagnosticoreRoot, "output", "tcga_brca_tp53_tiles_manifest_external20.csv"),
      path.join(diagnosticoreRoot, "output", "tcga_brca_tp53_tiles_manifest_full_200.csv"),
    ].join(",");
  env.MDT_ADK_MODEL_PROVIDER = env.MDT_ADK_MODEL_PROVIDER || "ollama_chat";
  env.MDT_ADK_MODEL_NAME = env.MDT_ADK_MODEL_NAME || "qwen2.5:7b-instruct";

  const medgemmaRef = resolveLocalModelRef(
    env.MDT_MEDGEMMA_LOCAL_MODEL_ID || env.MDT_MEDGEMMA_MODEL_NAME || "../../models/medgemma-4b-it",
    dataDir,
    "medgemma-4b-it"
  );
  const medasrRef = resolveLocalModelRef(
    env.MDT_MEDASR_LOCAL_MODEL_ID || "../../models/medasr",
    dataDir,
    "medasr"
  );

  if (medgemmaRef) {
    env.MDT_MEDGEMMA_LOCAL_MODEL_ID = medgemmaRef;
    env.MDT_MEDGEMMA_MODEL_NAME = env.MDT_MEDGEMMA_MODEL_NAME || medgemmaRef;
  }
  if (medasrRef) {
    env.MDT_MEDASR_LOCAL_MODEL_ID = medasrRef;
  }
  env.MDT_MEDASR_LOCAL_ALLOW_TEXT_FALLBACK =
    env.MDT_MEDASR_LOCAL_ALLOW_TEXT_FALLBACK || "true";

  if (env.MDT_MEDGEMMA_LOCAL_MODEL_ID && !fs.existsSync(env.MDT_MEDGEMMA_LOCAL_MODEL_ID)) {
    console.warn(
      `[backend] MedGemma local path not found: ${env.MDT_MEDGEMMA_LOCAL_MODEL_ID}`
    );
  } else {
    console.log(`[backend] MedGemma local path: ${env.MDT_MEDGEMMA_LOCAL_MODEL_ID}`);
  }
  if (env.MDT_MEDASR_LOCAL_MODEL_ID && !fs.existsSync(env.MDT_MEDASR_LOCAL_MODEL_ID)) {
    console.warn(`[backend] MedASR local path not found: ${env.MDT_MEDASR_LOCAL_MODEL_ID}`);
  } else {
    console.log(`[backend] MedASR local path: ${env.MDT_MEDASR_LOCAL_MODEL_ID}`);
  }
  if (!fs.existsSync(env.MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV)) {
    console.warn(
      `[backend] DiagnostiCore case predictions not found: ${env.MDT_DIAGNOSTICORE_CASE_PREDICTIONS_CSV}`
    );
  }
  if (!fs.existsSync(env.MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON)) {
    console.warn(
      `[backend] DiagnostiCore clinical report not found: ${env.MDT_DIAGNOSTICORE_CLINICAL_REPORT_JSON}`
    );
  }
  if (!fs.existsSync(env.MDT_DIAGNOSTICORE_DEEPZOOM_DIR)) {
    console.warn(
      `[backend] DiagnostiCore DeepZoom dir not found: ${env.MDT_DIAGNOSTICORE_DEEPZOOM_DIR}`
    );
  }

  env.PYTHONUNBUFFERED = "1";

  return env;
}

function backendUrl() {
  return `http://${BACKEND_HOST}:${BACKEND_PORT}`;
}

function ollamaUrl() {
  return `http://${OLLAMA_HOST}:${OLLAMA_PORT}`;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForBackendReady(timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  const url = `${backendUrl()}/health`;

  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      // continue polling
    }
    await delay(1000);
  }
  return false;
}

async function waitForOllamaReady(timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  const url = `${ollamaUrl()}/api/tags`;

  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      // continue polling
    }
    await delay(1000);
  }
  return false;
}

function attachBackendLogs(child) {
  if (!child) return;
  if (child.stdout) {
    child.stdout.on("data", (buf) => {
      process.stdout.write(`[backend] ${buf}`);
    });
  }
  if (child.stderr) {
    child.stderr.on("data", (buf) => {
      process.stderr.write(`[backend] ${buf}`);
    });
  }
}

function attachOllamaLogs(child) {
  if (!child) return;
  if (child.stdout) {
    child.stdout.on("data", (buf) => {
      process.stdout.write(`[ollama] ${buf}`);
    });
  }
  if (child.stderr) {
    child.stderr.on("data", (buf) => {
      process.stderr.write(`[ollama] ${buf}`);
    });
  }
}

async function ensureOllamaReadyIfNeeded() {
  if (!shouldUseAdkLocalMode()) {
    return;
  }

  const alreadyReady = await waitForOllamaReady(1500);
  if (alreadyReady) {
    return;
  }

  const child = spawn(ollamaCommand(), ["serve"], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  ollamaProcess = child;
  attachOllamaLogs(child);

  child.once("error", (err) => {
    console.error(`[ollama] failed to start: ${err.message}`);
  });
  child.on("exit", (code, signal) => {
    if (appQuitting) return;
    console.error(`[ollama] exited unexpectedly (code=${code}, signal=${signal})`);
  });

  const ready = await waitForOllamaReady(60000);
  if (!ready) {
    throw new Error(
      "Ollama did not become ready. For ADK local mode, install/start Ollama and pull the configured model."
    );
  }
}

function startBackend(dataDir) {
  return new Promise((resolve, reject) => {
    const cwd = backendServiceDir();
    if (!fs.existsSync(cwd)) {
      reject(new Error(`Backend service directory not found: ${cwd}`));
      return;
    }

    const args = [
      "-m",
      "uvicorn",
      "main:app",
      "--host",
      BACKEND_HOST,
      "--port",
      String(BACKEND_PORT),
    ];

    const pythonBin = pythonCommand();
    const child = spawn(pythonBin, args, {
      cwd,
      env: backendEnv(dataDir),
      stdio: ["ignore", "pipe", "pipe"],
    });
    backendProcess = child;
    attachBackendLogs(child);

    child.once("error", (err) => {
      reject(
        new Error(
          `Failed to start backend process with '${pythonBin}'. ` +
            `Set VOXELOMICS_PYTHON_BIN if needed. Root cause: ${err.message}`
        )
      );
    });
    child.once("spawn", () => {
      resolve();
    });

    child.on("exit", (code, signal) => {
      if (appQuitting) return;
      console.error(`[backend] exited unexpectedly (code=${code}, signal=${signal})`);
    });
  });
}

function stopBackend() {
  if (!backendProcess || backendProcess.killed) return;
  const pid = backendProcess.pid;
  if (!pid) return;

  if (process.platform === "win32") {
    spawn("taskkill", ["/PID", String(pid), "/T", "/F"]);
  } else {
    backendProcess.kill("SIGTERM");
  }
}

function stopOllamaIfManaged() {
  if (!ollamaProcess || ollamaProcess.killed) return;
  const pid = ollamaProcess.pid;
  if (!pid) return;

  if (process.platform === "win32") {
    spawn("taskkill", ["/PID", String(pid), "/T", "/F"]);
  } else {
    ollamaProcess.kill("SIGTERM");
  }
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 520,
    height: 320,
    frame: false,
    transparent: false,
    alwaysOnTop: false,
    resizable: false,
    movable: true,
    show: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
    },
  });
  splashWindow.loadFile(path.join(__dirname, "splash.html"));
}

function createMainWindow() {
  const iconPath = appIconPath() || undefined;
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1080,
    minHeight: 720,
    show: false,
    backgroundColor: "#f5f7fb",
    icon: iconPath,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  mainWindow.webContents.on(
    "did-fail-load",
    (_event, errorCode, errorDescription, validatedURL) => {
      console.error(
        `[renderer] did-fail-load code=${errorCode} desc=${errorDescription} url=${validatedURL}`
      );
    }
  );

  mainWindow.webContents.on("console-message", (_event, level, message) => {
    process.stdout.write(`[renderer:${level}] ${message}\n`);
  });
}

async function loadRenderer() {
  const explicitUrl = process.env.VOXELOMICS_RENDERER_URL;
  if (explicitUrl) {
    await mainWindow.loadURL(explicitUrl);
    return;
  }

  if (!app.isPackaged) {
    const devUrl = "http://127.0.0.1:5173";
    try {
      const res = await fetch(devUrl);
      if (res.ok) {
        await mainWindow.loadURL(devUrl);
        return;
      }
    } catch {
      // fall through to built dist
    }

    const builtIndex = path.join(appRootDir(), "dist", "index.html");
    if (fs.existsSync(builtIndex)) {
      await mainWindow.loadFile(builtIndex);
      return;
    }

    throw new Error(
      "Renderer not available. Start Vite on http://127.0.0.1:5173 or run `npm run build` to generate dist/index.html."
    );
  }

  if (app.isPackaged) {
    await mainWindow.loadFile(path.join(appRootDir(), "dist", "index.html"));
    return;
  }
}

async function promptForDataDirectoryAfterLaunch() {
  const settings = loadDesktopSettings();
  if (settings.initialDataDirPromptCompleted) {
    return;
  }

  const selected = await chooseDataDirectory(currentDataDir || defaultDataDir());
  const nextDataDir = selected || currentDataDir || defaultDataDir();

  fs.mkdirSync(nextDataDir, { recursive: true });
  const changed = nextDataDir !== currentDataDir;
  currentDataDir = nextDataDir;
  saveDesktopSettings({
    ...settings,
    dataDir: nextDataDir,
    initialDataDirPromptCompleted: true,
  });

  if (!changed) {
    return;
  }

  stopBackend();
  await delay(600);
  await startBackend(nextDataDir);
  const ready = await waitForBackendReady(45000);
  if (!ready) {
    dialog.showErrorBox(
      "Backend Restart Failed",
      "The backend did not become healthy after changing the data folder."
    );
  }
}

async function bootstrapDesktop() {
  applyAppIcon();
  createSplashWindow();
  createMainWindow();

  const { dataDir, hasPersistedChoice } = resolveDataDirectory();
  currentDataDir = dataDir;

  try {
    await ensureOllamaReadyIfNeeded();
    await startBackend(currentDataDir);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    dialog.showErrorBox("Backend Startup Failed", message);
  }

  const backendReady = await waitForBackendReady();
  if (!backendReady) {
    if (shouldUseAdkLocalMode()) {
      forcedExecutionMode = "local";
      stopBackend();
      await delay(600);
      try {
        await startBackend(currentDataDir);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        dialog.showErrorBox("Backend Startup Failed", message);
      }
      const fallbackReady = await waitForBackendReady(45000);
      if (!fallbackReady) {
        dialog.showErrorBox(
          "Backend Health Check Failed",
          "Voxelomics backend did not become healthy in adk_local or fallback local mode."
        );
      } else {
        dialog.showMessageBox({
          type: "warning",
          title: "Fallback Mode Activated",
          message:
            "ADK local mode did not become healthy, so Voxelomics switched to local orchestrator mode.",
        });
      }
    } else {
      dialog.showErrorBox(
        "Backend Health Check Failed",
        "Voxelomics backend did not become healthy. Verify Python dependencies in backend/mdt-command-service."
      );
    }
  }

  let mainWindowShown = false;
  const revealMainWindow = () => {
    if (mainWindowShown || !mainWindow || mainWindow.isDestroyed()) return;
    mainWindowShown = true;
    if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close();
    mainWindow.show();
    if (!hasPersistedChoice) {
      setTimeout(() => {
        void promptForDataDirectoryAfterLaunch();
      }, 900);
    }
  };

  // Attach listeners before loading renderer to avoid missing early events.
  mainWindow.once("ready-to-show", revealMainWindow);
  mainWindow.webContents.once("did-finish-load", () => {
    // Fallback if ready-to-show does not fire.
    setTimeout(revealMainWindow, 100);
  });

  await loadRenderer();

  // Final safety net in case neither event fires due renderer issues.
  setTimeout(revealMainWindow, 1500);
}

ipcMain.handle("desktop:get-app-info", async () => {
  return {
    backendUrl: backendUrl(),
    ollamaUrl: ollamaUrl(),
    dataDir: currentDataDir,
    executionMode: configuredExecutionMode(),
    packaged: app.isPackaged,
    appVersion: app.getVersion(),
  };
});

ipcMain.handle("desktop:choose-data-directory", async () => {
  const selected = await chooseDataDirectory(currentDataDir || defaultDataDir());
  if (!selected) {
    return { changed: false, dataDir: currentDataDir };
  }

  fs.mkdirSync(selected, { recursive: true });
  currentDataDir = selected;
  saveDesktopSettings({
    ...loadDesktopSettings(),
    dataDir: selected,
    initialDataDirPromptCompleted: true,
  });

  stopBackend();
  await delay(600);
  await startBackend(selected);
  const ready = await waitForBackendReady(45000);
  return { changed: true, dataDir: selected, backendReady: ready };
});

ipcMain.handle("desktop:save-report", async (_event, payload) => {
  const format = payload?.format === "pdf" ? "pdf" : "txt";
  const defaultFileNameRaw =
    typeof payload?.defaultFileName === "string" && payload.defaultFileName.trim()
      ? payload.defaultFileName.trim()
      : `MDT-Report.${format}`;
  const safeDefaultFileName = defaultFileNameRaw.replace(/[\\/:"*?<>|]+/g, "_");
  const defaultDir = app.getPath("downloads");
  const defaultPath = path.join(defaultDir, safeDefaultFileName);

  const saveResult = await dialog.showSaveDialog(mainWindow || null, {
    title: `Save ${format.toUpperCase()} Report`,
    defaultPath,
    filters:
      format === "pdf"
        ? [{ name: "PDF", extensions: ["pdf"] }]
        : [{ name: "Text", extensions: ["txt"] }],
  });

  if (saveResult.canceled || !saveResult.filePath) {
    return { success: false, canceled: true };
  }

  const filePath = saveResult.filePath;
  if (format === "txt") {
    const textContent = typeof payload?.textContent === "string" ? payload.textContent : "";
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, textContent, "utf-8");
    return { success: true, canceled: false, path: filePath };
  }

  const bytes = Array.isArray(payload?.binaryBytes) ? payload.binaryBytes : null;
  if (!bytes) {
    throw new Error("Missing binary bytes for PDF save.");
  }
  const buffer = Buffer.from(bytes);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, buffer);
  return { success: true, canceled: false, path: filePath };
});

app.on("before-quit", () => {
  appQuitting = true;
  stopBackend();
  stopOllamaIfManaged();
});

app.whenReady().then(async () => {
  await bootstrapDesktop();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      bootstrapDesktop();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
