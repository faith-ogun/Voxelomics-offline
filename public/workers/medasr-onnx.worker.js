/* eslint-disable no-restricted-globals */
let sessionPromise = null;
let vocabPromise = null;
let configKey = "";

const FALLBACK_ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";

const toErrorString = (value) => {
  if (!value) return "Unknown error";
  if (value instanceof Error) return value.message || String(value);
  return String(value);
};

const collapseWhitespace = (text) => text.replace(/\s+/g, " ").trim();

const normalizeTokenText = (token) => {
  if (!token) return "";
  let out = String(token);
  if (
    out === "<epsilon>" ||
    out === "<s>" ||
    out === "</s>" ||
    out === "<unk>" ||
    out === "<pad>"
  ) {
    return "";
  }
  out = out.replace(/<[^>]+>/g, "");
  out = out.replace(/▁/g, " ");
  return out;
};

const decodeCtcGreedy = (logitsData, dims, idToToken) => {
  let timeSteps = 0;
  let vocabSize = 0;
  let baseOffset = 0;

  if (dims.length === 3) {
    // [batch, time, vocab]
    timeSteps = dims[1];
    vocabSize = dims[2];
    baseOffset = 0;
  } else if (dims.length === 2) {
    // [time, vocab]
    timeSteps = dims[0];
    vocabSize = dims[1];
    baseOffset = 0;
  } else {
    throw new Error(`Unexpected logits rank: ${JSON.stringify(dims)}`);
  }

  let previous = -1;
  const tokenPieces = [];
  const blankId = 0;

  for (let t = 0; t < timeSteps; t += 1) {
    const rowOffset = baseOffset + t * vocabSize;
    let bestId = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let v = 0; v < vocabSize; v += 1) {
      const score = logitsData[rowOffset + v];
      if (score > bestScore) {
        bestScore = score;
        bestId = v;
      }
    }
    if (bestId === blankId) {
      previous = -1;
      continue;
    }
    if (bestId === previous) continue;
    previous = bestId;
    tokenPieces.push(normalizeTokenText(idToToken[bestId] || ""));
  }

  return collapseWhitespace(tokenPieces.join(""));
};

const parseVocabPayload = (payload) => {
  if (!payload) return null;
  if (Array.isArray(payload)) {
    if (payload.length && Array.isArray(payload[0])) {
      return payload.map((entry) => String(entry[0] ?? ""));
    }
    return payload.map((entry) => String(entry ?? ""));
  }
  if (Array.isArray(payload.id_to_token)) {
    return payload.id_to_token.map((entry) => String(entry ?? ""));
  }
  if (payload.model && Array.isArray(payload.model.vocab)) {
    return payload.model.vocab.map((entry) =>
      Array.isArray(entry) ? String(entry[0] ?? "") : String(entry ?? "")
    );
  }
  return null;
};

const ensureOrtRuntime = async (ortUrl) => {
  if (self.ort && self.ort.InferenceSession) return self.ort;

  const candidates = [ortUrl, FALLBACK_ORT_CDN].filter(Boolean);
  let loaded = false;
  const errors = [];
  for (const candidate of candidates) {
    try {
      importScripts(candidate);
      if (self.ort && self.ort.InferenceSession) {
        loaded = true;
        if (self.ort?.env?.wasm) {
          const slash = candidate.lastIndexOf("/");
          if (slash > 0) self.ort.env.wasm.wasmPaths = candidate.slice(0, slash + 1);
          // Electron file:// environments are more stable with non-threaded wasm init.
          self.ort.env.wasm.numThreads = 1;
          self.ort.env.wasm.proxy = false;
        }
        break;
      }
    } catch (err) {
      errors.push(`${candidate}: ${toErrorString(err)}`);
    }
  }

  if (!loaded) {
    throw new Error(
      `Failed to load onnxruntime-web. Tried local path '${ortUrl}' and CDN fallback. ` +
        `Provide local ort assets under /public/vendor/onnxruntime-web. ` +
        `${errors.join(" | ")}`
    );
  }
  return self.ort;
};

const ensureRuntime = async (modelUrl, vocabUrl, ortUrl, executionProviders) => {
  const providerKey = Array.isArray(executionProviders) ? executionProviders.join(",") : "";
  const newKey = `${modelUrl}::${vocabUrl}::${ortUrl}::${providerKey}`;
  if (newKey !== configKey) {
    configKey = newKey;
    sessionPromise = null;
    vocabPromise = null;
  }

  const ort = await ensureOrtRuntime(ortUrl);

  if (!sessionPromise) {
    sessionPromise = (async () => {
      const modelResp = await fetch(modelUrl);
      if (!modelResp.ok) {
        throw new Error(`Failed to load ONNX model: ${modelUrl} (${modelResp.status})`);
      }
      const modelBytes = new Uint8Array(await modelResp.arrayBuffer());
      const attempts = [];
      if (Array.isArray(executionProviders) && executionProviders.length) {
        attempts.push(executionProviders);
      }
      if (!attempts.some((value) => JSON.stringify(value) === JSON.stringify(["webgpu"]))) {
        attempts.push(["webgpu"]);
      }
      if (!attempts.some((value) => JSON.stringify(value) === JSON.stringify(["wasm"]))) {
        attempts.push(["wasm"]);
      }
      let lastErr = null;
      for (const providers of attempts) {
        try {
          return await ort.InferenceSession.create(modelBytes, {
            executionProviders: providers,
            graphOptimizationLevel: "all",
          });
        } catch (err) {
          lastErr = err;
        }
      }
      throw new Error(`Failed to initialize ONNX session. ${toErrorString(lastErr)}`);
    })();
  }

  if (!vocabPromise) {
    vocabPromise = (async () => {
      const res = await fetch(vocabUrl);
      if (!res.ok) {
        throw new Error(`Failed to load vocab JSON: ${vocabUrl} (${res.status})`);
      }
      const payload = await res.json();
      const parsed = parseVocabPayload(payload);
      if (!parsed || !parsed.length) {
        throw new Error("Invalid vocab JSON: expected id_to_token mapping.");
      }
      return parsed;
    })();
  }

  const [session, idToToken] = await Promise.all([sessionPromise, vocabPromise]);
  return { ort, session, idToToken };
};

const runSingleChunk = async (ort, session, idToToken, audioPcm) => {
  const inputValues = new ort.Tensor("float32", audioPcm, [1, audioPcm.length]);
  const inputLengths = new ort.Tensor(
    "int64",
    new BigInt64Array([BigInt(audioPcm.length)]),
    [1]
  );
  const outputs = await session.run({
    input_values: inputValues,
    input_lengths: inputLengths,
  });
  const logitsTensor = outputs.logits || outputs[Object.keys(outputs)[0]];
  if (!logitsTensor) {
    throw new Error("ONNX inference returned no logits tensor.");
  }
  return decodeCtcGreedy(logitsTensor.data, logitsTensor.dims, idToToken);
};

self.onmessage = async (event) => {
  const payload = event.data || {};
  const id = payload.id;
  if (payload.type !== "transcribe") return;

  try {
    const modelUrl = payload.model_url || "/models/medasr.onnx";
    const vocabUrl = payload.vocab_url || "/models/medasr_vocab.json";
    const ortUrl = payload.ort_url || "./vendor/onnxruntime-web/ort.min.js";
    const executionProviders =
      Array.isArray(payload.execution_providers) && payload.execution_providers.length
        ? payload.execution_providers
        : ["webgpu", "wasm"];
    const sampleRate = Number(payload.sample_rate || 16000);
    if (sampleRate !== 16000) {
      throw new Error(`Unsupported sample rate ${sampleRate}. Expected 16000 Hz.`);
    }

    const audioPcm =
      payload.audio_pcm instanceof Float32Array
        ? payload.audio_pcm
        : new Float32Array(payload.audio_pcm || []);
    if (!audioPcm.length) {
      throw new Error("Received empty PCM audio.");
    }

    const { ort, session, idToToken } = await ensureRuntime(
      modelUrl,
      vocabUrl,
      ortUrl,
      executionProviders
    );
    const chunkSeconds = Math.max(4, Number(payload.chunk_seconds || 10));
    const overlapSeconds = Math.max(0, Math.min(3, Number(payload.overlap_seconds || 1)));
    const chunkSamples = Math.max(sampleRate, Math.floor(chunkSeconds * sampleRate));
    const overlapSamples = Math.floor(overlapSeconds * sampleRate);
    const stepSamples = Math.max(1, chunkSamples - overlapSamples);

    const pieces = [];
    if (audioPcm.length <= chunkSamples) {
      pieces.push(await runSingleChunk(ort, session, idToToken, audioPcm));
    } else {
      for (let start = 0; start < audioPcm.length; start += stepSamples) {
        const end = Math.min(start + chunkSamples, audioPcm.length);
        const chunk = audioPcm.subarray(start, end);
        const text = await runSingleChunk(ort, session, idToToken, chunk);
        if (text) pieces.push(text);
        if (end >= audioPcm.length) break;
      }
    }

    const transcript = collapseWhitespace(pieces.join(" "));
    if (!transcript) {
      throw new Error("Decoded transcript was empty.");
    }

    self.postMessage({
      type: "transcribe:ok",
      id,
      transcript,
      engine: "medasr-onnx-webgpu",
    });
  } catch (err) {
    self.postMessage({
      type: "transcribe:error",
      id,
      error: toErrorString(err),
    });
  }
};
