// Browser scorer backed by YOUR fine-tuned kill detector (image-classification).
//
// Drop-in replacement for app/static/scorer.js once you've trained and pushed a
// model with the notebook. Unlike the generative-VLM scorer, this uses the
// `image-classification` pipeline, which Transformers.js fully supports — so it's
// both more reliable and (with a small fine-tuned model) much more accurate.
//
// To use it:
//   1. Copy this file over app/static/scorer.js
//   2. Run the app with:  -e VLM_MODEL_ID="your-username/montage-kill-detector"
//
// The model must have a label named "kill" (the notebook produces kill / no_kill).

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js";

const KILL_LABEL = "kill";

let _classifier = null;

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function getClassifier(modelId, onStatus) {
  if (_classifier) return _classifier;
  onStatus("Loading your kill detector in the browser (one-time download)…");
  _classifier = await pipeline("image-classification", modelId, {
    device: "webgpu",
    progress_callback: (p) => {
      if (p && p.status === "progress" && p.file) {
        onStatus(`Downloading model: ${p.file} (${Math.round(p.progress || 0)}%)`);
      }
    },
  });
  return _classifier;
}

function killScore(output) {
  // output: [{label, score}, ...]. Find the "kill" probability.
  const hit = (output || []).find(
    (o) => String(o.label).toLowerCase() === KILL_LABEL,
  );
  return hit ? hit.score : null;
}

/**
 * Score every frame in the manifest with the fine-tuned classifier.
 * @returns {Promise<Object<number, number>>} map of frame index -> kill probability 0..1
 */
export async function scoreJob(manifest, { onStatus = () => {} } = {}) {
  const classifier = await getClassifier(manifest.model_id, onStatus);
  const scores = {};
  const frames = manifest.frames || [];
  let yes = 0;

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const url = new URL(f.url, location.origin).href;
    onStatus(
      `Scoring moment ${i + 1} of ${frames.length} ` +
      `(t=${f.timestamp.toFixed(1)}s) — ${yes} kill${yes === 1 ? "" : "s"} so far`,
    );
    try {
      const out = await classifier(url, { top_k: 5 });
      const s = killScore(out);
      if (s !== null) {
        scores[f.index] = s;
        if (s >= 0.5) yes++;
      }
    } catch (err) {
      console.warn(`Frame scoring failed at index ${f.index}:`, err);
    }
  }
  onStatus(`Done — ${yes} kill moment${yes === 1 ? "" : "s"} found.`);
  return scores;
}
