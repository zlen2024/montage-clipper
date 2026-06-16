// Browser-side highlight scoring via CLIP zero-shot image classification (WebGPU).
//
// The server samples candidate frames and pauses; this module loads a small CLIP
// model into the user's GPU and asks, for each frame, how much it looks like intense
// gameplay action vs. an idle/menu screen. The "action" probability is the epicness
// score (0..1) the server uses to re-rank.
//
// No cloud inference — the model runs entirely on the user's hardware. We use CLIP
// (zero-shot classification) rather than a generative VLM because Transformers.js
// supports it directly and it returns a clean probability with no text parsing.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js";

let _classifier = null;

// CLIP compares each frame against these text descriptions; index 0 = "epic".
const LABELS = [
  "intense video game action with a gunfight, explosion, or kill",
  "a calm, idle, empty, loading, or menu screen with no action",
];

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function getClassifier(modelId, onStatus) {
  if (_classifier) return _classifier;
  _classifier = await pipeline("zero-shot-image-classification", modelId, {
    device: "webgpu",
    progress_callback: (p) => {
      if (p && p.status === "progress" && p.file) {
        onStatus(`Downloading model: ${p.file} (${Math.round(p.progress || 0)}%)`);
      }
    },
  });
  return _classifier;
}

// Pick up to k representative frames (middle-out) to keep things fast.
function pickFrames(urls, k) {
  if (urls.length <= k) return urls.slice();
  const mid = Math.floor(urls.length / 2);
  const idx = new Set([mid]);
  let off = 1;
  while (idx.size < k) {
    if (mid - off >= 0) idx.add(mid - off);
    if (idx.size < k && mid + off < urls.length) idx.add(mid + off);
    off++;
  }
  return [...idx].sort((a, b) => a - b).map((i) => urls[i]);
}

function actionScore(output) {
  // output: [{label, score}, ...], scores sum ~1 across the candidate labels.
  const hit = (output || []).find((o) => o.label === LABELS[0]);
  return hit ? hit.score : null;
}

/**
 * Score every segment in a frames manifest.
 * A segment's epicness is the max "action" probability across its sampled frames
 * (a clip is epic if any moment in it is action-packed).
 * @returns {Promise<Object<number, number>>} map of segment index -> epicness (0..1)
 */
export async function scoreJob(manifest, { onStatus = () => {} } = {}) {
  const classifier = await getClassifier(manifest.model_id, onStatus);
  const scores = {};
  const segments = manifest.segments || [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (!seg.frame_urls || seg.frame_urls.length === 0) continue;

    onStatus(`Scoring highlight ${i + 1} of ${segments.length}…`);
    let best = null;
    for (const url of pickFrames(seg.frame_urls, 2)) {
      try {
        const out = await classifier(new URL(url, location.origin).href, LABELS);
        const s = actionScore(out);
        if (s !== null) best = best === null ? s : Math.max(best, s);
      } catch (err) {
        console.warn("Frame scoring failed", err);
      }
    }
    if (best !== null) scores[i] = best;
  }
  return scores;
}
