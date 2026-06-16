// Browser-side highlight scoring with SmolVLM via Transformers.js + WebGPU.
//
// The server samples candidate frames and pauses; this module loads an
// open-weights vision-language model into the user's GPU, rates each segment's
// "epicness", and returns a {segmentIndex: 0..1} map for the server to re-rank.
//
// No cloud inference is involved — the model runs entirely on the user's hardware.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js";

let _generator = null;

const PROMPT =
  "Rate how action-packed or epic this gameplay moment is on a scale from 0 to 10. " +
  "Reply with only a single number.";

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function getGenerator(modelId, onStatus) {
  if (_generator) return _generator;
  _generator = await pipeline("image-text-to-text", modelId, {
    device: "webgpu",
    dtype: "q4",
    progress_callback: (p) => {
      if (p && p.status === "progress" && p.file) {
        onStatus(`Downloading model: ${p.file} (${Math.round(p.progress || 0)}%)`);
      }
    },
  });
  return _generator;
}

function parseScore(text) {
  const m = String(text).match(/(\d+(?:\.\d+)?)/);
  if (!m) return null;
  let v = parseFloat(m[1]);
  if (Number.isNaN(v)) return null;
  v = Math.max(0, Math.min(10, v));
  return v / 10;
}

function extractText(out) {
  try {
    const g = out[0].generated_text;
    if (typeof g === "string") return g;
    if (Array.isArray(g)) {
      const last = g[g.length - 1];
      if (typeof last.content === "string") return last.content;
      if (Array.isArray(last.content)) return last.content.map((c) => c.text || "").join(" ");
    }
  } catch (_) {
    /* fall through */
  }
  return "";
}

// Pick up to k representative frames (middle-out) to keep cost down.
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

/**
 * Score every segment in a frames manifest.
 * @returns {Promise<Object<number, number>>} map of segment index -> epicness (0..1)
 */
export async function scoreJob(manifest, { onStatus = () => {} } = {}) {
  const generator = await getGenerator(manifest.model_id, onStatus);
  const scores = {};
  const segments = manifest.segments || [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (!seg.frame_urls || seg.frame_urls.length === 0) continue;

    onStatus(`Scoring highlight ${i + 1} of ${segments.length}…`);
    let sum = 0;
    let count = 0;
    for (const url of pickFrames(seg.frame_urls, 2)) {
      try {
        const messages = [
          {
            role: "user",
            content: [
              { type: "image", image: new URL(url, location.origin).href },
              { type: "text", text: PROMPT },
            ],
          },
        ];
        const out = await generator(messages, { max_new_tokens: 12, do_sample: false });
        const s = parseScore(extractText(out));
        if (s !== null) {
          sum += s;
          count++;
        }
      } catch (err) {
        console.warn("Frame scoring failed", err);
      }
    }
    if (count > 0) scores[i] = sum / count;
  }
  return scores;
}
