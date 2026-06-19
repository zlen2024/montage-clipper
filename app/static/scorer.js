// Browser-side kill detection with a vision-language model (WebGPU).
//
// The server samples one frame every N seconds across the WHOLE video and
// pauses. This module loads a vision-language model (default LFM2.5-VL-450M)
// into the user's GPU and asks the model, for each frame, a single yes/no
// question: "is there active combat or a kill happening?" YES becomes a high
// epicness score, NO a low one. The server then clips a window around every YES.
//
// The model id comes from the server (VLM_MODEL_ID) and must be an ONNX /
// Transformers.js-compatible repo (i.e. one with an onnx/ folder).
//
// NOTE: LFM2.5-VL needs Transformers.js v4+ (v3 doesn't know its image
// processor). Use v4's SELF-CONTAINED browser bundle: dist/transformers.min.js
// (dist/transformers.web.js externalizes onnxruntime-web and needs a bundler).
//
// We score each frame independently. The audio loudness path is the Free tier
// only — the AI tier lets the model decide every moment.

import {
  AutoProcessor,
  AutoModelForImageTextToText,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js";

const PROMPT =
  "Look at this video game screenshot. Is there active combat happening right " +
  "now — shooting, an explosion, a player aiming a gun at an enemy, a kill " +
  "feed notification, or someone being eliminated? Answer with only YES or NO.";

let _processor = null;
let _model = null;
// Processors differ in argument order across model families: LFM2-VL is
// (images, text); SmolVLM is (text, images). Detect once, then reuse.
let _argOrder = null;

async function runProcessor(processor, text, image) {
  if (_argOrder === "text_img") return processor(text, [image]);
  if (_argOrder === "img_text") return processor([image], text);
  try {
    const out = await processor([image], text); // LFM2-VL order
    _argOrder = "img_text";
    return out;
  } catch (_) {
    const out = await processor(text, [image]); // SmolVLM order
    _argOrder = "text_img";
    return out;
  }
}

export function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function loadModel(modelId, onStatus) {
  if (_model) return { model: _model, processor: _processor };

  const progress_callback = (p) => {
    if (!p) return;
    if (p.status === "progress" && p.file) {
      onStatus(`Downloading model: ${p.file} (${Math.round(p.progress || 0)}%)`);
    } else if (p.status === "ready") {
      onStatus("Model ready — scoring video…");
    }
  };

  onStatus("Loading vision model in your browser (one-time download)…");
  _processor = await AutoProcessor.from_pretrained(modelId, { progress_callback });
  _model = await AutoModelForImageTextToText.from_pretrained(modelId, {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      decoder_model_merged: "q4",
    },
    device: "webgpu",
    progress_callback,
  });
  return { model: _model, processor: _processor };
}

async function askYesNo(modelId, imageUrl, onStatus) {
  const { model, processor } = await loadModel(modelId, onStatus);
  const image = await RawImage.read(imageUrl);

  const messages = [
    {
      role: "user",
      content: [{ type: "image" }, { type: "text", text: PROMPT }],
    },
  ];
  const text = processor.apply_chat_template(messages, {
    add_generation_prompt: true,
  });

  const inputs = await runProcessor(processor, text, image);
  const generated_ids = await model.generate({
    ...inputs,
    max_new_tokens: 5,   // we only need YES / NO
    do_sample: false,
  });
  const decoded = processor.batch_decode(
    generated_ids.slice(null, [inputs.input_ids.dims.at(-1), null]),
    { skip_special_tokens: true },
  );
  return (decoded[0] || "").trim();
}

function parseYesNo(text) {
  const t = String(text).toLowerCase().trim();
  if (t.startsWith("yes")) return 0.9;
  if (t.startsWith("no")) return 0.1;
  if (/\byes\b/.test(t)) return 0.7;
  if (/\bno\b/.test(t)) return 0.2;
  return null;
}

/**
 * Score every frame in the manifest. Each frame represents one timestamp in
 * the source video; the server decides what to do with the score map.
 *
 * @returns {Promise<Object<number, number>>} map of frame index -> epicness 0..1
 */
export async function scoreJob(manifest, { onStatus = () => {} } = {}) {
  const scores = {};
  const frames = manifest.frames || [];
  let yesCount = 0;

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const url = new URL(f.url, location.origin).href;
    onStatus(
      `Asking the model: moment ${i + 1} of ${frames.length} ` +
      `(t=${f.timestamp.toFixed(1)}s) — ${yesCount} kill${yesCount === 1 ? "" : "s"} so far`,
    );
    try {
      const answer = await askYesNo(manifest.model_id, url, onStatus);
      const score = parseYesNo(answer);
      if (score !== null) {
        scores[f.index] = score;
        if (score >= 0.5) yesCount++;
      }
    } catch (err) {
      console.warn(`Frame scoring failed at index ${f.index}:`, err);
    }
  }
  onStatus(`Done — ${yesCount} kill moment${yesCount === 1 ? "" : "s"} found.`);
  return scores;
}
