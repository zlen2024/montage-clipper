// Browser-side highlight scoring with a real vision-language model (WebGPU).
//
// Unlike CLIP (which only measures image–text similarity), SmolVLM actually
// understands a screenshot and answers arbitrary questions about it. We ask it
// a single yes/no question per highlight — "is this active combat?" — and
// convert the answer to a 0..1 epicness score. The server then re-ranks
// candidates by combining this with the audio loudness.
//
// To keep it fast we only score ONE frame per audio-detected segment (the
// middle one, which is usually the climax) rather than every sampled frame.
//
// Runs entirely on the user's GPU. No cloud inference cost.

import {
  AutoProcessor,
  AutoModelForVision2Seq,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js";

const PROMPT =
  "Look at this video game screenshot. Is there active combat happening right " +
  "now — shooting, an explosion, a player aiming a gun at an enemy, a kill " +
  "feed notification, or someone being eliminated? Answer with only YES or NO.";

let _processor = null;
let _model = null;

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
      onStatus("Model ready — scoring highlights…");
    }
  };

  onStatus("Loading vision model in your browser (one-time download)…");
  _processor = await AutoProcessor.from_pretrained(modelId, { progress_callback });
  _model = await AutoModelForVision2Seq.from_pretrained(modelId, {
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

  const inputs = await processor(text, [image], { do_image_splitting: false });
  const generated_ids = await model.generate({
    ...inputs,
    max_new_tokens: 12,
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
  // Sometimes the model hedges — "There is..." / "It appears..." — fall back to
  // a substring check and return null if we genuinely can't tell.
  if (/\byes\b/.test(t)) return 0.7;
  if (/\bno\b/.test(t)) return 0.2;
  return null;
}

/**
 * Score every segment in a frames manifest by asking the VLM one yes/no
 * question about its middle frame.
 *
 * @returns {Promise<Object<number, number>>} map of segment index -> epicness 0..1
 */
export async function scoreJob(manifest, { onStatus = () => {} } = {}) {
  const scores = {};
  const segments = manifest.segments || [];

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (!seg.frame_urls || seg.frame_urls.length === 0) continue;

    // The middle frame is usually the climax — sufficient signal at ~1/Nth the cost.
    const mid = Math.floor(seg.frame_urls.length / 2);
    const url = new URL(seg.frame_urls[mid], location.origin).href;

    onStatus(`Asking the model about highlight ${i + 1} of ${segments.length}…`);
    try {
      const answer = await askYesNo(manifest.model_id, url, onStatus);
      const score = parseYesNo(answer);
      if (score !== null) {
        scores[i] = score;
        console.log(`segment ${i}: "${answer}" -> ${score}`);
      } else {
        console.log(`segment ${i}: unparseable answer "${answer}"`);
      }
    } catch (err) {
      console.warn(`Frame scoring failed for segment ${i}:`, err);
    }
  }
  return scores;
}
