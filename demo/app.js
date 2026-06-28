"use strict";

// Served from the same origin as the API, so relative paths work.
const API_BASE = "";

const form = document.getElementById("generate-form");
const promptInput = document.getElementById("prompt");
const button = document.getElementById("generate-button");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");

const sliders = ["max_new_tokens", "temperature", "top_k", "top_p"];

// Keep each slider's live value label in sync.
for (const id of sliders) {
  const input = document.getElementById(id);
  const label = document.getElementById(`${id}_value`);
  const update = () => {
    label.textContent = input.value;
  };
  input.addEventListener("input", update);
  update();
}

function readParams() {
  return {
    prompt: promptInput.value,
    max_new_tokens: Number(document.getElementById("max_new_tokens").value),
    temperature: Number(document.getElementById("temperature").value),
    top_k: Number(document.getElementById("top_k").value),
    top_p: Number(document.getElementById("top_p").value),
  };
}

function setBusy(isBusy, message) {
  button.disabled = isBusy;
  statusEl.textContent = message || "";
  statusEl.classList.remove("error");
}

function showError(message) {
  statusEl.textContent = message;
  statusEl.classList.add("error");
}

// Parse a Server-Sent Events stream from a fetch ReadableStream and invoke
// onToken for each generated text delta.
async function streamGeneration(params, onToken) {
  const response = await fetch(`${API_BASE}/generate/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Request failed (${response.status}): ${detail}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const line = event.trim();
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice("data:".length).trim());
      if (payload.token) onToken(payload.token);
    }
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const params = readParams();
  outputEl.textContent = params.prompt;
  setBusy(true, "Generating...");

  const startedAt = performance.now();
  let tokenCount = 0;
  try {
    await streamGeneration(params, (token) => {
      outputEl.textContent += token;
      tokenCount += 1;
    });
    const seconds = (performance.now() - startedAt) / 1000;
    setBusy(false, `Done - ${tokenCount} chunks in ${seconds.toFixed(1)}s`);
  } catch (err) {
    setBusy(false, "");
    showError(err.message);
  }
});
