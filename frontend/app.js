/* ── Config ─────────────────────────────────────────────────────── */
const API = "";          // same origin; change to "http://localhost:8000" for dev
const CELL = 28;         // attention heatmap cell size in px

/* ── State ──────────────────────────────────────────────────────── */
let attnData = null;     // last /attention response
let lossChart = null;
let pplChart  = null;
let confChart = null;

/* ── Helpers ─────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

function setLoading(btn, loading) {
  if (loading) {
    btn.disabled = true;
    btn.dataset.orig = btn.textContent;
    btn.innerHTML = '<span class="spinner"></span>Working…';
  } else {
    btn.disabled = false;
    btn.textContent = btn.dataset.orig || btn.textContent;
  }
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

/* ── Tab switching ───────────────────────────────────────────────── */
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");

    // Lazy-load data when tab becomes visible
    if (btn.dataset.tab === "training") loadTrainingLogs();
    if (btn.dataset.tab === "info")     loadModelInfo();
  });
});

/* ── Model switcher ──────────────────────────────────────────────── */
document.querySelectorAll(".switch-btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    const key = btn.dataset.model;
    const badge = $("model-badge");
    badge.textContent = "Switching…";
    badge.className = "badge badge-loading";
    try {
      await apiFetch("/switch-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: key }),
      });
      document.querySelectorAll(".switch-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      await checkModel();
    } catch (e) {
      badge.textContent = e.message;
      badge.className = "badge badge-error";
    }
  });
});

/* ── Model badge ─────────────────────────────────────────────────── */
async function checkModel() {
  const badge = $("model-badge");
  try {
    const data = await apiFetch("/model-info");
    if (data.loaded) {
      badge.textContent = `${data.active_tokenizer.toUpperCase()} · ${data.num_params_fmt} params`;
      badge.className = "badge badge-ready";
    } else {
      badge.textContent = "Model not trained";
      badge.className = "badge badge-error";
    }
  } catch {
    badge.textContent = "API unreachable";
    badge.className = "badge badge-error";
  }
}

/* ── Confidence ──────────────────────────────────────────────────── */
const confBtn = $("conf-btn");
bindSlider("conf-temp", "conf-temp-val");
bindSlider("conf-topk", "conf-topk-val");
bindSlider("conf-maxt", "conf-maxt-val");

confBtn.addEventListener("click", async () => {
  const prompt = $("conf-prompt").value.trim();
  if (!prompt) return;

  $("conf-output").classList.add("hidden");
  $("conf-legend").classList.add("hidden");
  $("conf-chart-wrap").classList.add("hidden");
  setLoading(confBtn, true);

  try {
    const data = await apiFetch("/generate-with-confidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_tokens: Number($("conf-maxt").value),
        temperature: Number($("conf-temp").value),
        top_k:       Number($("conf-topk").value),
      }),
    });

    renderConfidence(data);
  } catch (e) {
    const out = $("conf-output");
    out.textContent = "Error: " + e.message;
    out.classList.remove("hidden");
  } finally {
    setLoading(confBtn, false);
  }
});

function entropyToColor(entropy, maxEntropy) {
  // 0 = confident (green), 1 = uncertain (red)
  const t = Math.min(entropy / maxEntropy, 1);
  // green #16a34a → yellow #f59e0b → red #dc2626
  let r, g, b;
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(22  + s * (245 - 22));
    g = Math.round(163 + s * (158 - 163));
    b = Math.round(74  + s * (11  - 74));
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(245 + s * (220 - 245));
    g = Math.round(158 + s * (38  - 158));
    b = Math.round(11  + s * (38  - 11));
  }
  return `rgb(${r},${g},${b})`;
}

function renderConfidence(data) {
  const out = $("conf-output");
  out.innerHTML = "";

  // Prompt text (static, grey)
  const promptSpan = document.createElement("span");
  promptSpan.className = "conf-prompt-text";
  promptSpan.textContent = data.prompt;
  out.appendChild(promptSpan);

  const maxE = data.max_entropy;

  data.tokens.forEach(({ token, entropy }) => {
    const span = document.createElement("span");
    span.className = "conf-token";
    span.textContent = token;
    span.style.color = entropyToColor(entropy, maxE);
    span.title = `entropy: ${entropy.toFixed(3)} nats`;
    out.appendChild(span);
  });

  out.classList.remove("hidden");
  $("conf-legend").classList.remove("hidden");

  // Entropy-over-time chart
  if (confChart) confChart.destroy();
  const labels = data.tokens.map((_, i) => i + 1);
  const entropies = data.tokens.map(t => t.entropy);
  const colors = entropies.map(e => entropyToColor(e, maxE));

  confChart = new Chart($("conf-chart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Entropy (nats)",
        data: entropies,
        backgroundColor: colors,
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { title: { display: true, text: "Entropy (nats)" }, beginAtZero: true },
      },
    },
  });

  $("conf-chart-wrap").classList.remove("hidden");
}

/* ── Generate ────────────────────────────────────────────────────── */
const genBtn = $("gen-btn");

// Sync slider labels
function bindSlider(inputId, labelId) {
  const input = $(inputId), label = $(labelId);
  input.addEventListener("input", () => { label.textContent = input.value; });
}
bindSlider("gen-temp",  "temp-val");
bindSlider("gen-topk",  "topk-val");
bindSlider("gen-maxt",  "maxt-val");

genBtn.addEventListener("click", async () => {
  const prompt = $("gen-prompt").value.trim();
  if (!prompt) return;

  const output = $("gen-output");
  output.classList.add("hidden");
  setLoading(genBtn, true);

  try {
    const data = await apiFetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_tokens: Number($("gen-maxt").value),
        temperature: Number($("gen-temp").value),
        top_k:       Number($("gen-topk").value),
      }),
    });
    output.textContent = data.generated;
    output.classList.remove("hidden");
  } catch (e) {
    output.textContent = "Error: " + e.message;
    output.classList.remove("hidden");
  } finally {
    setLoading(genBtn, false);
  }
});

/* ── Attention ───────────────────────────────────────────────────── */
const attnBtn  = $("attn-btn");

attnBtn.addEventListener("click", async () => {
  const text = $("attn-text").value.trim();
  if (!text) return;

  setLoading(attnBtn, true);
  $("attn-controls").classList.add("hidden");
  $("attn-container").classList.add("hidden");

  try {
    attnData = await apiFetch("/attention", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    // Populate layer / head selects
    const layerSel = $("attn-layer");
    const headSel  = $("attn-head");
    layerSel.innerHTML = "";
    headSel.innerHTML  = "";

    for (let i = 0; i < attnData.n_layers; i++) {
      const opt = document.createElement("option");
      opt.value = i; opt.textContent = `Layer ${i}`;
      layerSel.appendChild(opt);
    }
    for (let i = 0; i < attnData.n_heads; i++) {
      const opt = document.createElement("option");
      opt.value = i; opt.textContent = `Head ${i}`;
      headSel.appendChild(opt);
    }

    $("attn-controls").classList.remove("hidden");
    $("attn-container").classList.remove("hidden");
    drawHeatmap();
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    setLoading(attnBtn, false);
  }
});

$("attn-layer").addEventListener("change", drawHeatmap);
$("attn-head").addEventListener("change",  drawHeatmap);

function drawHeatmap() {
  if (!attnData) return;

  const layer  = Number($("attn-layer").value);
  const head   = Number($("attn-head").value);
  const matrix = attnData.attention_weights[layer][head]; // [T][T]
  const tokens = attnData.tokens;
  const T      = tokens.length;

  const canvas  = $("attn-canvas");
  const size    = T * CELL;
  canvas.width  = size;
  canvas.height = size;
  canvas.style.width  = size + "px";
  canvas.style.height = size + "px";

  const ctx = canvas.getContext("2d");

  for (let row = 0; row < T; row++) {
    for (let col = 0; col < T; col++) {
      const val = matrix[row][col];           // 0–1
      const b   = Math.round(val * 255);
      const g   = Math.round(val * 120);
      ctx.fillStyle = `rgb(0, ${g}, ${b})`;
      ctx.fillRect(col * CELL, row * CELL, CELL, CELL);
    }
  }

  // Token labels (x-axis, bottom)
  const row = $("attn-tokens");
  row.innerHTML = "";
  tokens.forEach(ch => {
    const div = document.createElement("div");
    div.className = "token-cell";
    div.style.width = CELL + "px";
    div.textContent = ch === " " ? "·" : ch;
    row.appendChild(div);
  });
}

/* ── Training logs ───────────────────────────────────────────────── */
$("refresh-logs").addEventListener("click", loadTrainingLogs);

async function loadTrainingLogs() {
  try {
    const data = await apiFetch("/training-logs");
    renderTraining(data.logs);
  } catch (e) {
    $("no-logs").textContent = "Error: " + e.message;
    $("no-logs").classList.remove("hidden");
  }
}

function renderTraining(logs) {
  if (!logs || logs.length === 0) {
    $("no-logs").classList.remove("hidden");
    $("stats-row").classList.add("hidden");
    $("charts-area").classList.add("hidden");
    return;
  }

  $("no-logs").classList.add("hidden");
  $("stats-row").classList.remove("hidden");
  $("charts-area").classList.remove("hidden");

  // Stats
  const bestVal = Math.min(...logs.map(l => l.val_loss));
  const bestPpl = Math.min(...logs.map(l => l.perplexity));
  const steps   = logs[logs.length - 1].step;
  $("stat-best-val").textContent = bestVal.toFixed(4);
  $("stat-best-ppl").textContent = bestPpl.toFixed(1);
  $("stat-steps").textContent    = steps.toLocaleString();

  const labels     = logs.map(l => l.step);
  const trainLoss  = logs.map(l => l.train_loss);
  const valLoss    = logs.map(l => l.val_loss);
  const perplexity = logs.map(l => l.perplexity);

  const chartOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: "top" } },
    elements: { point: { radius: 3 } },
    scales: {
      x: { title: { display: true, text: "Step" } },
    },
  };

  // Loss chart
  if (lossChart) lossChart.destroy();
  lossChart = new Chart($("loss-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Train loss", data: trainLoss, borderColor: "#2563eb", backgroundColor: "rgba(37,99,235,.08)", tension: .3 },
        { label: "Val loss",   data: valLoss,   borderColor: "#16a34a", backgroundColor: "rgba(22,163,74,.08)",  tension: .3 },
      ],
    },
    options: { ...chartOpts, scales: { ...chartOpts.scales, y: { title: { display: true, text: "Cross-entropy loss" } } } },
  });

  // Perplexity chart
  if (pplChart) pplChart.destroy();
  pplChart = new Chart($("ppl-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Perplexity", data: perplexity, borderColor: "#9333ea", backgroundColor: "rgba(147,51,234,.08)", tension: .3 },
      ],
    },
    options: { ...chartOpts, scales: { ...chartOpts.scales, y: { title: { display: true, text: "Perplexity" } } } },
  });
}

/* ── Model info ──────────────────────────────────────────────────── */
async function loadModelInfo() {
  try {
    const data = await apiFetch("/model-info");
    renderModelInfo(data);
  } catch (e) {
    $("info-not-loaded").textContent = "Error: " + e.message;
  }
}

function renderModelInfo(data) {
  if (!data.loaded) {
    $("info-not-loaded").classList.remove("hidden");
    $("info-grid").classList.add("hidden");
    return;
  }

  $("info-not-loaded").classList.add("hidden");
  $("info-grid").classList.remove("hidden");

  const items = [
    { key: "Parameters",      value: data.num_params_fmt },
    { key: "Vocab size",      value: data.vocab_size },
    { key: "Context length",  value: data.block_size },
    { key: "Embedding dim",   value: data.n_embd },
    { key: "Attention heads", value: data.n_heads },
    { key: "Layers",          value: data.n_layers },
    { key: "Dropout",         value: data.dropout },
    { key: "Architecture",    value: "Decoder-only" },
  ];

  $("info-grid").innerHTML = items.map(({ key, value }) => `
    <div class="info-item">
      <div class="info-key">${key}</div>
      <div class="info-value">${value}</div>
    </div>`).join("");
}

/* ── Init ────────────────────────────────────────────────────────── */
checkModel();
loadModelInfo();
