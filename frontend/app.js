const API_BASE = "http://127.0.0.1:8000";

const healthBtn = document.getElementById("healthBtn");
const generateBtn = document.getElementById("generateBtn");
const streamBtn = document.getElementById("streamBtn");
const serverDot = document.getElementById("serverDot");
const runningTag = document.getElementById("runningTag");
const previewBox = document.getElementById("previewBox");
const logArea = document.getElementById("logArea");

const promptEl = document.getElementById("prompt");
const widthEl = document.getElementById("width");
const heightEl = document.getElementById("height");
const maxIterationsEl = document.getElementById("maxIterations");
const minIterationsEl = document.getElementById("minIterations");
const targetScoreEl = document.getElementById("targetScore");
const addReferenceBtn = document.getElementById("addReferenceBtn");
const referenceListEl = document.getElementById("referenceList");

const MAX_REFERENCE_IMAGES = 5;
const REFERENCE_UPLOAD_ENDPOINT = `${API_BASE}/api/v1/reference-images/upload`;

function createReferenceRow({ mode = "url", url = "", description = "" } = {}) {
  const row = document.createElement("div");
  row.className = "reference-row";
  row.innerHTML = `
    <div class="reference-headline">
      <label class="field reference-mode-field">
        <span>来源</span>
        <select class="ref-mode">
          <option value="url">在线链接</option>
          <option value="upload">本地上传</option>
        </select>
      </label>
      <button type="button" class="remove-ref-btn">移除</button>
    </div>
    <div class="reference-grid">
      <div class="reference-panel reference-url-panel">
        <label class="field reference-field">
          <span>图片链接</span>
          <input class="ref-url" type="url" placeholder="https://example.com/image.jpg" value="${url}" />
        </label>
      </div>
      <div class="reference-panel reference-upload-panel hidden">
        <label class="field reference-field">
          <span>本地图片</span>
          <input class="ref-file" type="file" accept="image/*" />
        </label>
        <p class="file-name">未选择文件</p>
      </div>
      <label class="field reference-field reference-desc-field">
        <span>图片描述</span>
        <input class="ref-desc" type="text" maxlength="500" placeholder="例如：蓝色霓虹人物半身像" value="${description}" />
      </label>
    </div>
  `;

  const modeSelect = row.querySelector(".ref-mode");
  const urlPanel = row.querySelector(".reference-url-panel");
  const uploadPanel = row.querySelector(".reference-upload-panel");
  const fileInput = row.querySelector(".ref-file");
  const fileName = row.querySelector(".file-name");
  let selectedFile = null;

  function syncMode() {
    const isUpload = modeSelect.value === "upload";
    urlPanel.classList.toggle("hidden", isUpload);
    uploadPanel.classList.toggle("hidden", !isUpload);
  }

  modeSelect.value = mode;
  syncMode();

  modeSelect.addEventListener("change", syncMode);
  fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files?.[0] || null;
    fileName.textContent = selectedFile ? selectedFile.name : "未选择文件";
  });

  row.getReferenceData = () => ({
    mode: modeSelect.value,
    url: row.querySelector(".ref-url")?.value.trim() || "",
    description: row.querySelector(".ref-desc")?.value.trim() || "",
    file: selectedFile,
  });

  const removeBtn = row.querySelector(".remove-ref-btn");
  removeBtn.addEventListener("click", () => {
    row.remove();
    updateReferenceButtonState();
  });

  referenceListEl.appendChild(row);
  updateReferenceButtonState();
}

function updateReferenceButtonState() {
  const count = referenceListEl.querySelectorAll(".reference-row").length;
  addReferenceBtn.disabled = count >= MAX_REFERENCE_IMAGES;
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error(`无法读取文件 ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function uploadReferenceFile(file, description) {
  const dataUrl = await readFileAsDataUrl(file);
  const resp = await fetch(REFERENCE_UPLOAD_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      mime_type: file.type || "image/png",
      data_url: dataUrl,
      description,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`上传失败 HTTP ${resp.status}: ${text}`);
  }

  const payload = await resp.json();
  return payload.url;
}

async function collectReferenceImages() {
  const rows = referenceListEl.querySelectorAll(".reference-row");
  const references = [];

  for (const row of rows) {
    const referenceData = row.getReferenceData?.();
    if (!referenceData) {
      continue;
    }

    const description = referenceData.description || "未提供描述";
    if (referenceData.mode === "upload") {
      if (!referenceData.file) {
        appendLog("跳过未选择文件的本地参考图");
        continue;
      }
      const uploadedUrl = await uploadReferenceFile(
        referenceData.file,
        description,
      );
      references.push({ url: uploadedUrl, description });
      continue;
    }

    const url = referenceData.url;
    if (!url) {
      continue;
    }
    try {
      const parsed = new URL(url);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        appendLog(`忽略非法参考图 URL（仅支持 http/https）: ${url}`);
        continue;
      }
      references.push({ url, description });
    } catch {
      appendLog(`忽略非法参考图 URL: ${url}`);
    }
  }

  return references;
}

function getPayload() {
  const referenceImages = collectReferenceImages();
  return {
    prompt: promptEl.value.trim() || "做一张当代艺术展海报，极简、不要按钮",
    width: Number(widthEl.value) || 768,
    height: Number(heightEl.value) || 1152,
    max_iterations: Number(maxIterationsEl.value) || 2,
    min_iterations: Number(minIterationsEl.value) || 0,
    target_score: Number(targetScoreEl.value) || 85,
    reference_images: referenceImages,
  };
}

function setRunning(running, label = "Idle") {
  runningTag.textContent = label;
  runningTag.classList.toggle("running", running);
}

function appendLog(message, replace = false) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  logArea.textContent = replace
    ? line
    : `${logArea.textContent}${logArea.textContent ? "\n" : ""}${line}`;
  logArea.scrollTop = logArea.scrollHeight;
}

function renderPreviewFromImageUrl(imageUrl) {
  previewBox.innerHTML = "";
  const img = document.createElement("img");
  img.alt = "poster result";
  img.src = imageUrl.startsWith("http") ? imageUrl : `${API_BASE}${imageUrl}`;
  previewBox.appendChild(img);
}

function renderPreviewFromBase64(finalImage) {
  previewBox.innerHTML = "";
  const img = document.createElement("img");
  img.alt = "poster result";
  img.src = `data:image/png;base64,${finalImage}`;
  previewBox.appendChild(img);
}

function applyServerStatus(ok) {
  serverDot.classList.remove("ok", "bad");
  serverDot.classList.add(ok ? "ok" : "bad");
}

async function checkHealth() {
  setRunning(true, "Checking");
  appendLog("请求健康检查 /health");
  try {
    const resp = await fetch(`${API_BASE}/health`);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    applyServerStatus(true);
    appendLog(`服务在线: ${JSON.stringify(data)}`);
  } catch (err) {
    applyServerStatus(false);
    appendLog(`健康检查失败: ${err.message}`);
  } finally {
    setRunning(false);
  }
}

async function runGenerate() {
  setRunning(true, "Generating");
  appendLog("调用快速生成 /api/v1/generate");

  try {
    const payload = getPayload();
    payload.reference_images = await collectReferenceImages();
    appendLog(`携带参考图数量: ${payload.reference_images.length}`);
    const resp = await fetch(`${API_BASE}/api/v1/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    appendLog(`生成完成: job_id=${data.job_id}, score=${data.score ?? "N/A"}`);
    updateCurrentState(data);

    if (data.image_url) {
      renderPreviewFromImageUrl(data.image_url);
      appendLog(`预览地址: ${data.image_url}`);
    } else if (data.final_image) {
      renderPreviewFromBase64(data.final_image);
      appendLog("已使用 final_image(base64) 预览");
    } else {
      appendLog("返回中未包含可预览图像");
    }
  } catch (err) {
    appendLog(`快速生成失败: ${err.message}`);
  } finally {
    setRunning(false);
  }
}

async function runStream() {
  setRunning(true, "Streaming");
  appendLog("调用流式生成 /api/v1/generate/stream");

  try {
    const payload = getPayload();
    payload.reference_images = await collectReferenceImages();
    appendLog(`携带参考图数量: ${payload.reference_images.length}`);
    const resp = await fetch(`${API_BASE}/api/v1/generate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";

      for (const chunk of chunks) {
        const lines = chunk.split("\n");
        let eventName = "message";
        let dataLine = "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
          }
          if (line.startsWith("data:")) {
            dataLine += line.slice(5).trim();
          }
        }

        if (!dataLine) {
          continue;
        }

        let payload;
        try {
          payload = JSON.parse(dataLine);
        } catch {
          appendLog(`SSE(${eventName}): ${dataLine}`);
          continue;
        }

        appendLog(`SSE(${eventName}): ${Object.keys(payload).join(", ")}`);

        if (eventName === "render_preview" && payload.image_url) {
          renderPreviewFromImageUrl(payload.image_url);
          updateCurrentState(payload);
        }

        if (eventName === "final_output") {
          if (payload.image_url) {
            renderPreviewFromImageUrl(payload.image_url);
          } else if (payload.final_image) {
            renderPreviewFromBase64(payload.final_image);
          }
          updateCurrentState(payload);
          appendLog(`流式生成完成: score=${payload.score ?? "N/A"}`);
        }
      }
    }
  } catch (err) {
    appendLog(`流式生成失败: ${err.message}`);
  } finally {
    setRunning(false);
  }
}

// ── Refine state (Phase 1-4) ──

const refinePanel = document.getElementById("refinePanel");
const refinePromptEl = document.getElementById("refinePrompt");
const refineBtn = document.getElementById("refineBtn");
const refineStatus = document.getElementById("refineStatus");

let currentJobId = "";
let currentHtmlUrl = "";
let currentLayoutHtml = "";
let currentImageUrl = "";
let currentWidth = 768;
let currentHeight = 1152;
let currentIteration = 0;

function updateRefineButtonState() {
  const hasHtml = !!(currentHtmlUrl || currentLayoutHtml);
  const hasPrompt = refinePromptEl.value.trim().length > 0;
  refineBtn.disabled = !(hasHtml && hasPrompt);
}

function updateCurrentState(data) {
  if (data.job_id) currentJobId = data.job_id;
  if (data.html_url) {
    currentHtmlUrl = data.html_url;
  } else if (currentJobId && data.image_url) {
    // Infer html_url from image_url when not provided.
    // e.g. /assets/abc_1.png → /assets/abc_1.html
    const imgUrl = data.image_url || "";
    const htmlGuess = imgUrl.replace(/\.(png|jpg|jpeg)$/, ".html");
    if (htmlGuess !== imgUrl) {
      currentHtmlUrl = htmlGuess;
    }
  }
  if (data.layout_html) currentLayoutHtml = data.layout_html;
  if (data.image_url) currentImageUrl = data.image_url;
  if (data.width) currentWidth = data.width;
  if (data.height) currentHeight = data.height;
  if (data.iteration !== undefined) currentIteration = data.iteration;
  updateRefineButtonState();
}

async function runRefine() {
  const prompt = refinePromptEl.value.trim();
  if (!prompt) {
    appendLog("请先填写微调提示词");
    return;
  }
  if (!currentHtmlUrl && !currentLayoutHtml) {
    appendLog("暂无可微调的 HTML，请先生成一张海报");
    return;
  }

  setRunning(true, "Refining");
  refineStatus.textContent = "微调中...";
  refineStatus.classList.add("running");
  refineStatus.classList.remove("hidden");
  appendLog(`微调请求: "${prompt}"`);

  try {
    const payload = {
      job_id: currentJobId,
      html_url: currentHtmlUrl || undefined,
      layout_html: currentHtmlUrl ? undefined : currentLayoutHtml,
      prompt,
      width: currentWidth,
      height: currentHeight,
      iteration: currentIteration + 1,
    };
    Object.keys(payload).forEach((k) => {
      if (payload[k] === undefined) delete payload[k];
    });

    const resp = await fetch(`${API_BASE}/api/v1/refine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    appendLog(`微调完成: iteration=${data.iteration}, warnings=${data.warnings?.length || 0}`);
    if (data.warnings && data.warnings.length > 0) {
      appendLog(`警告: ${data.warnings.join("; ")}`);
    }
    if (data.critique) {
      appendLog(`VLM评分: ${data.critique.score}, ${data.critique.revision_focus || "N/A"}`);
    }

    // Update state and preview.
    currentHtmlUrl = data.html_url;
    currentLayoutHtml = data.layout_html;
    if (data.image_url) currentImageUrl = data.image_url;
    currentIteration = data.iteration;
    updateRefineButtonState();

    if (data.image_url) {
      renderPreviewFromImageUrl(data.image_url);
    } else if (data.final_image) {
      renderPreviewFromBase64(data.final_image);
    }

    refineStatus.textContent = "完成";
  } catch (err) {
    appendLog(`微调失败: ${err.message}`);
    refineStatus.textContent = "失败";
  } finally {
    setRunning(false);
    refineStatus.classList.remove("running");
    setTimeout(() => refineStatus.classList.add("hidden"), 2000);
  }
}

refineBtn.addEventListener("click", runRefine);
refinePromptEl.addEventListener("input", updateRefineButtonState);

// ── Preset buttons ──

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const promptText = btn.dataset.prompt;
    if (promptText) {
      promptEl.value = promptText;
    }
  });
});

// ── Wire up entries ──

healthBtn.addEventListener("click", checkHealth);
generateBtn.addEventListener("click", runGenerate);
streamBtn.addEventListener("click", runStream);
addReferenceBtn.addEventListener("click", () => {
  createReferenceRow();
});

createReferenceRow();
checkHealth();
