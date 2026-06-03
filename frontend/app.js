const API_BASE = "http://127.0.0.1:8000";

const healthBtn = document.getElementById("healthBtn");
const generateBtn = document.getElementById("generateBtn");
const streamBtn = document.getElementById("streamBtn");
const clearLogBtn = document.getElementById("clearLogBtn");
const serverDot = document.getElementById("serverDot");
const serverText = document.getElementById("serverText");
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

const refinePromptEl = document.getElementById("refinePrompt");
const refineBtn = document.getElementById("refineBtn");
const refineStatus = document.getElementById("refineStatus");

const progressBar = document.getElementById("progressBar");
const progressPercent = document.getElementById("progressPercent");
const progressMessage = document.getElementById("progressMessage");
const progressSteps = document.getElementById("progressSteps");
const jobIdValue = document.getElementById("jobIdValue");
const iterationValue = document.getElementById("iterationValue");
const stageValue = document.getElementById("stageValue");
const scoreValue = document.getElementById("scoreValue");

const MAX_REFERENCE_IMAGES = 5;
const REFERENCE_UPLOAD_ENDPOINT = `${API_BASE}/api/v1/reference-images/upload`;

const STEP_ORDER = ["content", "style", "layout", "render", "critique", "final"];
const STEP_LABELS = {
  content: "内容解析",
  style: "风格规划",
  layout: "HTML 布局",
  render: "渲染预览",
  critique: "视觉评审",
  final: "生成完成",
};
const AGENT_TO_STEP = {
  ContentExtractor: "content",
  StyleDirector: "style",
  SpatialLayoutPlanner: "layout",
  HTMLPainter: "render",
  VLMCritic: "critique",
};
const STEP_PROGRESS = {
  idle: 0,
  content: 12,
  style: 28,
  layout: 46,
  render: 64,
  critique: 82,
  final: 100,
};

let currentJobId = "";
let currentHtmlUrl = "";
let currentLayoutHtml = "";
let currentImageUrl = "";
let currentWidth = 768;
let currentHeight = 1152;
let currentIteration = 0;
let currentAbortController = null;
let activeStep = "idle";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setControlsDisabled(disabled) {
  generateBtn.disabled = disabled;
  streamBtn.disabled = disabled;
  healthBtn.disabled = disabled;
  addReferenceBtn.disabled =
    disabled ||
    referenceListEl.querySelectorAll(".reference-row").length >=
      MAX_REFERENCE_IMAGES;
  referenceListEl
    .querySelectorAll("input, select, button")
    .forEach((element) => {
      element.disabled = disabled;
    });
  updateRefineButtonState();
}

function setRunning(state, label = "空闲") {
  runningTag.textContent = label;
  runningTag.classList.remove("running", "done", "error");
  if (state === "running") {
    runningTag.classList.add("running");
  }
  if (state === "done") {
    runningTag.classList.add("done");
  }
  if (state === "error") {
    runningTag.classList.add("error");
  }
}

function appendLog(message, replace = false) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  logArea.textContent = replace
    ? line
    : `${logArea.textContent}${logArea.textContent ? "\n" : ""}${line}`;
  logArea.scrollTop = logArea.scrollHeight;
}

function clampProgress(value) {
  return Math.max(0, Math.min(100, Number(value) || 0));
}

function setProgress(step, message, options = {}) {
  const resolvedStep = step || activeStep || "content";
  activeStep = resolvedStep;
  const activeIndex = STEP_ORDER.indexOf(resolvedStep);
  const progressValue = clampProgress(
    options.progress ?? STEP_PROGRESS[resolvedStep] ?? STEP_PROGRESS.idle,
  );

  progressBar.style.width = `${progressValue}%`;
  progressPercent.textContent = `${progressValue}%`;
  progressMessage.textContent = message || "准备就绪";
  stageValue.textContent =
    resolvedStep && STEP_LABELS[resolvedStep] ? STEP_LABELS[resolvedStep] : "--";

  progressSteps.querySelectorAll(".step").forEach((item) => {
    const itemStep = item.dataset.step;
    const itemIndex = STEP_ORDER.indexOf(itemStep);
    item.classList.remove("active", "done", "error");
    if (options.error && itemStep === resolvedStep) {
      item.classList.add("error");
      return;
    }
    if (resolvedStep === "final" && !options.error) {
      item.classList.add("done");
      return;
    }
    if (activeIndex >= 0 && itemIndex < activeIndex) {
      item.classList.add("done");
    }
    if (itemStep === resolvedStep && !options.error) {
      item.classList.add("active");
    }
  });
}

function resetProgress() {
  activeStep = "idle";
  jobIdValue.textContent = "--";
  iterationValue.textContent = "--";
  scoreValue.textContent = "--";
  setProgress("idle", "准备就绪", { progress: 0 });
}

function applyServerStatus(ok, text) {
  serverDot.classList.remove("ok", "bad");
  serverDot.classList.add(ok ? "ok" : "bad");
  serverText.textContent = text || (ok ? "服务在线" : "服务不可用");
}

function absoluteAssetUrl(url) {
  if (!url) {
    return "";
  }
  return url.startsWith("http") ? url : `${API_BASE}${url}`;
}

function renderPreviewFromImageUrl(imageUrl) {
  previewBox.innerHTML = "";
  const img = document.createElement("img");
  img.alt = "生成的海报预览";
  img.src = absoluteAssetUrl(imageUrl);
  previewBox.appendChild(img);
}

function renderPreviewFromBase64(finalImage) {
  previewBox.innerHTML = "";
  const img = document.createElement("img");
  img.alt = "生成的海报预览";
  img.src = `data:image/png;base64,${finalImage}`;
  previewBox.appendChild(img);
}

function showEmptyPreview(title = "暂无预览", detail = "") {
  const detailMarkup = detail ? `<span>${escapeHtml(detail)}</span>` : "";
  previewBox.innerHTML = `
    <div class="preview-empty">
      <strong>${escapeHtml(title)}</strong>
      ${detailMarkup}
    </div>
  `;
}

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
          <input class="ref-url" type="url" placeholder="https://example.com/image.jpg" value="${escapeHtml(url)}" />
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
        <input class="ref-desc" type="text" maxlength="500" placeholder="例如：蓝色霓虹人物半身像" value="${escapeHtml(description)}" />
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

  row.querySelector(".remove-ref-btn").addEventListener("click", () => {
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
        appendLog(`忽略非法参考图 URL，仅支持 http/https: ${url}`);
        continue;
      }
      references.push({ url, description });
    } catch {
      appendLog(`忽略非法参考图 URL: ${url}`);
    }
  }

  return references;
}

function getBasePayload() {
  return {
    prompt:
      promptEl.value.trim() ||
      "做一张当代艺术展海报，极简、不要按钮",
    width: Number(widthEl.value) || 768,
    height: Number(heightEl.value) || 1152,
    max_iterations: Number(maxIterationsEl.value) || 2,
    min_iterations: Number(minIterationsEl.value) || 0,
    target_score: Number(targetScoreEl.value) || 85,
    reference_images: [],
  };
}

async function getPayloadWithReferences() {
  const payload = getBasePayload();
  setProgress("content", "正在处理参考图片", { progress: 6 });
  payload.reference_images = await collectReferenceImages();
  appendLog(`参考图片数量: ${payload.reference_images.length}`);
  return payload;
}

function updateRefineButtonState() {
  const hasHtml = !!(currentHtmlUrl || currentLayoutHtml);
  const hasPrompt = refinePromptEl.value.trim().length > 0;
  const busy = runningTag.classList.contains("running");
  refineBtn.disabled = !(hasHtml && hasPrompt) || busy;
}

function updateCurrentState(data) {
  if (!data || typeof data !== "object") {
    return;
  }

  if (data.job_id) {
    currentJobId = data.job_id;
    jobIdValue.textContent = data.job_id;
  }
  if (data.html_url) {
    currentHtmlUrl = data.html_url;
  } else if (currentJobId && data.image_url) {
    const htmlGuess = data.image_url.replace(/\.(png|jpg|jpeg)$/i, ".html");
    if (htmlGuess !== data.image_url) {
      currentHtmlUrl = htmlGuess;
    }
  }
  if (data.layout_html) {
    currentLayoutHtml = data.layout_html;
  }
  if (data.image_url) {
    currentImageUrl = data.image_url;
  }
  if (data.width) {
    currentWidth = data.width;
  }
  if (data.height) {
    currentHeight = data.height;
  }
  if (data.render_result?.width) {
    currentWidth = data.render_result.width;
  }
  if (data.render_result?.height) {
    currentHeight = data.render_result.height;
  }
  if (data.iteration !== undefined) {
    currentIteration = data.iteration;
    iterationValue.textContent = String(data.iteration);
  }
  if (data.score !== undefined && data.score !== null) {
    scoreValue.textContent = String(data.score);
  }
  if (Array.isArray(data.critiques) && data.critiques.length > 0) {
    const latest = data.critiques[data.critiques.length - 1];
    if (latest?.score !== undefined && latest.score !== null) {
      scoreValue.textContent = String(latest.score);
    }
  }
  updateRefineButtonState();
}

function applyPreviewPayload(payload) {
  if (payload.image_url) {
    renderPreviewFromImageUrl(payload.image_url);
  } else if (payload.final_image) {
    renderPreviewFromBase64(payload.final_image);
  } else if (payload.render_result?.image_url) {
    renderPreviewFromImageUrl(payload.render_result.image_url);
  } else if (payload.render_result?.image_base64) {
    renderPreviewFromBase64(payload.render_result.image_base64);
  }
}

async function checkHealth() {
  setRunning("running", "检查中");
  appendLog("请求健康检查 /health");
  try {
    const resp = await fetch(`${API_BASE}/health`);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    applyServerStatus(true, "服务在线");
    appendLog(`服务在线: ${JSON.stringify(data)}`);
    setRunning("done", "服务在线");
  } catch (err) {
    applyServerStatus(false, "服务不可用");
    appendLog(`健康检查失败: ${err.message}`);
    setRunning("error", "连接失败");
  } finally {
    setTimeout(() => {
      if (!runningTag.classList.contains("running")) {
        setRunning("idle", "空闲");
      }
    }, 1200);
  }
}

function startJob(label) {
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();
  setControlsDisabled(true);
  setRunning("running", label);
  resetProgress();
  showEmptyPreview("正在生成");
  return currentAbortController.signal;
}

function finishJob(label = "已完成") {
  setRunning("done", label);
  setControlsDisabled(false);
  currentAbortController = null;
  setTimeout(() => {
    if (!runningTag.classList.contains("running")) {
      setRunning("idle", "空闲");
      updateRefineButtonState();
    }
  }, 1600);
}

function failJob(message) {
  setRunning("error", "失败");
  setControlsDisabled(false);
  currentAbortController = null;
  setProgress(activeStep === "idle" ? "content" : activeStep, message, {
    progress: Number.parseInt(progressPercent.textContent, 10) || 0,
    error: true,
  });
  updateRefineButtonState();
}

async function runGenerate() {
  const signal = startJob("快速生成中");
  setProgress("content", "提交同步生成请求", { progress: 8 });
  appendLog("调用快速生成 /api/v1/generate");

  try {
    const payload = await getPayloadWithReferences();
    currentWidth = payload.width;
    currentHeight = payload.height;
    setProgress("render", "后端正在生成海报，同步模式将在完成后返回", {
      progress: 42,
    });

    const resp = await fetch(`${API_BASE}/api/v1/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    updateCurrentState(data);
    applyPreviewPayload(data);
    setProgress("final", `生成完成，评分 ${data.score ?? "N/A"}`, {
      progress: 100,
    });
    appendLog(`生成完成: job_id=${data.job_id}, score=${data.score ?? "N/A"}`);
    if (data.image_url) {
      appendLog(`预览地址: ${data.image_url}`);
    }
    finishJob("已完成");
  } catch (err) {
    if (err.name === "AbortError") {
      appendLog("请求已取消");
      return;
    }
    appendLog(`快速生成失败: ${err.message}`);
    failJob(`快速生成失败: ${err.message}`);
  }
}

function readSseChunk(chunk) {
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

  return { eventName, dataLine };
}

function handleStreamEvent(eventName, payload) {
  if (eventName === "job_started") {
    updateCurrentState(payload);
    setProgress("content", "任务已启动，正在解析内容", { progress: 10 });
    appendLog(`任务启动: ${payload.job_id}`);
    return;
  }

  if (eventName === "agent_start") {
    const step = AGENT_TO_STEP[payload.agent] || "content";
    const label = STEP_LABELS[step] || payload.agent;
    setProgress(step, payload.message || `${label}中`, {
      progress: STEP_PROGRESS[step],
    });
    appendLog(`${label}开始: ${payload.message || payload.agent}`);
    return;
  }

  if (eventName === "agent_complete") {
    const step = AGENT_TO_STEP[payload.agent] || "content";
    const label = STEP_LABELS[step] || payload.agent;
    updateCurrentState(payload.result || payload);
    setProgress(step, `${label}完成`, {
      progress: Math.min((STEP_PROGRESS[step] || 0) + 9, 95),
    });
    appendLog(`${label}完成`);
    return;
  }

  if (eventName === "render_preview") {
    updateCurrentState(payload);
    applyPreviewPayload(payload);
    setProgress("render", `第 ${payload.iteration ?? 0} 次渲染完成`, {
      progress: 72,
    });
    appendLog(`渲染预览完成: iteration=${payload.iteration ?? 0}`);
    return;
  }

  if (eventName === "critique") {
    updateCurrentState(payload);
    setProgress("critique", `视觉评审完成，评分 ${payload.score ?? "N/A"}`, {
      progress: 88,
    });
    const focus = payload.revision_focus ? `, focus=${payload.revision_focus}` : "";
    appendLog(`视觉评审: score=${payload.score ?? "N/A"}${focus}`);
    return;
  }

  if (eventName === "warning") {
    appendLog(`警告: ${payload.message || JSON.stringify(payload)}`);
    return;
  }

  if (eventName === "final_output") {
    updateCurrentState(payload);
    applyPreviewPayload(payload);
    setProgress("final", `生成完成，评分 ${payload.score ?? "N/A"}`, {
      progress: 100,
    });
    appendLog(`流式生成完成: score=${payload.score ?? "N/A"}`);
    return;
  }

  if (eventName === "job_finished") {
    setProgress("final", "任务已结束", { progress: 100 });
    appendLog(`任务结束: ${payload.job_id || currentJobId}`);
    return;
  }

  if (eventName === "error") {
    throw new Error(payload.message || "流式生成失败");
  }

  appendLog(`SSE(${eventName}): ${JSON.stringify(payload)}`);
}

async function runStream() {
  const signal = startJob("流式生成中");
  setProgress("content", "准备发送流式请求", { progress: 5 });
  appendLog("调用流式生成 /api/v1/generate/stream");

  try {
    const payload = await getPayloadWithReferences();
    currentWidth = payload.width;
    currentHeight = payload.height;

    const resp = await fetch(`${API_BASE}/api/v1/generate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
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
        const { eventName, dataLine } = readSseChunk(chunk);
        if (!dataLine) {
          continue;
        }

        let parsedPayload;
        try {
          parsedPayload = JSON.parse(dataLine);
        } catch {
          appendLog(`SSE(${eventName}): ${dataLine}`);
          continue;
        }

        handleStreamEvent(eventName, parsedPayload);
      }
    }

    finishJob("已完成");
  } catch (err) {
    if (err.name === "AbortError") {
      appendLog("请求已取消");
      return;
    }
    appendLog(`流式生成失败: ${err.message}`);
    failJob(`流式生成失败: ${err.message}`);
  }
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

  setControlsDisabled(true);
  setRunning("running", "微调中");
  refineStatus.textContent = "微调中";
  refineStatus.classList.remove("hidden", "done", "error");
  appendLog(`微调请求: "${prompt}"`);
  setProgress("layout", "正在微调 HTML 布局", { progress: 48 });

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
    Object.keys(payload).forEach((key) => {
      if (payload[key] === undefined) {
        delete payload[key];
      }
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
    updateCurrentState(data);
    applyPreviewPayload(data);
    if (data.critique?.score !== undefined) {
      scoreValue.textContent = String(data.critique.score);
    }
    if (data.warnings?.length) {
      appendLog(`警告: ${data.warnings.join("; ")}`);
    }
    if (data.critique) {
      appendLog(
        `微调评审: score=${data.critique.score}, focus=${data.critique.revision_focus || "N/A"}`,
      );
    }
    setProgress("final", "微调完成", { progress: 100 });
    refineStatus.textContent = "完成";
    refineStatus.classList.add("done");
    finishJob("微调完成");
  } catch (err) {
    appendLog(`微调失败: ${err.message}`);
    refineStatus.textContent = "失败";
    refineStatus.classList.add("error");
    failJob(`微调失败: ${err.message}`);
  } finally {
    setTimeout(() => {
      refineStatus.classList.add("hidden");
      refineStatus.classList.remove("done", "error");
    }, 2200);
  }
}

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const promptText = btn.dataset.prompt;
    if (promptText) {
      promptEl.value = promptText;
    }
  });
});

healthBtn.addEventListener("click", checkHealth);
generateBtn.addEventListener("click", runGenerate);
streamBtn.addEventListener("click", runStream);
addReferenceBtn.addEventListener("click", () => createReferenceRow());
refineBtn.addEventListener("click", runRefine);
refinePromptEl.addEventListener("input", updateRefineButtonState);
clearLogBtn.addEventListener("click", () => {
  logArea.textContent = "";
});

createReferenceRow();
resetProgress();
checkHealth();
