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
const targetScoreEl = document.getElementById("targetScore");

function getPayload() {
  return {
    prompt: promptEl.value.trim() || "制作一张科技风 AI 会议海报",
    width: Number(widthEl.value) || 768,
    height: Number(heightEl.value) || 1152,
    max_iterations: Number(maxIterationsEl.value) || 2,
    target_score: Number(targetScoreEl.value) || 85,
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
    const resp = await fetch(`${API_BASE}/api/v1/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getPayload()),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    appendLog(`生成完成: job_id=${data.job_id}, score=${data.score ?? "N/A"}`);

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
    const resp = await fetch(`${API_BASE}/api/v1/generate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getPayload()),
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
        }

        if (eventName === "final_output") {
          if (payload.image_url) {
            renderPreviewFromImageUrl(payload.image_url);
          } else if (payload.final_image) {
            renderPreviewFromBase64(payload.final_image);
          }
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

healthBtn.addEventListener("click", checkHealth);
generateBtn.addEventListener("click", runGenerate);
streamBtn.addEventListener("click", runStream);

checkHealth();
