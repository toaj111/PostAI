# PostAI — 多 Agent 智能海报生成

## 环境要求

## 快速开始

### 方式一：uv（推荐）

```bash
# 1. 创建虚拟环境 & 安装依赖
uv venv --python 3.13
uv pip install -e "backend[dev]"

# 2. 安装 Playwright 浏览器
uv run playwright install chromium

# 3. 启动服务
uv run uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000

# 4. 运行测试
uv run pytest --rootdir backend -c backend/pyproject.toml -v
```

### 方式二：默认 venv + pip

```bash
# 1. 创建虚拟环境（需要 python.org Python 3.11 ~ 3.13）
python -m venv .venv

# 2. 激活
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. 安装依赖
pip install -e "backend[dev]"

# 4. 安装 Playwright 浏览器
playwright install chromium

# 5. 启动服务
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000

# 6. 运行测试
cd backend && python -m pytest -v
```

```bash
cp backend/.env.example backend/.env
```

编辑 `backend/.env`，填入你的 API Key：

```ini
# 文本 LLM — 生成 ContentPlan / StyleGuide / HTML 布局
LLM_API_KEY=你的-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_RESPONSE_FORMAT=json_object

# 视觉模型 — 评审渲染后的海报图片
VISION_API_KEY=你的-qwen-vl-api-key
VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_MODEL=qwen3-vl-flash

# VLM 深度推理（看图→思考→评价）
VISION_ENABLE_THINKING=true
VISION_THINKING_BUDGET=8192

# 模型失败时回退到本地规则 Agent（默认 true）
ALLOW_MODEL_FALLBACK=true
```

不填 API Key 也能用——系统会自动回退到本地规则生成（不调用 LLM/VLM）。

### 调用示例

#### 命令行

```powershell
# Windows PowerShell（中文必须 UTF-8 编码）
$body = '{"prompt":"给我一个卡通画的爱丽丝梦游仙境的海报","width":768,"height":1152,"max_iterations":2,"min_iterations":1}'
$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$resp = Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/generate `
  -Method Post -ContentType "application/json; charset=utf-8" -Body $bodyBytes
```

```bash
# curl (Git Bash / Linux)
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"prompt":"请介绍你自己","width":768,"height":1152,"max_iterations":2,"min_iterations":1}'
```

#### 网页演示

```bash
cd frontend
python -m http.server 5500
```

浏览器访问`http://127.0.0.1:5500`

## API

| 方法   | 路径                      | 说明                                               |
| ------ | ------------------------- | -------------------------------------------------- |
| `POST` | `/api/v1/generate`        | 同步生成海报，返回 JSON（含 `final_image` base64） |
| `POST` | `/api/v1/generate/stream` | SSE 流式生成，实时推送各阶段进度                   |
| `GET`  | `/health`                 | 健康检查                                           |

### 请求体

```json
{
  "prompt": "制作一张科技风 AI 会议海报",
  "width": 768,
  "height": 1152,
  "max_iterations": 3,
  "min_iterations": 1,
  "target_score": 85
}
```

| 字段             | 默认值 | 说明                                                |
| ---------------- | ------ | --------------------------------------------------- |
| `prompt`         | (必填) | 海报主题，中文/英文均可                             |
| `width`          | 1024   | 画布宽度 (256–4096)                                 |
| `height`         | 1536   | 画布高度 (256–4096)                                 |
| `max_iterations` | 3      | 最大迭代次数 (1–5)                                  |
| `min_iterations` | 0      | 最少 VLM 评审次数，0=达标即停，1=至少一轮反馈后再停 |
| `target_score`   | 85     | 目标评分，VLM 达到此分且 min_iterations 满足时停止  |

### 响应体

```json
{
  "job_id": "a4779c96...",
  "final_image": "iVBORw0KGgo...",
  "image_url": "/assets/a4779c96...png",
  "html_url": "/assets/a4779c96_0.html",
  "score": 85,
  "layout_html": "<!DOCTYPE html>...",
  "content_plan": { ... },
  "style": { ... },
  "critiques": [ { "score": 72, "vision_description": "...", "suggestions": [...] } ],
  "warnings": []
}
```

---

## 架构

```
        ┌──────────────────┐
        │  GenerateRequest │
        └────────┬─────────┘
                 │
    ┌────────────┼────────────┐
    │    ContentExtractor     │  → ContentPlan (JSON, DeepSeek)
    ├─────────────────────────┤
    │    StyleDirector        │  → StyleGuide (JSON, DeepSeek)
    ├─────────────────────────┤
    │  SpatialLayoutPlanner   │  → HTML+CSS 文档 (DeepSeek, force_raw)
    ├─────────────────────────┤
    │      HTMLPainter        │  → PNG 截图 (Playwright 无头浏览器)
    ├─────────────────────────┤
    │    HeuristicVLMCritic   │  → CritiqueResult (Qwen VL + enable_thinking)
    └────────────┬────────────┘
                 │
        ┌────────┴─────────┐
        │   Router (iterate │  score ≥ target ? final : re-layout)
        └──────────────────┘
```

- **ContentExtractor** — 从用户 prompt 提取海报元素（标题/副标题/主图/CTA）
- **StyleDirector** — 决定配色、字体风格、mood
- **SpatialLayoutPlanner** — **直接输出完整 HTML+CSS 文档**，不再用自定义 schema，LLM 原生理解 CSS
- **HTMLPainter** — Playwright 无头浏览器渲染 HTML → PNG 截图
- **HeuristicVLMCritic** — Qwen VL 看图评审，`enable_thinking` 模式深度推理，输出评分 + 自然语言改进建议
- **Router** — 决定 finalize 还是再迭代一轮

---

## 项目结构

```
PostAI/
├── README.md
├── HTML_PAINTER_PLAN.md          # HTML/CSS 渲染器改进方案
├── PHASE_PLAN.md                 # 早期分阶段方案（PainterPlan 阶段）
├── backend/
│   ├── .env                      # API Key 配置（不提交 git）
│   ├── .env.example              # 配置模板
│   ├── pyproject.toml            # 项目配置 + 依赖
│   ├── app/
│   │   ├── main.py               # FastAPI 应用入口
│   │   ├── api/
│   │   │   ├── routes_generate.py  # POST /generate, /generate/stream
│   │   │   └── middleware.py       # 请求/响应日志中间件
│   │   ├── agents/
│   │   │   ├── content_extractor.py   # ContentPlan Agent
│   │   │   ├── style_director.py      # StyleGuide Agent
│   │   │   ├── layout_planner.py      # HTML 布局 Agent
│   │   │   └── vlm_critic.py          # VLM 评审 Agent
│   │   ├── core/
│   │   │   ├── config.py          # 配置加载
│   │   │   ├── errors.py          # 自定义异常
│   │   │   ├── events.py          # SSE 事件模型
│   │   │   ├── llm_client.py      # OpenAI 兼容 API 客户端
│   │   │   └── logging.py         # 日志配置
│   │   ├── orchestration/
│   │   │   ├── graph_runner.py    # 管线编排器
│   │   │   ├── retry.py           # 异步重试工具
│   │   │   └── router.py          # 迭代路由决策
│   │   ├── render/
│   │   │   ├── html_painter.py    # Playwright HTML→PNG 渲染器
│   │   │   └── asset_store.py     # PNG 文件存储
│   │   └── schemas/
│   │       ├── agents.py          # ContentPlan / StyleGuide / CritiqueResult
│   │       ├── api.py             # GenerateRequest / GenerateResponse
│   │       ├── layout.py          # 旧 LayoutTree（逐步废弃中）
│   │       └── state.py           # GraphState / RenderResult
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api_routes.py
│   │   ├── test_html_painter.py
│   │   ├── test_llm_agents.py
│   │   ├── test_llm_client.py
│   │   ├── test_layout_adjustments.py
│   │   ├── test_retry.py
│   │   ├── test_router.py
│   │   ├── test_schema.py
│   │   ├── test_sse_events.py
│   │   ├── test_state_machine.py
│   │   ├── test_vision_client.py
│   │   └── test_vlm_critic.py
│   └── generated/                 # 生成的 PNG 海报
└── logs/
    └── postai.log                 # 运行日志
```

---
