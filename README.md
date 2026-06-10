# PostAI — 多 Agent 智能海报生成

从自然语言需求出发，多 Agent 协作生成**可编辑 HTML/CSS 海报**与 **PNG 成品**。

核心思路：不把海报生成当作"一次文生图"，而是拆成内容策划 → 插图生成 → 艺术指导 → HTML/CSS 布局 → 浏览器渲染 → VLM 视觉评审 → 迭代修正的流水线。即使没有远程模型 API，也能通过 fallback 规则引擎跑通完整流程。

> 📄 **项目论文**：[docs/PostAI_Experiment_Report.md](docs/PostAI_Experiment_Report.md)  
> 📊 **演示 PPT**：[slides/slides.md](slides/slides.md)（Slidev）  
> 📝 **演讲稿**：[speech.md](speech.md)  
> 🔍 **架构分析**：[analysis.md](analysis.md)

---

## 快速开始

### 方式一：uv（推荐）

```bash
# 1. 创建虚拟环境 & 安装依赖
uv venv --python 3.13
uv pip install -e "backend[dev]"

# 2. 安装 Playwright 浏览器
uv run playwright install chromium

# 3. （可选）配置 API Key
cp backend/.env.example backend/.env
# 编辑 backend/.env 填入模型 API Key；不填也能跑（自动回退到本地规则引擎）

# 4. 启动服务
uv run uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000

# 5. 运行测试
uv run pytest --rootdir backend -c backend/pyproject.toml -v
```

### 方式二：默认 venv + pip

```bash
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -e "backend[dev]"
playwright install chromium
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

### 网页演示

```bash
cd frontend
python -m http.server 5500
# 浏览器访问 http://127.0.0.1:5500
```

---

## 模型配置

编辑 `backend/.env`，填入你的 API Key：

```ini
# ── 文本 LLM — ContentExtractor / StyleDirector / SpatialLayoutPlanner ──
LLM_API_KEY=你的-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_RESPONSE_FORMAT=json_object

# ── 视觉 VLM — VLMCritic 评审渲染后的海报 ──
VISION_API_KEY=你的-api-key
VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_MODEL=qwen3-vl-flash

# ── 文生图 — IllustrationAgent 生成插图资产 ──
# 不填则自动跳过，不影响海报主流程
IMAGE_API_KEY=你的-api-key
IMAGE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
IMAGE_MODEL=qwen-image-2.0
IMAGE_SIZE=1024x1024
IMAGE_TIMEOUT_SECONDS=120

# ── VLM 深度推理（看图 → 思考 → 评价）──
VISION_ENABLE_THINKING=true
VISION_THINKING_BUDGET=8192

# ── 模型失败时回退到本地规则 Agent ──
ALLOW_MODEL_FALLBACK=false
```

> 三套模型配置（LLM / Vision / Image）彼此独立，可灵活混用不同供应商。不填任何 API Key 也能用——系统自动回退到本地规则引擎和启发式评审。

---

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/generate` | 同步生成海报，返回 JSON（含 `final_image` base64） |
| `POST` | `/api/v1/generate/stream` | SSE 流式生成，实时推送各阶段进度 |
| `POST` | `/api/v1/refine` | 增量修改已有 HTML，自动渲染 + 评审 |
| `POST` | `/api/v1/reference-images/upload` | 上传参考图（base64 data URL），返回 asset URL |
| `GET` | `/assets/{filename}` | 访问生成的 HTML / PNG / 插图 / 参考图 |
| `GET` | `/health` | 健康检查 |

### 请求体

```json
{
  "prompt": "制作一张科技风 AI 会议海报",
  "width": 768,
  "height": 1152,
  "max_iterations": 3,
  "min_iterations": 1,
  "target_score": 85,
  "enable_generated_illustrations": true,
  "max_generated_illustrations": 3,
  "reference_images": []
}
```

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | （必填） | 海报主题，中文/英文均可 |
| `width` | 1024 | 画布宽度（256–4096） |
| `height` | 1536 | 画布高度（256–4096） |
| `max_iterations` | 3 | 最大迭代次数（1–5） |
| `min_iterations` | 0 | 最少 VLM 评审次数，0=达标即停，1=至少一轮反馈后再停 |
| `target_score` | 85 | 目标评分，VLM 达到此分且 min_iterations 满足时停止 |
| `enable_generated_illustrations` | true | 是否启用 AI 插图生成 |
| `max_generated_illustrations` | 3 | 最多生成插图数（0–5） |
| `reference_images` | [] | 参考图列表（每项含 url + description） |

### 响应体

```json
{
  "job_id": "a4779c96...",
  "final_image": "iVBORw0KGgo...",
  "image_url": "/assets/a4779c96_0.png",
  "html_url": "/assets/a4779c96_0.html",
  "score": 85,
  "layout_html": "<!DOCTYPE html>...",
  "poster_brief": { "poster_intent": {...}, "content_strategy": {...}, "messages": [...] },
  "art_direction": { "poster_language": {...}, "color_system": {...}, "typography": {...} },
  "generated_illustrations": [ { "id": "...", "url": "...", "status": "generated" } ],
  "render_result": { "image_base64": "...", "image_url": "...", "width": 768, "height": 1152 },
  "critiques": [ { "score": 72, "revision_focus": "layout", "structured_issues": [...], "vision_description": "..." } ],
  "warnings": []
}
```

> 响应同时包含 V1 兼容字段（`content_plan`, `style`）和 V2 富信息字段（`poster_brief`, `art_direction`）。

---

## 架构

### 流水线

```
GenerateRequest
  │
  ├─ 1. ContentExtractor        （文本 LLM：prompt → PosterBriefV2 → ContentPlan）
  ├─ 2. IllustrationAgent       （图像 API：visual_subjects → 插图资产，非阻塞）
  ├─ 3. StyleDirector           （文本 LLM：brief → ArtDirectionV2 → StyleGuide）
  │
  └─ [迭代循环] ─────────────────────────────────────────────────────┐
       ├─ 4. SpatialLayoutPlanner  （文本 LLM：brief + art_direction│
       │                             + VLM feedback → HTML/CSS）     │
       ├─ 5. HTMLPainter           （Playwright：HTML → PNG 截图）   │
       ├─ 6. VLMCritic             （视觉 VLM / 启发式：PNG → 评审） │
       └─ 7. Router                （决定 final / layout / style /   │
                                     content / render）              │
            ├─ final  → 结束                                        │
            ├─ layout → 回到步骤 4                                    │
            ├─ style  → 回到步骤 3                                    │
            └─ content → 回到步骤 1                                   │
       ──────────────────────────────────────────────────────────────┘
  │
  ▼
GenerateResponse（HTML + PNG + critique history）
```

### 各 Agent 职责

| Agent | 输入 | 输出 | 模型 | Fallback |
|-------|------|------|------|----------|
| **ContentExtractor** | user_prompt + reference_images | PosterBriefV2 → ContentPlan | 文本 LLM | 关键词检测 + 模板化元素 |
| **IllustrationAgent** | visual_subjects | 插图资产列表（generated/failed/skipped） | 图像 API | 跳过，不阻塞主流程 |
| **StyleDirector** | poster_brief + prompt | ArtDirectionV2 → StyleGuide | 文本 LLM | 7 套 poster_type 模板 |
| **SpatialLayoutPlanner** | brief + art_direction + feedback_history + assets | 完整 HTML/CSS 文档 | 文本 LLM（force_raw） | 5 套 HTML 模板 |
| **HTMLPainter** | HTML + canvas 尺寸 | PNG 截图（base64 + URL） | Playwright Chromium | — |
| **VLMCritic** | PNG + HTML + brief + art_direction | CritiqueResult（score + rubric + revision_focus） | 视觉 VLM（enable_thinking） | 启发式评分（检查文本/结构/CSS/图层） |
| **Router** | feedback_history + iteration_count + target_score | RouteDecision（final/layout/style/content） | 规则引擎 | — |
| **HTMLRefiner** | 当前 HTML + 修改指令 | 修订后 HTML | 文本 LLM（force_raw） | 返回原 HTML + warning |

### 关键设计决策

- **HTML/CSS 作为中间表示**：比自定义几何 schema 表达力更强，天然支持字体/图层/网格/裁切/纹理，且 HTML 就是可编辑源文件
- **V1/V2 Schema 并存**：PosterBriefV2 和 ContentPlan、ArtDirectionV2 和 StyleGuide 通过转换器双向兼容，允许各 Agent 独立演进
- **VLM thinking 模式**：评审时先做 chain-of-thought 推理再输出 JSON，推理链存入 `vision_reasoning`，提升评分准确度
- **Canvas guard**：注入 `!important` CSS 锁定画布尺寸，防止 LLM 响应式代码导致截图异常
- **revision_focus 路由**：VLM 评审后精确指定下一步回到哪个阶段（而非笼统"重做"），Router 五层优先级决策
- **三套模型独立配置**：文本 LLM、视觉 VLM、图像生成各自独立的 API key / base URL / model

---

## 项目结构

```
PostAI/
├── README.md
├── speech.md                        # 项目汇报演讲稿
├── analysis.md                      # 完整架构分析文档
├── backend/
│   ├── .env                         # API Key 配置（不提交 git）
│   ├── .env.example                 # 配置模板
│   ├── pyproject.toml               # 项目配置 + 依赖
│   ├── design.md                    # 早期设计文档
│   ├── app/
│   │   ├── main.py                  # FastAPI 应用入口 + 静态文件挂载
│   │   ├── api/
│   │   │   ├── routes_generate.py   # POST /generate, /generate/stream, /refine, /reference-images/upload
│   │   │   └── middleware.py        # 请求/响应日志中间件
│   │   ├── agents/
│   │   │   ├── content_extractor.py   # 内容提取 Agent（PosterBriefV2）
│   │   │   ├── illustration_agent.py  # 插图生成 Agent（非阻塞）
│   │   │   ├── style_director.py      # 风格指导 Agent（ArtDirectionV2）
│   │   │   ├── layout_planner.py      # HTML/CSS 布局 Agent
│   │   │   ├── vlm_critic.py          # VLM 视觉评审 Agent
│   │   │   └── html_refiner.py        # HTML 增量修改 Agent（patch / full）
│   │   ├── core/
│   │   │   ├── config.py            # 配置加载（Settings dataclass + .env）
│   │   │   ├── errors.py            # 异常类型层次（PostAIError 基类）
│   │   │   ├── events.py            # SSE 事件模型 + 格式化
│   │   │   ├── image_client.py      # OpenAI-compatible 图像生成客户端
│   │   │   ├── llm_client.py        # 文本/视觉 LLM 统一客户端（parse + parse_vision）
│   │   │   └── logging.py           # 日志配置（stdout + 可选文件）
│   │   ├── orchestration/
│   │   │   ├── graph_runner.py      # 主编排器（同步 + SSE 流式）
│   │   │   ├── retry.py             # 异步重试工具（指数退避）
│   │   │   └── router.py            # 五层优先级路由决策
│   │   ├── render/
│   │   │   ├── html_painter.py      # Playwright HTML→PNG 渲染器 + 5 套 fallback 模板
│   │   │   ├── asset_store.py       # 资产持久化（PNG/HTML/插图/参考图）
│   │   │   └── interface.py         # Renderer Protocol（抽象接口）
│   │   └── schemas/
│   │       ├── agents.py            # PosterBriefV2 / ArtDirectionV2 / CritiqueResult 等全部 Agent schema + V1↔V2 转换器
│   │       ├── api.py               # GenerateRequest / GenerateResponse / RefineRequest 等
│   │       ├── layout.py            # CanvasSpec / LayoutTree（legacy）
│   │       └── state.py             # GraphState / RenderResult / GeneratedIllustration 等
│   └── tests/
│       ├── conftest.py
│       ├── test_state_machine.py    # 完整 fallback 流程
│       ├── test_api_routes.py       # 同步/流式/参考图上传
│       ├── test_router.py           # 路由决策
│       ├── test_html_painter.py     # 渲染 + canvas guard + asset 解析
│       ├── test_vlm_critic.py       # VLM 解析 + 启发式评审
│       ├── test_golden_prompts.py   # 多类型端到端生成
│       ├── test_illustration_agent.py
│       ├── test_llm_agents.py       # 各 Agent LLM 路径
│       ├── test_llm_client.py       # 文本 LLM 客户端
│       ├── test_vision_client.py    # 视觉 LLM 客户端
│       ├── test_image_client.py     # 图像生成客户端
│       ├── test_asset_store.py      # 资产持久化
│       ├── test_refine.py           # HTML refine
│       ├── test_schema.py           # Schema 验证
│       ├── test_sse_events.py       # SSE 事件
│       ├── test_layout_adjustments.py
│       ├── test_prompt_baseline.py
│       └── test_retry.py
├── docs/
│   ├── PostAI_Experiment_Report.md  # 项目论文
│   └── 实验报告.md
├── slides/
│   ├── slides.md                    # 演示 PPT（Slidev）
│   └── ...
├── frontend/
│   ├── index.html                   # 网页演示入口
│   ├── app.js                       # 前端逻辑
│   └── styles.css
├── generated/                       # 生成的 HTML + PNG 资产
└── logs/                            # 运行日志
```
