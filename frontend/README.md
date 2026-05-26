# Frontend Demo

一个零依赖静态演示页，用于联调 PostAI 后端接口。

## 功能入口

- 健康检查：`GET /health`
- 快速生成：`POST /api/v1/generate`
- 流式生成：`POST /api/v1/generate/stream`

## 使用方式

1. 先启动后端（默认 `http://127.0.0.1:8000`）。
2. 进入 `frontend/` 目录。
3. 用任意静态服务器打开本目录，例如：

```bash
cd frontend
python -m http.server 5500
```

4. 浏览器访问 `http://127.0.0.1:5500`。

## 文件说明

- `index.html`：页面结构
- `styles.css`：视觉样式与响应式布局
- `app.js`：接口调用与流式事件解析
